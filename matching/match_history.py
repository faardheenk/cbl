#!/usr/bin/env python3
"""
Match History Layer — Pre-places rows based on a previous output file.

When a user saves their reconciliation output after moving rows between
buckets, that output becomes the source of truth (the "matrix"). This
module reads the previous output, extracts fingerprints from each sheet,
and automatically places matching rows into their previously assigned
buckets before the comparison passes run.
"""

import pandas as pd
import logging
import io
import datetime as dt

logger = logging.getLogger(__name__)

# ── Fingerprint Exclude List ─────────────────────────────────────────────
# Only columns the BACKEND creates during preprocessing, tracking, and passes.
# The frontend MUST adopt this same list (plus any frontend-only columns).
#
# After exclusion, the fingerprint is built from:
#   - Original data columns from the uploaded Excel file
#   - Mapped columns (PlacingNo, PolicyNo, ClientName, ProcessedAmount, etc.)
#
# For insurer rows the _INSURER suffix is stripped BEFORE this list is applied,
# so entries like "PlacingNo_Clean" cover both sides.
FINGERPRINT_EXCLUDE_COLUMNS = {
    # ── preprocess() — cleaned / computed columns ──
    "PlacingNo_Clean",
    "PolicyNo_Clean",
    "PolicyNo_2_Clean",           # insurer-side (after _INSURER strip)
    "ProcessedAmount_Clean",
    "ClientName_Clean",

    # ── initialize_tracking() — match state columns ──
    "match_status",
    "match_pass",
    "match_reason",
    "matched_insurer_indices",
    "matched_amtdue_total",
    "Amount Difference",
    "partial_candidates_indices",
    "match_resolved_in_pass",
    "partial_resolved_in_pass",

    # ── passes — columns set during matching ──
    "group_id",
    "corporate_root",
    "match_confidence",

    # ── internal / temporary ──
    "_source_sheet",
    "_fingerprint",

    # ── user annotations (preserved via history, not part of identity) ──
    "Remarks",
}

# Sentinel value for no-match pre-placed rows — prevents passes from processing them.
# Must be renamed back to "No Match" after all passes via finalize_history_no_match().
HISTORY_NO_MATCH_SENTINEL = "_History_No_Match"

# Sentinel prefix for dynamic bucket pre-placed rows.
# Converted back to the BucketKey after all passes via finalize_history_dynamic_buckets().
DYNAMIC_BUCKET_SENTINEL_PREFIX = "_DynamicBucket_"


def generate_fingerprint(row: pd.Series) -> str:
    """Generate a fingerprint for a row by concatenating all non-metadata
    column values, sorted alphabetically by column name.

    Value formatting rules (aligned with frontend):
      - NaN / None → ""
      - datetime / Timestamp → DD/MM/YYYY  (e.g. "03/06/2025")
      - float equal to int → str(int)       (e.g. 5000.0 → "5000")
      - everything else → str(val)
    """
    keys = sorted([k for k in row.index if k not in FINGERPRINT_EXCLUDE_COLUMNS])
    values = []
    for k in keys:
        val = row[k]
        if pd.isna(val) or val is None:
            values.append("")
        elif isinstance(val, (pd.Timestamp, dt.datetime)):
            values.append(val.strftime("%d/%m/%Y"))
        elif isinstance(val, dt.date):
            values.append(val.strftime("%d/%m/%Y"))
        elif isinstance(val, float) and val == int(val):
            values.append(str(int(val)))
        else:
            values.append(str(val))
    return "|".join(values)


def generate_fingerprints_for_df(df: pd.DataFrame) -> pd.Series:
    """Generate fingerprints for all rows in a DataFrame."""
    return df.apply(generate_fingerprint, axis=1)


def read_previous_output(prev_output_source):
    """Read a previous output file and extract placement records.

    Each sheet maps to a target bucket. Rows are grouped by group_id
    within each sheet. Fingerprints (_fingerprint, _fingerprint_INSURER)
    identify which rows from the new data should be pre-placed.

    Args:
        prev_output_source: File content (bytes), file path (str), or None.

    Returns:
        list[dict]: Placement records, each with target_bucket,
            cbl_fingerprints, insurer_fingerprints, group_id,
            cbl_remarks, insurer_remarks.
    """
    if prev_output_source is None:
        return []

    try:
        if isinstance(prev_output_source, bytes):
            xls = pd.ExcelFile(io.BytesIO(prev_output_source))
        else:
            xls = pd.ExcelFile(prev_output_source)
    except Exception as e:
        logger.warning(f"[MATRIX] Could not read previous output: {e}")
        return []

    SHEET_TO_BUCKET = {
        "Exact Matches": "exact",
        "Partial Matches": "partial",
        "No Matches CBL": "no-match",
    }
    SKIP_SHEETS = {"No Matches Insurer", "_BucketConfig"}

    placements = []

    for sheet_name in xls.sheet_names:
        if sheet_name in SKIP_SHEETS:
            continue

        target_bucket = SHEET_TO_BUCKET.get(sheet_name, sheet_name)

        try:
            df = pd.read_excel(xls, sheet_name=sheet_name)
        except Exception as e:
            logger.warning(f"[MATRIX] Could not read sheet '{sheet_name}': {e}")
            continue

        if df.empty:
            continue

        has_cbl_fp = "_fingerprint" in df.columns
        has_ins_fp = "_fingerprint_INSURER" in df.columns

        if not has_cbl_fp and not has_ins_fp:
            logger.warning(f"[MATRIX] Sheet '{sheet_name}' has no fingerprint columns — skipping")
            continue

        has_group = "group_id" in df.columns
        has_remarks = "Remarks" in df.columns
        has_ins_remarks = "Remarks_INSURER" in df.columns

        grouped_indices = {}
        ungrouped_indices = []

        for idx in df.index:
            gk = None
            if has_group:
                val = df.at[idx, "group_id"]
                if pd.notna(val) and str(val).strip():
                    gk = str(val).strip()
            if gk is not None:
                grouped_indices.setdefault(gk, []).append(idx)
            else:
                ungrouped_indices.append(idx)

        for group_id, indices in grouped_indices.items():
            subset = df.loc[indices]
            cbl_fps = []
            ins_fps = []
            cbl_remarks_list = []
            ins_remarks_list = []

            if has_cbl_fp:
                cbl_fps = [fp for fp in subset["_fingerprint"] if pd.notna(fp) and str(fp).strip()]
            if has_ins_fp:
                ins_fps = [fp for fp in subset["_fingerprint_INSURER"] if pd.notna(fp) and str(fp).strip()]
            if has_remarks:
                cbl_remarks_list = [r if pd.notna(r) else "" for r in subset["Remarks"]]
            if has_ins_remarks:
                ins_remarks_list = [r if pd.notna(r) else "" for r in subset["Remarks_INSURER"]]

            if cbl_fps or ins_fps:
                placements.append({
                    "target_bucket": target_bucket,
                    "cbl_fingerprints": cbl_fps,
                    "insurer_fingerprints": ins_fps,
                    "group_id": group_id,
                    "cbl_remarks": cbl_remarks_list,
                    "insurer_remarks": ins_remarks_list,
                })

        for idx in ungrouped_indices:
            row = df.loc[idx]
            cbl_fps = []
            ins_fps = []

            if has_cbl_fp and pd.notna(row.get("_fingerprint")) and str(row["_fingerprint"]).strip():
                cbl_fps = [str(row["_fingerprint"])]
            if has_ins_fp and pd.notna(row.get("_fingerprint_INSURER")) and str(row["_fingerprint_INSURER"]).strip():
                ins_fps = [str(row["_fingerprint_INSURER"])]

            cbl_rem = [str(row.get("Remarks", ""))] if has_remarks else []
            ins_rem = [str(row.get("Remarks_INSURER", ""))] if has_ins_remarks else []

            if cbl_fps or ins_fps:
                placements.append({
                    "target_bucket": target_bucket,
                    "cbl_fingerprints": cbl_fps,
                    "insurer_fingerprints": ins_fps,
                    "group_id": None,
                    "cbl_remarks": cbl_rem,
                    "insurer_remarks": ins_rem,
                })

    logger.info(f"[MATRIX] Read {len(placements)} placement records from previous output ({len(xls.sheet_names)} sheets)")
    return placements


def apply_previous_output(cbl_df, insurer_df, prev_output_source, global_tracker=None, dynamic_buckets=None):
    """
    Pre-place rows based on a previous output file (the matrix).

    Called AFTER preprocessing + initialize_tracking, BEFORE any matching passes.

    Args:
        cbl_df: Preprocessed CBL DataFrame with tracking columns initialized.
        insurer_df: Preprocessed insurer DataFrame (columns have _INSURER suffix).
        prev_output_source: Previous output file content (bytes), path (str), or None.
        global_tracker: GlobalMatchTracker instance.
        dynamic_buckets: list of {"BucketName": str, "BucketKey": str, "Rematch": bool} or None.

    Returns:
        tuple: (cbl_df, insurer_df, summary_dict, rematch_stash, insurer_only_placements)
    """
    placements = read_previous_output(prev_output_source)

    if not placements:
        logger.info("[MATRIX] No placement records found — skipping pre-placement")
        return cbl_df, insurer_df, {"exact": 0, "partial": 0, "no-match": 0}, [], {}

    logger.info(f"[MATRIX] Found {len(placements)} placement records — matching against new data...")

    if "_fingerprint" not in cbl_df.columns or "_fingerprint_INSURER" not in insurer_df.columns:
        logger.warning("[MATRIX] Canonical fingerprints not found — generating on the fly")
        cbl_df["_fingerprint"] = generate_fingerprints_for_df(cbl_df)
        insurer_cols_stripped = {
            col: col[:-len("_INSURER")] if col.endswith("_INSURER") else col
            for col in insurer_df.columns
        }
        fp_insurer_source = insurer_df.rename(columns=insurer_cols_stripped)
        insurer_df["_fingerprint_INSURER"] = generate_fingerprints_for_df(fp_insurer_source)

    cbl_fp_map = {}
    for idx, fp in cbl_df["_fingerprint"].items():
        cbl_fp_map.setdefault(fp, []).append(idx)

    insurer_fp_map = {}
    for idx, fp in insurer_df["_fingerprint_INSURER"].items():
        insurer_fp_map.setdefault(fp, []).append(idx)

    logger.info(f"[MATRIX] Unique CBL fingerprints: {len(cbl_fp_map)}, Unique insurer fingerprints: {len(insurer_fp_map)}")

    claimed_cbl = set()
    claimed_insurer = set()
    summary = {"exact": 0, "partial": 0, "no-match": 0}
    group_counter = 0

    dynamic_bucket_keys = set()
    rematch_bucket_keys = set()
    bucket_name_to_key = {}
    if dynamic_buckets:
        dynamic_bucket_keys = {b["BucketKey"] for b in dynamic_buckets}
        rematch_bucket_keys = {b["BucketKey"] for b in dynamic_buckets if b.get("Rematch", False)}
        bucket_name_to_key = {b["BucketName"]: b["BucketKey"] for b in dynamic_buckets}
        for key in dynamic_bucket_keys:
            summary[key] = 0
    valid_targets = {"exact", "partial", "no-match"} | dynamic_bucket_keys

    rematch_stash = []
    insurer_only_placements = {}

    for record in placements:
        target = record["target_bucket"]
        target = bucket_name_to_key.get(target, target)
        if target not in valid_targets:
            logger.warning(f"[MATRIX] Unknown target bucket '{target}' — skipping")
            continue

        entry_cbl_indices = []
        entry_insurer_indices = []
        cbl_remarks_list = record.get("cbl_remarks", [])
        ins_remarks_list = record.get("insurer_remarks", [])

        for fp_idx, fp in enumerate(record["cbl_fingerprints"]):
            if fp in cbl_fp_map:
                for idx in cbl_fp_map[fp]:
                    if idx not in claimed_cbl:
                        entry_cbl_indices.append(idx)
                        claimed_cbl.add(idx)
                        if fp_idx < len(cbl_remarks_list) and cbl_remarks_list[fp_idx]:
                            cbl_df.at[idx, "Remarks"] = cbl_remarks_list[fp_idx]
                        break

        for fp_idx, fp in enumerate(record["insurer_fingerprints"]):
            if fp in insurer_fp_map:
                for idx in insurer_fp_map[fp]:
                    if idx not in claimed_insurer:
                        entry_insurer_indices.append(idx)
                        claimed_insurer.add(idx)
                        if fp_idx < len(ins_remarks_list) and ins_remarks_list[fp_idx]:
                            insurer_df.at[idx, "Remarks_INSURER"] = ins_remarks_list[fp_idx]
                        break

        if not entry_cbl_indices and not entry_insurer_indices:
            continue

        # Insurer-only placement to a dynamic bucket
        if not entry_cbl_indices and entry_insurer_indices and target in dynamic_bucket_keys:
            insurer_only_placements.setdefault(target, set()).update(entry_insurer_indices)
            if global_tracker:
                global_tracker.mark_matrix_used(entry_insurer_indices)
            logger.info(f"[MATRIX] Insurer-only: {len(entry_insurer_indices)} insurer rows -> {target}")
            continue

        # Rematchable bucket — stash for re-matching
        if target in rematch_bucket_keys:
            rematch_stash.append({
                "target_bucket": target,
                "cbl_indices": list(entry_cbl_indices),
                "insurer_indices": list(entry_insurer_indices),
            })
            for cbl_idx in entry_cbl_indices:
                claimed_cbl.discard(cbl_idx)
            for ins_idx in entry_insurer_indices:
                claimed_insurer.discard(ins_idx)
            summary[target] += len(entry_cbl_indices)
            continue

        # Standard placement
        if target == "exact":
            match_status = "Exact Match"
        elif target == "partial":
            match_status = "Partial Match"
        elif target == "no-match":
            match_status = HISTORY_NO_MATCH_SENTINEL
        else:
            match_status = f"{DYNAMIC_BUCKET_SENTINEL_PREFIX}{target}"

        group_id = record.get("group_id")
        if group_id is None and len(entry_cbl_indices) > 1:
            group_id = f"MATRIX_{group_counter}"
            group_counter += 1

        for cbl_idx in entry_cbl_indices:
            cbl_df.at[cbl_idx, "match_status"] = match_status
            cbl_df.at[cbl_idx, "matched_insurer_indices"] = list(entry_insurer_indices)
            cbl_df.at[cbl_idx, "match_reason"] = f"Matrix pre-placed ({target})"
            cbl_df.at[cbl_idx, "match_pass"] = ["matrix"]
            cbl_df.at[cbl_idx, "match_resolved_in_pass"] = "matrix"
            if group_id is not None:
                cbl_df.at[cbl_idx, "group_id"] = group_id

        if global_tracker and entry_insurer_indices:
            global_tracker.mark_matrix_used(entry_insurer_indices)

        summary[target] += len(entry_cbl_indices)

    total = sum(summary.values())
    rematch_cbl_count = sum(len(s["cbl_indices"]) for s in rematch_stash)
    insurer_only_count = sum(len(v) for v in insurer_only_placements.values())
    logger.info(f"[MATRIX] ========== MATRIX PRE-PLACEMENT SUMMARY ==========")
    logger.info(f"[MATRIX] Exact:    {summary['exact']} CBL rows pre-placed")
    logger.info(f"[MATRIX] Partial:  {summary['partial']} CBL rows pre-placed")
    logger.info(f"[MATRIX] No Match: {summary['no-match']} CBL rows pre-placed")
    for key in dynamic_bucket_keys:
        if summary.get(key, 0) > 0:
            label = f"{key} (rematch)" if key in rematch_bucket_keys else key
            logger.info(f"[MATRIX] {label}: {summary[key]} CBL rows — {'stashed for re-matching' if key in rematch_bucket_keys else 'pre-placed'}")
        if key in insurer_only_placements:
            logger.info(f"[MATRIX] {key}: {len(insurer_only_placements[key])} insurer-only rows placed")
    logger.info(f"[MATRIX] Total:    {total} CBL rows processed ({rematch_cbl_count} stashed for re-matching)")
    if insurer_only_count:
        logger.info(f"[MATRIX] Insurer-only placements: {insurer_only_count} insurer rows across {len(insurer_only_placements)} bucket(s)")
    logger.info(f"[MATRIX] =====================================================")

    return cbl_df, insurer_df, summary, rematch_stash, insurer_only_placements


def finalize_history_no_match(cbl_df):
    """
    Convert sentinel _History_No_Match status back to "No Match" after all passes.

    Must be called after all matching passes complete but before output generation.

    Args:
        cbl_df: CBL DataFrame that may contain sentinel values.

    Returns:
        cbl_df: Updated DataFrame with sentinels replaced.
    """
    mask = cbl_df["match_status"] == HISTORY_NO_MATCH_SENTINEL
    count = mask.sum()
    if count > 0:
        cbl_df.loc[mask, "match_status"] = "No Match"
        logger.info(f"Finalized {count} history no-match rows")
    return cbl_df


def finalize_history_dynamic_buckets(cbl_df):
    """
    Convert _DynamicBucket_* sentinels back to their BucketKey values after all passes.

    Must be called after all matching passes complete but before output generation.

    Args:
        cbl_df: CBL DataFrame that may contain dynamic bucket sentinel values.

    Returns:
        cbl_df: Updated DataFrame with sentinels replaced by BucketKey values.
    """
    mask = cbl_df["match_status"].str.startswith(DYNAMIC_BUCKET_SENTINEL_PREFIX, na=False)
    count = mask.sum()
    if count > 0:
        cbl_df.loc[mask, "match_status"] = cbl_df.loc[mask, "match_status"].str.replace(
            DYNAMIC_BUCKET_SENTINEL_PREFIX, "", n=1
        )
        logger.info(f"Finalized {count} history dynamic-bucket rows")
    return cbl_df


def finalize_rematch_buckets(cbl_df, rematch_stash):
    """
    Place stashed rematch rows back into their original dynamic bucket
    if the matching passes did not match them.

    Rows that were matched by a pass (match_status != 'No Match') are left
    where the pass put them. Rows still unmatched are returned to their
    original bucket so the user sees them in the same place as before.

    Must be called after all matching passes and after finalize_history_no_match.

    Args:
        cbl_df: CBL DataFrame after all passes have run.
        rematch_stash: list of dicts from apply_match_history, each with
            target_bucket, cbl_indices, insurer_indices.

    Returns:
        cbl_df: Updated DataFrame.
    """
    if not rematch_stash:
        return cbl_df

    matched_by_pass = 0
    returned_to_bucket = 0

    for stash_entry in rematch_stash:
        target = stash_entry["target_bucket"]
        cbl_indices = stash_entry["cbl_indices"]
        insurer_indices = stash_entry["insurer_indices"]

        for cbl_idx in cbl_indices:
            current_status = cbl_df.at[cbl_idx, "match_status"]
            if current_status == "No Match":
                cbl_df.at[cbl_idx, "match_status"] = target
                cbl_df.at[cbl_idx, "matched_insurer_indices"] = list(insurer_indices)
                cbl_df.at[cbl_idx, "match_reason"] = f"History rematch fallback ({target})"
                cbl_df.at[cbl_idx, "match_pass"] = ["history-rematch"]
                cbl_df.at[cbl_idx, "match_resolved_in_pass"] = "history-rematch"
                returned_to_bucket += 1
                logger.info(f"[REMATCH] CBL idx={cbl_idx} still unmatched — returned to bucket '{target}'")
            else:
                matched_by_pass += 1
                logger.info(f"[REMATCH] CBL idx={cbl_idx} matched by pass (status='{current_status}') — keeping")

    logger.info(f"[REMATCH] ========== REMATCH SUMMARY ==========")
    logger.info(f"[REMATCH] Matched by passes:    {matched_by_pass}")
    logger.info(f"[REMATCH] Returned to bucket:   {returned_to_bucket}")
    logger.info(f"[REMATCH] ==========================================")

    return cbl_df
