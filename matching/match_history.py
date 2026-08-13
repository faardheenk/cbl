#!/usr/bin/env python3
"""
Match History Layer — Pre-places rows based on a previous output file.

When a user saves their reconciliation output after moving rows between
buckets, that output becomes the source of truth (the "matrix"). This
module reads the previous output, identifies which of its rows are present
in the new data, and places them back into the bucket the user filed them
in before the comparison passes run.

Rows are identified by a business key built from the normalised identity
columns (see CBL_KEY_COLUMNS). The key is recomputed from both the previous
output and the new data on every run, so there is no stored identity string
that can go stale when this code changes.
"""

import ast
import pandas as pd
import logging
import io
import math
import datetime as dt

logger = logging.getLogger(__name__)

# ── Row Identity ─────────────────────────────────────────────────────────
# A row is identified by the columns preprocess() has already normalised —
# uppercased, trimmed, regex-cleaned and numerically coerced. These are the
# same columns the matching passes treat as a row's identity, and they are
# written into every output sheet.
#
# Deliberately NOT included:
#   - Presentation columns (brokerage, dates, currency) — these change
#     without the row becoming a different transaction.
#   - Annotation columns the broker fills in ("Timing Difference",
#     "Correction to be done by CBL", Remarks) — annotating a row must not
#     change its identity.
#
# The key is always recomputed on both sides from these columns. Never
# persist a key and read it back: a stored key is produced by whatever
# version of this code wrote the file, and would silently stop matching
# the moment the normalisation here changes.
CBL_KEY_COLUMNS = ["PlacingNo_Clean", "PolicyNo_Clean", "ProcessedAmount_Clean"]
INSURER_KEY_COLUMNS = [
    "PlacingNo_Clean_INSURER",
    "PolicyNo_Clean_INSURER",
    "ProcessedAmount_Clean_INSURER",
]

# ── Fingerprint Exclude List ─────────────────────────────────────────────
# NOTE: fingerprints are no longer used to match rows — that is done by the
# business key above. generate_fingerprint() is retained only to populate the
# _fingerprint / _fingerprint_INSURER columns that the frontend still reads.
#
# Only columns the BACKEND creates during preprocessing, tracking, and passes.
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


def build_row_key(row, columns):
    """Build a row's identity key, or None if the row carries no identity.

    Values are normalised so that a value written to Excel and read back
    produces the same key:
      - NaN / None / non-finite → ""
      - whole-number float → str(int)   (1908.0 → "1908")
      - other float → 2dp               (these are currency amounts)
      - everything else → str(val).strip()

    Returns None when every component is empty — such a row has no identity
    and must not be matched, or all identity-less rows would collide.
    """
    parts = []
    for col in columns:
        val = row.get(col)
        if val is None or (not isinstance(val, (list, dict, set)) and pd.isna(val)):
            parts.append("")
        elif isinstance(val, float):
            if not math.isfinite(val):
                parts.append("")
            elif val == int(val):
                parts.append(str(int(val)))
            else:
                parts.append(f"{val:.2f}")
        else:
            parts.append(str(val).strip())

    if not any(parts):
        return None
    return "|".join(parts)


def build_key_map(df, columns):
    """Map row key -> list of DataFrame indices."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        logger.warning(f"[MATRIX] Identity columns missing {missing} — cannot match rows")
        return {}

    key_map = {}
    for idx in df.index:
        key = build_row_key(df.loc[idx], columns)
        if key is not None:
            key_map.setdefault(key, []).append(idx)
    return key_map


def read_bucket_config(xls):
    """Map output sheet name -> BucketKey using the _BucketConfig sheet.

    Excel truncates sheet names at 31 characters, so a bucket named
    "Correction to be done by insurer" is written to a sheet called
    "Correction to be done by insure". _BucketConfig records the real
    mapping, which makes a previous output self-describing rather than
    dependent on the caller passing a matching dynamic_buckets list.
    """
    if "_BucketConfig" not in xls.sheet_names:
        return {}
    try:
        cfg = pd.read_excel(xls, sheet_name="_BucketConfig")
    except Exception as e:
        logger.warning(f"[MATRIX] Could not read _BucketConfig: {e}")
        return {}

    if not {"SheetName", "BucketKey"}.issubset(cfg.columns):
        return {}

    return {
        str(r["SheetName"]).strip(): str(r["BucketKey"]).strip()
        for _, r in cfg.iterrows()
        if pd.notna(r["SheetName"]) and pd.notna(r["BucketKey"])
    }


def count_insurer_indices(value):
    """How many insurer rows a previous output's row claims.

    matched_insurer_indices round-trips through Excel as the repr of a list.
    The indices themselves point into the previous run's insurer file and are
    meaningless now — only the count is used, to tell how many continuation
    rows follow an anchor.
    """
    if value is None or (not isinstance(value, (list, tuple, set)) and pd.isna(value)):
        return 0
    if isinstance(value, (list, tuple, set)):
        return len(value)
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return 0
    return len(parsed) if isinstance(parsed, (list, tuple, set)) else 0


def read_previous_output(prev_output_source):
    """Read a previous output file and extract placement records.

    Each sheet maps to a target bucket — via _BucketConfig where present,
    which survives Excel's 31-character sheet name truncation. Rows are
    grouped by group_id within each sheet, and identified by the business
    key built from their normalised identity columns.

    Group ids are only used to keep rows that belong together in one
    placement, so the id read from the file is discarded and a fresh
    MATRIX_GRP_n is issued. The ids in the previous output were minted by
    that run's own sequential counters (MATCH_n, MERGED_GROUP_n,
    NAME_GROUP_n), and this run's counters restart at 1 — reusing them
    would let an unrelated match land in a restored group. Re-issuing
    rather than prefixing keeps this stable across generations: a
    previous output may legitimately contain both "MATCH_68" and a
    restored "MATRIX_..._MATCH_68", which any prefix scheme would
    eventually collapse back together.

    Args:
        prev_output_source: File content (bytes), file path (str), or None.

    Returns:
        list[dict]: Placement records, each with target_bucket, cbl_keys,
            insurer_keys, group_id, cbl_remarks, insurer_remarks.
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
    SKIP_SHEETS = {"No Matches Insurer", "_BucketConfig", "Summary"}

    sheet_to_bucket_key = read_bucket_config(xls)
    placements = []
    restored_group_counter = 0

    for sheet_name in xls.sheet_names:
        if sheet_name in SKIP_SHEETS:
            continue

        # Base buckets are fixed; dynamic buckets resolve through
        # _BucketConfig, falling back to the sheet name itself.
        target_bucket = (
            SHEET_TO_BUCKET.get(sheet_name)
            or sheet_to_bucket_key.get(sheet_name, sheet_name)
        )

        try:
            df = pd.read_excel(xls, sheet_name=sheet_name)
        except Exception as e:
            logger.warning(f"[MATRIX] Could not read sheet '{sheet_name}': {e}")
            continue

        if df.empty:
            continue

        has_cbl_key = all(c in df.columns for c in CBL_KEY_COLUMNS)
        has_ins_key = all(c in df.columns for c in INSURER_KEY_COLUMNS)

        if not has_cbl_key and not has_ins_key:
            logger.warning(f"[MATRIX] Sheet '{sheet_name}' has no identity columns — skipping")
            continue

        has_group = "group_id" in df.columns
        has_remarks = "Remarks" in df.columns
        has_ins_remarks = "Remarks_INSURER" in df.columns

        has_indices = "matched_insurer_indices" in df.columns

        # Rows that belong to one match are tied together by group_id, but
        # only cluster matches ever got one. A combination match (one CBL row
        # against several insurer rows) is written as an anchor row plus
        # insurer-only continuation rows, historically with no group_id at
        # all — reading those as independent records silently drops their
        # amounts from the match. The anchor states how many insurer rows it
        # holds, so claim exactly that many following CBL-less rows.
        grouped_indices = {}
        ungrouped_blocks = []
        owed = 0

        for idx in df.index:
            gk = None
            if has_group:
                val = df.at[idx, "group_id"]
                if pd.notna(val) and str(val).strip():
                    gk = str(val).strip()

            if gk is not None:
                grouped_indices.setdefault(gk, []).append(idx)
                owed = 0
                continue

            if has_cbl_key and build_row_key(df.loc[idx], CBL_KEY_COLUMNS) is not None:
                ungrouped_blocks.append([idx])
                owed = max(0, count_insurer_indices(df.at[idx, "matched_insurer_indices"]) - 1) if has_indices else 0
                continue

            is_insurer_row = has_ins_key and build_row_key(df.loc[idx], INSURER_KEY_COLUMNS) is not None
            if is_insurer_row and owed > 0 and ungrouped_blocks:
                ungrouped_blocks[-1].append(idx)
                owed -= 1
            else:
                # A standalone insurer-only placement, not a continuation.
                ungrouped_blocks.append([idx])
                owed = 0

        def keys_for(indices, columns, enabled):
            if not enabled:
                return []
            keys = []
            for i in indices:
                key = build_row_key(df.loc[i], columns)
                if key is not None:
                    keys.append(key)
            return keys

        for indices in grouped_indices.values():
            cbl_keys = keys_for(indices, CBL_KEY_COLUMNS, has_cbl_key)
            ins_keys = keys_for(indices, INSURER_KEY_COLUMNS, has_ins_key)
            if not cbl_keys and not ins_keys:
                continue

            restored_group_counter += 1
            subset = df.loc[indices]
            placements.append({
                "target_bucket": target_bucket,
                "cbl_keys": cbl_keys,
                "insurer_keys": ins_keys,
                "group_id": f"MATRIX_GRP_{restored_group_counter}",
                "cbl_remarks": [r if pd.notna(r) else "" for r in subset["Remarks"]] if has_remarks else [],
                "insurer_remarks": [r if pd.notna(r) else "" for r in subset["Remarks_INSURER"]] if has_ins_remarks else [],
            })

        for block in ungrouped_blocks:
            cbl_keys = keys_for(block, CBL_KEY_COLUMNS, has_cbl_key)
            ins_keys = keys_for(block, INSURER_KEY_COLUMNS, has_ins_key)
            if not cbl_keys and not ins_keys:
                continue

            # A multi-row block is one match spread over several rows. Give it
            # an id so it is written back as a group and stays readable next run.
            group_id = None
            if len(block) > 1:
                restored_group_counter += 1
                group_id = f"MATRIX_GRP_{restored_group_counter}"

            subset = df.loc[block]
            placements.append({
                "target_bucket": target_bucket,
                "cbl_keys": cbl_keys,
                "insurer_keys": ins_keys,
                "group_id": group_id,
                "cbl_remarks": [r if pd.notna(r) else "" for r in subset["Remarks"]] if has_remarks else [],
                "insurer_remarks": [r if pd.notna(r) else "" for r in subset["Remarks_INSURER"]] if has_ins_remarks else [],
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

    # Keys are computed fresh on both sides by the same code, so there is no
    # stored identity that can drift out of step with this implementation.
    cbl_key_map = build_key_map(cbl_df, CBL_KEY_COLUMNS)
    insurer_key_map = build_key_map(insurer_df, INSURER_KEY_COLUMNS)

    logger.info(f"[MATRIX] Unique CBL keys: {len(cbl_key_map)}, Unique insurer keys: {len(insurer_key_map)}")

    if not cbl_key_map and not insurer_key_map:
        logger.warning("[MATRIX] No identity keys could be built — skipping pre-placement")
        return cbl_df, insurer_df, {"exact": 0, "partial": 0, "no-match": 0}, [], {}

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

        for key_idx, key in enumerate(record["cbl_keys"]):
            for idx in cbl_key_map.get(key, []):
                if idx not in claimed_cbl:
                    entry_cbl_indices.append(idx)
                    claimed_cbl.add(idx)
                    if key_idx < len(cbl_remarks_list) and cbl_remarks_list[key_idx]:
                        cbl_df.at[idx, "Remarks"] = cbl_remarks_list[key_idx]
                    break

        for key_idx, key in enumerate(record["insurer_keys"]):
            for idx in insurer_key_map.get(key, []):
                if idx not in claimed_insurer:
                    entry_insurer_indices.append(idx)
                    claimed_insurer.add(idx)
                    if key_idx < len(ins_remarks_list) and ins_remarks_list[key_idx]:
                        insurer_df.at[idx, "Remarks_INSURER"] = ins_remarks_list[key_idx]
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

        # Rematchable — stash for re-matching instead of pre-placing.
        # Includes rematch dynamic buckets, partial, and no-match:
        # partial/no-match rows should be re-evaluated against fresh insurer data
        # each run rather than locked into stale results.
        if target in rematch_bucket_keys or target in ("partial", "no-match"):
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

        # Standard placement (exact matches and non-rematch dynamic buckets)
        if target == "exact":
            match_status = "Exact Match"
        else:
            match_status = f"{DYNAMIC_BUCKET_SENTINEL_PREFIX}{target}"

        group_id = record.get("group_id")
        if group_id is None and len(entry_cbl_indices) > 1:
            # Ungrouped record whose key matched several CBL rows. Distinct
            # prefix from the MATRIX_GRP_n issued by read_previous_output so
            # the two allocators cannot collide.
            group_id = f"MATRIX_KEY_{group_counter}"
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
    logger.info(f"[MATRIX] Partial:  {summary['partial']} CBL rows stashed for re-matching")
    logger.info(f"[MATRIX] No Match: {summary['no-match']} CBL rows stashed for re-matching")
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


def finalize_rematch_buckets(cbl_df, rematch_stash, global_tracker=None):
    """
    Place stashed rematch rows back into their original bucket
    if the matching passes did not match them.

    Rows that were matched by a pass (match_status != 'No Match') are left
    where the pass put them. Rows still unmatched are returned to their
    original bucket so the user sees them in the same place as before.

    Insurer indices that were claimed by a pass during rematching are
    excluded from the fallback to prevent the same insurer row appearing
    in multiple places in the output.

    Must be called after all matching passes and after finalize_history_no_match.

    Args:
        cbl_df: CBL DataFrame after all passes have run.
        rematch_stash: list of dicts from apply_previous_output, each with
            target_bucket, cbl_indices, insurer_indices.
        global_tracker: GlobalMatchTracker instance for checking claimed insurer rows.

    Returns:
        cbl_df: Updated DataFrame.
    """
    if not rematch_stash:
        return cbl_df

    all_claimed_insurer = set()
    if global_tracker:
        all_claimed_insurer = (
            global_tracker.matrix_used_insurer
            | global_tracker.exact_used_insurer
            | global_tracker.partial_used_insurer
        )

    matched_by_pass = 0
    returned_to_bucket = 0
    demoted_to_no_match = 0

    FALLBACK_STATUS = {
        "partial": "Partial Match",
        "no-match": "No Match",
    }

    for stash_entry in rematch_stash:
        target = stash_entry["target_bucket"]
        cbl_indices = stash_entry["cbl_indices"]
        insurer_indices = stash_entry["insurer_indices"]

        for cbl_idx in cbl_indices:
            current_status = cbl_df.at[cbl_idx, "match_status"]
            if current_status != "No Match":
                matched_by_pass += 1
                logger.info(f"[REMATCH] CBL idx={cbl_idx} matched by pass (status='{current_status}') — keeping")
                continue

            available_insurer = [idx for idx in insurer_indices if idx not in all_claimed_insurer]

            if target == "partial" and not available_insurer:
                # Every insurer row that justified the partial was claimed by
                # a stronger match this run, so there is nothing left to be
                # partial against. Record that plainly here rather than let
                # output generation silently demote a insurer-less
                # "Partial Match" and leave a contradictory match_reason.
                cbl_df.at[cbl_idx, "match_status"] = "No Match"
                cbl_df.at[cbl_idx, "match_reason"] = (
                    "Matrix rematch fallback (partial) — insurer rows reclaimed by a stronger match"
                )
                demoted_to_no_match += 1
                logger.info(f"[REMATCH] CBL idx={cbl_idx} was partial but all insurer rows reclaimed — now No Match")
            else:
                cbl_df.at[cbl_idx, "match_status"] = FALLBACK_STATUS.get(target, target)
                cbl_df.at[cbl_idx, "match_reason"] = f"Matrix rematch fallback ({target})"
                returned_to_bucket += 1
                logger.info(f"[REMATCH] CBL idx={cbl_idx} still unmatched — returned to bucket '{target}' with {len(available_insurer)}/{len(insurer_indices)} insurer rows")

            cbl_df.at[cbl_idx, "matched_insurer_indices"] = available_insurer
            cbl_df.at[cbl_idx, "match_pass"] = ["matrix-rematch"]
            cbl_df.at[cbl_idx, "match_resolved_in_pass"] = "matrix-rematch"

    logger.info(f"[REMATCH] ========== REMATCH SUMMARY ==========")
    logger.info(f"[REMATCH] Matched by passes:    {matched_by_pass}")
    logger.info(f"[REMATCH] Returned to bucket:   {returned_to_bucket}")
    if demoted_to_no_match:
        logger.info(f"[REMATCH] Partial -> No Match:  {demoted_to_no_match} (insurer rows reclaimed)")
    logger.info(f"[REMATCH] ==========================================")

    return cbl_df
