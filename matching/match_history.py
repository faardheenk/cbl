#!/usr/bin/env python3
"""
Match History Layer — Pre-places rows based on previous manual user moves.

When a user manually moves rows between buckets in the reconciliation UI,
fingerprints are recorded in history.xlsx. This module reads that history,
regenerates fingerprints from new data, and automatically places matching
rows into their previously assigned buckets before the comparison passes run.
"""

import pandas as pd
import json
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


def read_match_history(history_source) -> list:
    """Read match history entries from history.xlsx.

    Args:
        history_source: File path (str), file content (bytes), or None.

    Returns:
        list[dict]: Parsed history entries.
    """
    if history_source is None:
        logger.info("[HISTORY] history_source is None — no history file provided")
        return []

    logger.info(f"[HISTORY] Reading history file...")

    try:
        if isinstance(history_source, bytes):
            df = pd.read_excel(io.BytesIO(history_source), sheet_name="MatchHistory")
        else:
            df = pd.read_excel(history_source, sheet_name="MatchHistory")
    except FileNotFoundError:
        logger.info("[HISTORY] History file not found — no pre-placement will be applied")
        return []
    except ValueError as e:
        logger.info(f"[HISTORY] History sheet not found: {e}")
        return []
    except Exception as e:
        logger.warning(f"[HISTORY] Could not read match history: {e}")
        return []

    if df.empty:
        logger.info("[HISTORY] MatchHistory sheet is empty")
        return []

    logger.info(f"[HISTORY] MatchHistory sheet has {len(df)} rows, columns: {list(df.columns)}")

    # JSON array fields that can be split across continuation rows
    _ARRAY_FIELDS = [
        ("CblFingerprints", "cbl_fingerprints"),
        ("InsurerFingerprints", "insurer_fingerprints"),
        ("CblRemarks", "cbl_remarks"),
        ("InsurerRemarks", "insurer_remarks"),
        ("TargetCblFingerprints", "target_cbl_fingerprints"),
        ("TargetInsurerFingerprints", "target_insurer_fingerprints"),
        ("OrphanedCblFingerprints", "orphaned_cbl_fingerprints"),
        ("OrphanedInsurerFingerprints", "orphaned_insurer_fingerprints"),
    ]

    def _parse_json_array(row, col_name):
        try:
            return json.loads(row[col_name]) if pd.notna(row.get(col_name)) else []
        except (json.JSONDecodeError, TypeError):
            return []

    entries = []
    current_entry = None
    entry_idx = -1

    for i, (_, row) in enumerate(df.iterrows()):
        action_type = str(row.get("ActionType", "move")).strip().lower() if pd.notna(row.get("ActionType")) else "move"

        if action_type == "continuation":
            if current_entry is None:
                logger.warning(f"[HISTORY] Row #{i} is a continuation with no preceding entry — skipping")
                continue

            for excel_col, key in _ARRAY_FIELDS:
                current_entry[key] += _parse_json_array(row, excel_col)

            logger.info(f"[HISTORY] Row #{i} continuation merged into entry #{entry_idx}")
            continue

        # New entry — flush previous
        if current_entry is not None:
            entries.append(current_entry)

        entry_idx += 1

        try:
            cbl_fps = json.loads(row["CblFingerprints"]) if pd.notna(row.get("CblFingerprints")) else []
            ins_fps = json.loads(row["InsurerFingerprints"]) if pd.notna(row.get("InsurerFingerprints")) else []
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"[HISTORY] Skipping malformed history entry #{entry_idx}: {e}")
            current_entry = None
            continue

        cbl_remarks = _parse_json_array(row, "CblRemarks")
        ins_remarks = _parse_json_array(row, "InsurerRemarks")
        target_cbl_fps = _parse_json_array(row, "TargetCblFingerprints")
        target_ins_fps = _parse_json_array(row, "TargetInsurerFingerprints")
        orphaned_cbl_fps = _parse_json_array(row, "OrphanedCblFingerprints")
        orphaned_ins_fps = _parse_json_array(row, "OrphanedInsurerFingerprints")

        current_entry = {
            "action_type": action_type,
            "cbl_fingerprints": cbl_fps,
            "insurer_fingerprints": ins_fps,
            "cbl_remarks": cbl_remarks,
            "insurer_remarks": ins_remarks,
            "from_bucket": row.get("FromBucket", ""),
            "target_bucket": row.get("TargetBucket", ""),
            "timestamp": row.get("Timestamp", ""),
            "target_cbl_fingerprints": target_cbl_fps,
            "target_insurer_fingerprints": target_ins_fps,
            "orphaned_cbl_fingerprints": orphaned_cbl_fps,
            "orphaned_insurer_fingerprints": orphaned_ins_fps,
        }

        logger.info(f"[HISTORY] Entry #{entry_idx} ({action_type}): from='{current_entry['from_bucket']}' -> target='{current_entry['target_bucket']}' | "
                     f"CBL fps={len(cbl_fps)}, Insurer fps={len(ins_fps)}")
        if cbl_fps:
            preview = cbl_fps[0][:80] + "..." if len(cbl_fps[0]) > 80 else cbl_fps[0]
            logger.info(f"[HISTORY]   CBL fp[0]: {preview}")
        if ins_fps:
            preview = ins_fps[0][:80] + "..." if len(ins_fps[0]) > 80 else ins_fps[0]
            logger.info(f"[HISTORY]   Insurer fp[0]: {preview}")
        if action_type == "regroup":
            logger.info(f"[HISTORY]   Regroup target CBL fps={len(target_cbl_fps)}, target Insurer fps={len(target_ins_fps)}, "
                         f"orphaned CBL fps={len(orphaned_cbl_fps)}, orphaned Insurer fps={len(orphaned_ins_fps)}")

    # Flush the last entry
    if current_entry is not None:
        entries.append(current_entry)

    return entries


def apply_match_history(cbl_df, insurer_df, history_source, global_tracker=None, dynamic_buckets=None):
    """
    Apply match history to pre-place rows in their correct buckets.

    Called AFTER preprocessing + initialize_tracking, BEFORE any matching passes.

    Args:
        cbl_df: Preprocessed CBL DataFrame with tracking columns initialized.
        insurer_df: Preprocessed insurer DataFrame (columns have _INSURER suffix).
        history_source: Path to history.xlsx (str), file content (bytes), or None.
        global_tracker: GlobalMatchTracker instance.
        dynamic_buckets: list of {"BucketName": str, "BucketKey": str} or None.

    Returns:
        tuple: (cbl_df, insurer_df, summary_dict, rematch_stash, insurer_only_placements)
    """
    history = read_match_history(history_source)

    if not history:
        logger.info("[HISTORY] No match history found — skipping pre-placement")
        return cbl_df, insurer_df, {"exact": 0, "partial": 0, "no-match": 0}, [], {}

    logger.info(f"[HISTORY] Found {len(history)} history entries — using canonical fingerprints...")

    # --- Use pre-generated canonical fingerprints ---
    # Fingerprints are generated by the orchestrator before this function is called.
    # CBL fingerprints are in "_fingerprint", insurer fingerprints in "_fingerprint_INSURER".
    if "_fingerprint" not in cbl_df.columns or "_fingerprint_INSURER" not in insurer_df.columns:
        logger.warning("[HISTORY] Canonical fingerprints not found — generating on the fly")
        cbl_df["_fingerprint"] = generate_fingerprints_for_df(cbl_df)
        insurer_cols_stripped = {
            col: col[:-len("_INSURER")] if col.endswith("_INSURER") else col
            for col in insurer_df.columns
        }
        fp_insurer_source = insurer_df.rename(columns=insurer_cols_stripped)
        insurer_df["_fingerprint_INSURER"] = generate_fingerprints_for_df(fp_insurer_source)

    # --- Log sample fingerprints ---
    logger.info(f"[HISTORY] Using {len(cbl_df)} CBL fingerprints, {len(insurer_df)} insurer fingerprints")
    for i in range(min(3, len(cbl_df))):
        fp = cbl_df["_fingerprint"].iloc[i]
        preview = fp[:120] + "..." if len(fp) > 120 else fp
        logger.info(f"[HISTORY] CBL row {cbl_df.index[i]} fp: {preview}")
    for i in range(min(3, len(insurer_df))):
        fp = insurer_df["_fingerprint_INSURER"].iloc[i]
        preview = fp[:120] + "..." if len(fp) > 120 else fp
        logger.info(f"[HISTORY] Insurer row {insurer_df.index[i]} fp: {preview}")

    # --- Build lookup maps: fingerprint -> [indices] ---
    cbl_fp_map = {}
    for idx, fp in cbl_df["_fingerprint"].items():
        cbl_fp_map.setdefault(fp, []).append(idx)

    insurer_fp_map = {}
    for idx, fp in insurer_df["_fingerprint_INSURER"].items():
        insurer_fp_map.setdefault(fp, []).append(idx)

    logger.info(f"[HISTORY] Unique CBL fingerprints: {len(cbl_fp_map)}, Unique insurer fingerprints: {len(insurer_fp_map)}")

    # --- Match history entries against new data ---
    claimed_cbl = set()
    claimed_insurer = set()
    summary = {"exact": 0, "partial": 0, "no-match": 0}
    history_group_counter = 0

    # Build set of valid target bucket keys (fixed + dynamic)
    dynamic_bucket_keys = set()
    rematch_bucket_keys = set()
    if dynamic_buckets:
        dynamic_bucket_keys = {b["BucketKey"] for b in dynamic_buckets}
        rematch_bucket_keys = {b["BucketKey"] for b in dynamic_buckets if b.get("Rematch", False)}
        for key in dynamic_bucket_keys:
            summary[key] = 0
    valid_targets = {"exact", "partial", "no-match"} | dynamic_bucket_keys

    rematch_stash = []
    insurer_only_placements = {}  # bucket_key -> set of insurer indices (insurer-only moves)

    for entry_idx, entry in enumerate(history):
        target = entry["target_bucket"]
        if target not in valid_targets:
            logger.warning(f"[HISTORY] Unknown target bucket '{target}' — skipping entry #{entry_idx}")
            continue

        action_type = entry.get("action_type", "move")

        entry_cbl_indices = []
        entry_insurer_indices = []
        cbl_remarks_list = entry.get("cbl_remarks", [])
        ins_remarks_list = entry.get("insurer_remarks", [])

        # Match CBL fingerprints (one match per fingerprint)
        cbl_matched = 0
        cbl_missed = 0
        cbl_reclaimed = 0
        for fp_idx, fp in enumerate(entry["cbl_fingerprints"]):
            if fp in cbl_fp_map:
                found_unclaimed = False
                for idx in cbl_fp_map[fp]:
                    if idx not in claimed_cbl:
                        entry_cbl_indices.append(idx)
                        claimed_cbl.add(idx)
                        cbl_matched += 1
                        found_unclaimed = True
                        if fp_idx < len(cbl_remarks_list) and cbl_remarks_list[fp_idx]:
                            cbl_df.at[idx, "Remarks"] = cbl_remarks_list[fp_idx]
                        break
                if not found_unclaimed:
                    idx = cbl_fp_map[fp][0]
                    entry_cbl_indices.append(idx)
                    cbl_reclaimed += 1
                    if fp_idx < len(cbl_remarks_list) and cbl_remarks_list[fp_idx]:
                        cbl_df.at[idx, "Remarks"] = cbl_remarks_list[fp_idx]
            else:
                cbl_missed += 1
                preview = fp[:80] + "..." if len(fp) > 80 else fp
                logger.info(f"[HISTORY] Entry #{entry_idx} CBL fp NOT FOUND: {preview}")

        # Match insurer fingerprints (one match per fingerprint)
        ins_matched = 0
        ins_missed = 0
        ins_reclaimed = 0
        for fp_idx, fp in enumerate(entry["insurer_fingerprints"]):
            if fp in insurer_fp_map:
                found_unclaimed = False
                for idx in insurer_fp_map[fp]:
                    if idx not in claimed_insurer:
                        entry_insurer_indices.append(idx)
                        claimed_insurer.add(idx)
                        ins_matched += 1
                        found_unclaimed = True
                        if fp_idx < len(ins_remarks_list) and ins_remarks_list[fp_idx]:
                            insurer_df.at[idx, "Remarks_INSURER"] = ins_remarks_list[fp_idx]
                        break
                if not found_unclaimed:
                    idx = insurer_fp_map[fp][0]
                    entry_insurer_indices.append(idx)
                    ins_reclaimed += 1
                    if fp_idx < len(ins_remarks_list) and ins_remarks_list[fp_idx]:
                        insurer_df.at[idx, "Remarks_INSURER"] = ins_remarks_list[fp_idx]
            else:
                ins_missed += 1
                preview = fp[:80] + "..." if len(fp) > 80 else fp
                logger.info(f"[HISTORY] Entry #{entry_idx} Insurer fp NOT FOUND: {preview}")

        if cbl_reclaimed or ins_reclaimed:
            logger.info(f"[HISTORY] Entry #{entry_idx} sequential op: re-claimed {cbl_reclaimed} CBL, {ins_reclaimed} insurer rows from earlier entries")

        logger.info(f"[HISTORY] Entry #{entry_idx} ({action_type} -> {target}): CBL {cbl_matched + cbl_reclaimed}/{len(entry['cbl_fingerprints'])} matched, "
                     f"Insurer {ins_matched + ins_reclaimed}/{len(entry['insurer_fingerprints'])} matched")

        if not entry_cbl_indices and not entry_insurer_indices:
            logger.info(f"[HISTORY] Entry #{entry_idx}: NO matches at all — skipping")
            continue

        # ─── REGROUP: merge moved rows into the target group ────────────
        if action_type == "regroup":
            target_cbl_indices = []
            target_insurer_indices = []

            for fp in entry.get("target_cbl_fingerprints", []):
                if fp in cbl_fp_map:
                    found = False
                    for idx in cbl_fp_map[fp]:
                        if idx not in claimed_cbl:
                            target_cbl_indices.append(idx)
                            claimed_cbl.add(idx)
                            found = True
                            break
                    if not found:
                        target_cbl_indices.append(cbl_fp_map[fp][0])

            for fp in entry.get("target_insurer_fingerprints", []):
                if fp in insurer_fp_map:
                    found = False
                    for idx in insurer_fp_map[fp]:
                        if idx not in claimed_insurer:
                            target_insurer_indices.append(idx)
                            claimed_insurer.add(idx)
                            found = True
                            break
                    if not found:
                        target_insurer_indices.append(insurer_fp_map[fp][0])

            # Orphaned rows go to no-match
            orphaned_cbl_indices = []
            orphaned_insurer_indices = []

            for fp in entry.get("orphaned_cbl_fingerprints", []):
                if fp in cbl_fp_map:
                    found = False
                    for idx in cbl_fp_map[fp]:
                        if idx not in claimed_cbl:
                            orphaned_cbl_indices.append(idx)
                            claimed_cbl.add(idx)
                            found = True
                            break
                    if not found:
                        orphaned_cbl_indices.append(cbl_fp_map[fp][0])

            for fp in entry.get("orphaned_insurer_fingerprints", []):
                if fp in insurer_fp_map:
                    found = False
                    for idx in insurer_fp_map[fp]:
                        if idx not in claimed_insurer:
                            orphaned_insurer_indices.append(idx)
                            claimed_insurer.add(idx)
                            found = True
                            break
                    if not found:
                        orphaned_insurer_indices.append(insurer_fp_map[fp][0])

            logger.info(f"[HISTORY] Entry #{entry_idx} regroup: target CBL={len(target_cbl_indices)}, "
                         f"target Insurer={len(target_insurer_indices)}, "
                         f"orphaned CBL={len(orphaned_cbl_indices)}, orphaned Insurer={len(orphaned_insurer_indices)}")

            # Merged group = moved rows + target group rows
            merged_cbl_indices = entry_cbl_indices + target_cbl_indices
            merged_insurer_indices = entry_insurer_indices + target_insurer_indices

            if target == "no-match":
                match_status = HISTORY_NO_MATCH_SENTINEL
            elif target == "exact":
                match_status = "Exact Match"
            elif target == "partial":
                match_status = "Partial Match"
            else:
                match_status = f"{DYNAMIC_BUCKET_SENTINEL_PREFIX}{target}"

            group_id = f"HISTORY_REGROUP_{history_group_counter}"
            history_group_counter += 1

            for cbl_idx in merged_cbl_indices:
                cbl_df.at[cbl_idx, "match_status"] = match_status
                cbl_df.at[cbl_idx, "matched_insurer_indices"] = list(merged_insurer_indices)
                cbl_df.at[cbl_idx, "match_reason"] = f"History regroup ({target})"
                cbl_df.at[cbl_idx, "match_pass"] = ["history"]
                cbl_df.at[cbl_idx, "match_resolved_in_pass"] = "history"
                cbl_df.at[cbl_idx, "group_id"] = group_id
                logger.info(f"[HISTORY] Regroup-placed CBL idx={cbl_idx} -> {match_status} | group={group_id}")

            if global_tracker and merged_insurer_indices:
                global_tracker.mark_matrix_used(merged_insurer_indices)

            # Place orphaned rows into no-match
            for cbl_idx in orphaned_cbl_indices:
                cbl_df.at[cbl_idx, "match_status"] = HISTORY_NO_MATCH_SENTINEL
                cbl_df.at[cbl_idx, "match_reason"] = "History regroup orphan (no-match)"
                cbl_df.at[cbl_idx, "match_pass"] = ["history"]
                logger.info(f"[HISTORY] Orphaned CBL idx={cbl_idx} -> No Match")

            if global_tracker and orphaned_insurer_indices:
                global_tracker.mark_matrix_used(orphaned_insurer_indices)

            summary[target] += len(merged_cbl_indices)
            summary["no-match"] += len(orphaned_cbl_indices)
            continue

        # ─── MOVE: standard bucket pre-placement ────────────────────────
        if target in ("exact", "partial") or target in dynamic_bucket_keys:
            # Insurer-only move (no CBL rows) to a dynamic bucket — always place
            # directly, even for rematchable buckets (re-matching is CBL-centric)
            if not entry_cbl_indices and entry_insurer_indices and target in dynamic_bucket_keys:
                insurer_only_placements.setdefault(target, set()).update(entry_insurer_indices)
                if global_tracker:
                    global_tracker.mark_matrix_used(entry_insurer_indices)
                logger.info(f"[HISTORY] Insurer-only placement: {len(entry_insurer_indices)} insurer rows -> {target}")
                continue

            # Rematchable buckets: stash rows for re-matching instead of pre-placing
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
                logger.info(f"[HISTORY] Stashed {len(entry_cbl_indices)} CBL + {len(entry_insurer_indices)} insurer rows for re-matching (bucket={target})")
                summary[target] += len(entry_cbl_indices)
                continue

            if target == "exact":
                match_status = "Exact Match"
            elif target == "partial":
                match_status = "Partial Match"
            else:
                match_status = f"{DYNAMIC_BUCKET_SENTINEL_PREFIX}{target}"

            group_id = None
            if len(entry_cbl_indices) > 1:
                group_id = f"HISTORY_{history_group_counter}"
                history_group_counter += 1

            for cbl_idx in entry_cbl_indices:
                cbl_df.at[cbl_idx, "match_status"] = match_status
                cbl_df.at[cbl_idx, "matched_insurer_indices"] = list(entry_insurer_indices)
                cbl_df.at[cbl_idx, "match_reason"] = f"History pre-placed ({target})"
                cbl_df.at[cbl_idx, "match_pass"] = ["history"]
                cbl_df.at[cbl_idx, "match_resolved_in_pass"] = "history"
                if group_id is not None:
                    cbl_df.at[cbl_idx, "group_id"] = group_id
                logger.info(f"[HISTORY] Pre-placed CBL idx={cbl_idx} -> {match_status} | insurer_indices={entry_insurer_indices}")

            if global_tracker and entry_insurer_indices:
                global_tracker.mark_matrix_used(entry_insurer_indices)
                logger.info(f"[HISTORY] Registered {len(entry_insurer_indices)} insurer indices in GlobalTracker")

        elif target == "no-match":
            for cbl_idx in entry_cbl_indices:
                cbl_df.at[cbl_idx, "match_status"] = HISTORY_NO_MATCH_SENTINEL
                cbl_df.at[cbl_idx, "match_reason"] = "History pre-placed (no-match)"
                cbl_df.at[cbl_idx, "match_pass"] = ["history"]
                logger.info(f"[HISTORY] Pre-placed CBL idx={cbl_idx} -> No Match (sentinel)")

            if global_tracker and entry_insurer_indices:
                global_tracker.mark_matrix_used(entry_insurer_indices)
                logger.info(f"[HISTORY] Registered {len(entry_insurer_indices)} insurer indices in GlobalTracker")

        summary[target] += len(entry_cbl_indices)

    # --- Summary ---
    total = sum(summary.values())
    rematch_cbl_count = sum(len(s["cbl_indices"]) for s in rematch_stash)
    insurer_only_count = sum(len(v) for v in insurer_only_placements.values())
    logger.info(f"[HISTORY] ========== MATCH HISTORY SUMMARY ==========")
    logger.info(f"[HISTORY] Exact:    {summary['exact']} CBL rows pre-placed")
    logger.info(f"[HISTORY] Partial:  {summary['partial']} CBL rows pre-placed")
    logger.info(f"[HISTORY] No Match: {summary['no-match']} CBL rows pre-placed")
    for key in dynamic_bucket_keys:
        if summary.get(key, 0) > 0:
            label = f"{key} (rematch)" if key in rematch_bucket_keys else key
            logger.info(f"[HISTORY] {label}: {summary[key]} CBL rows — {'stashed for re-matching' if key in rematch_bucket_keys else 'pre-placed'}")
        if key in insurer_only_placements:
            logger.info(f"[HISTORY] {key}: {len(insurer_only_placements[key])} insurer-only rows placed")
    logger.info(f"[HISTORY] Total:    {total} CBL rows processed ({rematch_cbl_count} stashed for re-matching)")
    if insurer_only_count:
        logger.info(f"[HISTORY] Insurer-only placements: {insurer_only_count} insurer rows across {len(insurer_only_placements)} bucket(s)")
    if total == 0 and insurer_only_count == 0:
        logger.info("[HISTORY] No fingerprint matches found — all rows will go through normal comparison")
        logger.info("[HISTORY] This means the fingerprints from history.xlsx do NOT match any regenerated fingerprints.")
        logger.info("[HISTORY] Check that the frontend and backend use the same exclude list and value formatting.")
    logger.info(f"[HISTORY] =============================================")

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
