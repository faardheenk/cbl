#!/usr/bin/env python3
"""
Local test script for match history fingerprint generation and matching.

Usage:
  1. Place cbl.xlsx, el.xlsx, and history.xlsx in the data/ folder
  2. Run: python test_history_local.py
"""

import logging
import sys
import os

# Configure logging to show everything
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Paths — adjust if your files are elsewhere
CBL_FILE = "data/cbl.xlsx"
INSURER_FILE = "data/el.xlsx"
HISTORY_FILE = "data/history.xlsx"

# Column mappings for EAGLE (adjust for your insurer)
COLUMN_MAPPINGS = {
    'cbl_mappings': {
        'Placing/Endorsement No.': 'PlacingNo',
        'Policy No.': 'PolicyNo',
        'Client Name': 'ClientName',
        'Balance Net of Brokerage': 'ProcessedAmount',
    },
    'insurer_mappings': {
        'Policy Ref': 'PlacingNo',
        'Policy Number': 'PolicyNo_1',
        'Insured Name': 'ClientName',
        'Equivalent in MUR': 'ProcessedAmount',
    },
}


def main():
    # Validate files exist
    for path in [CBL_FILE, INSURER_FILE, HISTORY_FILE]:
        if not os.path.exists(path):
            logger.error(f"File not found: {path}")
            sys.exit(1)

    from matching.data_processing import preprocess, initialize_tracking, read_excel_with_smart_headers
    from matching.match_history import (
        generate_fingerprint,
        generate_fingerprints_for_df,
        read_match_history,
        FINGERPRINT_EXCLUDE_COLUMNS,
    )

    # ── Step 1: Read Excel files ────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Reading Excel files")
    logger.info("=" * 60)

    with open(CBL_FILE, 'rb') as f:
        cbl_bytes = f.read()
    with open(INSURER_FILE, 'rb') as f:
        insurer_bytes = f.read()

    cbl_df = read_excel_with_smart_headers(cbl_bytes, usecols=lambda x: not str(x).startswith('Unnamed:'))
    insurer_df = read_excel_with_smart_headers(insurer_bytes, usecols=lambda x: not str(x).startswith('Unnamed:'))

    logger.info(f"CBL: {len(cbl_df)} rows, columns: {list(cbl_df.columns)}")
    logger.info(f"Insurer: {len(insurer_df)} rows, columns: {list(insurer_df.columns)}")

    # ── Step 2: Preprocess ──────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2: Preprocessing (rename + clean)")
    logger.info("=" * 60)

    clean_cbl, clean_insurer = preprocess(cbl_df, insurer_df, COLUMN_MAPPINGS)
    clean_cbl = initialize_tracking(clean_cbl)

    logger.info(f"CBL after preprocess: {len(clean_cbl)} rows, columns: {list(clean_cbl.columns)}")
    logger.info(f"Insurer after preprocess: {len(clean_insurer)} rows, columns: {list(clean_insurer.columns)}")

    # ── Step 3: Generate CBL fingerprints (preprocessed) ────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3: Generating CBL fingerprints (preprocessed + date fix)")
    logger.info("=" * 60)

    cbl_cols_used = sorted([k for k in clean_cbl.columns if k not in FINGERPRINT_EXCLUDE_COLUMNS])
    cbl_cols_excluded = sorted([k for k in clean_cbl.columns if k in FINGERPRINT_EXCLUDE_COLUMNS])
    logger.info(f"Columns INCLUDED ({len(cbl_cols_used)}): {cbl_cols_used}")
    logger.info(f"Columns EXCLUDED ({len(cbl_cols_excluded)}): {cbl_cols_excluded}")

    clean_cbl["_fingerprint"] = generate_fingerprints_for_df(clean_cbl)

    logger.info(f"\nSample CBL fingerprints (first 5):")
    for i in range(min(5, len(clean_cbl))):
        fp = clean_cbl["_fingerprint"].iloc[i]
        logger.info(f"  Row {clean_cbl.index[i]}: {fp[:120]}{'...' if len(fp) > 120 else ''}")

    # ── Step 4: Generate Insurer fingerprints (preprocessed) ────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 4: Generating Insurer fingerprints (preprocessed + date fix)")
    logger.info("=" * 60)

    insurer_cols_stripped = {
        col: col[:-len("_INSURER")] if col.endswith("_INSURER") else col
        for col in clean_insurer.columns
    }
    insurer_stripped = clean_insurer.rename(columns=insurer_cols_stripped)

    ins_cols_used = sorted([k for k in insurer_stripped.columns if k not in FINGERPRINT_EXCLUDE_COLUMNS])
    ins_cols_excluded = sorted([k for k in insurer_stripped.columns if k in FINGERPRINT_EXCLUDE_COLUMNS])
    logger.info(f"Columns INCLUDED ({len(ins_cols_used)}): {ins_cols_used}")
    logger.info(f"Columns EXCLUDED ({len(ins_cols_excluded)}): {ins_cols_excluded}")

    clean_insurer["_fingerprint"] = generate_fingerprints_for_df(insurer_stripped)

    logger.info(f"\nSample Insurer fingerprints (first 5):")
    for i in range(min(5, len(clean_insurer))):
        fp = clean_insurer["_fingerprint"].iloc[i]
        logger.info(f"  Row {clean_insurer.index[i]}: {fp[:120]}{'...' if len(fp) > 120 else ''}")

    # ── Step 5: Read history.xlsx ───────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 5: Reading history.xlsx")
    logger.info("=" * 60)

    history = read_match_history(HISTORY_FILE)
    logger.info(f"History entries: {len(history)}")

    if not history:
        logger.info("No history entries found — nothing to match.")
        return

    # ── Step 6: Build lookup maps ───────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 6: Building fingerprint lookup maps")
    logger.info("=" * 60)

    cbl_fp_map = {}
    for idx, fp in clean_cbl["_fingerprint"].items():
        cbl_fp_map.setdefault(fp, []).append(idx)

    insurer_fp_map = {}
    for idx, fp in clean_insurer["_fingerprint"].items():
        insurer_fp_map.setdefault(fp, []).append(idx)

    logger.info(f"Unique CBL fingerprints: {len(cbl_fp_map)}")
    logger.info(f"Unique Insurer fingerprints: {len(insurer_fp_map)}")

    # ── Step 7: Match history fingerprints ──────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 7: Matching history fingerprints against data")
    logger.info("=" * 60)

    for entry_idx, entry in enumerate(history):
        target = entry["target_bucket"]
        logger.info(f"\n--- Entry #{entry_idx}: {entry['from_bucket']} -> {target} ---")
        logger.info(f"  CBL fingerprints in history: {len(entry['cbl_fingerprints'])}")
        logger.info(f"  Insurer fingerprints in history: {len(entry['insurer_fingerprints'])}")

        # Match CBL fingerprints
        logger.info(f"\n  CBL fingerprint matching:")
        for fp_idx, fp in enumerate(entry["cbl_fingerprints"]):
            if fp in cbl_fp_map:
                matched_indices = cbl_fp_map[fp]
                logger.info(f"    [MATCH] fp[{fp_idx}] -> data row(s) {matched_indices}")
            else:
                logger.info(f"    [MISS]  fp[{fp_idx}]: {fp[:100]}{'...' if len(fp) > 100 else ''}")
                # Try to find closest match for debugging
                best_match = None
                best_overlap = 0
                fp_parts = set(fp.split("|"))
                for data_fp in list(cbl_fp_map.keys())[:500]:  # check first 500
                    data_parts = set(data_fp.split("|"))
                    overlap = len(fp_parts & data_parts)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_match = data_fp
                if best_match:
                    logger.info(f"            Closest CBL fp ({best_overlap}/{len(fp_parts)} parts overlap):")
                    logger.info(f"            {best_match[:100]}{'...' if len(best_match) > 100 else ''}")
                    # Show differing parts
                    hist_parts = fp.split("|")
                    data_parts = best_match.split("|")
                    if len(hist_parts) == len(data_parts):
                        diffs = []
                        for j, (h, d) in enumerate(zip(hist_parts, data_parts)):
                            if h != d:
                                diffs.append(f"  pos {j}: history='{h}' vs data='{d}'")
                        if diffs:
                            logger.info(f"            Differences ({len(diffs)}):")
                            for diff in diffs[:10]:
                                logger.info(f"            {diff}")
                    else:
                        logger.info(f"            Part count differs: history={len(hist_parts)} vs data={len(data_parts)}")

        # Match Insurer fingerprints
        logger.info(f"\n  Insurer fingerprint matching:")
        for fp_idx, fp in enumerate(entry["insurer_fingerprints"]):
            if fp in insurer_fp_map:
                matched_indices = insurer_fp_map[fp]
                logger.info(f"    [MATCH] fp[{fp_idx}] -> data row(s) {matched_indices}")
            else:
                logger.info(f"    [MISS]  fp[{fp_idx}]: {fp[:100]}{'...' if len(fp) > 100 else ''}")
                # Try to find closest match
                best_match = None
                best_overlap = 0
                fp_parts = set(fp.split("|"))
                for data_fp in list(insurer_fp_map.keys())[:500]:
                    data_parts = set(data_fp.split("|"))
                    overlap = len(fp_parts & data_parts)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_match = data_fp
                if best_match:
                    logger.info(f"            Closest Insurer fp ({best_overlap}/{len(fp_parts)} parts overlap):")
                    logger.info(f"            {best_match[:100]}{'...' if len(best_match) > 100 else ''}")
                    hist_parts = fp.split("|")
                    data_parts = best_match.split("|")
                    if len(hist_parts) == len(data_parts):
                        diffs = []
                        for j, (h, d) in enumerate(zip(hist_parts, data_parts)):
                            if h != d:
                                diffs.append(f"  pos {j}: history='{h}' vs data='{d}'")
                        if diffs:
                            logger.info(f"            Differences ({len(diffs)}):")
                            for diff in diffs[:10]:
                                logger.info(f"            {diff}")
                    else:
                        logger.info(f"            Part count differs: history={len(hist_parts)} vs data={len(data_parts)}")

    # ── Summary ─────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("DONE")
    logger.info("=" * 60)

    # Cleanup
    clean_cbl.drop(columns=["_fingerprint"], inplace=True)
    clean_insurer.drop(columns=["_fingerprint"], inplace=True)


if __name__ == "__main__":
    main()
