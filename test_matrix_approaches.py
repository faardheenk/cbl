#!/usr/bin/env python3
"""Regression check for the matrix (pre-placement) layer.

Runs the real pipeline for each insurer profile with its previous output as
the matrix, then reports what the matrix actually achieved and asserts the
properties that must not regress:

  * no CBL row appears in more than one sheet
  * the matrix pre-places rows rather than silently doing nothing

The count of insurer rows appearing in more than one sheet is reported but
not asserted. Insurer exports contain genuinely duplicated rows — MUA has
146 rows sharing an identity with another — so two value-identical rows
landing in different sheets is correct behaviour, not double-placement.
Watch the number for sudden jumps rather than expecting zero.

Bucket fidelity — how many of the previous output's rows return to the
bucket the broker filed them in — is reported for reference. It is not
asserted, because rows in rematchable buckets are *meant* to move when
fresh insurer data resolves them.

Usage:  python test_matrix_approaches.py [swan|mua|all]
"""

import io
import logging
import os
import sys

import pandas as pd

import local_profiles as profiles
from matching.match_history import (
    CBL_KEY_COLUMNS,
    INSURER_KEY_COLUMNS,
    build_row_key,
    read_bucket_config,
)
from matching.orchestrator import run_matching_process

logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join("data", "comparison")

# Sheets that hold no CBL rows of their own.
NON_BUCKET_SHEETS = {"Summary", "_BucketConfig"}
INSURER_ONLY_SHEET = "No Matches Insurer"

SHEET_TO_BUCKET = {
    "Exact Matches": "exact",
    "Partial Matches": "partial",
    "No Matches CBL": "no-match",
}


def sheet_frames(output_content, include_insurer_sheet=True):
    """Yield (sheet_name, DataFrame) for every bucket sheet in an output."""
    xls = pd.ExcelFile(io.BytesIO(output_content))
    for name in xls.sheet_names:
        if name in NON_BUCKET_SHEETS:
            continue
        if not include_insurer_sheet and name == INSURER_ONLY_SHEET:
            continue
        yield name, pd.read_excel(xls, sheet_name=name)


def rows_by_key(output_content, key_columns, include_insurer_sheet):
    """Map row key -> set of sheets the row appears in."""
    seen = {}
    for name, df in sheet_frames(output_content, include_insurer_sheet):
        if not all(c in df.columns for c in key_columns):
            continue
        for i in df.index:
            key = build_row_key(df.loc[i], key_columns)
            if key:
                seen.setdefault(key, set()).add(name)
    return seen


def matrix_placed_count(output_content):
    """Rows carrying a matrix provenance marker."""
    placed = 0
    for _, df in sheet_frames(output_content, include_insurer_sheet=False):
        if "match_reason" in df.columns:
            placed += df["match_reason"].astype(str).str.startswith("Matrix").sum()
    return int(placed)


def bucket_fidelity(profile, output_content):
    """(same_bucket, different_bucket, absent) for the previous output's CBL rows."""
    prev = pd.ExcelFile(profile["prev_output"])
    sheet_to_key = read_bucket_config(prev)
    landed = rows_by_key(output_content, CBL_KEY_COLUMNS, include_insurer_sheet=False)

    same = different = absent = 0
    for sheet in prev.sheet_names:
        if sheet in NON_BUCKET_SHEETS or sheet == INSURER_ONLY_SHEET:
            continue
        expected = {sheet, SHEET_TO_BUCKET.get(sheet) or sheet_to_key.get(sheet, sheet)}

        df = pd.read_excel(prev, sheet_name=sheet)
        if df.empty or not all(c in df.columns for c in CBL_KEY_COLUMNS):
            continue

        for i in df.index:
            key = build_row_key(df.loc[i], CBL_KEY_COLUMNS)
            if not key:
                continue
            got = landed.get(key)
            if not got:
                absent += 1
            elif got & expected:
                same += 1
            else:
                different += 1
    return same, different, absent


def check_profile(name):
    """Run one profile and report. Returns a list of failure messages."""
    profile = profiles.PROFILES[name]

    missing = profiles.missing_files(profile)
    if missing:
        print(f"\n{name.upper()}: SKIPPED — missing {missing}")
        return []

    print()
    print("=" * 70)
    print(f"  {name.upper()}")
    print("=" * 70)

    cbl_content, insurer_content, prev_output_content = profiles.load_inputs(profile)
    column_mappings = profiles.build_mappings(profile, cbl_content, insurer_content)

    result = run_matching_process(
        column_mappings=column_mappings,
        cbl_file=cbl_content,
        insurer_file=insurer_content,
        output_file=f"{name}_output.xlsx",
        prev_output_file=prev_output_content,
        dynamic_buckets=profiles.DYNAMIC_BUCKETS,
    )
    output = result["output_content"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"output_{name}.xlsx")
    with open(path, "wb") as f:
        f.write(output)

    print("\nSheet row counts")
    for sheet, df in sheet_frames(output):
        print(f"  {sheet:<36}{len(df):>8}")

    placed = matrix_placed_count(output)
    same, different, absent = bucket_fidelity(profile, output)
    considered = same + different

    cbl_rows = rows_by_key(output, CBL_KEY_COLUMNS, include_insurer_sheet=False)
    insurer_rows = rows_by_key(output, INSURER_KEY_COLUMNS, include_insurer_sheet=True)
    cbl_multi = [k for k, v in cbl_rows.items() if len(v) > 1]
    insurer_multi = [k for k, v in insurer_rows.items() if len(v) > 1]

    stats = result["cbl_stats"]
    fidelity = f"{same}/{considered} ({same / considered * 100:.1f}%)" if considered else "n/a"
    print("\nMatrix")
    print(f"  rows pre-placed by matrix           {placed:>8}")
    print(f"  bucket fidelity                     {fidelity:>8}")
    print(f"  moved bucket (rematch or upgrade)   {different:>8}")
    print(f"  not present in new CBL data         {absent:>8}")

    print("\nMatch statistics")
    print(f"  exact / partial / no match          {stats['exact_matches']:>8}"
          f" / {stats['partial_matches']} / {stats['no_matches']}")
    print(f"  insurer exact match rate            {result['insurer_stats']['exact_match_rate']:>7.1f}%")

    failures = []
    if cbl_multi:
        failures.append(f"{name}: {len(cbl_multi)} CBL row(s) appear in more than one sheet")
    if prev_output_content is not None and placed == 0:
        failures.append(f"{name}: matrix pre-placed nothing — the previous output had no effect")

    print("\nChecks")
    print(f"  CBL rows in >1 sheet                {len(cbl_multi):>8}   (must be 0)")
    print(f"  insurer rows in >1 sheet            {len(insurer_multi):>8}   (informational — source duplicates)")
    print(f"  wrote {path}")
    return failures


def main():
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    which = (sys.argv[1] if len(sys.argv) > 1 else "all").lower()
    if which == "all":
        names = list(profiles.PROFILES)
    else:
        names = [profiles.resolve(which)[0]]

    failures = []
    for name in names:
        failures.extend(check_profile(name))

    print()
    print("=" * 70)
    if failures:
        print("FAILED")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASSED — no row duplication, matrix is placing rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
