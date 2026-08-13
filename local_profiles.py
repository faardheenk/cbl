#!/usr/bin/env python3
"""Local test fixtures — per-insurer input files and column mappings.

Shared by process_local.py and test_matrix_approaches.py so the two cannot
drift apart. Development only: in production SharePoint supplies the files,
the column mappings and the bucket list at runtime.
"""

import os

from matching.data_processing import (
    create_dynamic_column_mappings,
    read_excel_with_smart_headers,
)

DATA_DIR = "data"

# The CBL export has the same shape for every insurer.
CBL_MAPPINGS = {
    "Placing/Endorsement No.": "PlacingNo",
    "Policy No.": "PolicyNo",
    "Client Name": "ClientName",
    "Balance Net of Brokerage": "ProcessedAmount",
}

# Mirrors what get_dynamic_buckets() returns from SharePoint.
DYNAMIC_BUCKETS = [
    {"BucketName": "Timing Differences", "BucketKey": "timing_differences", "Rematch": True},
    {"BucketName": "Allocation Issues", "BucketKey": "allocation_issues", "Rematch": False},
    {"BucketName": "Correction to be done by CBL", "BucketKey": "correction_to_be_done_by_cbl", "Rematch": False},
    {"BucketName": "Correction to be done by insurer", "BucketKey": "correction_to_be_done_by_insurer", "Rematch": False},
    {"BucketName": "Mise en demeure", "BucketKey": "mise_en_demeure", "Rematch": False},
    {"BucketName": "Miscellaneous", "BucketKey": "miscellaneous", "Rematch": False},
]

# Only the insurer export differs between profiles.
PROFILES = {
    "swan": {
        "cbl": os.path.join(DATA_DIR, "cbl.xlsx"),
        "insurer": os.path.join(DATA_DIR, "insurer.xlsx"),
        "prev_output": os.path.join(DATA_DIR, "prev_output.xlsx"),
        "insurer_mappings": {
            "BRKREF": "PlacingNo",
            "POLSER": "PolicyNo_1",
            "NAME": "ClientName",
            "AMTDUE": "ProcessedAmount",
        },
    },
    "mua": {
        "cbl": os.path.join(DATA_DIR, "cbl_mua.xlsx"),
        "insurer": os.path.join(DATA_DIR, "insurer_mua.xlsx"),
        "prev_output": os.path.join(DATA_DIR, "prev_output_mua.xlsx"),
        "insurer_mappings": {
            "Amount (Balance)": "ProcessedAmount",
            "Underwriting  (Reference)": "PolicyNo_1",
            "Insured": "ClientName",
            "Details": ["PolicyNo_2", "PlacingNo"],
        },
    },
}


def resolve(name):
    """Return the profile for `name`, or raise with the valid options."""
    key = name.lower()
    if key not in PROFILES:
        raise SystemExit(f"Unknown profile '{name}'. Available: {', '.join(PROFILES)}")
    return key, PROFILES[key]


def missing_files(profile, require_prev_output=True):
    """List the profile's input files that are not on disk."""
    needed = ["cbl", "insurer"] + (["prev_output"] if require_prev_output else [])
    return [profile[k] for k in needed if not os.path.exists(profile[k])]


def load_inputs(profile):
    """Read the profile's files. prev_output is None when absent."""
    with open(profile["cbl"], "rb") as f:
        cbl_content = f.read()
    with open(profile["insurer"], "rb") as f:
        insurer_content = f.read()

    prev_output_content = None
    if os.path.exists(profile["prev_output"]):
        with open(profile["prev_output"], "rb") as f:
            prev_output_content = f.read()

    return cbl_content, insurer_content, prev_output_content


def build_mappings(profile, cbl_content, insurer_content):
    """Build column mappings the same way the SharePoint runner does."""
    cbl_df = read_excel_with_smart_headers(cbl_content)
    insurer_df = read_excel_with_smart_headers(insurer_content)
    return create_dynamic_column_mappings(
        cbl_columns=list(cbl_df.columns),
        insurer_columns=list(insurer_df.columns),
        custom_mappings={
            "cbl_mappings": CBL_MAPPINGS,
            "insurer_mappings": profile["insurer_mappings"],
        },
    )
