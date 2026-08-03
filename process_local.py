#!/usr/bin/env python3
"""Process local CBL and insurer Excel files through the matching engine."""

import os
import logging
from matching.orchestrator import run_matching_process
from matching.data_processing import create_dynamic_column_mappings, read_excel_with_smart_headers

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def main():
    cbl_path = os.path.join("data", "cbl.xlsx")
    insurer_path = os.path.join("data", "insurer.xlsx")
    history_path = os.path.join("data", "history.xlsx")
    insurer_name = "JUBILEE"

    cbl_custom_mappings = {
        "Placing/Endorsement No.": "PlacingNo",
        "Policy No.": "PolicyNo",
        "Client Name": "ClientName",
        "Balance Net of Brokerage": "ProcessedAmount",
    }

    insurer_custom_mappings = {
        "Policy No": "PolicyNo_1",
        "Insured Name": "ClientName",
        "Total": "ProcessedAmount",
        "Broker Reference": "PlacingNo",
    }

    dynamic_buckets = [
        {"BucketName": "Timing Differences", "BucketKey": "timing_differences", "Rematch": True},
        {"BucketName": "Allocation Issues", "BucketKey": "allocation_issues", "Rematch": False},
        {"BucketName": "Correction to be done by CBL", "BucketKey": "correction_to_be_done_by_cbl", "Rematch": False},
        {"BucketName": "Correction to be done by insurer", "BucketKey": "correction_to_be_done_by_insurer", "Rematch": False},
        {"BucketName": "Mise en demeure", "BucketKey": "mise_en_demeure", "Rematch": False},
        {"BucketName": "Miscellaneous", "BucketKey": "miscellaneous", "Rematch": False},
    ]

    with open(cbl_path, "rb") as f:
        cbl_content = f.read()
    with open(insurer_path, "rb") as f:
        insurer_content = f.read()

    history_content = None
    if os.path.exists(history_path):
        with open(history_path, "rb") as f:
            history_content = f.read()
        logger.info(f"History file loaded: {history_path}")
    else:
        logger.info(f"No history file found at {history_path} — skipping")

    cbl_df = read_excel_with_smart_headers(cbl_content)
    insurer_df = read_excel_with_smart_headers(insurer_content)

    logger.info(f"CBL columns: {list(cbl_df.columns)}")
    logger.info(f"Insurer columns: {list(insurer_df.columns)}")

    column_mappings = create_dynamic_column_mappings(
        cbl_columns=list(cbl_df.columns),
        insurer_columns=list(insurer_df.columns),
        custom_mappings={
            "cbl_mappings": cbl_custom_mappings,
            "insurer_mappings": insurer_custom_mappings,
        },
    )

    logger.info(f"Final CBL mappings: {column_mappings['cbl_mappings']}")
    logger.info(f"Final Insurer mappings: {column_mappings['insurer_mappings']}")

    result = run_matching_process(
        column_mappings=column_mappings,
        cbl_file=cbl_content,
        insurer_file=insurer_content,
        output_file=f"{insurer_name}_output.xlsx",
        history_file=history_content,
        dynamic_buckets=dynamic_buckets,
    )

    output_path = os.path.join("data", "output.xlsx")
    with open(output_path, "wb") as f:
        f.write(result["output_content"])

    logger.info(f"\nResults saved to {output_path}")
    logger.info(f"CBL Exact Matches: {result['cbl_stats']['exact_matches']}")
    logger.info(f"CBL Partial Matches: {result['cbl_stats']['partial_matches']}")
    logger.info(f"CBL No Matches: {result['cbl_stats']['no_matches']}")
    logger.info(f"Insurer Match Rate: {result['insurer_stats']['exact_match_rate']:.1f}%")


if __name__ == "__main__":
    main()
