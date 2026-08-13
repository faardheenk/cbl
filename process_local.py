#!/usr/bin/env python3
"""Run the matching engine against local Excel files.

Usage:
    python process_local.py                  # swan
    python process_local.py mua
    python process_local.py swan --no-matrix # ignore the previous output

Inputs and column mappings come from local_profiles.py. Results are written
to data/output_<profile>.xlsx.
"""

import argparse
import logging
import os
import sys

import local_profiles as profiles
from matching.orchestrator import run_matching_process

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "profile",
        nargs="?",
        default="swan",
        help=f"insurer profile ({', '.join(profiles.PROFILES)})",
    )
    parser.add_argument(
        "--no-matrix",
        action="store_true",
        help="run without the previous output, as if this were a first run",
    )
    args = parser.parse_args()

    name, profile = profiles.resolve(args.profile)

    missing = profiles.missing_files(profile, require_prev_output=False)
    if missing:
        logger.error(f"Missing input files: {missing}")
        return 1

    cbl_content, insurer_content, prev_output_content = profiles.load_inputs(profile)

    if args.no_matrix:
        prev_output_content = None
        logger.info("Matrix disabled (--no-matrix)")
    elif prev_output_content is None:
        logger.info(f"No previous output at {profile['prev_output']} — running without matrix")
    else:
        logger.info(f"Previous output (matrix) loaded: {profile['prev_output']}")

    column_mappings = profiles.build_mappings(profile, cbl_content, insurer_content)
    logger.info(f"CBL mappings:     {column_mappings['cbl_mappings']}")
    logger.info(f"Insurer mappings: {column_mappings['insurer_mappings']}")

    result = run_matching_process(
        column_mappings=column_mappings,
        cbl_file=cbl_content,
        insurer_file=insurer_content,
        output_file=f"{name}_output.xlsx",
        prev_output_file=prev_output_content,
        dynamic_buckets=profiles.DYNAMIC_BUCKETS,
    )

    output_path = os.path.join(profiles.DATA_DIR, f"output_{name}.xlsx")
    with open(output_path, "wb") as f:
        f.write(result["output_content"])

    cbl_stats = result["cbl_stats"]
    logger.info(f"\nResults saved to {output_path}")
    logger.info(f"CBL Exact Matches:   {cbl_stats['exact_matches']}")
    logger.info(f"CBL Partial Matches: {cbl_stats['partial_matches']}")
    logger.info(f"CBL No Matches:      {cbl_stats['no_matches']}")
    logger.info(f"Insurer Match Rate:  {result['insurer_stats']['exact_match_rate']:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
