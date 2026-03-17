#!/usr/bin/env python3

import pandas as pd
import logging
from .utils import add_pass, extract_policy_tokens
from .matching_engine import (
    classify_amount_match,
    _apply_exact_match,
    _handle_conflict_resolution,
)

logger = logging.getLogger(__name__)


def pass2(cbl_df, insurer_df, tolerance=50, global_tracker=None):
    """Pass 2: Matching by Policy Number and Amount (Name matching removed)."""
    logger.info("\n=== Pass 2: Matching by Policy Number and Amount ===")
    total_records = len(cbl_df[cbl_df["match_status"].isin(["No Match", "Partial Match"])])
    exact_matches = 0
    partial_matches = 0
    processed = 0

    logger.info(f"Pass 2 starting with global tracker: {global_tracker.get_usage_summary()}")

    # Use GlobalMatchTracker for consistent filtering (same logic as Pass 3)
    # For Pass 2, we exclude exact and matrix matches but allow partial matches to be upgraded
    already_matched_insurer = global_tracker.exact_used_insurer | global_tracker.matrix_used_insurer
    fallback_rows = insurer_df[~insurer_df.index.isin(already_matched_insurer)].copy()
    logger.info(f"Pass 2: Using global tracker - excluding {len(already_matched_insurer)} exact/matrix used insurer rows")
    logger.info(f"Pass 2: Available insurer rows for policy number matching: {len(fallback_rows)}")

    # Phase 1: Collect all potential matches without applying them yet
    potential_matches = []

    for i, row in cbl_df[cbl_df["match_status"].isin(["No Match", "Partial Match"])].iterrows():
        # Skip rows already resolved by match history — user manual placements are authoritative
        if row.get("match_resolved_in_pass") == "history":
            continue

        processed += 1
        if processed % 50 == 0:
            logger.info(f"Progress: {processed}/{total_records} records processed")

        add_pass(cbl_df, i, 2)
        tokens = extract_policy_tokens(row["PolicyNo_Clean"])
        cbl_amt = row["ProcessedAmount_Clean"]

        matched_indices = []

        for j, insurer_row in fallback_rows.iterrows():
            # Collect all available policy values for this insurer row
            insurer_policy_values = []

            # Add PolicyNo_1 if available
            if pd.notna(insurer_row.get("PolicyNo_Clean_INSURER")) and str(insurer_row["PolicyNo_Clean_INSURER"]).strip():
                insurer_policy_values.append(str(insurer_row["PolicyNo_Clean_INSURER"]).strip())

            # Add PolicyNo_2 if available
            if "PolicyNo_2_Clean_INSURER" in insurer_row.index:
                if pd.notna(insurer_row["PolicyNo_2_Clean_INSURER"]) and str(insurer_row["PolicyNo_2_Clean_INSURER"]).strip():
                    insurer_policy_values.append(str(insurer_row["PolicyNo_2_Clean_INSURER"]).strip())

            # Simple policy matching - check if any insurer policy is in CBL tokens
            policy_match = False
            for insurer_policy in insurer_policy_values:
                if insurer_policy in tokens:
                    policy_match = True
                    logger.debug(f"Pass 2 CBL {i}: Policy match found - '{insurer_policy}' in tokens {tokens}")
                    break

            if policy_match:
                matched_indices.append(j)

        total_amt = fallback_rows.loc[matched_indices, "ProcessedAmount_Clean_INSURER"].sum()

        if matched_indices:
            # Classify the amount match using graduated confidence levels
            match_type, difference, confidence = classify_amount_match(cbl_amt, total_amt, tolerance)

            # Determine if it's single or cumulative amount match
            if len(matched_indices) == 1:
                amount_match_type = 'Single Amount Match'
            else:
                amount_match_type = 'Cumulative Amount Match'

            # Simple policy match info
            policy_strategy_info = " (Policy Match)"

            if match_type in ["PERFECT_MATCH", "EXACT_MATCH"]:
                # Exact match found
                match_reason = f'Policy Number{policy_strategy_info} + {amount_match_type} ({confidence} Confidence)'
                potential_matches.append({
                    'cbl_index': i,
                    'match_type': 'exact',
                    'insurer_indices': matched_indices,
                    'match_reason': match_reason,
                    'confidence_level': confidence,
                    'amount_difference': difference,
                    'total_amount': total_amt
                })
            # REMOVED: CLOSE_MATCH and REVIEW_REQUIRED/INVESTIGATION_REQUIRED logic
            # Let Phase 3 name grouping handle all amount mismatches

    # Phase 2: Resolve conflicts and apply matches
    logger.info("\n=== Pass 2 Phase 2: Resolving conflicts and applying matches ===")

    # Sort potential matches: exact matches first, then partial matches
    # Within each type, sort by number of insurer indices (larger combinations get priority)
    potential_matches.sort(key=lambda x: (
        0 if x['match_type'] == 'exact' else 1,
        -len(x['insurer_indices'])  # Negative for descending order (larger combinations first)
    ))

    # Use GlobalMatchTracker for consistent tracking
    for match in potential_matches:
        cbl_index = match['cbl_index']
        match_type = match['match_type']
        insurer_indices = match['insurer_indices']

        # Use GlobalMatchTracker for conflict detection
        can_claim_all, available_indices, conflicts = global_tracker.can_cbl_claim_insurer(
            cbl_index, insurer_indices, 'exact' if match_type == 'exact' else 'partial'
        )

        if conflicts:
            # Use helper function for conflict resolution
            exact_added, partial_added = _handle_conflict_resolution(
                cbl_df, insurer_df, match, None, tolerance, 2, global_tracker, fallback_rows
            )
            exact_matches += exact_added
            partial_matches += partial_added
        else:
            # Apply the match using helper functions
            # Pass 2 only creates exact matches (no partial matches)
            if match_type == 'exact':
                exact_matches += _apply_exact_match(
                    cbl_df, cbl_index, match['match_reason'], insurer_indices,
                    match['total_amount'], [], 2, global_tracker,
                    confidence_level=match.get('confidence_level'),
                    amount_difference=match.get('amount_difference')
                )
            else:
                # This should never happen as Pass 2 only creates exact matches
                logger.error(f"Pass 2: Unexpected match_type '{match_type}' - skipping")
                continue

    logger.info(f"✓ Pass 2 complete: {exact_matches} exact matches, {partial_matches} partial matches")
    return cbl_df
