#!/usr/bin/env python3

import pandas as pd
import logging
from itertools import combinations
from .utils import add_pass
from .matching_engine import (
    classify_amount_match,
    validate_substring_match,
    _apply_exact_match,
    _handle_conflict_resolution,
)

logger = logging.getLogger(__name__)


def pass1(cbl_df, insurer_df, tolerance=50, global_tracker=None):
    """Pass 1: Matching by Placing Number and ProcessedAmount."""
    logger.info("\n=== Pass 1: Matching by Placing Number and Amount ===")
    total_records = len(cbl_df)
    exact_matches = 0
    partial_matches = 0

    logger.info(f"Pass 1 starting with global tracker: {global_tracker.get_usage_summary()}")

    # Pre-compute string conversions for performance optimization
    logger.info("Pre-computing insurer placing strings for substring matching...")
    insurer_placing_strings = insurer_df["PlacingNo_Clean_INSURER"].astype(str)
    # Cache valid placing strings (length >= 10) to avoid repeated checks
    valid_insurer_mask = (insurer_placing_strings != 'nan') & (insurer_placing_strings.str.len() >= 10)

    # Phase 1: Collect all potential matches without applying them yet
    potential_matches = []

    for i, row in cbl_df.iterrows():
        if i % 100 == 0:
            logger.info(f"Progress: {i+1}/{total_records} records processed")

        # Skip rows already resolved by match history — user manual placements are authoritative
        if row.get("match_resolved_in_pass") == "history":
            continue

        add_pass(cbl_df, i, 1)

        placing = row["PlacingNo_Clean"]
        amt1 = row["ProcessedAmount_Clean"]

        # Validate input data
        if pd.isna(placing) or placing == "" or str(placing).strip() == "":
            logger.warning(f"Record {i}: Empty or invalid placing number, skipping")
            continue

        if pd.isna(amt1):
            logger.warning(f"Record {i}: Invalid amount ({amt1}), skipping")
            continue

        insurer_matches = insurer_df[insurer_df["PlacingNo_Clean_INSURER"] == placing]

        # If no exact matches, try enhanced substring matching with quality controls
        overlap_details = {}  # Store overlap info for later use in match reasons
        if insurer_matches.empty:
            placing_str = str(placing).strip()

            # Only proceed if CBL placing is long enough (>= 10 chars)
            if len(placing_str) >= 10:
                logger.debug(f"Record {i}: No exact matches, trying quality-controlled substring matching for '{placing_str}'")
                qualified_indices = []
                rejected_count = 0

                # Check each insurer placing number with quality validation
                for idx in insurer_df.index:
                    insurer_placing = str(insurer_df.at[idx, "PlacingNo_Clean_INSURER"])

                    # Skip NaN or invalid insurer placing numbers
                    if pd.isna(insurer_placing) or insurer_placing == 'nan' or not insurer_placing.strip():
                        continue

                    # Validate substring match quality
                    is_valid, overlap_info = validate_substring_match(placing_str, insurer_placing.strip())

                    if is_valid:
                        qualified_indices.append(idx)
                        overlap_details[idx] = overlap_info  # Store for later use in match reasons
                        logger.debug(f"Record {i}: Qualified substring match at index {idx}: {overlap_info}")
                    else:
                        rejected_count += 1
                        logger.debug(f"Record {i}: Rejected substring match at index {idx}: {overlap_info}")

                # Create matches dataframe from qualified indices
                insurer_matches = insurer_df.loc[qualified_indices] if qualified_indices else pd.DataFrame()

                if not insurer_matches.empty:
                    logger.info(f"Record {i}: Found {len(insurer_matches)} qualified substring matches (rejected {rejected_count} poor quality matches)")
                elif rejected_count > 0:
                    logger.info(f"Record {i}: No qualified substring matches found (rejected {rejected_count} poor quality matches)")
            else:
                logger.debug(f"Record {i}: Skipping substring matching - placing too short ({len(placing_str)} chars)")

        # First comparison - exact matches
        exact_match_indices = None
        exact_partial_count = 0
        insurer_indices = []  # Initialize insurer_indices outside the if block

        if not insurer_matches.empty:
            # Ensure unique indices to prevent duplicates in combinations
            unique_indices = []
            unique_amounts = []
            seen_indices = set()

            for idx, amt in zip(insurer_matches.index.tolist(), insurer_matches["ProcessedAmount_Clean_INSURER"].tolist()):
                if idx not in seen_indices:
                    unique_indices.append(idx)
                    unique_amounts.append(amt)
                    seen_indices.add(idx)

            insurer_indices = unique_indices
            insurer_amounts = unique_amounts
            exact_partial_count = len(insurer_indices)

            # Check individual matches first with graduated confidence levels
            best_match = None
            for j, amt2 in zip(insurer_indices, insurer_amounts):
                if pd.notna(amt2):
                    match_type, difference, confidence = classify_amount_match(amt1, amt2, tolerance)

                    if match_type in ["PERFECT_MATCH", "EXACT_MATCH"]:
                        # Auto-approve exact matches only - no close matches
                        # Include overlap info if this came from substring matching
                        overlap_info = overlap_details.get(j, "")
                        overlap_suffix = f" ({overlap_info})" if overlap_info else ""

                        best_match = {
                            'indices': [j],
                            'type': 'exact',
                            'confidence': confidence,
                            'difference': difference,
                            'reason': f'Placing Number{overlap_suffix} + Single Amount Match ({confidence} Confidence)'
                        }
                        break
                    # REMOVED: CLOSE_MATCH logic - let Phase 3 name grouping handle amount mismatches

            # Set exact_match_indices based on best match found
            if best_match and best_match['type'] == 'exact':
                exact_match_indices = best_match['indices']
                exact_match_confidence = best_match['confidence']
                exact_match_difference = best_match['difference']
                exact_match_reason = best_match['reason']
            else:
                exact_match_indices = None

        # Second comparison - combinations (smart selection)
        combination_match_indices = None
        combination_partial_count = 0
        if not insurer_matches.empty and exact_match_indices is None:
            # Only try combinations if no exact match was found
            # Exclude any indices that were used in exact matches
            available_indices = [idx for idx in insurer_indices if idx not in (exact_match_indices or [])]
            available_amounts = [amt for idx, amt in zip(insurer_indices, insurer_amounts) if idx in available_indices]

            combination_partial_count = len(available_indices)

            # Smart selection: limit to 50 most promising items
            max_items_to_consider = 20
            target = -amt1  # We want sum(insurer_amounts) to be close to -amt1

            if len(available_indices) > max_items_to_consider:
                # Sort by how close each amount gets us to the target
                sorted_pairs = sorted(zip(available_indices, available_amounts),
                                    key=lambda x: abs(x[1] - target))

                # Take the 50 most promising items
                limited_indices = [pair[0] for pair in sorted_pairs[:max_items_to_consider]]
                limited_amounts = [pair[1] for pair in sorted_pairs[:max_items_to_consider]]

                logger.info(f"Record {i}: Selected {max_items_to_consider} most promising items from {len(available_indices)} total")
                logger.info(f"Target amount: {target}, Selected amounts: {limited_amounts}")
            else:
                limited_indices = available_indices
                limited_amounts = available_amounts

            # Try combinations with the limited set (max 5 items per combination for business reality)
            max_combination_size = min(5, len(limited_indices))

            for r in range(2, max_combination_size + 1):
                for combination in combinations(zip(limited_indices, limited_amounts), r):
                    combination_indices, combination_amounts = zip(*combination)

                    # Filter out NaN values and validate amounts
                    valid_amounts = [amt for amt in combination_amounts if pd.notna(amt)]
                    if len(valid_amounts) != len(combination_amounts):
                        logger.warning(f"Record {i}: Skipping combination with NaN values: {combination_amounts}")
                        continue

                    total_amount = sum(valid_amounts)
                    if pd.notna(total_amount):
                        match_type, difference, confidence = classify_amount_match(amt1, total_amount, tolerance)

                        if match_type in ["PERFECT_MATCH", "EXACT_MATCH"]:
                            combination_match_indices = list(combination_indices)
                            combination_match_confidence = confidence
                            combination_match_difference = difference

                            # Include overlap info if any of the combination items came from substring matching
                            overlap_infos = [overlap_details.get(idx, "") for idx in combination_indices if overlap_details.get(idx, "")]
                            overlap_suffix = f" ({'; '.join(set(overlap_infos))})" if overlap_infos else ""

                            combination_match_reason = f'Placing Number{overlap_suffix} + Cumulative Amount Match ({confidence} Confidence)'
                            logger.info(f"Record {i}: Found combination match with {r} items, total: {total_amount}, confidence: {confidence}")
                            break
                        # REMOVED: CLOSE_MATCH logic - let Phase 3 name grouping handle amount mismatches
                if combination_match_indices is not None:
                    break

        # Log results for each comparison method
        logger.info(f"\nComparison results for CBL record {i}:")
        logger.info(f"Exact comparison: {1 if exact_match_indices else 0} exact matches, {exact_partial_count} partial matches")
        logger.info(f"Combination comparison: {1 if combination_match_indices else 0} exact matches, {combination_partial_count} partial matches")

        # Store potential matches for later resolution
        if exact_match_indices is not None:
            potential_matches.append({
                'cbl_index': i,
                'match_type': 'exact',
                'insurer_indices': exact_match_indices,
                'match_reason': exact_match_reason,
                'confidence_level': exact_match_confidence,
                'amount_difference': exact_match_difference,
                'fallback_indices': [idx for idx in insurer_indices if idx not in exact_match_indices]
            })
        elif combination_match_indices is not None:
            # Combination matches are always exact (no close matches anymore)
            potential_matches.append({
                'cbl_index': i,
                'match_type': 'combination',
                'insurer_indices': combination_match_indices,
                'match_reason': combination_match_reason,
                'confidence_level': combination_match_confidence,
                'amount_difference': combination_match_difference,
                'fallback_indices': [idx for idx in insurer_indices if idx not in combination_match_indices]
            })
        # REMOVED: Partial match logic for REVIEW_REQUIRED/INVESTIGATION_REQUIRED
        # Let Phase 3 name grouping handle all amount mismatches

    # Phase 2: Resolve conflicts by prioritizing combination matches
    logger.info("\n=== Phase 2: Resolving conflicts and applying matches ===")

    # Sort potential matches: exact matches first, then combinations, then partial matches
    # Within each type, sort by number of insurer indices (larger combinations get priority)
    potential_matches.sort(key=lambda x: (
        0 if x['match_type'] == 'exact' else (1 if x['match_type'] == 'combination' else 2),
        -len(x['insurer_indices'])  # Negative for descending order (larger combinations first)
    ))

    # Use GlobalMatchTracker for consistent tracking
    for match in potential_matches:
        cbl_index = match['cbl_index']
        match_type = match['match_type']
        insurer_indices = match['insurer_indices']

        # Use GlobalMatchTracker for conflict detection
        can_claim_all, available_indices, conflicts = global_tracker.can_cbl_claim_insurer(
            cbl_index, insurer_indices, 'exact' if match_type in ['exact', 'combination'] else 'partial'
        )

        if conflicts:
            # Use helper function for conflict resolution
            exact_added, partial_added = _handle_conflict_resolution(
                cbl_df, insurer_df, match, None, tolerance, 1, global_tracker
            )
            exact_matches += exact_added
            partial_matches += partial_added
        else:
            # Apply the match using helper functions
            # Pass 1 only creates exact matches (no partial matches)
            if match_type in ['exact', 'combination']:
                total_amount = sum(insurer_df.loc[insurer_indices, "ProcessedAmount_Clean_INSURER"])
                exact_matches += _apply_exact_match(
                    cbl_df, cbl_index, match['match_reason'], insurer_indices,
                    total_amount, match.get('fallback_indices', []), 1, global_tracker,
                    confidence_level=match.get('confidence_level'),
                    amount_difference=match.get('amount_difference')
                )
            else:
                # This should never happen as Pass 1 only creates exact matches
                logger.error(f"Pass 1: Unexpected match_type '{match_type}' - skipping")
                continue

    logger.info(f"✓ Pass 1 complete: {exact_matches} exact matches, {partial_matches} partial matches")
    return cbl_df
