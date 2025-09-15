#!/usr/bin/env python3

import pandas as pd
import logging
from itertools import combinations
from fuzzywuzzy import fuzz
from .utils import add_pass, extract_policy_tokens, oriupdate_others_after_upgrade

logger = logging.getLogger(__name__)


def classify_amount_match(amt1, amt2, tolerance):
    """
    Classify amount matching with business-relevant confidence levels.
    
    Args:
        amt1: CBL amount (usually negative)
        amt2: Insurer amount (usually positive) 
        tolerance: Base tolerance for exact matches
        
    Returns:
        tuple: (match_type, difference, confidence_level)
    """
    difference = abs(amt1 + amt2)
    
    if difference <= tolerance * 0.1:  # Within 10% of tolerance
        return "PERFECT_MATCH", difference, "Perfect"
    elif difference <= tolerance:  # Within tolerance
        return "EXACT_MATCH", difference, "High"
    elif difference <= tolerance * 2:  # Within 2x tolerance  
        return "CLOSE_MATCH", difference, "Medium"
    elif difference <= tolerance * 5:  # Within 5x tolerance
        return "REVIEW_REQUIRED", difference, "Low"
    elif difference <= tolerance * 10:  # Within 10x tolerance
        return "INVESTIGATION_REQUIRED", difference, "Very Low"
    else:
        return "NO_MATCH", difference, "None"


def _apply_exact_match(cbl_df, cbl_index, match_reason, insurer_indices, total_amount, fallback_indices, pass_number, confidence_level=None, amount_difference=None):
    """Apply an exact match to a CBL record."""
    cbl_df.at[cbl_index, "match_status"] = "Exact Match"
    cbl_df.at[cbl_index, "match_reason"] = match_reason
    cbl_df.at[cbl_index, "matched_insurer_indices"] = insurer_indices
    cbl_df.at[cbl_index, "matched_amtdue_total"] = total_amount
    cbl_df.at[cbl_index, "partial_candidates_indices"] = fallback_indices or []
    cbl_df.at[cbl_index, "match_resolved_in_pass"] = pass_number
    
    # Add confidence and difference information
    if confidence_level is not None:
        cbl_df.at[cbl_index, "match_confidence"] = confidence_level
    if amount_difference is not None:
        cbl_df.at[cbl_index, "amount_difference"] = amount_difference
        
    return 1  # Return count for exact matches


def _apply_partial_match(cbl_df, cbl_index, match_reason, insurer_indices, total_amount, pass_number, confidence_level=None, amount_difference=None):
    """Apply a partial match to a CBL record."""
    cbl_df.at[cbl_index, "match_status"] = "Partial Match"
    cbl_df.at[cbl_index, "match_reason"] = match_reason
    cbl_df.at[cbl_index, "matched_insurer_indices"] = insurer_indices
    cbl_df.at[cbl_index, "matched_amtdue_total"] = total_amount
    cbl_df.at[cbl_index, "partial_candidates_indices"] = []
    cbl_df.at[cbl_index, "partial_resolved_in_pass"] = pass_number
    
    # Add confidence and difference information
    if confidence_level is not None:
        cbl_df.at[cbl_index, "match_confidence"] = confidence_level
    if amount_difference is not None:
        cbl_df.at[cbl_index, "amount_difference"] = amount_difference
        
    return 1  # Return count for partial matches


def _apply_no_match(cbl_df, cbl_index, match_reason):
    """Apply a no match status to a CBL record."""
    cbl_df.at[cbl_index, "match_status"] = "No Match"
    cbl_df.at[cbl_index, "match_reason"] = match_reason
    cbl_df.at[cbl_index, "matched_insurer_indices"] = []
    cbl_df.at[cbl_index, "matched_amtdue_total"] = None
    cbl_df.at[cbl_index, "partial_candidates_indices"] = []


def _handle_conflict_resolution(cbl_df, insurer_df, match, used_insurer_indices, tolerance, pass_number, fallback_rows=None):
    """
    Handle conflict resolution with fallback logic.
    
    Args:
        cbl_df: CBL dataframe
        insurer_df: Insurer dataframe (or fallback_rows for Pass 2)
        match: Match dictionary with conflict
        used_insurer_indices: Set of already used insurer indices
        tolerance: Tolerance for amount matching
        pass_number: Which pass is calling this function
        fallback_rows: Optional fallback rows dataframe (for Pass 2)
        
    Returns:
        tuple: (exact_matches_added, partial_matches_added)
    """
    cbl_index = match['cbl_index']
    match_type = match['match_type']
    insurer_indices = match['insurer_indices']
    
    logger.info(f"Pass {pass_number} Record {cbl_index}: Handling conflicts for {match_type} match")
    
    # Try fallback indices first (better business alternatives)
    available_indices = []
    if 'fallback_indices' in match and match['fallback_indices']:
        available_fallback = [idx for idx in match['fallback_indices'] if idx not in used_insurer_indices]
        if available_fallback:
            logger.info(f"Record {cbl_index}: Using fallback indices {available_fallback}")
            available_indices = available_fallback
    
    # If no fallback available, use remaining original indices
    if not available_indices:
        available_indices = [idx for idx in insurer_indices if idx not in used_insurer_indices]
        if available_indices:
            logger.info(f"Record {cbl_index}: Using remaining original indices {available_indices}")
    
    if not available_indices:
        # All potential indices are already used - mark as No Match
        logger.info(f"Record {cbl_index}: All potential indices used - marking as No Match")
        _apply_no_match(cbl_df, cbl_index, match['match_reason'])
        return 0, 0
    
    # Calculate amounts using the appropriate dataframe
    data_source = fallback_rows if fallback_rows is not None else insurer_df
    cbl_amount = cbl_df.at[cbl_index, "Amount_Clean"]
    available_amounts = data_source.loc[available_indices, "Amount_Clean_INSURER"]
    total_available_amount = available_amounts.sum()
    
    # Check if fallback indices create a perfect match
    if -tolerance <= (cbl_amount + total_available_amount) <= tolerance:
        # Upgrade to exact match!
        logger.info(f"Record {cbl_index}: Fallback indices upgraded to Exact Match!")
        used_insurer_indices.update(available_indices)
        return _apply_exact_match(cbl_df, cbl_index, f"{match['match_reason']} (Fallback Match)", 
                                 available_indices, total_available_amount, [], pass_number), 0
    else:
        # Apply as partial match
        logger.info(f"Record {cbl_index}: Fallback indices as Partial Match")
        used_insurer_indices.update(available_indices)
        return 0, _apply_partial_match(cbl_df, cbl_index, f"{match['match_reason']} (Fallback Partial)",
                                      available_indices, total_available_amount, pass_number)


def pass1(cbl_df, insurer_df, tolerance=100):
    """Pass 1: Matching by Placing Number and Amount."""
    logger.info("\n=== Pass 1: Matching by Placing Number and Amount ===")
    total_records = len(cbl_df)
    exact_matches = 0
    partial_matches = 0

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
            
        add_pass(cbl_df, i, 1)
        
        placing = row["PlacingNo_Clean"]
        amt1 = row["Amount_Clean"]
        
        # Validate input data
        if pd.isna(placing) or placing == "" or str(placing).strip() == "":
            logger.warning(f"Record {i}: Empty or invalid placing number, skipping")
            continue
            
        if pd.isna(amt1):
            logger.warning(f"Record {i}: Invalid amount ({amt1}), skipping")
            continue

        insurer_matches = insurer_df[insurer_df["PlacingNo_Clean_INSURER"] == placing]
        
        # If no exact matches, try substring matching (optimized)
        if insurer_matches.empty:
            placing_str = str(placing).strip()
            
            # Only proceed if CBL placing is long enough (>= 10 chars)
            if len(placing_str) >= 10:
                # Use vectorized operations for better performance
                # Check if CBL placing contains insurer placing OR insurer placing contains CBL placing
                contains_cbl_mask = insurer_placing_strings.str.contains(placing_str, na=False, regex=False)
                
                # For the reverse check (insurer contains CBL), we need to check each valid string
                valid_insurer_strings = insurer_placing_strings[valid_insurer_mask]
                contained_in_cbl_indices = valid_insurer_strings[valid_insurer_strings.apply(lambda x: x in placing_str)].index
                contained_in_cbl_mask = insurer_df.index.isin(contained_in_cbl_indices)
                
                # Combine masks: (CBL contains insurer) OR (insurer contains CBL)
                substring_mask = (contains_cbl_mask & valid_insurer_mask) | contained_in_cbl_mask
                insurer_matches = insurer_df[substring_mask]
                
                if not insurer_matches.empty:
                    logger.info(f"Record {i}: Found {len(insurer_matches)} substring matches for placing '{placing_str}'")
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
            
            for idx, amt in zip(insurer_matches.index.tolist(), insurer_matches["Amount_Clean_INSURER"].tolist()):
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
                        # Auto-approve exact matches
                        best_match = {
                            'indices': [j], 
                            'type': 'exact',
                            'confidence': confidence,
                            'difference': difference,
                            'reason': f'Placing Number + Single Amount Match ({confidence} Confidence, Diff: ${difference:.2f})'
                        }
                        break
                    elif match_type == "CLOSE_MATCH" and best_match is None:
                        # Consider close matches if no exact match found
                        best_match = {
                            'indices': [j],
                            'type': 'close', 
                            'confidence': confidence,
                            'difference': difference,
                            'reason': f'Placing Number + Close Amount Match ({confidence} Confidence, Diff: ${difference:.2f})'
                        }
            
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
                            combination_match_reason = f'Placing Number + Cumulative Amount Match ({confidence} Confidence, Diff: ${difference:.2f})'
                            logger.info(f"Record {i}: Found combination match with {r} items, total: {total_amount}, confidence: {confidence}")
                            break
                        elif match_type == "CLOSE_MATCH" and combination_match_indices is None:
                            # Store close combination match as backup
                            combination_match_indices = list(combination_indices)
                            combination_match_confidence = confidence
                            combination_match_difference = difference
                            combination_match_reason = f'Placing Number + Close Cumulative Match ({confidence} Confidence, Diff: ${difference:.2f})'
                            logger.info(f"Record {i}: Found close combination match with {r} items, total: {total_amount}, confidence: {confidence}")
                            # Don't break - continue looking for exact matches
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
            potential_matches.append({
                'cbl_index': i,
                'match_type': 'combination',
                'insurer_indices': combination_match_indices,
                'match_reason': combination_match_reason,
                'confidence_level': combination_match_confidence,
                'amount_difference': combination_match_difference,
                'fallback_indices': [idx for idx in insurer_indices if idx not in combination_match_indices]
            })
        elif not insurer_matches.empty:
            # Only create partial match if there are some valid matches (even if not perfect)
            # Use graduated classification for partial matches too
            reasonable_matches = []
            best_partial_confidence = None
            best_partial_difference = None
            
            for idx, amt in zip(insurer_indices, insurer_amounts):
                if pd.notna(amt):
                    match_type, difference, confidence = classify_amount_match(amt1, amt, tolerance)
                    
                    if match_type in ["REVIEW_REQUIRED", "INVESTIGATION_REQUIRED"]:
                        reasonable_matches.append(idx)
                        # Track the best (lowest difference) partial match for reporting
                        if best_partial_difference is None or difference < best_partial_difference:
                            best_partial_confidence = confidence
                            best_partial_difference = difference
            
            if reasonable_matches:
                # Partial match - there are some reasonable matches
                # For partial matches, fallback should be ALL other insurer indices with same placing number
                # that weren't selected as reasonable matches
                fallback_candidates = [idx for idx in insurer_indices if idx not in reasonable_matches]
                
                partial_reason = f'Placing Number Match ({best_partial_confidence} Confidence, Diff: ${best_partial_difference:.2f})'
                
                potential_matches.append({
                    'cbl_index': i,
                    'match_type': 'partial',
                    'insurer_indices': reasonable_matches,
                    'match_reason': partial_reason,
                    'confidence_level': best_partial_confidence,
                    'amount_difference': best_partial_difference,
                    'fallback_indices': fallback_candidates
                })
            # If no reasonable matches, don't create any match (will be No Match)
            # This ensures that only rows with actual reasonable matches are flagged as Partial Match

    # Phase 2: Resolve conflicts by prioritizing combination matches
    logger.info("\n=== Phase 2: Resolving conflicts and applying matches ===")
    
    # Sort potential matches: exact matches first, then combinations, then partial matches
    # Within each type, sort by number of insurer indices (larger combinations get priority)
    potential_matches.sort(key=lambda x: (
        0 if x['match_type'] == 'exact' else (1 if x['match_type'] == 'combination' else 2),
        -len(x['insurer_indices'])  # Negative for descending order (larger combinations first)
    ))
    
    # Track used insurer indices
    used_insurer_indices = set()
    
    for match in potential_matches:
        cbl_index = match['cbl_index']
        match_type = match['match_type']
        insurer_indices = match['insurer_indices']
        
        # Check if any of the insurer indices are already used
        conflicting_indices = set(insurer_indices) & used_insurer_indices
        
        if conflicting_indices:
            # Use helper function for conflict resolution
            exact_added, partial_added = _handle_conflict_resolution(
                cbl_df, insurer_df, match, used_insurer_indices, tolerance, 1
            )
            exact_matches += exact_added
            partial_matches += partial_added
        else:
            # Apply the match using helper functions
            if match_type in ['exact', 'combination']:
                total_amount = sum(insurer_df.loc[insurer_indices, "Amount_Clean_INSURER"])
                exact_matches += _apply_exact_match(
                    cbl_df, cbl_index, match['match_reason'], insurer_indices, 
                    total_amount, match.get('fallback_indices', []), 1,
                    confidence_level=match.get('confidence_level'),
                    amount_difference=match.get('amount_difference')
                )
                used_insurer_indices.update(insurer_indices)
            else:  # partial
                total_amount = insurer_df.loc[insurer_indices, "Amount_Clean_INSURER"].sum()
                partial_matches += _apply_partial_match(
                    cbl_df, cbl_index, match['match_reason'], insurer_indices, total_amount, 1,
                    confidence_level=match.get('confidence_level'),
                    amount_difference=match.get('amount_difference')
                )
                used_insurer_indices.update(insurer_indices)

    logger.info(f"✓ Pass 1 complete: {exact_matches} exact matches, {partial_matches} partial matches")
    return cbl_df


def pass2(cbl_df, insurer_df, tolerance=100, name_threshold=95):
    """Pass 2: Matching by Policy Number and Name."""
    logger.info("\n=== Pass 2: Matching by Policy Number and Name ===")
    total_records = len(cbl_df[cbl_df["match_status"].isin(["No Match", "Partial Match"])])
    exact_matches = 0
    partial_matches = 0
    processed = 0

    # Track which insurer rows have been used for partial matches
    partial_used_insurer_indices = set()
    
    # Get all insurer indices that have been used for partial matches in previous passes
    for indices in cbl_df[cbl_df["match_status"] == "Partial Match"]["matched_insurer_indices"]:
        if isinstance(indices, list):
            partial_used_insurer_indices.update(indices)
        elif pd.notna(indices):
            partial_used_insurer_indices.add(indices)

    fallback_index_pool = set()

    for i, row in cbl_df.iterrows():
        fallback = row.get("partial_candidates_indices")
        if isinstance(fallback, list):
            fallback_index_pool.update(fallback)

    # Filter out insurer indices that have already been used for partial matches
    available_fallback_indices = fallback_index_pool - partial_used_insurer_indices
    fallback_rows = insurer_df.loc[list(available_fallback_indices)]
    logger.info(f"Found {len(available_fallback_indices)} potential matches from Pass 1 (excluding already used partial matches)")

    # Phase 1: Collect all potential matches without applying them yet
    potential_matches = []
    
    for i, row in cbl_df[cbl_df["match_status"].isin(["No Match", "Partial Match"])].iterrows():
        processed += 1
        if processed % 50 == 0:
            logger.info(f"Progress: {processed}/{total_records} records processed")

        add_pass(cbl_df, i, 2)
        tokens = extract_policy_tokens(row["PolicyNo"])
        cbl_name = str(row["ClientName"]).upper().strip()
        cbl_amt = row["Amount_Clean"]

        matched_indices = []
        name_scores = []

        for j, insurer_row in fallback_rows.iterrows():
            insurer_name = str(insurer_row["ClientName_INSURER"]).upper().strip()
            name_score = fuzz.partial_ratio(cbl_name, insurer_name)
            
            # Check PolicyNo_1 match (only if it's not empty)
            policy_no_1_match = False
            if pd.notna(insurer_row["PolicyNo_Clean_INSURER"]) and insurer_row["PolicyNo_Clean_INSURER"]:
                policy_no_1_match = insurer_row["PolicyNo_Clean_INSURER"] in tokens
            
            # Check PolicyNo_2 match (only if it exists and is not empty)
            policy_no_2_match = False
            if "PolicyNo_2_Clean_INSURER" in insurer_row.index and pd.notna(insurer_row["PolicyNo_2_Clean_INSURER"]) and insurer_row["PolicyNo_2_Clean_INSURER"]:
                policy_no_2_match = insurer_row["PolicyNo_2_Clean_INSURER"] in tokens
            
            policy_match = policy_no_1_match or policy_no_2_match

            if policy_match and name_score >= name_threshold:
                matched_indices.append(j)
                name_scores.append(name_score)

        total_amt = fallback_rows.loc[matched_indices, "Amount_Clean_INSURER"].sum()
        highest_name_score = max(name_scores) if name_scores else 0

        if matched_indices:
            # Classify the amount match using graduated confidence levels
            match_type, difference, confidence = classify_amount_match(cbl_amt, total_amt, tolerance)
            
            # Determine if it's single or cumulative amount match
            if len(matched_indices) == 1:
                amount_match_type = 'Single Amount Match'
            else:
                amount_match_type = 'Cumulative Amount Match'
            
            if match_type in ["PERFECT_MATCH", "EXACT_MATCH"]:
                # Exact match found
                match_reason = f'Policy + Name Match (CS: {highest_name_score}%) + {amount_match_type} ({confidence} Confidence, Diff: ${difference:.2f})'
                potential_matches.append({
                    'cbl_index': i,
                    'match_type': 'exact',
                    'insurer_indices': matched_indices,
                    'match_reason': match_reason,
                    'confidence_level': confidence,
                    'amount_difference': difference,
                    'total_amount': total_amt,
                    'name_score': highest_name_score
                })
            elif match_type == "CLOSE_MATCH":
                # Close match - treat as exact but with medium confidence
                match_reason = f'Policy + Name Match (CS: {highest_name_score}%) + Close {amount_match_type} ({confidence} Confidence, Diff: ${difference:.2f})'
                potential_matches.append({
                    'cbl_index': i,
                    'match_type': 'exact',  # Still treat as exact for processing
                    'insurer_indices': matched_indices,
                    'match_reason': match_reason,
                    'confidence_level': confidence,
                    'amount_difference': difference,
                    'total_amount': total_amt,
                    'name_score': highest_name_score
                })
            elif match_type in ["REVIEW_REQUIRED", "INVESTIGATION_REQUIRED"]:
                # Partial match found
                match_reason = f'Policy + Name Match (CS: {highest_name_score}%) + {amount_match_type} ({confidence} Confidence, Diff: ${difference:.2f})'
                potential_matches.append({
                    'cbl_index': i,
                    'match_type': 'partial',
                    'insurer_indices': matched_indices,
                    'match_reason': match_reason,
                    'confidence_level': confidence,
                    'amount_difference': difference,
                    'total_amount': total_amt,
                    'name_score': highest_name_score
                })
            # If NO_MATCH, don't add to potential matches

    # Phase 2: Resolve conflicts and apply matches
    logger.info("\n=== Pass 2 Phase 2: Resolving conflicts and applying matches ===")
    
    # Sort potential matches: exact matches first, then partial matches
    # Within each type, sort by number of insurer indices (larger combinations get priority)
    potential_matches.sort(key=lambda x: (
        0 if x['match_type'] == 'exact' else 1,
        -len(x['insurer_indices'])  # Negative for descending order (larger combinations first)
    ))
    
    # Track used insurer indices
    used_insurer_indices = set()
    
    for match in potential_matches:
        cbl_index = match['cbl_index']
        match_type = match['match_type']
        insurer_indices = match['insurer_indices']
        
        # Check if any of the insurer indices are already used
        conflicting_indices = set(insurer_indices) & used_insurer_indices
        
        if conflicting_indices:
            # Use helper function for conflict resolution
            exact_added, partial_added = _handle_conflict_resolution(
                cbl_df, insurer_df, match, used_insurer_indices, tolerance, 2, fallback_rows
            )
            exact_matches += exact_added
            partial_matches += partial_added
        else:
            # Apply the match using helper functions
            if match_type == 'exact':
                exact_matches += _apply_exact_match(
                    cbl_df, cbl_index, match['match_reason'], insurer_indices, 
                    match['total_amount'], [], 2,
                    confidence_level=match.get('confidence_level'),
                    amount_difference=match.get('amount_difference')
                )
                used_insurer_indices.update(insurer_indices)
                
                # Update other CBL rows that might be affected
                oriupdate_others_after_upgrade(cbl_df, cbl_index, insurer_indices)
            else:  # partial
                total_amount = fallback_rows.loc[insurer_indices, "Amount_Clean_INSURER"].sum()
                partial_matches += _apply_partial_match(
                    cbl_df, cbl_index, match['match_reason'], insurer_indices, total_amount, 2,
                    confidence_level=match.get('confidence_level'),
                    amount_difference=match.get('amount_difference')
                )
                used_insurer_indices.update(insurer_indices)

    logger.info(f"✓ Pass 2 complete: {exact_matches} exact matches, {partial_matches} partial matches")
    return cbl_df


def pass3(cbl_df, insurer_df, tolerance=100, fuzzy_threshold=95):
    """Pass 3: Final matching by Name and Amount (Row-by-Row + Cumulative + Improved Group Matching)."""
    logger.info("\n=== Pass 3: Final matching by Name and Amount (Row-by-Row + Cumulative + Improved Group Matching) ===")
    total_records = len(cbl_df[cbl_df["match_status"].isin(["No Match", "Partial Match"])])
    exact_matches = 0
    partial_matches = 0
    processed = 0

    # Get indices of insurer rows that were already matched in previous passes
    already_matched_insurer = set()
    for indices in cbl_df[cbl_df["match_status"] == "Exact Match"]["matched_insurer_indices"]:
        if isinstance(indices, list):
            already_matched_insurer.update(indices)
        elif pd.notna(indices):
            already_matched_insurer.add(indices)
    
    # Get indices of insurer rows that have been used for partial matches
    partial_used_insurer = set()
    for indices in cbl_df[cbl_df["match_status"] == "Partial Match"]["matched_insurer_indices"]:
        if isinstance(indices, list):
            partial_used_insurer.update(indices)
        elif pd.notna(indices):
            partial_used_insurer.add(indices)

    # Filter out insurer rows already used in exact matches. We allow using rows that
    # are currently in partial matches, because step 3 may upgrade them to exact.
    available_insurer = insurer_df[~insurer_df.index.isin(already_matched_insurer)].copy()
    
    # Pre-calculate name scores for all insurer rows
    insurer_names = available_insurer["ClientName_INSURER"].fillna("").str.upper().str.strip()
    insurer_amounts = available_insurer["Amount_Clean_INSURER"]

    # Phase 1: Collect all potential matches without applying them yet
    potential_matches = []
    
    # IMPROVED GROUP MATCHING: First, try to find group matches
    logger.info("=== Phase 1: Collecting group matches ===")
    
    # Group CBL rows by name (only unmatched and partial matches)
    cbl_groups = {}
    for idx, row in cbl_df.iterrows():
        if row['match_status'] in ['No Match', 'Partial Match']:
            name = str(row['ClientName']).upper().strip()
            if name not in cbl_groups:
                cbl_groups[name] = []
            cbl_groups[name].append(idx)
    
    # Group insurer rows by name
    insurer_groups = {}
    for idx, row in available_insurer.iterrows():
        name = str(row['ClientName_INSURER']).upper().strip()
        if name not in insurer_groups:
            insurer_groups[name] = []
        insurer_groups[name].append(idx)
    
    logger.info(f"Found {len(cbl_groups)} CBL name groups and {len(insurer_groups)} insurer name groups")
    
    # Find matching groups
    group_matches = []
    
    for cbl_name, cbl_indices in cbl_groups.items():
        for insurer_name, insurer_indices in insurer_groups.items():
            # Check if names are similar enough
            name_score = fuzz.partial_ratio(cbl_name, insurer_name)
            if name_score >= fuzzy_threshold:
                # Calculate totals
                cbl_total = cbl_df.loc[cbl_indices, 'Amount_Clean'].sum()
                insurer_total = available_insurer.loc[insurer_indices, 'Amount_Clean_INSURER'].sum()
                difference = cbl_total + insurer_total

                if -tolerance <= difference <= tolerance:
                    group_matches.append({
                        'cbl_name': cbl_name,
                        'insurer_name': insurer_name,
                        'cbl_indices': cbl_indices,
                        'insurer_indices': insurer_indices,
                        'cbl_total': cbl_total,
                        'insurer_total': insurer_total,
                        'difference': difference,
                        'name_score': name_score
                    })
    
    logger.info(f"Found {len(group_matches)} potential group matches")
    
    # Debug: Log details about group matching
    logger.info(f"DEBUG: CBL groups found: {len(cbl_groups)}")
    logger.info(f"DEBUG: Insurer groups found: {len(insurer_groups)}")
    multi_cbl_groups = {k: v for k, v in cbl_groups.items() if len(v) > 1}
    multi_insurer_groups = {k: v for k, v in insurer_groups.items() if len(v) > 1}
    logger.info(f"DEBUG: CBL groups with multiple rows: {len(multi_cbl_groups)}")
    logger.info(f"DEBUG: Insurer groups with multiple rows: {len(multi_insurer_groups)}")
    
    if group_matches:
        logger.info(f"DEBUG: First few group matches:")
        for i, match in enumerate(group_matches[:3]):
            logger.info(f"  {i+1}. {match['cbl_name'][:30]}... -> {match['insurer_name'][:30]}... (Score: {match['name_score']}%)")
    else:
        logger.info("DEBUG: No group matches found - checking why...")
        if multi_cbl_groups and multi_insurer_groups:
            logger.info(f"DEBUG: Sample CBL groups: {list(multi_cbl_groups.keys())[:3]}")
            logger.info(f"DEBUG: Sample insurer groups: {list(multi_insurer_groups.keys())[:3]}")
        else:
            logger.info(f"DEBUG: No multi-row groups found - CBL: {len(multi_cbl_groups)}, Insurer: {len(multi_insurer_groups)}")
    
    # Add group matches to potential matches
    for match in group_matches:
        potential_matches.append({
            'match_type': 'group',
            'cbl_indices': match['cbl_indices'],
            'insurer_indices': match['insurer_indices'],
            'match_reason': f'Name Group Match (CS: {match["name_score"]}%)',
            'cbl_total': match['cbl_total'],
            'insurer_total': match['insurer_total'],
            'difference': match['difference'],
            'name_score': match['name_score']
        })
    
    # Now process remaining unmatched records with traditional row-by-row and cumulative matching
    logger.info("=== Phase 1: Collecting individual matches ===")
    
    for i, row in cbl_df[cbl_df["match_status"].isin(["No Match", "Partial Match"])].iterrows():
        processed += 1
        if processed % 50 == 0:
            logger.info(f"Progress: {processed}/{total_records} records processed")

        add_pass(cbl_df, i, 3)

        cbl_name = str(row["ClientName"]).upper().strip() if pd.notna(row["ClientName"]) else ""
        cbl_amt = row["Amount_Clean"]   

        # Skip if CBL name is empty
        if not cbl_name:
            continue

        # Calculate name scores for all insurer rows at once
        name_scores = insurer_names.apply(lambda x: fuzz.partial_ratio(cbl_name, x) if x else 0)
        name_matches = name_scores[name_scores >= fuzzy_threshold]
        
        if name_matches.empty:
            continue

        # Get all matching rows and their amounts
        matching_rows = available_insurer.loc[name_matches.index]
        matching_amounts = insurer_amounts.loc[name_matches.index]
        
        # Track processed rows to avoid duplicates
        processed_rows = set()
        matched_indices = []
        
        # First: Try row-by-row matching with graduated confidence
        best_individual_match = None
        for idx, amt in zip(matching_rows.index, matching_amounts):
            if idx not in processed_rows and pd.notna(amt):
                match_type, difference, confidence = classify_amount_match(cbl_amt, amt, tolerance)
                
                if match_type in ["PERFECT_MATCH", "EXACT_MATCH"]:
                    matched_indices.append(idx)
                    processed_rows.add(idx)
                elif match_type == "CLOSE_MATCH" and best_individual_match is None:
                    # Store close match as backup
                    best_individual_match = {
                        'indices': [idx],
                        'confidence': confidence,
                        'difference': difference,
                        'type': 'close_individual'
                    }
        
        # Second: Try cumulative matching with remaining unprocessed rows
        unprocessed_indices = [idx for idx in name_matches.index if idx not in processed_rows]
        best_cumulative_match = None
        
        if unprocessed_indices:
            unprocessed_amounts = matching_amounts[unprocessed_indices]
            # Filter out NaN values before summing
            valid_amounts = unprocessed_amounts[unprocessed_amounts.notna()]

            if not valid_amounts.empty:
                total_unprocessed_amount = valid_amounts.sum()
                match_type, difference, confidence = classify_amount_match(cbl_amt, total_unprocessed_amount, tolerance)
                
                if match_type in ["PERFECT_MATCH", "EXACT_MATCH"]:
                    matched_indices.extend(unprocessed_indices)
                    processed_rows.update(unprocessed_indices)
                elif match_type == "CLOSE_MATCH" and not matched_indices:
                    # Store close cumulative match as backup if no exact individual matches
                    best_cumulative_match = {
                        'indices': unprocessed_indices,
                        'confidence': confidence,
                        'difference': difference,
                        'type': 'close_cumulative'
                    }
        
        # If no exact matches found, use the best close match
        if not matched_indices:
            if best_individual_match and (not best_cumulative_match or best_individual_match['difference'] <= best_cumulative_match['difference']):
                matched_indices = best_individual_match['indices']
                match_confidence = best_individual_match['confidence']
                match_difference = best_individual_match['difference']
                close_match_type = 'individual'
            elif best_cumulative_match:
                matched_indices = best_cumulative_match['indices']
                match_confidence = best_cumulative_match['confidence']
                match_difference = best_cumulative_match['difference']
                close_match_type = 'cumulative'
        
        # Add to potential matches
        if matched_indices and cbl_df.at[i, "match_status"] != "Exact Match":
            # Get the highest name score among the matched indices
            highest_name_score = name_scores[matched_indices].max()
            
            # Calculate total amount and determine confidence
            total_amount = sum(insurer_df.loc[matched_indices, "Amount_Clean_INSURER"])
            final_match_type, final_difference, final_confidence = classify_amount_match(cbl_amt, total_amount, tolerance)
            
            # Determine match reason based on how the match was found
            if len(matched_indices) == 1:
                base_reason = "Single Amount Match"
            else:
                base_reason = "Cumulative Amount Match"
            
            # Check if this was a close match that we accepted
            if 'match_confidence' in locals():
                # This was a close match
                if close_match_type == 'individual':
                    match_reason = f"Name Match (CS: {highest_name_score}%) + Close {base_reason} ({match_confidence} Confidence, Diff: ${match_difference:.2f})"
                else:
                    match_reason = f"Name Match (CS: {highest_name_score}%) + Close {base_reason} ({match_confidence} Confidence, Diff: ${match_difference:.2f})"
                confidence_to_use = match_confidence
                difference_to_use = match_difference
            else:
                # This was an exact match
                match_reason = f"Name Match (CS: {highest_name_score}%) + {base_reason} ({final_confidence} Confidence, Diff: ${final_difference:.2f})"
                confidence_to_use = final_confidence
                difference_to_use = final_difference
            
            potential_matches.append({
                'match_type': 'exact',
                'cbl_index': i,
                'insurer_indices': matched_indices,
                'match_reason': match_reason,
                'confidence_level': confidence_to_use,
                'amount_difference': difference_to_use,
                'total_amount': total_amount,
                'name_score': highest_name_score
            })
        else:
            # If no exact matches found, mark as partial match with all similar names
            # But only if they haven't been used for partial matches before
            available_partial_indices = [idx for idx in name_matches.index if idx not in partial_used_insurer]
            
            if available_partial_indices:
                # Get the highest name score among the available partial indices
                highest_name_score = name_scores[available_partial_indices].max()
                
                # Find the best partial match by amount difference
                best_partial_difference = None
                best_partial_confidence = None
                for idx in available_partial_indices:
                    amt = insurer_amounts.loc[idx]
                    if pd.notna(amt):
                        match_type, difference, confidence = classify_amount_match(cbl_amt, amt, tolerance)
                        if match_type in ["REVIEW_REQUIRED", "INVESTIGATION_REQUIRED"]:
                            if best_partial_difference is None or difference < best_partial_difference:
                                best_partial_difference = difference
                                best_partial_confidence = confidence
                
                if best_partial_confidence:
                    match_reason = f"Name Match (CS: {highest_name_score}%) ({best_partial_confidence} Confidence, Diff: ${best_partial_difference:.2f})"
                else:
                    match_reason = f"Name Match (CS: {highest_name_score}%) (Amount mismatch)"
                    best_partial_confidence = "Very Low"
                    best_partial_difference = abs(cbl_amt + insurer_amounts.loc[available_partial_indices[0]])
                
                potential_matches.append({
                    'match_type': 'partial',
                    'cbl_index': i,
                    'insurer_indices': available_partial_indices,
                    'match_reason': match_reason,
                    'confidence_level': best_partial_confidence,
                    'amount_difference': best_partial_difference,
                    'total_amount': None,
                    'name_score': highest_name_score
                })

    # Phase 2: Resolve conflicts and apply matches
    logger.info("\n=== Phase 2: Resolving conflicts and applying matches ===")
    
    # Sort potential matches: group matches first, then exact matches, then partial matches
    # Within each type, sort by number of insurer indices (larger groups get priority)
    potential_matches.sort(key=lambda x: (
        0 if x['match_type'] == 'group' else (1 if x['match_type'] == 'exact' else 2),
        -len(x['insurer_indices'])  # Negative for descending order (larger groups first)
    ))
    
    # Track used insurer indices
    used_insurer_indices = set()
    
    for match in potential_matches:
        match_type = match['match_type']
        insurer_indices = match['insurer_indices']
        
        # Check if any of the insurer indices are already used
        conflicting_indices = set(insurer_indices) & used_insurer_indices
        
        if conflicting_indices:
            if match_type == 'group':
                # For group matches, filter out conflicted indices and apply with available ones
                available_insurer_indices = [idx for idx in insurer_indices if idx not in used_insurer_indices]
                
                if available_insurer_indices:
                    logger.info(f"Pass 3 Group Match: Applying partial group match with {len(available_insurer_indices)}/{len(insurer_indices)} available insurer indices (conflicts: {conflicting_indices})")
                    
                    # Update the match with only available indices
                    match['insurer_indices'] = available_insurer_indices
                    # Recalculate totals with available indices
                    match['insurer_total'] = available_insurer.loc[available_insurer_indices, 'Amount_Clean_INSURER'].sum()
                    match['difference'] = match['cbl_total'] + match['insurer_total']
                else:
                    logger.info(f"Pass 3 Group Match: Skipping group match - all insurer indices already used (conflicts: {conflicting_indices})")
                    continue
            else:
                # Use helper function for conflict resolution (non-group matches)
                exact_added, partial_added = _handle_conflict_resolution(
                    cbl_df, available_insurer, match, used_insurer_indices, tolerance, 3
                )
                exact_matches += exact_added
                partial_matches += partial_added
        else:
            # Apply the match
            if match_type == 'group':
                # Update insurer_indices to reflect any filtering from conflict resolution
                insurer_indices = match['insurer_indices']
                logger.info(f"Applying group match: {len(match['cbl_indices'])} CBL rows vs {len(insurer_indices)} insurer rows (Total: {match['cbl_total']:.2f} + {match['insurer_total']:.2f} = {match['difference']:.2f})")
                
                # For group matches, we need to distribute the insurer indices among CBL rows
                # This prevents the same insurer rows from being assigned to multiple CBL rows
                cbl_indices = match['cbl_indices']
                
                # If we have more CBL rows than insurer rows, some CBL rows will share insurer rows
                # If we have more insurer rows than CBL rows, some insurer rows will be unused
                # We'll distribute them as evenly as possible
                
                # Create a mapping of CBL indices to their assigned insurer indices
                cbl_to_insurer_mapping = {}
                
                if len(cbl_indices) <= len(insurer_indices):
                    # More insurer rows than CBL rows - distribute insurer rows among CBL rows
                    for i, cbl_idx in enumerate(cbl_indices):
                        # Each CBL row gets one insurer row
                        cbl_to_insurer_mapping[cbl_idx] = [insurer_indices[i]]
                else:
                    # More CBL rows than insurer rows - distribute insurer rows as evenly as possible
                    # Ensure every CBL row gets at least one insurer index
                    insurer_per_cbl = len(insurer_indices) // len(cbl_indices)
                    remaining_insurers = len(insurer_indices) % len(cbl_indices)
                    
                    # If insurer_per_cbl is 0, we have more CBL rows than insurer rows
                    # This creates a problematic scenario where we can't properly distribute insurers
                    # without creating duplicates. We should skip this group match to avoid issues.
                    if insurer_per_cbl == 0:
                        # Skip this group match to avoid creating duplicates
                        logger.warning(f"Skipping group match with {len(cbl_indices)} CBL rows vs {len(insurer_indices)} insurer rows to avoid duplicates")
                        continue
                    else:
                        # Normal distribution
                        insurer_idx = 0
                        for i, cbl_idx in enumerate(cbl_indices):
                            # Calculate how many insurer rows this CBL row should get
                            num_insurers = insurer_per_cbl + (1 if i < remaining_insurers else 0)
                            
                            # Assign the insurer rows
                            assigned_insurers = insurer_indices[insurer_idx:insurer_idx + num_insurers]
                            cbl_to_insurer_mapping[cbl_idx] = assigned_insurers
                            insurer_idx += num_insurers
                
                # Apply the matches with proper distribution
                for cbl_idx in cbl_indices:
                    if cbl_df.at[cbl_idx, 'match_status'] in ['No Match', 'Partial Match']:
                        assigned_insurer_indices = cbl_to_insurer_mapping[cbl_idx]
                        assigned_insurer_total = available_insurer.loc[assigned_insurer_indices, "Amount_Clean_INSURER"].sum()
                        
                        cbl_df.at[cbl_idx, 'match_status'] = 'Exact Match'
                        cbl_df.at[cbl_idx, 'match_reason'] = match['match_reason']
                        cbl_df.at[cbl_idx, 'matched_insurer_indices'] = assigned_insurer_indices
                        cbl_df.at[cbl_idx, 'matched_amtdue_total'] = assigned_insurer_total
                        cbl_df.at[cbl_idx, 'match_resolved_in_pass'] = 3
                        cbl_df.at[cbl_idx, 'partial_candidates_indices'] = []
                        exact_matches += 1
                
                used_insurer_indices.update(match['insurer_indices'])
                
            elif match_type == 'exact':
                exact_matches += _apply_exact_match(
                    cbl_df, match['cbl_index'], match['match_reason'], insurer_indices, 
                    match['total_amount'], [], 3,
                    confidence_level=match.get('confidence_level'),
                    amount_difference=match.get('amount_difference')
                )
                used_insurer_indices.update(insurer_indices)
                
                # Update other CBL rows that might be affected
                oriupdate_others_after_upgrade(cbl_df, match['cbl_index'], insurer_indices)
                
            else:  # partial
                total_amount = available_insurer.loc[insurer_indices, "Amount_Clean_INSURER"].sum()
                partial_matches += _apply_partial_match(
                    cbl_df, match['cbl_index'], match['match_reason'], insurer_indices, total_amount, 3,
                    confidence_level=match.get('confidence_level'),
                    amount_difference=match.get('amount_difference')
                )
                used_insurer_indices.update(insurer_indices)

    logger.info(f"✓ Pass 3 complete: {exact_matches} exact matches, {partial_matches} partial matches (including improved group matching)")
    return cbl_df
