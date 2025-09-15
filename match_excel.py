#!/usr/bin/env python3

import pandas as pd
import re
import argparse
from fuzzywuzzy import fuzz
import sys
import logging
import os
from itertools import combinations
from matrix import matrix_pass, build_key

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

def validate_column_mappings(cbl_df, insurer_df, column_mappings):
    """Validate that required columns exist in the dataframes after mapping."""
    logger.info("Validating column mappings...")
    
    # Check CBL mappings
    cbl_column_map = column_mappings['cbl_mappings']
    missing_cbl_columns = []
    for original_col, mapped_col in cbl_column_map.items():
        if original_col not in cbl_df.columns:
            missing_cbl_columns.append(original_col)
    
    if missing_cbl_columns:
        logger.warning(f"Missing CBL columns in source file: {missing_cbl_columns}")
    
    # Check insurer mappings
    insurer_column_map = column_mappings['insurer_mappings']
    missing_insurer_columns = []
    for original_col, mapped_col in insurer_column_map.items():
        if original_col not in insurer_df.columns:
            missing_insurer_columns.append(original_col)
    
    if missing_insurer_columns:
        logger.warning(f"Missing insurer columns in source file: {missing_insurer_columns}")
    
    # Log available columns for debugging
    logger.info(f"Available CBL columns: {list(cbl_df.columns)}")
    logger.info(f"Available insurer columns: {list(insurer_df.columns)}")
    
    return len(missing_cbl_columns) == 0 and len(missing_insurer_columns) == 0


def preprocess(cbl_df, insurer_df, column_mappings):
    logger.info("\n=== Starting Data Preprocessing ===")
    # Define a regular expression to match special characters and whitespace
    pattern = r'[^a-zA-Z0-9]'

    # Clean up unnamed columns that are likely empty
    unnamed_cols = [col for col in cbl_df.columns if col.startswith('Unnamed:')]
    if unnamed_cols:
        logger.info(f"Removing {len(unnamed_cols)} unnamed columns from CBL data")
        cbl_df = cbl_df.drop(columns=unnamed_cols)
    
    unnamed_cols_insurer = [col for col in insurer_df.columns if col.startswith('Unnamed:')]
    if unnamed_cols_insurer:
        logger.info(f"Removing {len(unnamed_cols_insurer)} unnamed columns from insurer data")
        insurer_df = insurer_df.drop(columns=unnamed_cols_insurer)

    # Validate column mappings first
    validate_column_mappings(cbl_df, insurer_df, column_mappings)

    # Normalize column names - Example mapping
    cbl_column_map = column_mappings['cbl_mappings']

    insurer_column_map = column_mappings['insurer_mappings']

    # Rename columns (only for columns that exist)
    existing_cbl_columns = {k: v for k, v in cbl_column_map.items() if k in cbl_df.columns}
    existing_insurer_columns = {k: v for k, v in insurer_column_map.items() if k in insurer_df.columns}
    
    cbl_df = cbl_df.rename(columns=existing_cbl_columns)
    insurer_df = insurer_df.rename(columns=existing_insurer_columns)
    
    # Log which columns were successfully renamed
    logger.info(f"Successfully renamed CBL columns: {list(existing_cbl_columns.values())}")
    logger.info(f"Successfully renamed insurer columns: {list(existing_insurer_columns.values())}")

    # Add _INSURER suffix to all insurer columns
    insurer_columns = list(insurer_df.columns)
    insurer_column_suffix_map = {col: col + '_INSURER' for col in insurer_columns}
    insurer_df = insurer_df.rename(columns=insurer_column_suffix_map)
    
    # Log available columns after renaming for debugging
    logger.info(f"Available insurer columns after renaming: {list(insurer_df.columns)}")

    # Create missing required columns with default values
    required_insurer_columns = ['PlacingNo_INSURER', 'PolicyNo_1_INSURER', 'Amount_INSURER', 'ClientName_INSURER']
    optional_insurer_columns = ['PolicyNo_2_INSURER']
    
    for col in required_insurer_columns:
        if col not in insurer_df.columns:
            logger.warning(f"Required column {col} not found - creating with empty values")
            insurer_df[col] = ""
    
    for col in optional_insurer_columns:
        if col not in insurer_df.columns:
            logger.info(f"Optional column {col} not found - creating with empty values")
            insurer_df[col] = ""

    # Clean and process data
    logger.info(f"DEBUG: Before data cleaning - CBL rows: {len(cbl_df)}, Insurer rows: {len(insurer_df)}")
    
    cbl_df["PlacingNo_Clean"] = cbl_df["PlacingNo"].str.upper().str.strip()
    cbl_df["PlacingNo_Clean"] = cbl_df["PlacingNo_Clean"].str.replace(pattern, '', regex=True)
    cbl_df["Amount_Clean"] = pd.to_numeric(cbl_df["Amount"], errors="coerce")

    insurer_df["PlacingNo_Clean_INSURER"] = insurer_df["PlacingNo_INSURER"].astype(str).str.upper().str.strip()
    insurer_df["PlacingNo_Clean_INSURER"] = insurer_df["PlacingNo_Clean_INSURER"].str.replace(pattern, '', regex=True)
    insurer_df["PolicyNo_Clean_INSURER"] = insurer_df["PolicyNo_1_INSURER"].astype(str).str.split(".").str[0]
    # Replace NaN values with empty string
    insurer_df["PolicyNo_Clean_INSURER"] = insurer_df["PolicyNo_Clean_INSURER"].fillna("")
    
    logger.info(f"DEBUG: After data cleaning - CBL rows: {len(cbl_df)}, Insurer rows: {len(insurer_df)}")
    
    # Handle optional PolicyNo_2 column
    if "PolicyNo_2_INSURER" in insurer_df.columns:
        insurer_df["PolicyNo_2_Clean_INSURER"] = insurer_df["PolicyNo_2_INSURER"].astype(str)
        # Replace NaN values with empty string
        insurer_df["PolicyNo_2_Clean_INSURER"] = insurer_df["PolicyNo_2_Clean_INSURER"].fillna("")
    else:
        # Create empty PolicyNo_2 column if it doesn't exist
        insurer_df["PolicyNo_2_Clean_INSURER"] = ""
        logger.info("PolicyNo_2_INSURER column not found in insurer file - treating as optional")
    
    insurer_df["Amount_Clean_INSURER"] = pd.to_numeric(insurer_df["Amount_INSURER"], errors="coerce")

    # Matrix Key
    cols_cbl =  [ 'PlacingNo', 'PolicyNo', 'ClientName', 'Amount' ]
    cols_insurer = [ 'PlacingNo_INSURER', 'PolicyNo_1_INSURER', 'ClientName_INSURER', 'Amount_INSURER' ]

    cbl_df["MatrixKey"] = cbl_df.apply(lambda row: build_key(row, cols_cbl), axis=1)
    insurer_df["MatrixKey_INSURER"] = insurer_df.apply(lambda row: build_key(row, cols_insurer), axis=1)

    logger.info(f"✓ Preprocessing complete: {len(cbl_df)} CBL records, {len(insurer_df)} insurer records")
    return cbl_df, insurer_df

def initialize_tracking(cbl_df):
    logger.info("Initializing tracking columns...")
    cbl_df["match_status"] = "No Match"
    cbl_df["match_pass"] = [[] for _ in range(len(cbl_df))]
    cbl_df["match_reason"] = ""
    cbl_df["matched_insurer_indices"] = [[] for _ in range(len(cbl_df))]
    cbl_df["matched_amtdue_total"] = None
    cbl_df["partial_candidates_indices"] = [[] for _ in range(len(cbl_df))]
    cbl_df["match_resolved_in_pass"] = None
    cbl_df["partial_resolved_in_pass"] = None
    return cbl_df

def add_pass(cbl_df, row_index, pass_number):
    existing = cbl_df.at[row_index, "match_pass"]
    if isinstance(existing, list):
        if pass_number not in existing:
            cbl_df.at[row_index, "match_pass"] = existing + [pass_number]
    elif isinstance(existing, int):
        if existing != pass_number:
            cbl_df.at[row_index, "match_pass"] = [existing, pass_number]
    else:
        cbl_df.at[row_index, "match_pass"] = [pass_number]

def pass1(cbl_df, insurer_df, tolerance=100):
    logger.info("\n=== Pass 1: Matching by Placing Number and Amount ===")
    total_records = len(cbl_df)
    exact_matches = 0
    partial_matches = 0
    
    # Track which insurer rows have been used for partial matches
    partial_used_insurer_indices = set()

    # Phase 1: Collect all potential matches without applying them yet
    potential_matches = []
    
    for i, row in cbl_df.iterrows():
        if i % 100 == 0:
            logger.info(f"Progress: {i+1}/{total_records} records processed")
            
        add_pass(cbl_df, i, 1)
        
        placing = row["PlacingNo_Clean"]
        amt1 = row["Amount_Clean"]

        insurer_matches = insurer_df[insurer_df["PlacingNo_Clean_INSURER"] == placing]
        
        # If no exact matches, try substring matching
        if insurer_matches.empty:
            insurer_placing_clean = insurer_df["PlacingNo_Clean_INSURER"].astype(str)
            placing_str = str(placing)
            
            # Check if CBL placing contains insurer placing OR insurer placing contains CBL placing
            # Only consider strings with at least 10 characters to avoid false matches
            substring_mask = insurer_placing_clean.apply(
                lambda x: (placing_str.find(x) != -1 or x.find(placing_str) != -1) 
                if x != 'nan' and len(x.strip()) >= 10 and len(placing_str.strip()) >= 10 
                else False
            )
            insurer_matches = insurer_df[substring_mask]

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
            
            # Check individual matches first
            for j, amt2 in zip(insurer_indices, insurer_amounts):
                if -tolerance <= (amt1 + amt2) <= tolerance:
                    exact_match_indices = [j]
                    break
        
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
            
            # Try combinations with the limited set (max 10 items per combination for performance)
            max_combination_size = min(10, len(limited_indices))
            
            for r in range(2, max_combination_size + 1):
                for combination in combinations(zip(limited_indices, limited_amounts), r):
                    combination_indices, combination_amounts = zip(*combination)
                    total_amount = sum(combination_amounts)
                    if -tolerance <= (amt1 + total_amount) <= tolerance:
                        combination_match_indices = list(combination_indices)
                        logger.info(f"Record {i}: Found combination match with {r} items, total: {total_amount}")
                        break
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
                'match_reason': 'Placing Number + Single Amount Match',
                'fallback_indices': [idx for idx in insurer_indices if idx not in exact_match_indices]
            })
        elif combination_match_indices is not None:
            potential_matches.append({
                'cbl_index': i,
                'match_type': 'combination',
                'insurer_indices': combination_match_indices,
                'match_reason': 'Placing Number + Cumulative Amount Match',
                'fallback_indices': [idx for idx in insurer_indices if idx not in combination_match_indices]
            })
        elif not insurer_matches.empty:
            # Only create partial match if there are some valid matches (even if not perfect)
            # Check if any insurer amounts are within a reasonable range (e.g., within 10x tolerance)
            reasonable_matches = []
            for idx, amt in zip(insurer_indices, insurer_amounts):
                # Check if the amount is within a reasonable range (not completely different)
                if pd.notna(amt) and abs(amt1 + amt) <= tolerance * 10:  # 10x tolerance for partial matches
                    reasonable_matches.append(idx)
            
            if reasonable_matches:
                # Partial match - there are some reasonable matches
                potential_matches.append({
                    'cbl_index': i,
                    'match_type': 'partial',
                    'insurer_indices': reasonable_matches,
                    'match_reason': 'Placing Number Match (Amount mismatch)',
                    'fallback_indices': reasonable_matches
                })
            # If no reasonable matches, don't create any match (will be No Match)
            # This ensures that only rows with actual reasonable matches are flagged as Partial Match

    # Phase 2: Resolve conflicts by prioritizing combination matches
    logger.info("\n=== Phase 2: Resolving conflicts and applying matches ===")
    
    # Sort potential matches: exact matches first, then combinations, then partial matches
    # Within each type, sort by number of insurer indices (smaller individual matches get priority)
    potential_matches.sort(key=lambda x: (
        0 if x['match_type'] == 'exact' else (1 if x['match_type'] == 'combination' else 2),
        len(x['insurer_indices'])  # Positive for ascending order (individual matches first)
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
            logger.info(f"Record {cbl_index}: Skipping {match_type} match due to conflicts with indices {conflicting_indices}")
            # Mark as partial match with remaining available indices
            available_indices = [idx for idx in insurer_indices if idx not in used_insurer_indices]
            if available_indices:
                partial_matches += 1
                cbl_df.at[cbl_index, "match_status"] = "Partial Match"
                cbl_df.at[cbl_index, "match_reason"] = f"{match['match_reason']}"
                cbl_df.at[cbl_index, "matched_insurer_indices"] = available_indices
                cbl_df.at[cbl_index, "matched_amtdue_total"] = insurer_df.loc[available_indices, "Amount_Clean_INSURER"].tolist()
                cbl_df.at[cbl_index, "partial_candidates_indices"] = available_indices
                cbl_df.at[cbl_index, "partial_resolved_in_pass"] = 1
                used_insurer_indices.update(available_indices)
            else:
                # All potential indices are already used - mark as No Match
                logger.info(f"Record {cbl_index}: All potential indices used by other records - marking as No Match")
                cbl_df.at[cbl_index, "match_status"] = "No Match"
                cbl_df.at[cbl_index, "match_reason"] = f"{match['match_reason']}"
                cbl_df.at[cbl_index, "matched_insurer_indices"] = []
                cbl_df.at[cbl_index, "matched_amtdue_total"] = None
                cbl_df.at[cbl_index, "partial_candidates_indices"] = []
        else:
            # Apply the match
            if match_type == 'exact':
                exact_matches += 1
                cbl_df.at[cbl_index, "match_status"] = "Exact Match"
                cbl_df.at[cbl_index, "match_reason"] = match['match_reason']
                cbl_df.at[cbl_index, "matched_insurer_indices"] = insurer_indices
                cbl_df.at[cbl_index, "matched_amtdue_total"] = sum(insurer_df.loc[insurer_indices, "Amount_Clean_INSURER"])
                cbl_df.at[cbl_index, "partial_candidates_indices"] = match['fallback_indices']
                cbl_df.at[cbl_index, "match_resolved_in_pass"] = 1
                used_insurer_indices.update(insurer_indices)
            elif match_type == 'combination':
                # Combination matches should be treated as partial matches if there are conflicts
                # Check if any of the insurer indices are already used
                conflicting_indices = set(insurer_indices) & used_insurer_indices
                if conflicting_indices:
                    # Mark as partial match with remaining available indices
                    available_indices = [idx for idx in insurer_indices if idx not in used_insurer_indices]
                    if available_indices:
                        partial_matches += 1
                        cbl_df.at[cbl_index, "match_status"] = "Partial Match"
                        cbl_df.at[cbl_index, "match_reason"] = f"{match['match_reason']}"
                        cbl_df.at[cbl_index, "matched_insurer_indices"] = available_indices
                        cbl_df.at[cbl_index, "matched_amtdue_total"] = insurer_df.loc[available_indices, "Amount_Clean_INSURER"].tolist()
                        cbl_df.at[cbl_index, "partial_candidates_indices"] = available_indices
                        cbl_df.at[cbl_index, "partial_resolved_in_pass"] = 1
                        used_insurer_indices.update(available_indices)
                    else:
                        # All potential indices are already used - mark as No Match
                        logger.info(f"Record {cbl_index}: All potential indices used by other records - marking as No Match")
                        cbl_df.at[cbl_index, "match_status"] = "No Match"
                        cbl_df.at[cbl_index, "match_reason"] = f"{match['match_reason']}"
                        cbl_df.at[cbl_index, "matched_insurer_indices"] = []
                        cbl_df.at[cbl_index, "matched_amtdue_total"] = None
                        cbl_df.at[cbl_index, "partial_candidates_indices"] = []
                else:
                    # No conflicts - apply as exact match
                    exact_matches += 1
                    cbl_df.at[cbl_index, "match_status"] = "Exact Match"
                    cbl_df.at[cbl_index, "match_reason"] = match['match_reason']
                    cbl_df.at[cbl_index, "matched_insurer_indices"] = insurer_indices
                    cbl_df.at[cbl_index, "matched_amtdue_total"] = sum(insurer_df.loc[insurer_indices, "Amount_Clean_INSURER"])
                    cbl_df.at[cbl_index, "partial_candidates_indices"] = match['fallback_indices']
                    cbl_df.at[cbl_index, "match_resolved_in_pass"] = 1
                    used_insurer_indices.update(insurer_indices)
            else:  # partial
                partial_matches += 1
                cbl_df.at[cbl_index, "match_status"] = "Partial Match"
                cbl_df.at[cbl_index, "match_reason"] = match['match_reason']
                cbl_df.at[cbl_index, "matched_insurer_indices"] = insurer_indices
                cbl_df.at[cbl_index, "matched_amtdue_total"] = insurer_df.loc[insurer_indices, "Amount_Clean_INSURER"].tolist()
                cbl_df.at[cbl_index, "partial_candidates_indices"] = insurer_indices
                cbl_df.at[cbl_index, "partial_resolved_in_pass"] = 1
                used_insurer_indices.update(insurer_indices)

    logger.info(f"✓ Pass 1 complete: {exact_matches} exact matches, {partial_matches} partial matches")
    return cbl_df

def extract_policy_tokens(policy_str):
    if pd.isna(policy_str):
        return []
    
    policy_str = str(policy_str).strip()
    
    # If it contains multiple '/' characters, just remove special characters
    if policy_str.count('/') > 1:
         # Split by spaces and clean each token
        tokens = policy_str.split()
        return [re.sub(r'[^a-zA-Z0-9]', '', token) for token in tokens]
    
    # If it's a pure number (no letters, no special characters), keep it as is
    if policy_str.isdigit():
        return [policy_str]
    
    # For all other cases, extract 4-5 digit numbers
    return re.findall(r'\b\d{4,5}\b', policy_str)

def oriupdate_others_after_upgrade(cbl_df, upgraded_row_index, used_insurer_indices):
    for insurer_idx in used_insurer_indices:
        for j, other_row in cbl_df.iterrows():
            if j == upgraded_row_index:
                continue

            partials = other_row.get("partial_candidates_indices", [])
            if insurer_idx in partials:
                updated_partials = [idx for idx in partials if idx != insurer_idx]
                cbl_df.at[j, "partial_candidates_indices"] = updated_partials

            if other_row["match_status"] != "Exact Match": 
                matched_indices = other_row.get("matched_insurer_indices", [])
                if insurer_idx in matched_indices:
                    try:
                        idx_to_remove = matched_indices.index(insurer_idx)
                        updated_matched = matched_indices[:idx_to_remove] + matched_indices[idx_to_remove+1:]
                        cbl_df.at[j, "matched_insurer_indices"] = updated_matched
    
                        matched_amounts = other_row.get("matched_amtdue_total", [])
                        if isinstance(matched_amounts, list) and len(matched_amounts) > idx_to_remove:
                            updated_amounts = matched_amounts[:idx_to_remove] + matched_amounts[idx_to_remove+1:]
                            cbl_df.at[j, "matched_amtdue_total"] = updated_amounts
                    except ValueError:
                        pass

def pass2(cbl_df, insurer_df, tolerance=100, name_threshold=95):
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

        if matched_indices and -tolerance <= (cbl_amt + total_amt) <= tolerance:
            # Exact match found
            # Determine if it's single or cumulative amount match
            if len(matched_indices) == 1:
                amount_match_type = 'Single Amount Match'
            else:
                amount_match_type = 'Cumulative Amount Match'
            
            potential_matches.append({
                'cbl_index': i,
                'match_type': 'exact',
                'insurer_indices': matched_indices,
                'match_reason': f'Policy + Name Match (CS: {highest_name_score}%) + {amount_match_type}',
                'total_amount': total_amt,
                'name_score': highest_name_score
            })
        elif matched_indices:
            # Partial match found
            potential_matches.append({
                'cbl_index': i,
                'match_type': 'partial',
                'insurer_indices': matched_indices,
                'match_reason': f'Policy + Name Match (CS: {highest_name_score}%) (Amount mismatch)',
                'total_amount': None,
                'name_score': highest_name_score
            })

    # Phase 2: Resolve conflicts and apply matches
    logger.info("\n=== Pass 2 Phase 2: Resolving conflicts and applying matches ===")
    
    # Sort potential matches: exact matches first, then partial matches
    # Within each type, sort by number of insurer indices (smaller individual matches get priority)
    potential_matches.sort(key=lambda x: (
        0 if x['match_type'] == 'exact' else 1,
        len(x['insurer_indices'])  # Positive for ascending order (individual matches first)
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
            logger.info(f"Pass 2 Record {cbl_index}: Skipping {match_type} match due to conflicts with indices {conflicting_indices}")
            # Mark as partial match with remaining available indices
            available_indices = [idx for idx in insurer_indices if idx not in used_insurer_indices]
            if available_indices:
                partial_matches += 1
                cbl_df.at[cbl_index, "match_status"] = "Partial Match"
                cbl_df.at[cbl_index, "match_reason"] = f"{match['match_reason']}"
                cbl_df.at[cbl_index, "matched_insurer_indices"] = available_indices
                cbl_df.at[cbl_index, "matched_amtdue_total"] = fallback_rows.loc[available_indices, "Amount_Clean_INSURER"].tolist()
                cbl_df.at[cbl_index, "partial_candidates_indices"] = []
                cbl_df.at[cbl_index, "partial_resolved_in_pass"] = 2
                used_insurer_indices.update(available_indices)
            else:
                # All potential indices are already used - mark as No Match
                logger.info(f"Pass 2 Record {cbl_index}: All potential indices used by other records - marking as No Match")
                cbl_df.at[cbl_index, "match_status"] = "No Match"
                cbl_df.at[cbl_index, "match_reason"] = f"{match['match_reason']}"
                cbl_df.at[cbl_index, "matched_insurer_indices"] = []
                cbl_df.at[cbl_index, "matched_amtdue_total"] = None
                cbl_df.at[cbl_index, "partial_candidates_indices"] = []
        else:
            # Apply the match
            if match_type == 'exact':
                exact_matches += 1
                cbl_df.at[cbl_index, "match_status"] = "Exact Match"
                cbl_df.at[cbl_index, "match_reason"] = match['match_reason']
                cbl_df.at[cbl_index, "matched_insurer_indices"] = insurer_indices
                cbl_df.at[cbl_index, "matched_amtdue_total"] = match['total_amount']
                cbl_df.at[cbl_index, "partial_candidates_indices"] = []
                cbl_df.at[cbl_index, "match_resolved_in_pass"] = 2
                used_insurer_indices.update(insurer_indices)
                
                # Update other CBL rows that might be affected
                oriupdate_others_after_upgrade(cbl_df, cbl_index, insurer_indices)
            else:  # partial
                partial_matches += 1
                cbl_df.at[cbl_index, "match_status"] = "Partial Match"
                cbl_df.at[cbl_index, "match_reason"] = match['match_reason']
                cbl_df.at[cbl_index, "matched_insurer_indices"] = insurer_indices
                cbl_df.at[cbl_index, "matched_amtdue_total"] = fallback_rows.loc[insurer_indices, "Amount_Clean_INSURER"].tolist()
                cbl_df.at[cbl_index, "partial_candidates_indices"] = []
                cbl_df.at[cbl_index, "partial_resolved_in_pass"] = 2
                used_insurer_indices.update(insurer_indices)

    logger.info(f"✓ Pass 2 complete: {exact_matches} exact matches, {partial_matches} partial matches")
    return cbl_df

def pass3(cbl_df, insurer_df, tolerance=100, fuzzy_threshold=95):
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

       
                # print("insurer_total --> ", insurer_total)
                # print("difference --> ", difference)
                
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
        
        # First: Try row-by-row matching
        for idx, amt in zip(matching_rows.index, matching_amounts):
            if idx not in processed_rows and pd.notna(amt) and -tolerance <= (cbl_amt + amt) <= tolerance:
                matched_indices.append(idx)
                processed_rows.add(idx)
        
        # Second: Try cumulative matching with remaining unprocessed rows
        unprocessed_indices = [idx for idx in name_matches.index if idx not in processed_rows]
        if unprocessed_indices:
            unprocessed_amounts = matching_amounts[unprocessed_indices]
            # Filter out NaN values before summing
            valid_amounts = unprocessed_amounts[unprocessed_amounts.notna()]
            print("valid_amounts --> ", valid_amounts)

            if not valid_amounts.empty:
                total_unprocessed_amount = valid_amounts.sum()

                print("total_unprocessed_amount --> ", total_unprocessed_amount)
                print("cbl_amt --> ", cbl_amt)
                print("difference --> ", cbl_amt + total_unprocessed_amount)
                
                if -tolerance <= (cbl_amt + total_unprocessed_amount) <= tolerance:
                    print('YESSSS MATCHED')
                    matched_indices.extend(unprocessed_indices)
                    processed_rows.update(unprocessed_indices)
        
        # Add to potential matches
        if matched_indices and cbl_df.at[i, "match_status"] != "Exact Match":
            # Get the highest name score among the matched indices
            highest_name_score = name_scores[matched_indices].max()
            
            print("matched --> ", cbl_df.loc[matched_indices])

            # Determine match reason based on how the match was found
            if len(matched_indices) == 1:
                match_reason = f"Name Match (CS: {highest_name_score}%) + Single Amount Match"
            else:
                match_reason = f"Name Match (CS: {highest_name_score}%) + Cumulative Amount Match"
            
            potential_matches.append({
                'match_type': 'exact',
                'cbl_index': i,
                'insurer_indices': matched_indices,
                'match_reason': match_reason,
                'total_amount': sum(insurer_df.loc[matched_indices, "Amount_Clean_INSURER"]),
                'name_score': highest_name_score
            })
        else:
            # If no exact matches found, mark as partial match with all similar names
            # But only if they haven't been used for partial matches before
            available_partial_indices = [idx for idx in name_matches.index if idx not in partial_used_insurer]
            
            if available_partial_indices:
                # Get the highest name score among the available partial indices
                highest_name_score = name_scores[available_partial_indices].max()
                
                potential_matches.append({
                    'match_type': 'partial',
                    'cbl_index': i,
                    'insurer_indices': available_partial_indices,
                    'match_reason': f"Name Match (CS: {highest_name_score}%) (Amount mismatch)",
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
                logger.info(f"Pass 3 Record {match['cbl_index']}: Skipping {match_type} match due to conflicts with indices {conflicting_indices}")
                # Mark as partial match with remaining available indices
                available_indices = [idx for idx in insurer_indices if idx not in used_insurer_indices]
                if available_indices:
                    partial_matches += 1
                    cbl_df.at[match['cbl_index'], "match_status"] = "Partial Match"
                    cbl_df.at[match['cbl_index'], "match_reason"] = f"{match['match_reason']}"
                    cbl_df.at[match['cbl_index'], "matched_insurer_indices"] = available_indices
                    cbl_df.at[match['cbl_index'], "matched_amtdue_total"] = available_insurer.loc[available_indices, "Amount_Clean_INSURER"].tolist()
                    cbl_df.at[match['cbl_index'], "partial_candidates_indices"] = []
                    cbl_df.at[match['cbl_index'], "partial_resolved_in_pass"] = 3
                    used_insurer_indices.update(available_indices)
                else:
                    # All potential indices are already used - mark as No Match
                    logger.info(f"Pass 3 Record {match['cbl_index']}: All potential indices used by other records - marking as No Match")
                    cbl_df.at[match['cbl_index'], "match_status"] = "No Match"
                    cbl_df.at[match['cbl_index'], "match_reason"] = f"{match['match_reason']}"
                    cbl_df.at[match['cbl_index'], "matched_insurer_indices"] = []
                    cbl_df.at[match['cbl_index'], "matched_amtdue_total"] = None
                    cbl_df.at[match['cbl_index'], "partial_candidates_indices"] = []
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
                exact_matches += 1
                cbl_df.at[match['cbl_index'], "match_status"] = "Exact Match"
                cbl_df.at[match['cbl_index'], "match_reason"] = match['match_reason']
                cbl_df.at[match['cbl_index'], "matched_insurer_indices"] = insurer_indices
                cbl_df.at[match['cbl_index'], "matched_amtdue_total"] = match['total_amount']
                cbl_df.at[match['cbl_index'], "partial_candidates_indices"] = []
                cbl_df.at[match['cbl_index'], "match_resolved_in_pass"] = 3
                used_insurer_indices.update(insurer_indices)
                
                # Update other CBL rows that might be affected
                oriupdate_others_after_upgrade(cbl_df, match['cbl_index'], insurer_indices)
                
            else:  # partial
                partial_matches += 1
                cbl_df.at[match['cbl_index'], "match_status"] = "Partial Match"
                cbl_df.at[match['cbl_index'], "match_reason"] = match['match_reason']
                cbl_df.at[match['cbl_index'], "matched_insurer_indices"] = insurer_indices
                cbl_df.at[match['cbl_index'], "matched_amtdue_total"] = available_insurer.loc[insurer_indices, "Amount_Clean_INSURER"].tolist()
                cbl_df.at[match['cbl_index'], "partial_candidates_indices"] = []
                cbl_df.at[match['cbl_index'], "partial_resolved_in_pass"] = 3
                used_insurer_indices.update(insurer_indices)

    logger.info(f"✓ Pass 3 complete: {exact_matches} exact matches, {partial_matches} partial matches (including improved group matching)")
    return cbl_df


def _extract_insurer_indices(cbl_row):
    """
    Extract insurer indices from a CBL row, handling both list and single value formats.
    
    Args:
        cbl_row: CBL row containing matched_insurer_indices
        
    Returns:
        set: Set of insurer indices
    """
    insurer_indices = cbl_row['matched_insurer_indices']
    if isinstance(insurer_indices, list):
        return set(insurer_indices)
    elif pd.notna(insurer_indices):
        return {insurer_indices}
    else:
        return set()


def _get_insurer_rows_for_group(group_cbl_rows, insurer_df):
    """
    Get all insurer rows that are matched to any CBL row in the group.
    
    Args:
        group_cbl_rows: List of CBL rows in the group
        insurer_df: Insurer dataframe
        
    Returns:
        list: List of insurer rows
    """
    all_insurer_indices = set()
    for cbl_row in group_cbl_rows:
        all_insurer_indices.update(_extract_insurer_indices(cbl_row))
    
    insurer_rows = []
    for insurer_idx in all_insurer_indices:
        insurer_row = insurer_df.iloc[insurer_idx]
        insurer_rows.append(insurer_row)
    
    return insurer_rows


def _create_zipped_row(cbl_row, insurer_row, cbl_cols, insurer_cols, preserve_match_info=True):
    """
    Create a single row that combines CBL and insurer data.
    
    Args:
        cbl_row: CBL row data (can be None)
        insurer_row: Insurer row data (can be None)
        cbl_cols: List of CBL column names
        insurer_cols: List of insurer column names
        preserve_match_info: Whether to preserve match-related columns when clearing CBL data
        
    Returns:
        dict: Combined row data
    """
    # Start with an empty row
    new_row = {}
    
    # Add CBL data if available
    if cbl_row is not None:
        for col in cbl_cols:
            new_row[col] = cbl_row[col]
    else:
        # Clear CBL data but preserve match info if requested
        for col in cbl_cols:
            if preserve_match_info and col in ['match_status', 'match_reason', 'matched_insurer_indices', 'matched_amtdue_total', 'partial_candidates_indices']:
                continue  # Keep match info
            new_row[col] = None
    
    # Add insurer data if available
    if insurer_row is not None:
        for col in insurer_cols:
            new_row[col] = insurer_row[col]
    else:
        # Clear insurer data
        for col in insurer_cols:
            new_row[col] = None
    
    # Handle MatrixKey: preserve from CBL row if available, clear if no CBL data
    if cbl_row is not None and 'MatrixKey' in cbl_row:
        new_row['MatrixKey'] = cbl_row['MatrixKey']
    else:
        # Clear MatrixKey if no CBL data (insurer-only rows)
        new_row['MatrixKey'] = None
    
    # Handle MatrixKey_INSURER: preserve from insurer row if available, clear if no insurer data
    if insurer_row is not None and 'MatrixKey_INSURER' in insurer_row:
        new_row['MatrixKey_INSURER'] = insurer_row['MatrixKey_INSURER']
    else:
        # Clear MatrixKey_INSURER if no insurer data (CBL-only rows)
        new_row['MatrixKey_INSURER'] = None
    
    return new_row


def _process_group_match(group_cbl_rows, insurer_rows, cbl_cols, insurer_cols):
    """
    Process a group match by zipping CBL and insurer rows together.
    
    Args:
        group_cbl_rows: List of CBL rows in the group
        insurer_rows: List of insurer rows in the group
        cbl_cols: List of CBL column names
        insurer_cols: List of insurer column names
        
    Returns:
        list: List of combined rows
    """
    combined_rows = []
    
    # Only create rows for CBL rows that have corresponding insurer rows
    # This prevents creating insurer-only rows that cause duplicates
    for i, cbl_row in enumerate(group_cbl_rows):
        if i < len(insurer_rows):
            # Get corresponding insurer row
            insurer_row = insurer_rows[i]
            
            # Create combined row
            combined_row = _create_zipped_row(cbl_row, insurer_row, cbl_cols, insurer_cols)
            combined_rows.append(combined_row)
    
    return combined_rows


def _process_individual_match(cbl_row, insurer_df, cbl_cols, insurer_cols):
    """
    Process an individual match by creating rows for each matched insurer.
    
    Args:
        cbl_row: CBL row with individual match
        insurer_df: Insurer dataframe
        cbl_cols: List of CBL column names
        insurer_cols: List of insurer column names
        
    Returns:
        list: List of combined rows
    """
    combined_rows = []
    insurer_indices = _extract_insurer_indices(cbl_row)
    
    # If no insurer indices, create a row with only CBL data
    if not insurer_indices:
        combined_row = _create_zipped_row(cbl_row, None, cbl_cols, insurer_cols)
        combined_rows.append(combined_row)
        return combined_rows
    
    for i, insurer_idx in enumerate(insurer_indices):
        # Get insurer row using DataFrame index directly
        insurer_row = insurer_df.iloc[insurer_idx]
        
        # For multiple insurers, only show CBL data in first row
        if i > 0:
            # For subsequent insurer rows, pass None for CBL data to clear MatrixKey
            cbl_row_copy = None
        else:
            cbl_row_copy = cbl_row
        
        # Create combined row
        combined_row = _create_zipped_row(cbl_row_copy, insurer_row, cbl_cols, insurer_cols)
        combined_rows.append(combined_row)
    
    return combined_rows


def _separate_group_and_individual_matches(cbl_subset):
    """
    Separate CBL rows into group matches and individual matches.
    
    Args:
        cbl_subset: CBL dataframe subset
        
    Returns:
        tuple: (group_matches_dict, individual_matches_list)
    """
    group_matches = {}
    individual_matches = []
    
    for _, cbl_row in cbl_subset.iterrows():
        match_reason = cbl_row.get('match_reason', '')
        if 'Name Group Match:' in match_reason:
            # Use match reason as group key
            group_key = match_reason
            if group_key not in group_matches:
                group_matches[group_key] = []
            group_matches[group_key].append(cbl_row)
        else:
            individual_matches.append(cbl_row)
    
    return group_matches, individual_matches


def explode_and_merge(cbl_subset, insurer_df):
    """
    Explode and merge CBL and insurer data into a combined dataframe.
    
    This function takes matched CBL records and their corresponding insurer records,
    then creates a combined output where:
    - Group matches are "zipped" together (CBL + insurer on same row where possible)
    - Individual matches show each CBL-insurer pair
    - The total rows = max(CBL_count, insurer_count) for group matches
    
    Args:
        cbl_subset: CBL dataframe with match information
        insurer_df: Insurer dataframe with insurer_row_index column
        
    Returns:
        pd.DataFrame: Combined dataframe with CBL and insurer data
    """
    cbl_copy = cbl_subset.copy()
    cbl_cols = list(cbl_copy.columns)
    insurer_cols = list(insurer_df.columns)
    
    # Separate group matches from individual matches
    group_matches, individual_matches = _separate_group_and_individual_matches(cbl_copy)
    
    exploded_rows = []
    
    # Process individual matches
    for cbl_row in individual_matches:
        individual_combined_rows = _process_individual_match(cbl_row, insurer_df, cbl_cols, insurer_cols)
        exploded_rows.extend(individual_combined_rows)
        

    # Process group matches
    for group_key, group_cbl_rows in group_matches.items():
        logger.info(f"Processing group match: {len(group_cbl_rows)} CBL rows")
        
        # Get all insurer rows for this group
        insurer_rows = _get_insurer_rows_for_group(group_cbl_rows, insurer_df)
        logger.info(f"Found {len(insurer_rows)} insurer rows for group")
        
        # Create zipped rows for group match
        group_combined_rows = _process_group_match(group_cbl_rows, insurer_rows, cbl_cols, insurer_cols)
        exploded_rows.extend(group_combined_rows)
    
    # Create result dataframe and reorder columns
    result_df = pd.DataFrame(exploded_rows)
    result_df = result_df[[col for col in cbl_cols if col in result_df.columns] + 
                         [col for col in insurer_cols if col in result_df.columns]]
    
    logger.info(f"Created {len(result_df)} combined rows from {len(cbl_copy)} CBL rows")
    return result_df


def run_matching_process(column_mappings, matrix_keys, cbl_file=None, insurer_file=None, output_file='output.xlsx', tolerance=100):
    """
    Run the matching process between CBL and insurer files.
    
    Args:
        cbl_file (str, optional): Path to the CBL Excel file. If None, will be prompted.
        insurer_file (str, optional): Path to the insurer Excel file. If None, will be prompted.
        output_file (str, optional): Output Excel file name. Defaults to 'output.xlsx'.
        tolerance (int, optional): Tolerance for amount matching. Defaults to 100.
    """
    logger.info("\n=== Starting Matching Process ===")
    
    if cbl_file is None or insurer_file is None:
        parser = argparse.ArgumentParser(description='Match data between two Excel files.')
        parser.add_argument('cbl_file', help='Path to the CBL Excel file')
        parser.add_argument('insurer_file', help='Path to the insurer Excel file')
        parser.add_argument('--output', '-o', default=output_file, help='Output Excel file name')
        args = parser.parse_args()
        cbl_file = args.cbl_file
        insurer_file = args.insurer_file
        output_file = args.output

    try:
        logger.info(f"Reading files: {cbl_file} and {insurer_file}")
        
        # Debug: Read raw files first to check original row counts
        cbl_raw = pd.read_excel(cbl_file)
        insurer_raw = pd.read_excel(insurer_file)
        logger.info(f"DEBUG: Raw CBL rows: {len(cbl_raw)}, Raw insurer rows: {len(insurer_raw)}")
        
        # Read Excel files with column filtering
        cbl_df = pd.read_excel(cbl_file,  usecols=lambda x: not x.startswith('Unnamed:') )
        insurer_df = pd.read_excel(insurer_file,  usecols=lambda x: not x.startswith('Unnamed:') )
        
        logger.info(f"DEBUG: After column filtering - CBL rows: {len(cbl_df)}, Insurer rows: {len(insurer_df)}")

        # Get the directory of the input files
        input_dir = os.path.dirname(os.path.abspath(cbl_file))
        output_path = os.path.join(input_dir, output_file)

        # Process the data
        clean_cbl, clean_insurer = preprocess(cbl_df, insurer_df, column_mappings)
        clean_cbl = initialize_tracking(clean_cbl)
        
        logger.info(f"DEBUG: After preprocessing - CBL rows: {len(clean_cbl)}, Insurer rows: {len(clean_insurer)}")

        # Run matching passes

        print("matrix_keys --> ", matrix_keys)
        clean_cbl, clean_insurer, matched_insurer_indices = matrix_pass(clean_cbl, clean_insurer, matrix_keys)

        # Run additional passes only on unmatched records
        if len(matched_insurer_indices) < len(clean_insurer):
            print("Running additional passes")
            
            # Helper function to check if required keys exist in column mappings
            def has_required_keys(cbl_required, insurer_required):
                cbl_mappings = column_mappings.get('cbl_mappings', {})
                insurer_mappings = column_mappings.get('insurer_mappings', {})
                
                # Check if all required CBL keys are mapped
                cbl_has_keys = all(any(target == key for target in cbl_mappings.values()) for key in cbl_required)
                
                # Check if all required insurer keys are mapped
                insurer_has_keys = all(any(target == key for target in insurer_mappings.values()) for key in insurer_required)
                
                return cbl_has_keys and insurer_has_keys
            
            # Pass 1: Requires PlacingNo and Amount
            if has_required_keys(['PlacingNo', 'Amount'], ['PlacingNo', 'Amount']):
                logger.info("✓ Pass 1: Required keys found in mappings - running Pass 1")
                clean_cbl = pass1(clean_cbl, clean_insurer, tolerance)
            else:
                logger.info("⚠ Pass 1: Required keys (PlacingNo, Amount) not found in mappings - skipping Pass 1")
            
            # Pass 2: Requires PolicyNo, ClientName, and Amount (CBL) + ClientName, Amount, and at least one PolicyNo (Insurer)
            cbl_has_pass2 = has_required_keys(['PolicyNo', 'ClientName', 'Amount'], [])
            insurer_has_pass2_base = has_required_keys([], ['ClientName', 'Amount'])
            
            # Check if insurer has at least one policy number column mapped
            insurer_mappings = column_mappings.get('insurer_mappings', {})
            has_policy1 = any(target == 'PolicyNo_1' for target in insurer_mappings.values())
            has_policy2 = any(target == 'PolicyNo_2' for target in insurer_mappings.values()) 
            insurer_has_policy = has_policy1 or has_policy2
            
            if cbl_has_pass2 and insurer_has_pass2_base and insurer_has_policy:
                logger.info("✓ Pass 2: Required keys found in mappings - running Pass 2")
                clean_cbl = pass2(clean_cbl, clean_insurer, tolerance)
            else:
                missing_keys = []
                if not cbl_has_pass2:
                    missing_keys.append("CBL: PolicyNo, ClientName, Amount")
                if not insurer_has_pass2_base:
                    missing_keys.append("Insurer: ClientName, Amount")
                if not insurer_has_policy:
                    missing_keys.append("Insurer: PolicyNo_1 or PolicyNo_2")
                logger.info(f"⚠ Pass 2: Required keys not found in mappings - skipping Pass 2. Missing: {'; '.join(missing_keys)}")
            
            # Pass 3: Requires ClientName and Amount
            if has_required_keys(['ClientName', 'Amount'], ['ClientName', 'Amount']):
                logger.info("✓ Pass 3: Required keys found in mappings - running Pass 3")
                clean_cbl = pass3(clean_cbl, clean_insurer, tolerance)
            else:
                logger.info("⚠ Pass 3: Required keys (ClientName, Amount) not found in mappings - skipping Pass 3")
                
        else:
            print("All records matched via matrix keys - skipping additional passes")

        # Track insurer indices by match type
        exact_match_insurer_indices = set()
        partial_match_insurer_indices = set()

        # Collect insurer indices from exact matches
        for indices in clean_cbl[clean_cbl["match_status"] == "Exact Match"]["matched_insurer_indices"]:
            if isinstance(indices, list):
                exact_match_insurer_indices.update(indices)
            elif pd.notna(indices):
                exact_match_insurer_indices.add(indices)
        
        # Collect insurer indices from partial matches
        for indices in clean_cbl[clean_cbl["match_status"] == "Partial Match"]["matched_insurer_indices"]:
            if isinstance(indices, list):
                partial_match_insurer_indices.update(indices)
            elif pd.notna(indices):
                partial_match_insurer_indices.add(indices)

        # Remove exact matches from partial matches to avoid double counting
        partial_match_insurer_indices = partial_match_insurer_indices - exact_match_insurer_indices

        # Calculate unmatched insurer indices BEFORE resetting index
        all_insurer_indices = set(clean_insurer.index)
        matched_insurer_indices = exact_match_insurer_indices | partial_match_insurer_indices
        unmatched_insurer_indices = all_insurer_indices - matched_insurer_indices

        # Calculate statistics BEFORE resetting index
        total_insurer_rows = len(clean_insurer)
        exact_match_insurer_count = len(exact_match_insurer_indices)
        partial_match_insurer_count = len(partial_match_insurer_indices)
        unmatched_insurer_count = len(unmatched_insurer_indices)
        
        logger.info(f"DEBUG: Statistics BEFORE reset_index:")
        logger.info(f"  - Total insurer rows: {total_insurer_rows}")
        logger.info(f"  - Exact match insurer indices: {exact_match_insurer_count}")
        logger.info(f"  - Partial match insurer indices: {partial_match_insurer_count}")
        logger.info(f"  - Unmatched insurer indices: {unmatched_insurer_count}")
        logger.info(f"  - Sum of all categories: {exact_match_insurer_count + partial_match_insurer_count + unmatched_insurer_count}")

        # Keep original index - no reset needed

        # Split clean_cbls
        exact_matches = clean_cbl[clean_cbl["match_status"] == "Exact Match"].copy() 
        partial_matches = clean_cbl[clean_cbl["match_status"] == "Partial Match"].copy()
        no_matches = clean_cbl[clean_cbl["match_status"] == "No Match"].copy()

        logger.info(f"DEBUG: Exact matches: {len(exact_matches)}")
        logger.info(f"DEBUG: Partial matches: {len(partial_matches)}")
        logger.info(f"DEBUG: No matches: {len(no_matches)}")

        # Explode matched_insurer_indices and merge with insurer
        exact_matches = explode_and_merge(exact_matches, clean_insurer)

        # Before exploding partials, ensure we don't carry any insurer indices that are exact-matched elsewhere
        if not partial_matches.empty:
            def _filter_partial_indices(row):
                indices = row.get("matched_insurer_indices", [])
                if not isinstance(indices, list):
                    indices = [indices]
                # remove any indices that are in exact_match_insurer_indices
                filtered = [idx for idx in indices if idx not in exact_match_insurer_indices]
                return filtered

            partial_matches = partial_matches.copy()
            partial_matches["matched_insurer_indices"] = partial_matches.apply(_filter_partial_indices, axis=1)
            
            # Fix: Check for rows with no insurer data and mark them as No Match
            def _has_insurer_data(row):
                insurer_indices = row.get("matched_insurer_indices", [])
                partial_candidates = row.get("partial_candidates_indices", [])
                
                # Check if insurer_indices is not empty
                has_insurer = (
                    isinstance(insurer_indices, list) and len(insurer_indices) > 0
                ) or (
                    not isinstance(insurer_indices, list) and pd.notna(insurer_indices)
                )
                
                # Check if partial_candidates is not empty
                has_partial = (
                    isinstance(partial_candidates, list) and len(partial_candidates) > 0
                ) or (
                    not isinstance(partial_candidates, list) and pd.notna(partial_candidates)
                )
                
                return has_insurer or has_partial
            
            # Identify rows that should be No Match instead of Partial Match
            should_be_no_match = partial_matches[~partial_matches.apply(_has_insurer_data, axis=1)]
            
            if not should_be_no_match.empty:
                logger.info(f"Found {len(should_be_no_match)} partial match rows with no insurer data - marking as No Match")
                for idx in should_be_no_match.index:
                    partial_matches.at[idx, "match_status"] = "No Match"
                    logger.info(f"Fixed row {idx}: {partial_matches.at[idx, 'MatrixKey']} - {partial_matches.at[idx, 'match_reason']}")
                
                # Move these rows from partial_matches to no_matches
                fixed_rows = partial_matches[partial_matches["match_status"] == "No Match"].copy()
                partial_matches = partial_matches[partial_matches["match_status"] == "Partial Match"].copy()
                no_matches = pd.concat([no_matches, fixed_rows], ignore_index=False)
            
            # Don't drop rows where no partials remain - include them as CBL-only rows
            # This ensures all partial match CBL rows are included in the output

        partial_matches = explode_and_merge(partial_matches, clean_insurer)
        # no_matches = explode_and_merge(no_matches, clean_insurer)

        # Create unmatched insurer records
        unmatched_insurer = clean_insurer.iloc[list(unmatched_insurer_indices)].copy()

        # Update clean_cbl to reflect the post-processing fixes
        # This ensures summary statistics are calculated correctly
        if 'should_be_no_match' in locals() and not should_be_no_match.empty:
            for idx in should_be_no_match.index:
                clean_cbl.at[idx, "match_status"] = "No Match"
        
        # Create separate CBL sheets (without insurer data merged)
        exact_matches_cbl_only = clean_cbl[clean_cbl["match_status"] == "Exact Match"].copy()
        partial_matches_cbl_only = clean_cbl[clean_cbl["match_status"] == "Partial Match"].copy()
        no_matches_cbl_only = clean_cbl[clean_cbl["match_status"] == "No Match"].copy()

        # Create separate insurer sheets
        exact_match_insurer_only = clean_insurer.iloc[list(exact_match_insurer_indices)].copy()
        partial_match_insurer_only = clean_insurer.iloc[list(partial_match_insurer_indices)].copy()
        not_found_insurer_only = clean_insurer.iloc[list(unmatched_insurer_indices)].copy()

        # Write to Excel with 6 separate sheets
        try: 
            with pd.ExcelWriter(output_path) as writer:
                # CBL sheets (CBL data only)
                exact_matches_cbl_only.to_excel(writer, sheet_name="exact match cbl", index=False)
                partial_matches_cbl_only.to_excel(writer, sheet_name="partial match cbl", index=False)
                no_matches_cbl_only.to_excel(writer, sheet_name="not found cbl", index=False)
                
                # Insurer sheets (insurer data only)
                exact_match_insurer_only.to_excel(writer, sheet_name="exact match insurer", index=False)
                partial_match_insurer_only.to_excel(writer, sheet_name="partial match insurer", index=False)
                not_found_insurer_only.to_excel(writer, sheet_name="not found insurer", index=False)
                
                # Original combined sheets (for reference)
                exact_matches.to_excel(writer, sheet_name="Exact Matches", index=False)
                partial_matches.to_excel(writer, sheet_name="Partial Matches", index=False)
                no_matches.to_excel(writer, sheet_name="No Matches CBL", index=False)
                unmatched_insurer.to_excel(writer, sheet_name="No Matches Insurer", index=False)
        except Exception as e:
            logger.error(f"Error writing to Excel: {str(e)}")
            raise

        logger.info(f"✓ Results saved to: {output_path}")

        # Statistics are already calculated above, just use them here
        # logger.info(f"DEBUG: Final statistics (already calculated):")
        # logger.info(f"  - Total insurer rows: {total_insurer_rows}")
        # logger.info(f"  - Exact match insurer indices: {exact_match_insurer_count}")
        # logger.info(f"  - Partial match insurer indices: {partial_match_insurer_count}")
        # logger.info(f"  - Unmatched insurer indices: {unmatched_insurer_count}")
        # logger.info(f"  - Sum of all categories: {exact_match_insurer_count + partial_match_insurer_count + unmatched_insurer_count}")

        logger.info("\n=== Final Results ===")
        logger.info(f"✓ CBL Records:")
        logger.info(f"  - Total CBL rows: {len(clean_cbl)}")
        logger.info(f"  - Exact matches: {len(clean_cbl[clean_cbl['match_status'] == 'Exact Match'])}")
        logger.info(f"  - Partial matches: {len(clean_cbl[clean_cbl['match_status'] == 'Partial Match'])}")
        logger.info(f"  - No matches: {len(clean_cbl[clean_cbl['match_status'] == 'No Match'])}")
        logger.info(f"✓ Insurer Records:")
        logger.info(f"  - Total insurer rows: {total_insurer_rows}")
        logger.info(f"  - Exact match insurer rows: {exact_match_insurer_count} ({exact_match_insurer_count/total_insurer_rows*100:.1f}%)")
        logger.info(f"  - Partial match insurer rows: {partial_match_insurer_count} ({partial_match_insurer_count/total_insurer_rows*100:.1f}%)")
        logger.info(f"  - Unmatched insurer rows: {unmatched_insurer_count} ({unmatched_insurer_count/total_insurer_rows*100:.1f}%)")
        logger.info(f"✓ Results saved to: {output_path}")

       
        # Calculate amounts for different match types (ensure numeric conversion)
        cbl_exact_amount = pd.to_numeric(clean_cbl[clean_cbl['match_status'] == 'Exact Match']['Amount'], errors='coerce').sum()
        cbl_partial_amount = pd.to_numeric(clean_cbl[clean_cbl['match_status'] == 'Partial Match']['Amount'], errors='coerce').sum()
        cbl_no_match_amount = pd.to_numeric(clean_cbl[clean_cbl['match_status'] == 'No Match']['Amount'], errors='coerce').sum()
        
        # Calculate insurer amounts (ensure numeric conversion)
        exact_match_insurer_amount = pd.to_numeric(clean_insurer.iloc[list(exact_match_insurer_indices)]['Amount_INSURER'], errors='coerce').sum()
        partial_match_insurer_amount = pd.to_numeric(clean_insurer.iloc[list(partial_match_insurer_indices)]['Amount_INSURER'], errors='coerce').sum()
        unmatched_insurer_amount = pd.to_numeric(clean_insurer.iloc[list(unmatched_insurer_indices)]['Amount_INSURER'], errors='coerce').sum()
        
        return {
            'exact_matches': exact_matches,
            'partial_matches': partial_matches,
            'no_matches': no_matches,
            'unmatched_insurer': unmatched_insurer,
            'output_file': output_path,
            'cbl_stats': {
                'exact_matches': len(clean_cbl[clean_cbl['match_status'] == 'Exact Match']),
                'partial_matches': len(clean_cbl[clean_cbl['match_status'] == 'Partial Match']),
                'no_matches': len(clean_cbl[clean_cbl['match_status'] == 'No Match']),
                'exact_match_amount': cbl_exact_amount,
                'partial_match_amount': cbl_partial_amount,
                'no_match_amount': cbl_no_match_amount
            },
            'insurer_stats': {
                'total_rows': total_insurer_rows,
                'exact_match_rows': exact_match_insurer_count,
                'partial_match_rows': partial_match_insurer_count,
                'unmatched_rows': unmatched_insurer_count,
                'exact_match_rate': exact_match_insurer_count/total_insurer_rows*100,
                'partial_match_rate': partial_match_insurer_count/total_insurer_rows*100,
                'unmatched_rate': unmatched_insurer_count/total_insurer_rows*100,
                'exact_match_amount': exact_match_insurer_amount,
                'partial_match_amount': partial_match_insurer_amount,
                'unmatched_amount': unmatched_insurer_amount
            }
        }

    except Exception as e:
        logger.error(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    run_matching_process() 