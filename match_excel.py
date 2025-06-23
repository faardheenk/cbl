#!/usr/bin/env python3

import pandas as pd
import re
import argparse
from fuzzywuzzy import fuzz
import sys
import logging
import os
from itertools import combinations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

def preprocess(cbl_df, swan_df):
    logger.info("\n=== Starting Data Preprocessing ===")
    # Define a regular expression to match special characters and whitespace
    pattern = r'[^a-zA-Z0-9]'

    # Normalize column names - Example mapping
    cbl_column_map = {
        "Placing/Endorsement No.": "PlacingNo",
        "Policy No.": "PolicyNo_1", 
        "Client Name": "ClientName", 
        "Balance (MUR) (Net of Brokerage)": "Amount"
    }

    swan_column_map = {
        "BRKREF": "PlacingNo",
        "NAME": "ClientName", 
        "DOCSER": "PolicyNo_1",
        "POLSER": "PolicyNo_2",
        "AMTDUE": "Amount",
    }

    # Rename columns
    cbl_df = cbl_df.rename(columns=cbl_column_map)
    swan_df = swan_df.rename(columns=swan_column_map)

    # Clean and process data
    cbl_df["PlacingNo_Clean"] = cbl_df["PlacingNo"].str.upper().str.strip()
    cbl_df["PlacingNo_Clean"] = cbl_df["PlacingNo_Clean"].str.replace(pattern, '', regex=True)
    cbl_df["Amount_Clean"] = pd.to_numeric(cbl_df["Amount"], errors="coerce")

    swan_df["PlacingNo_Clean"] = swan_df["PlacingNo"].astype(str).str.upper().str.strip()
    swan_df["PlacingNo_Clean"] = swan_df["PlacingNo_Clean"].str.replace(pattern, '', regex=True)
    swan_df["PolicyNo_1_Clean"] = swan_df["PolicyNo_1"].astype(str).str.split(".").str[0]
    swan_df["PolicyNo_2_Clean"] = swan_df["PolicyNo_2"].astype(str)
    swan_df["Amount_Clean"] = pd.to_numeric(swan_df["Amount"], errors="coerce")

    logger.info(f"✓ Preprocessing complete: {len(cbl_df)} CBL records, {len(swan_df)} SWAN records")
    return cbl_df, swan_df

def initialize_tracking(cbl_df):
    logger.info("Initializing tracking columns...")
    cbl_df["match_status"] = "No Match"
    cbl_df["match_pass"] = [[] for _ in range(len(cbl_df))]
    cbl_df["match_reason"] = ""
    cbl_df["matched_swan_indices"] = [[] for _ in range(len(cbl_df))]
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

def pass1(cbl_df, swan_df):
    logger.info("\n=== Pass 1: Matching by Placing Number and Amount ===")
    total_records = len(cbl_df)
    exact_matches = 0
    partial_matches = 0

    for i, row in cbl_df.iterrows():
        if i % 100 == 0:
            logger.info(f"Progress: {i+1}/{total_records} records processed")
            
        add_pass(cbl_df, i, 1)
        
        placing = row["PlacingNo_Clean"]
        amt1 = row["Amount_Clean"]

        swan_matches = swan_df[swan_df["PlacingNo_Clean"] == placing]

        # First comparison - exact matches
        exact_match_indices = None
        exact_partial_count = 0
        swan_indices = []  # Initialize swan_indices outside the if block
        
        if not swan_matches.empty:
            swan_indices = swan_matches.index.tolist()
            swan_amounts = swan_matches["Amount_Clean"].tolist()
            exact_partial_count = len(swan_indices)
            
            # Check individual matches first
            for j, amt2 in zip(swan_indices, swan_amounts):
                if -10 <= (amt1 + amt2) <= 10:
                    exact_match_indices = [j]
                    break
        
        # Second comparison - combinations
        combo_match_indices = None
        combo_partial_count = 0
        if not swan_matches.empty:
            combo_partial_count = len(swan_indices)
            for r in range(2, len(swan_indices) + 1):
                for combo in combinations(zip(swan_indices, swan_amounts), r):
                    combo_indices, combo_amounts = zip(*combo)
                    if -10 <= (amt1 + sum(combo_amounts)) <= 10:
                        combo_match_indices = list(combo_indices)
                        break
                if combo_match_indices is not None:
                    break

        # Third comparison - substring matches
        substring_match_indices = None
        substring_partial_count = 0
        # Get unmatched SWAN rows (those not in swan_matches)
        unmatched_swan = swan_df[~swan_df.index.isin(swan_matches.index)]
        substring_partial_count = len(unmatched_swan)
        for j, swan_row in unmatched_swan.iterrows():
            swan_placing = swan_row["PlacingNo_Clean"]
            if swan_placing and swan_placing in placing:  # Check if non-empty SWAN placing is substring of CBL placing
                logger.info(f"Found substring match: SWAN placing '{swan_placing}' is a substring of CBL placing '{placing}'")
                swan_amt = swan_row["Amount_Clean"]
                if -10 <= (amt1 + swan_amt) <= 10:  # Check amount match
                    substring_match_indices = [j]
                    break

        # Log results for each comparison method
        logger.info(f"\nComparison results for CBL record {i}:")
        logger.info(f"Exact comparison: {1 if exact_match_indices else 0} exact matches, {exact_partial_count} partial matches")
        logger.info(f"Combo comparison: {1 if combo_match_indices else 0} exact matches, {combo_partial_count} partial matches")
        logger.info(f"Substring comparison: {1 if substring_match_indices else 0} exact matches, {substring_partial_count} partial matches")

        # Combine all matches found
        all_matches = []
        partial_matches_by_method = {
            'exact': [],
            'combo': [],
            'substring': []
        }
        
        if exact_match_indices is not None:
            all_matches.extend(exact_match_indices)
        elif not swan_matches.empty:
            partial_matches_by_method['exact'] = swan_indices

        if combo_match_indices is not None:
            all_matches.extend(combo_match_indices)
        elif not swan_matches.empty:
            partial_matches_by_method['combo'] = swan_indices

        if substring_match_indices is not None:
            all_matches.extend(substring_match_indices)
        elif not unmatched_swan.empty:
            partial_matches_by_method['substring'] = unmatched_swan.index.tolist()

        # Set fallback indices (all unmatched SWAN records for this placing)
        if all_matches:
            fallback_indices = [idx for idx in swan_indices if idx not in all_matches]
        else:
            fallback_indices = swan_indices
        
        if all_matches:
            exact_matches += 1
            cbl_df.at[i, "match_status"] = "Exact Match"
            cbl_df.at[i, "match_reason"] = "Placing + Amount"
            cbl_df.at[i, "matched_swan_indices"] = all_matches
            cbl_df.at[i, "matched_amtdue_total"] = sum(swan_df.loc[all_matches, "Amount_Clean"])
            cbl_df.at[i, "partial_candidates_indices"] = fallback_indices
            cbl_df.at[i, "match_resolved_in_pass"] = 1
        elif not swan_matches.empty:
            partial_matches += 1
            cbl_df.at[i, "match_status"] = "Partial Match"
            
            # Build match reason based on which methods found potential matches
            match_reasons = []
            if partial_matches_by_method['exact']:
                match_reasons.append("Exact placing match")
            if partial_matches_by_method['combo']:
                match_reasons.append("Combo placing match")
            if partial_matches_by_method['substring']:
                match_reasons.append("Substring placing match")
            
            cbl_df.at[i, "match_reason"] = " + ".join(match_reasons) if match_reasons else "Placing only"
            cbl_df.at[i, "matched_swan_indices"] = fallback_indices
            cbl_df.at[i, "matched_amtdue_total"] = swan_df.loc[fallback_indices, "Amount_Clean"].tolist()
            cbl_df.at[i, "partial_candidates_indices"] = fallback_indices
            cbl_df.at[i, "partial_resolved_in_pass"] = 1

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

def update_others_after_upgrade(cbl_df, upgraded_row_index, used_swan_indices):
    for swan_idx in used_swan_indices:
        for j, other_row in cbl_df.iterrows():
            if j == upgraded_row_index:
                continue

            partials = other_row.get("partial_candidates_indices", [])
            if swan_idx in partials:
                updated_partials = [idx for idx in partials if idx != swan_idx]
                cbl_df.at[j, "partial_candidates_indices"] = updated_partials

            if other_row["match_status"] != "Exact Match": 
                matched_indices = other_row.get("matched_swan_indices", [])
                if swan_idx in matched_indices:
                    try:
                        idx_to_remove = matched_indices.index(swan_idx)
                        updated_matched = matched_indices[:idx_to_remove] + matched_indices[idx_to_remove+1:]
                        cbl_df.at[j, "matched_swan_indices"] = updated_matched
    
                        matched_amounts = other_row.get("matched_amtdue_total", [])
                        if isinstance(matched_amounts, list) and len(matched_amounts) > idx_to_remove:
                            updated_amounts = matched_amounts[:idx_to_remove] + matched_amounts[idx_to_remove+1:]
                            cbl_df.at[j, "matched_amtdue_total"] = updated_amounts
                    except ValueError:
                        pass

def pass2(cbl_df, swan_df, tolerance=10, name_threshold=95):
    logger.info("\n=== Pass 2: Matching by Policy Number and Fuzzy Name ===")
    total_records = len(cbl_df[cbl_df["match_status"].isin(["No Match", "Partial Match"])])
    exact_matches = 0
    partial_matches = 0
    processed = 0

    fallback_index_pool = set()

    for i, row in cbl_df.iterrows():
        fallback = row.get("partial_candidates_indices")
        if isinstance(fallback, list):
            fallback_index_pool.update(fallback)

    fallback_rows = swan_df.loc[list(fallback_index_pool)]
    logger.info(f"Found {len(fallback_index_pool)} potential matches from Pass 1")

    for i, row in cbl_df[cbl_df["match_status"].isin(["No Match", "Partial Match"])].iterrows():
        processed += 1
        if processed % 50 == 0:
            logger.info(f"Progress: {processed}/{total_records} records processed")

        add_pass(cbl_df, i, 2)
        tokens = extract_policy_tokens(row["PolicyNo_1"])
        cbl_name = str(row["ClientName"]).upper().strip()
        cbl_amt = row["Amount_Clean"]

        matched_indices = []
        fallback_indices = []

        for j, sw_row in fallback_rows.iterrows():
            sw_name = str(sw_row["ClientName"]).upper().strip()
            name_score = fuzz.partial_ratio(cbl_name, sw_name)
            policy_match = sw_row["PolicyNo_1_Clean"] in tokens or sw_row["PolicyNo_2_Clean"] in tokens

            if policy_match and name_score >= name_threshold:
                matched_indices.append(j)

        total_amt = fallback_rows.loc[matched_indices, "Amount_Clean"].sum()

        if matched_indices and -10 <= (cbl_amt + total_amt) <= 10:
            exact_matches += 1
            cbl_df.at[i, "match_status"] = "Exact Match"
            cbl_df.at[i, "match_reason"] = f"Policy + Name + Cumulative Amount"
            cbl_df.at[i, "matched_swan_indices"] = matched_indices
            cbl_df.at[i, "matched_amtdue_total"] = total_amt
            cbl_df.at[i, "partial_candidates_indices"] = []
            cbl_df.at[i, "match_resolved_in_pass"] = 2  

            update_others_after_upgrade(cbl_df, i, matched_indices)

        elif fallback_indices:
            partial_matches += 1
            if cbl_df.at[i, "partial_resolved_in_pass"] is None:
                cbl_df.at[i, "partial_resolved_in_pass"] = 2
                cbl_df.at[i, "match_status"] = "Partial Match"  
                cbl_df.at[i, "match_reason"] = f"Policy + Name match only (Amount mismatch)"
            # If partial_resolved_in_pass is not None, preserve the original match reason
            cbl_df.at[i, "matched_swan_indices"] = matched_indices + fallback_indices
            cbl_df.at[i, "matched_amtdue_total"] = fallback_rows.loc[matched_indices + fallback_indices, "Amount_Clean"].tolist()

    logger.info(f"✓ Pass 2 complete: {exact_matches} exact matches, {partial_matches} partial matches")
    return cbl_df

def pass3(cbl_df, swan_df, tolerance=100, fuzzy_threshold=95):
    logger.info("\n=== Pass 3: Final matching by Fuzzy Name and Amount (Row-by-Row + Cumulative) ===")
    total_records = len(cbl_df[cbl_df["match_status"].isin(["No Match", "Partial Match"])])
    exact_matches = 0
    partial_matches = 0
    processed = 0

    # Get indices of SWAN rows that were already matched in previous passes
    already_matched_swan = set()
    for indices in cbl_df[cbl_df["match_status"] == "Exact Match"]["matched_swan_indices"]:
        if isinstance(indices, list):
            already_matched_swan.update(indices)
        elif pd.notna(indices):
            already_matched_swan.add(indices)

    # Filter out already matched SWAN rows
    available_swan = swan_df[~swan_df.index.isin(already_matched_swan)].copy()
    
    # Pre-calculate name scores for all SWAN rows
    swan_names = available_swan["ClientName"].fillna("").str.upper().str.strip()
    swan_amounts = available_swan["Amount_Clean"]

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

        # Calculate name scores for all SWAN rows at once
        name_scores = swan_names.apply(lambda x: fuzz.partial_ratio(cbl_name, x) if x else 0)
        name_matches = name_scores[name_scores >= fuzzy_threshold]
        
        if name_matches.empty:
            continue

        # Get all matching rows and their amounts
        matching_rows = available_swan.loc[name_matches.index]
        matching_amounts = swan_amounts.loc[name_matches.index]
        
        # Track processed rows to avoid duplicates
        processed_rows = set()
        matched_indices = []
        
        # First: Try row-by-row matching
        for idx, amt in zip(matching_rows.index, matching_amounts):
            if idx not in processed_rows and pd.notna(amt) and -10 <= (cbl_amt + amt) <= 10:
                matched_indices.append(idx)
                processed_rows.add(idx)
        
        # Second: Try cumulative matching with remaining unprocessed rows
        unprocessed_indices = [idx for idx in name_matches.index if idx not in processed_rows]
        if unprocessed_indices:
            unprocessed_amounts = matching_amounts[unprocessed_indices]
            # Filter out NaN values before summing
            valid_amounts = unprocessed_amounts[unprocessed_amounts.notna()]
            if not valid_amounts.empty:
                total_unprocessed_amount = valid_amounts.sum()
                
                if -10 <= (cbl_amt + total_unprocessed_amount) <= 10:
                    matched_indices.extend(unprocessed_indices)
                    processed_rows.update(unprocessed_indices)
        
        # Update the CBL record based on matches found
        if matched_indices:
            exact_matches += 1
            cbl_df.at[i, "match_status"] = "Exact Match"
            if cbl_df.at[i, "match_resolved_in_pass"] is None:
                cbl_df.at[i, "match_resolved_in_pass"] = 3
                
            # Determine match reason based on how the match was found
            if len(matched_indices) == 1:
                match_reason = f"Fuzzy Name Match ≥ {fuzzy_threshold}% + Single Amount Match"
            else:
                match_reason = f"Fuzzy Name Match ≥ {fuzzy_threshold}% + Cumulative Amount Match"
            
            cbl_df.at[i, "match_reason"] = match_reason
            cbl_df.at[i, "matched_swan_indices"] = matched_indices
            cbl_df.at[i, "matched_amtdue_total"] = sum(swan_df.loc[matched_indices, "Amount_Clean"])
            cbl_df.at[i, "partial_candidates_indices"] = []

            update_others_after_upgrade(cbl_df, i, matched_indices)
        else:
            # If no exact matches found, mark as partial match with all similar names
            partial_matches += 1
            if cbl_df.at[i, "partial_resolved_in_pass"] is None:
                cbl_df.at[i, "partial_resolved_in_pass"] = 3
                cbl_df.at[i, "match_status"] = "Partial Match"
                cbl_df.at[i, "match_reason"] = f"Fuzzy Name Match ≥ {fuzzy_threshold}% (Amount mismatch)"
            # If partial_resolved_in_pass is not None, preserve the original match reason
            cbl_df.at[i, "matched_swan_indices"] = list(name_matches.index)
            cbl_df.at[i, "matched_amtdue_total"] = matching_amounts.tolist()
            cbl_df.at[i, "partial_candidates_indices"] = []

    logger.info(f"✓ Pass 3 complete: {exact_matches} exact matches, {partial_matches} partial matches")
    return cbl_df


# Function to explode and merge with SWAN
def explode_and_merge(cbl_subset, swan_df):
    cbl_copy = cbl_subset.copy()
    exploded_rows = []
    cbl_cols = list(cbl_copy.columns)
    swan_cols = [col for col in swan_df.columns if col != 'swan_row_index']

    for _, cbl_row in cbl_copy.iterrows():
        swan_indices = cbl_row['matched_swan_indices']
        if not isinstance(swan_indices, list):
            swan_indices = [swan_indices]
        for i, swan_idx in enumerate(swan_indices):
            new_row = cbl_row.copy()
            if i > 0:
                for col in cbl_cols:
                    if col != 'matched_swan_indices' and col != 'match_status' and col != 'match_reason':
                        new_row[col] = None
            # Add SWAN data, using _SWAN suffix if column exists in CBL
            swan_row = swan_df[swan_df['swan_row_index'] == swan_idx].iloc[0]
            for col in swan_cols:
                swan_col_name = col + '_SWAN' if col in cbl_cols else col
                new_row[swan_col_name] = swan_row[col]
            exploded_rows.append(new_row)
    result_df = pd.DataFrame(exploded_rows)
    # Reorder: CBL columns, then SWAN columns (with _SWAN suffix if needed)
    swan_cols_renamed = [col + '_SWAN' if col in cbl_cols else col for col in swan_cols]
    result_df = result_df[[col for col in cbl_cols if col in result_df.columns] + [col for col in swan_cols_renamed if col in result_df.columns]]
    return result_df


def run_matching_process(cbl_file=None, swan_file=None, output_file='output.xlsx'):
    """
    Run the matching process between CBL and SWAN files.
    
    Args:
        cbl_file (str, optional): Path to the CBL Excel file. If None, will be prompted.
        swan_file (str, optional): Path to the SWAN Excel file. If None, will be prompted.
        output_file (str, optional): Output Excel file name. Defaults to 'output.xlsx'.
    """
    logger.info("\n=== Starting Matching Process ===")
    
    if cbl_file is None or swan_file is None:
        parser = argparse.ArgumentParser(description='Match data between two Excel files.')
        parser.add_argument('cbl_file', help='Path to the CBL Excel file')
        parser.add_argument('swan_file', help='Path to the SWAN Excel file')
        parser.add_argument('--output', '-o', default=output_file, help='Output Excel file name')
        args = parser.parse_args()
        cbl_file = args.cbl_file
        swan_file = args.swan_file
        output_file = args.output

    try:
        logger.info(f"Reading files: {cbl_file} and {swan_file}")
        # Read Excel files
        cbl_df = pd.read_excel(cbl_file)
        swan_df = pd.read_excel(swan_file)

        # Get the directory of the input files
        input_dir = os.path.dirname(os.path.abspath(cbl_file))
        output_path = os.path.join(input_dir, output_file)

        # Process the data
        clean_cbl, clean_swan = preprocess(cbl_df, swan_df)
        clean_cbl = initialize_tracking(clean_cbl)
        
        # Run matching passes
        result = pass1(clean_cbl, clean_swan)
        result = pass2(result, clean_swan)
        result = pass3(result, clean_swan)

        # Track SWAN indices by match type
        exact_match_swan_indices = set()
        partial_match_swan_indices = set()

        # Collect SWAN indices from exact matches
        for indices in result[result["match_status"] == "Exact Match"]["matched_swan_indices"]:
            if isinstance(indices, list):
                exact_match_swan_indices.update(indices)
            elif pd.notna(indices):
                exact_match_swan_indices.add(indices)
        
        # Collect SWAN indices from partial matches
        for indices in result[result["match_status"] == "Partial Match"]["matched_swan_indices"]:
            if isinstance(indices, list):
                partial_match_swan_indices.update(indices)
            elif pd.notna(indices):
                partial_match_swan_indices.add(indices)

        # Remove exact matches from partial matches to avoid double counting
        partial_match_swan_indices = partial_match_swan_indices - exact_match_swan_indices

        # Calculate unmatched SWAN indices
        all_swan_indices = set(clean_swan.index)
        matched_swan_indices = exact_match_swan_indices | partial_match_swan_indices
        unmatched_swan_indices = all_swan_indices - matched_swan_indices

        # Reset index of swan_df so we can merge using row index
        clean_swan = clean_swan.reset_index().rename(columns={"index": "swan_row_index"})

        # Split results
        exact_matches = result[result["match_status"] == "Exact Match"].copy()
        partial_matches = result[result["match_status"] == "Partial Match"].copy()
        no_matches = result[result["match_status"] == "No Match"].copy()

        # Explode matched_swan_indices and merge with SWAN
        exact_matches = explode_and_merge(exact_matches, clean_swan)
        partial_matches = explode_and_merge(partial_matches, clean_swan)
        # no_matches = explode_and_merge(no_matches, clean_swan)

        # Create unmatched SWAN records
        unmatched_swan = clean_swan[clean_swan['swan_row_index'].isin(unmatched_swan_indices)].copy()

        # Write to Excel
        logger.info(f"\nWriting results to {output_path}")
        try: 
            with pd.ExcelWriter(output_path) as writer:
                exact_matches.to_excel(writer, sheet_name="Exact Matches", index=False)
                partial_matches.to_excel(writer, sheet_name="Partial Matches", index=False)
                no_matches.to_excel(writer, sheet_name="No Matches", index=False)
                unmatched_swan.to_excel(writer, sheet_name="Unmatched SWAN", index=False)
        except Exception as e:
            logger.error(f"Error writing to Excel: {str(e)}")
            raise

        logger.info(f"✓ Results saved to: {output_path}")

        # Calculate statistics
        total_swan_rows = len(clean_swan)
        exact_match_swan_count = len(exact_match_swan_indices)
        partial_match_swan_count = len(partial_match_swan_indices)
        unmatched_swan_count = len(unmatched_swan_indices)

        # logger.info("\n=== Final Results ===")
        # logger.info(f"✓ Exact matches: {len(exact_matches)}")
        # logger.info(f"✓ Partial matches: {len(partial_matches)}")
        # logger.info(f"✓ No matches: {len(no_matches)}")
        # logger.info(f"✓ Results saved to: {output_path}")
        logger.info("\n=== Final Results ===")
        logger.info(f"✓ CBL Records:")
        logger.info(f"  - Exact matches: {len(result[result['match_status'] == 'Exact Match'])}")
        logger.info(f"  - Partial matches: {len(result[result['match_status'] == 'Partial Match'])}")
        logger.info(f"  - No matches: {len(result[result['match_status'] == 'No Match'])}")
        logger.info(f"✓ SWAN Records:")
        logger.info(f"  - Total SWAN rows: {total_swan_rows}")
        logger.info(f"  - Exact match SWAN rows: {exact_match_swan_count} ({exact_match_swan_count/total_swan_rows*100:.1f}%)")
        logger.info(f"  - Partial match SWAN rows: {partial_match_swan_count} ({partial_match_swan_count/total_swan_rows*100:.1f}%)")
        logger.info(f"  - Unmatched SWAN rows: {unmatched_swan_count} ({unmatched_swan_count/total_swan_rows*100:.1f}%)")
        logger.info(f"✓ Results saved to: {output_path}")

        # return {
        #     'exact_matches': exact_matches,
        #     'partial_matches': partial_matches,
        #     'no_matches': no_matches,
        #     'output_file': output_path
        # }
        return {
            'exact_matches': exact_matches,
            'partial_matches': partial_matches,
            'no_matches': no_matches,
            'unmatched_swan': unmatched_swan,
            'output_file': output_path,
            'cbl_stats': {
                'exact_matches': len(result[result['match_status'] == 'Exact Match']),
                'partial_matches': len(result[result['match_status'] == 'Partial Match']),
                'no_matches': len(result[result['match_status'] == 'No Match'])
            },
            'swan_stats': {
                'total_rows': total_swan_rows,
                'exact_match_rows': exact_match_swan_count,
                'partial_match_rows': partial_match_swan_count,
                'unmatched_rows': unmatched_swan_count,
                'exact_match_rate': exact_match_swan_count/total_swan_rows*100,
                'partial_match_rate': partial_match_swan_count/total_swan_rows*100,
                'unmatched_rate': unmatched_swan_count/total_swan_rows*100
            }
        }

    except Exception as e:
        logger.error(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    run_matching_process() 