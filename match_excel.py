#!/usr/bin/env python3

import pandas as pd
import re
import argparse
from fuzzywuzzy import fuzz
import sys
import logging
import os

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

        exact_match_index = None
        fallback_indices = []

        for j, sw_row in swan_matches.iterrows():
            amt2 = sw_row["Amount_Clean"]
            if abs(amt1 + amt2) <= 10 and exact_match_index is None:
                exact_match_index = j
            else:
                fallback_indices.append(j)
        
        if exact_match_index is not None:
            exact_matches += 1
            cbl_df.at[i, "match_status"] = "Exact Match"
            cbl_df.at[i, "match_reason"] = "Placing + Amount"
            cbl_df.at[i, "matched_swan_indices"] = [exact_match_index]
            cbl_df.at[i, "matched_amtdue_total"] = swan_df.at[exact_match_index, "Amount_Clean"]
            cbl_df.at[i, "partial_candidates_indices"] = fallback_indices
            cbl_df.at[i, "match_resolved_in_pass"] = 1
        elif not swan_matches.empty:
            partial_matches += 1
            cbl_df.at[i, "match_status"] = "Partial Match"
            cbl_df.at[i, "match_reason"] = "Placing only"
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

def pass2(cbl_df, swan_df, tolerance=10, name_threshold=90):
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
            name_score = fuzz.token_sort_ratio(cbl_name, sw_name)
            policy_match = sw_row["PolicyNo_1_Clean"] in tokens or sw_row["PolicyNo_2_Clean"] in tokens

            if policy_match and name_score >= name_threshold:
                matched_indices.append(j)

        total_amt = fallback_rows.loc[matched_indices, "Amount_Clean"].sum()

        if matched_indices and abs(cbl_amt + total_amt) <= tolerance:
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
            cbl_df.at[i, "matched_swan_indices"] = matched_indices + fallback_indices
            cbl_df.at[i, "matched_amtdue_total"] = fallback_rows.loc[matched_indices + fallback_indices, "Amount_Clean"].tolist()

    logger.info(f"✓ Pass 2 complete: {exact_matches} exact matches, {partial_matches} partial matches")
    return cbl_df

def pass3(cbl_df, swan_df, tolerance=10, fuzzy_threshold=90):
    logger.info("\n=== Pass 3: Final matching by Name and Amount ===")
    total_records = len(cbl_df[cbl_df["match_status"].isin(["No Match", "Partial Match"])])
    exact_matches = 0
    partial_matches = 0
    processed = 0

    for i, row in cbl_df[cbl_df["match_status"].isin(["No Match", "Partial Match"])].iterrows():
        processed += 1
        if processed % 50 == 0:
            logger.info(f"Progress: {processed}/{total_records} records processed")

        add_pass(cbl_df, i, 3)

        cbl_name = str(row["ClientName"]).upper().strip()
        cbl_amt = row["Amount_Clean"]

        matched_index = None
        fallback_indices = []

        for j, sw_row in swan_df.iterrows():
            sw_name = str(sw_row["ClientName"]).upper().strip()
            sw_amt = sw_row["Amount_Clean"]

            name_score = fuzz.token_sort_ratio(cbl_name, sw_name)
            if name_score >= fuzzy_threshold:
                if abs(cbl_amt + sw_amt) <= tolerance:
                    matched_index = j
                    break
                else:
                    fallback_indices.append(j)

        if matched_index is not None:
            exact_matches += 1
            cbl_df.at[i, "match_status"] = "Exact Match"
            if cbl_df.at[i, "match_resolved_in_pass"] is None:
                cbl_df.at[i, "match_resolved_in_pass"] = 3
                
            cbl_df.at[i, "match_reason"] = f"Name Match ≥ {fuzzy_threshold}% + Amount Match (row-by-row)"
            cbl_df.at[i, "matched_swan_indices"] = [matched_index]
            cbl_df.at[i, "matched_amtdue_total"] = swan_df.at[matched_index, "Amount_Clean"]
            cbl_df.at[i, "partial_candidates_indices"] = []

            update_others_after_upgrade(cbl_df, i, [matched_index])
        elif fallback_indices:
            partial_matches += 1
            if cbl_df.at[i, "partial_resolved_in_pass"] is None:
                cbl_df.at[i, "partial_resolved_in_pass"] = 3
                
            cbl_df.at[i, "match_status"] = "Partial Match"
            cbl_df.at[i, "match_reason"] = f"Name Match ≥ {fuzzy_threshold}% (Amount mismatch)"
            cbl_df.at[i, "matched_swan_indices"] = fallback_indices
            cbl_df.at[i, "matched_amtdue_total"] = swan_df.loc[fallback_indices, "Amount_Clean"].tolist()
            cbl_df.at[i, "partial_candidates_indices"] = []

    logger.info(f"✓ Pass 3 complete: {exact_matches} exact matches, {partial_matches} partial matches")
    return cbl_df

# Function to explode and merge with SWAN
def explode_and_merge(cbl_subset, swan_df):
    exploded = cbl_subset.explode("matched_swan_indices")
    return exploded.merge(
        swan_df,
        left_on="matched_swan_indices",
        right_on="swan_row_index",
        how="left"
    )


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

        # Reset index of swan_df so we can merge using row index
        clean_swan = clean_swan.reset_index().rename(columns={"index": "swan_row_index"})

        # Split results
        exact_matches = result[result["match_status"] == "Exact Match"].copy()
        partial_matches = result[result["match_status"] == "Partial Match"].copy()
        no_matches = result[result["match_status"] == "No Match"].copy()

        # Explode matched_swan_indices and merge with SWAN
        exact_matches = explode_and_merge(exact_matches, clean_swan)
        partial_matches = explode_and_merge(partial_matches, clean_swan)
        no_matches = explode_and_merge(no_matches, clean_swan)


        # Write to Excel
        logger.info(f"\nWriting results to {output_path}")
        with pd.ExcelWriter(output_path) as writer:
            exact_matches.to_excel(writer, sheet_name="Exact Matches", index=False)
            partial_matches.to_excel(writer, sheet_name="Partial Matches", index=False)
            no_matches.to_excel(writer, sheet_name="No Matches", index=False)

        logger.info("\n=== Final Results ===")
        logger.info(f"✓ Exact matches: {len(exact_matches)}")
        logger.info(f"✓ Partial matches: {len(partial_matches)}")
        logger.info(f"✓ No matches: {len(no_matches)}")
        logger.info(f"✓ Results saved to: {output_path}")

        return {
            'exact_matches': exact_matches,
            'partial_matches': partial_matches,
            'no_matches': no_matches,
            'output_file': output_path
        }

    except Exception as e:
        logger.error(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    run_matching_process() 