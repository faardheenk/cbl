#!/usr/bin/env python3

import pandas as pd
import logging
from matrix import build_key

logger = logging.getLogger(__name__)

def create_dynamic_column_mappings(cbl_columns, insurer_columns, custom_mappings=None):
    """
    Create dynamic column mappings based on available columns.
    
    Args:
        cbl_columns: List of available CBL column names
        insurer_columns: List of available insurer column names
        custom_mappings: Optional custom mappings to override defaults
        
    Returns:
        dict: Column mappings dictionary
    """
    # Default mappings - these are the standard expected column names
    default_cbl_mappings = {
        'PlacingNo': 'PlacingNo',
        'PolicyNo': 'PolicyNo', 
        'ClientName': 'ClientName',
        'Amount': 'Amount'
    }
    
    default_insurer_mappings = {
        'PlacingNo': 'PlacingNo',
        'PolicyNo_1': 'PolicyNo_1',
        'PolicyNo_2': 'PolicyNo_2',
        'ClientName': 'ClientName', 
        'Amount': 'Amount'
    }
    
    # Start with defaults
    cbl_mappings = default_cbl_mappings.copy()
    insurer_mappings = default_insurer_mappings.copy()
    
    # Apply custom mappings if provided
    if custom_mappings:
        if 'cbl_mappings' in custom_mappings:
            cbl_mappings.update(custom_mappings['cbl_mappings'])
        if 'insurer_mappings' in custom_mappings:
            insurer_mappings.update(custom_mappings['insurer_mappings'])
    
    # Filter to only include mappings where source columns exist
    filtered_cbl_mappings = {k: v for k, v in cbl_mappings.items() if k in cbl_columns}
    filtered_insurer_mappings = {k: v for k, v in insurer_mappings.items() if k in insurer_columns}
    
    logger.info(f"Dynamic CBL mappings: {filtered_cbl_mappings}")
    logger.info(f"Dynamic insurer mappings: {filtered_insurer_mappings}")
    
    return {
        'cbl_mappings': filtered_cbl_mappings,
        'insurer_mappings': filtered_insurer_mappings
    }



def preprocess(cbl_df, insurer_df, column_mappings, matrix_key_columns=None):
    """
    Preprocess and clean the CBL and insurer dataframes.
    
    Args:
        cbl_df: CBL dataframe
        insurer_df: Insurer dataframe  
        column_mappings: Dictionary containing column mappings (already filtered by create_dynamic_column_mappings)
        matrix_key_columns: Optional dictionary specifying columns to use for matrix keys
        
    Returns:
        tuple: (processed_cbl_df, processed_insurer_df)
    """
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


    # Get column mappings (already filtered by create_dynamic_column_mappings)
    cbl_column_map = column_mappings['cbl_mappings']
    insurer_column_map = column_mappings['insurer_mappings']

    # Rename columns directly (no need to filter again since mappings are pre-filtered)
    cbl_df = cbl_df.rename(columns=cbl_column_map)
    insurer_df = insurer_df.rename(columns=insurer_column_map)
    
    # Log which columns were successfully renamed
    logger.info(f"Successfully renamed CBL columns: {list(cbl_column_map.values())}")
    logger.info(f"Successfully renamed insurer columns: {list(insurer_column_map.values())}")

    # Add _INSURER suffix to all insurer columns
    insurer_columns = list(insurer_df.columns)
    insurer_column_suffix_map = {col: col + '_INSURER' for col in insurer_columns}
    insurer_df = insurer_df.rename(columns=insurer_column_suffix_map)
    
    # Log available columns after renaming for debugging
    logger.info(f"Available insurer columns after renaming: {list(insurer_df.columns)}")

    # Clean and process data dynamically based on available columns
    logger.info(f"DEBUG: Before data cleaning - CBL rows: {len(cbl_df)}, Insurer rows: {len(insurer_df)}")
    
    # Clean CBL columns dynamically
    if "PlacingNo" in cbl_df.columns:
        cbl_df["PlacingNo_Clean"] = cbl_df["PlacingNo"].str.upper().str.strip()
        cbl_df["PlacingNo_Clean"] = cbl_df["PlacingNo_Clean"].str.replace(pattern, '', regex=True)
    
    if "Amount" in cbl_df.columns:
        cbl_df["Amount_Clean"] = pd.to_numeric(cbl_df["Amount"], errors="coerce")
        
    if "PolicyNo" in cbl_df.columns:
        cbl_df["PolicyNo_Clean"] = cbl_df["PolicyNo"].astype(str).str.split(".").str[0]
        cbl_df["PolicyNo_Clean"] = cbl_df["PolicyNo_Clean"].fillna("")
        
    if "ClientName" in cbl_df.columns:
        cbl_df["ClientName_Clean"] = cbl_df["ClientName"].astype(str).str.upper().str.strip()
        cbl_df["ClientName_Clean"] = cbl_df["ClientName_Clean"].str.replace(pattern, '', regex=True)

    # Clean insurer columns dynamically
    if "PlacingNo_INSURER" in insurer_df.columns:
        insurer_df["PlacingNo_Clean_INSURER"] = insurer_df["PlacingNo_INSURER"].astype(str).str.upper().str.strip()
        insurer_df["PlacingNo_Clean_INSURER"] = insurer_df["PlacingNo_Clean_INSURER"].str.replace(pattern, '', regex=True)
    
    if "PolicyNo_1_INSURER" in insurer_df.columns:
        insurer_df["PolicyNo_Clean_INSURER"] = insurer_df["PolicyNo_1_INSURER"].astype(str).str.split(".").str[0]
        # Replace NaN values with empty string
        insurer_df["PolicyNo_Clean_INSURER"] = insurer_df["PolicyNo_Clean_INSURER"].fillna("")
    
    if "Amount_INSURER" in insurer_df.columns:
        insurer_df["Amount_Clean_INSURER"] = pd.to_numeric(insurer_df["Amount_INSURER"], errors="coerce")
    
    logger.info(f"DEBUG: After data cleaning - CBL rows: {len(cbl_df)}, Insurer rows: {len(insurer_df)}")
    
    # Handle optional PolicyNo_2 column dynamically
    if "PolicyNo_2_INSURER" in insurer_df.columns:
        insurer_df["PolicyNo_2_Clean_INSURER"] = insurer_df["PolicyNo_2_INSURER"].astype(str)
        # Replace NaN values with empty string
        insurer_df["PolicyNo_2_Clean_INSURER"] = insurer_df["PolicyNo_2_Clean_INSURER"].fillna("")
    else:
        # Create empty PolicyNo_2 column if it doesn't exist (some matching passes may expect it)
        insurer_df["PolicyNo_2_Clean_INSURER"] = ""
        logger.info("PolicyNo_2_INSURER column not found - creating empty column for compatibility")

    # Matrix Key - build dynamically based on available columns
    default_cbl_cols = ['PlacingNo', 'PolicyNo', 'ClientName', 'Amount']
    default_insurer_cols = ['PlacingNo_INSURER', 'PolicyNo_1_INSURER', 'ClientName_INSURER', 'Amount_INSURER']

    if matrix_key_columns:
        default_cbl_cols = matrix_key_columns.get('cbl_columns', default_cbl_cols)
        default_insurer_cols = matrix_key_columns.get('insurer_columns', default_insurer_cols)
    
    # Use only columns that exist in the dataframes
    available_cbl_cols = [col for col in default_cbl_cols if col in cbl_df.columns]
    available_insurer_cols = [col for col in default_insurer_cols if col in insurer_df.columns]
    
    # Build matrix keys
    cbl_df["MatrixKey"] = ""
    insurer_df["MatrixKey_INSURER"] = ""

    if available_cbl_cols:
        cbl_df["MatrixKey"] = cbl_df.apply(lambda row: build_key(row, available_cbl_cols), axis=1)
        logger.info(f"Built CBL MatrixKey using columns: {available_cbl_cols}")
    else:
        logger.warning("No standard columns available for CBL MatrixKey - using empty string")
    
    if available_insurer_cols:
        insurer_df["MatrixKey_INSURER"] = insurer_df.apply(lambda row: build_key(row, available_insurer_cols), axis=1)
        logger.info(f"Built insurer MatrixKey using columns: {available_insurer_cols}")
    else:
        logger.warning("No standard columns available for insurer MatrixKey - using empty string")

    logger.info(f"✓ Preprocessing complete: {len(cbl_df)} CBL records, {len(insurer_df)} insurer records")
    return cbl_df, insurer_df


def initialize_tracking(cbl_df):
    """Initialize tracking columns for the matching process."""
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
