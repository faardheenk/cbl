#!/usr/bin/env python3

import pandas as pd
import logging
from .utils import _extract_insurer_indices, _get_insurer_rows_for_group, _separate_group_and_individual_matches

logger = logging.getLogger(__name__)


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
            if preserve_match_info and col in ['match_status', 'match_reason', 'matched_insurer_indices', 'matched_amtdue_total', 'Amount Difference', 'partial_candidates_indices']:
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
    
    Shows insurer data only once by pairing CBL rows with insurer rows:
    - Row 1: CBL_A + Insurer_D
    - Row 2: CBL_E + Insurer_F
    - Row 3: (empty CBL) + Insurer_G (if more insurers than CBL)
    
    Args:
        group_cbl_rows: List of CBL rows in the group
        insurer_rows: List of insurer rows in the group
        cbl_cols: List of CBL column names
        insurer_cols: List of insurer column names
        
    Returns:
        list: List of combined rows
    """
    combined_rows = []
    max_rows = max(len(group_cbl_rows), len(insurer_rows))
    
    for i in range(max_rows):
        # Get CBL row if available, otherwise None
        cbl_row = group_cbl_rows[i] if i < len(group_cbl_rows) else None
        
        # Get insurer row if available, otherwise None
        insurer_row = insurer_rows[i] if i < len(insurer_rows) else None
        
        # Create combined row (handles None for either CBL or insurer)
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
    
    # DEBUG: Log CBL row information
    cbl_client_name = cbl_row.get('ClientName', 'N/A')
    cbl_matrix_key = cbl_row.get('MatrixKey', 'N/A')
    logger.debug(f"Processing individual match for CBL {cbl_row.name}: ClientName='{cbl_client_name}', MatrixKey='{cbl_matrix_key}'")
    logger.debug(f"  matched_insurer_indices from CBL row: {insurer_indices}")
    logger.debug(f"  insurer_df index range: {insurer_df.index.min()} to {insurer_df.index.max()}")
    logger.debug(f"  insurer_df shape: {insurer_df.shape}")
    
    # If no insurer indices, create a row with only CBL data
    if not insurer_indices:
        combined_row = _create_zipped_row(cbl_row, None, cbl_cols, insurer_cols)
        combined_rows.append(combined_row)
        return combined_rows
    
    # DEDUPLICATION: Remove duplicates while preserving order
    # Convert set to list to preserve order, then use dict.fromkeys() to remove duplicates
    unique_indices = list(dict.fromkeys(list(insurer_indices)))
    
    # Log deduplication if it occurred
    if len(unique_indices) < len(insurer_indices):
        duplicates_removed = len(insurer_indices) - len(unique_indices)
        logger.info(f"CBL {cbl_row.name}: Removed {duplicates_removed} duplicate insurer indices. "
                   f"Original: {insurer_indices}, Deduplicated: {unique_indices}")
    
    logger.debug(f"  Processing {len(unique_indices)} unique insurer indices: {unique_indices}")
    
    for i, insurer_idx in enumerate(unique_indices):
        # Validate index exists in insurer_df
        if insurer_idx not in insurer_df.index:
            logger.error(f"CBL {cbl_row.name}: Insurer index {insurer_idx} NOT FOUND in insurer_df!")
            logger.error(f"  Available insurer indices: {list(insurer_df.index[:10])}... (showing first 10)")
            continue
        
        # Get insurer row using DataFrame index directly
        insurer_row = insurer_df.loc[insurer_idx]
        
        # DEBUG: Log insurer information
        insurer_client_name = insurer_row.get('ClientName_INSURER', 'N/A')
        insurer_matrix_key = insurer_row.get('MatrixKey_INSURER', 'N/A')
        logger.debug(f"  [{i}] Fetched insurer {insurer_idx}: ClientName='{insurer_client_name}', MatrixKey='{insurer_matrix_key}'")
        
        # For multiple insurers, only show CBL data in first row
        if i > 0:
            # For subsequent insurer rows, pass None for CBL data to clear MatrixKey
            cbl_row_copy = None
        else:
            cbl_row_copy = cbl_row
        
        # Create combined row
        combined_row = _create_zipped_row(cbl_row_copy, insurer_row, cbl_cols, insurer_cols)
        combined_rows.append(combined_row)
    
    logger.debug(f"  Created {len(combined_rows)} combined rows for CBL {cbl_row.name}")
    return combined_rows


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
    logger.info(f"\n=== explode_and_merge called ===")
    logger.info(f"CBL subset: {len(cbl_subset)} rows")
    logger.info(f"Insurer DF: {len(insurer_df)} rows, index range: {insurer_df.index.min()} to {insurer_df.index.max()}")
    
    # DEBUG: Check group_id values BEFORE any processing
    if 'group_id' in cbl_subset.columns:
        unique_groups = cbl_subset['group_id'].unique()
        nan_count = cbl_subset['group_id'].isna().sum()
        logger.info(f"DEBUG: group_id analysis BEFORE copy:")
        logger.info(f"  - Total unique group_ids: {len(unique_groups)}")
        logger.info(f"  - NaN group_ids: {nan_count}")
        logger.info(f"  - Sample group_ids (first 10): {list(unique_groups[:10])}")
        # Check for string 'nan'
        string_nan_count = (cbl_subset['group_id'] == 'nan').sum()
        logger.info(f"  - String 'nan' group_ids: {string_nan_count}")
    
    cbl_copy = cbl_subset.copy()
    cbl_cols = list(cbl_copy.columns)
    insurer_cols = list(insurer_df.columns)
    
    # Separate group matches from individual matches
    group_matches, individual_matches = _separate_group_and_individual_matches(cbl_copy)
    
    logger.info(f"Separated into {len(group_matches)} groups and {len(individual_matches)} individual matches")
    
    exploded_rows = []
    
    # Process individual matches
    logger.info(f"\n--- Processing {len(individual_matches)} individual matches ---")
    for cbl_row in individual_matches:
        individual_combined_rows = _process_individual_match(cbl_row, insurer_df, cbl_cols, insurer_cols)
        exploded_rows.extend(individual_combined_rows)
        

    # Process group matches
    logger.info(f"\n--- Processing {len(group_matches)} group matches ---")
    for group_key, group_cbl_rows in group_matches.items():
        logger.info(f"Processing group {group_key}: {len(group_cbl_rows)} CBL rows")
        
        # Get all insurer rows for this group
        insurer_rows = _get_insurer_rows_for_group(group_cbl_rows, insurer_df)
        logger.info(f"Found {len(insurer_rows)} insurer rows for group {group_key}")
        
        # Create zipped rows for group match
        group_combined_rows = _process_group_match(group_cbl_rows, insurer_rows, cbl_cols, insurer_cols)
        exploded_rows.extend(group_combined_rows)
    
    # Create result dataframe and reorder columns
    result_df = pd.DataFrame(exploded_rows)
    result_df = result_df[[col for col in cbl_cols if col in result_df.columns] + 
                         [col for col in insurer_cols if col in result_df.columns]]
    
    logger.info(f"✓ Created {len(result_df)} combined rows from {len(cbl_copy)} CBL rows")
    return result_df
