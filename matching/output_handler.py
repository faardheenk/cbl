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
