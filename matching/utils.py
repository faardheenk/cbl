#!/usr/bin/env python3

import pandas as pd
import re
import logging

logger = logging.getLogger(__name__)

def add_pass(cbl_df, row_index, pass_number):
    """Add a pass number to the tracking for a specific row."""
    existing = cbl_df.at[row_index, "match_pass"]
    if isinstance(existing, list):
        if pass_number not in existing:
            cbl_df.at[row_index, "match_pass"] = existing + [pass_number]
    elif isinstance(existing, int):
        if existing != pass_number:
            cbl_df.at[row_index, "match_pass"] = [existing, pass_number]
    else:
        cbl_df.at[row_index, "match_pass"] = [pass_number]


def extract_policy_tokens(policy_str):
    """Extract policy tokens from a policy string."""
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
    """Update other CBL rows after one row is upgraded to exact match."""
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


