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
    """Extract policy tokens from a policy string with simple and effective approach."""
    if pd.isna(policy_str):
        return []
    
    policy_str = str(policy_str).strip()
    
    if not policy_str:
        return []
    
    # Strategy 1: If it's a pure number, keep it as is (but check minimum length)
    if policy_str.isdigit():
        if len(policy_str) >= 4:  # Minimum 4 characters for numbers
            return [policy_str]
        else:
            return []
    
    # Strategy 2: For clean policy numbers, split on spaces and process each part
    # This handles cases like: "102901 2092 0902 PYTH" or "DNH018AC012999 PYHO24LI000903"
    space_parts = policy_str.split()
    tokens = []
    
    for part in space_parts:
        # Clean the part of any remaining symbols
        cleaned_part = re.sub(r'[^a-zA-Z0-9]', '', part)
        
        if cleaned_part and len(cleaned_part) >= 4:  # Minimum 4 characters
            tokens.append(cleaned_part.upper())
    
    # Special case: If we have a single part with '/' characters, combine it into one token
    # This handles cases like "ABC/DEF/123/456" -> "ABCDEF123456"
    if len(space_parts) == 1 and '/' in policy_str:
        combined_token = re.sub(r'[^a-zA-Z0-9]', '', policy_str)
        if len(combined_token) >= 4:
            return [combined_token.upper()]
    
    # Strategy 3: If we didn't get good results from space splitting, try reconstruction
    # This handles cases like: "PY HO 24LI 000 903 DN H0 18AC 012 999" and "DN-H0-18AC/012/999"
    if len(tokens) < 2 or any(len(token) < 8 for token in tokens) or (len(tokens) >= 3 and all(len(token) < 10 for token in tokens)):
        # Get all alphanumeric parts from the original string
        alphanumeric_parts = re.findall(r'[A-Z0-9]+', policy_str.upper())
        
        # Find letter-only parts (potential policy code starts)
        letter_parts = [part for part in alphanumeric_parts if re.match(r'^[A-Z]+$', part)]
        
        if len(letter_parts) >= 1:  # Changed from >= 2 to >= 1
            reconstructed_tokens = []
            
            # Build policy codes by combining each letter part with following parts
            for i, letter_part in enumerate(letter_parts):
                # Find this letter part in the original list
                letter_index = alphanumeric_parts.index(letter_part)
                
                # Collect parts starting from this letter part
                policy_parts = [letter_part]
                j = letter_index + 1
                
                # Look for the next letter part to know when to stop
                next_letter_index = len(alphanumeric_parts)
                if i + 1 < len(letter_parts):
                    next_letter_part = letter_parts[i + 1]
                    try:
                        next_letter_index = alphanumeric_parts.index(next_letter_part)
                    except ValueError:
                        next_letter_index = len(alphanumeric_parts)
                
                # Collect all parts until the next letter part
                while j < next_letter_index and j < len(alphanumeric_parts):
                    policy_parts.append(alphanumeric_parts[j])
                    j += 1
                
                # Create the policy code
                policy_code = ''.join(policy_parts)
                if len(policy_code) >= 6:
                    reconstructed_tokens.append(policy_code)
            
            # Use reconstructed tokens if they're better
            if len(reconstructed_tokens) > len(tokens):
                tokens = reconstructed_tokens
            elif len(reconstructed_tokens) == 1 and len(tokens) > 1:
                # If reconstruction gives us 1 token and we have multiple short tokens, prefer reconstruction
                tokens = reconstructed_tokens
    
    # Strategy 4: Fallback - extract any remaining meaningful patterns
    if len(tokens) < 2:
        # Extract 4+ digit numbers
        numbers = re.findall(r'\b\d{4,}\b', policy_str)
        tokens.extend(numbers)
        
        # Extract alphanumeric patterns
        alpha_numeric = re.findall(r'\b[A-Z]{2,}\d+[A-Z0-9]*\b', policy_str.upper())
        tokens.extend(alpha_numeric)
        
        # Extract mixed alphanumeric codes
        mixed_codes = re.findall(r'\b[A-Z0-9]{4,}\b', policy_str.upper())
        tokens.extend(mixed_codes)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_tokens = []
    for token in tokens:
        if token and token not in seen:
            seen.add(token)
            unique_tokens.append(token)
    
    return unique_tokens


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
    
    logger.debug(f"_get_insurer_rows_for_group: Collected {len(all_insurer_indices)} unique insurer indices: {all_insurer_indices}")

    insurer_rows = []
    for insurer_idx in all_insurer_indices:
        # FIX: Use .loc (label-based) instead of .iloc (position-based)
        # The matched_insurer_indices stores DataFrame index labels, not positions
        if insurer_idx not in insurer_df.index:
            logger.error(f"Insurer index {insurer_idx} not found in insurer_df! Available: {list(insurer_df.index[:10])}")
            continue
        insurer_row = insurer_df.loc[insurer_idx]
        insurer_rows.append(insurer_row)

    logger.debug(f"_get_insurer_rows_for_group: Returning {len(insurer_rows)} insurer rows")
    return insurer_rows


def _separate_group_and_individual_matches(cbl_subset):
    """
    Separate CBL rows into group matches and individual matches.
    
    Groups are identified by having a group_id value from Pass 3 Phase 3 name grouping (NAME_GROUP_N).
    
    The output handler's only job is to:
    - Check if row has group_id → group it (will be zipped in output)
    - Otherwise → individual match (will be exploded in output)
    
    Args:
        cbl_subset: CBL dataframe subset
        
    Returns:
        tuple: (group_matches_dict, individual_matches_list)
    """
    group_matches = {}
    individual_matches = []
    
    for _, cbl_row in cbl_subset.iterrows():
        # Simple check: does this row have a group_id?
        group_id = cbl_row.get('group_id', None)
        
        # CRITICAL: Check for NaN/None/empty/string 'nan' BEFORE using as dict key
        # Python quirk: NaN as dict key causes all NaN values to be grouped together
        # Also check for string 'nan' which can come from data processing
        if pd.isna(group_id) or group_id is None or group_id == '' or str(group_id).lower() == 'nan':
            # No valid group_id - individual match
            individual_matches.append(cbl_row)
        else:
            # This row is part of a group - add to group matches
            if group_id not in group_matches:
                group_matches[group_id] = []
            group_matches[group_id].append(cbl_row)
    
    return group_matches, individual_matches


