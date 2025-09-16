import pandas as pd
import re

def build_key(row, cols): 
   return  "#".join(str(row[col]).strip() if pd.notna(row[col]) else "" for col in cols)

def normalize_key(key):
    """Normalize a key to handle data type differences and escape characters"""
    if pd.isna(key):
        return ""
    
    # Convert to string and strip
    key_str = str(key).strip()
    
    # Remove .0 suffixes from numeric values
    key_str = re.sub(r'\.0(?=#|$)', '', key_str)
    
    # Normalize escape sequences - replace double backslashes with single
    key_str = key_str.replace('\\\\', '\\')
    
    return key_str

def split_matrix(matrix_key):
    """
    Split matrix key into left-hand side (CBL) and right-hand side (Insurer) parts.
    
    Args:
        matrix_key (str): Matrix key in format "cbl_part1,cbl_part2|insurer_part1,insurer_part2"
    
    Returns:
        tuple: (lhs_parts, rhs_parts) where each is a list of key parts
    """
    if not matrix_key or "|" not in matrix_key:
        raise ValueError(f"Invalid matrix key format: {matrix_key}. Expected format: 'cbl_keys|insurer_keys'")
    
    try:
        lhs_str, rhs_str = matrix_key.split("|")
        
        # Split by commas and clean up parts
        lhs_parts = [part.strip() for part in lhs_str.split(",") if part.strip()]
        rhs_parts = [part.strip() for part in rhs_str.split(",") if part.strip()]
        
        if not lhs_parts or not rhs_parts:
            raise ValueError(f"Matrix key has empty parts - LHS: {lhs_parts}, RHS: {rhs_parts}")
        
        return lhs_parts, rhs_parts
        
    except Exception as e:
        if "not enough values to unpack" in str(e):
            raise ValueError(f"Matrix key missing '|' separator: {matrix_key}")
        raise ValueError(f"Error parsing matrix key '{matrix_key}': {str(e)}")

def matrix_pass(cbl_df, insurer_df, matrix_keys, global_tracker=None):
    """
    Matrix pass with comprehensive global tracker integration.
    
    Args:
        cbl_df: CBL DataFrame
        insurer_df: Insurer DataFrame  
        matrix_keys: List of matrix key configurations
        global_tracker: GlobalMatchTracker instance for data integrity
    
    Returns:
        tuple: (updated_cbl_df, updated_insurer_df, matched_insurer_indices)
    """
    print(f"\n=== Matrix Pass with Global Tracker Integration ===")
    
    if global_tracker:
        print(f"🔧 Matrix pass starting with global tracker: {global_tracker.get_usage_summary()}")
    else:
        print("⚠️ Matrix pass running without global tracker - legacy mode")
    
    # Validate required columns exist
    if "MatrixKey" not in cbl_df.columns:
        print(f"❌ Matrix pass failed: 'MatrixKey' column not found in CBL DataFrame")
        print(f"Available CBL columns: {list(cbl_df.columns)}")
        return cbl_df, insurer_df, set()
    
    if "MatrixKey_INSURER" not in insurer_df.columns:
        print(f"❌ Matrix pass failed: 'MatrixKey_INSURER' column not found in Insurer DataFrame")
        print(f"Available Insurer columns: {list(insurer_df.columns)}")
        return cbl_df, insurer_df, set()
    
    # Validate matrix keys parameter
    if not matrix_keys:
        print(f"⚠️ Matrix pass: No matrix keys provided - skipping matrix matching")
        return cbl_df, insurer_df, set()
    
    try:
        # Create matrix indices
        cbl_df_matrix_index = cbl_df.set_index("MatrixKey")
        insurer_df_matrix_index = insurer_df.set_index("MatrixKey_INSURER")
        print(f"✅ Matrix indexing successful - CBL: {len(cbl_df_matrix_index)} rows, Insurer: {len(insurer_df_matrix_index)} rows")
    except Exception as e:
        print(f"❌ Matrix pass failed during indexing: {str(e)}")
        return cbl_df, insurer_df, set()

    # Track which insurer rows have been matched
    matched_insurer_indices = set()
    exact_matches_count = 0

    # Create normalized versions of the indices for comparison
    cbl_normalized_index = {normalize_key(key): key for key in cbl_df_matrix_index.index}
    insurer_normalized_index = {normalize_key(key): key for key in insurer_df_matrix_index.index}

    for i, matrix_item in enumerate(matrix_keys):
        try:
            if not isinstance(matrix_item, dict) or 'matrixKey' not in matrix_item:
                print(f"⚠️ Skipping invalid matrix item {i}: {matrix_item}")
                continue
            
            matrix_key = matrix_item['matrixKey']
            if not matrix_key or not isinstance(matrix_key, str):
                print(f"⚠️ Skipping empty or invalid matrix key {i}: {matrix_key}")
                continue
                
            print(f"\nProcessing matrix key {i+1}/{len(matrix_keys)}: {matrix_key}")
            lhs_parts, rhs_parts = split_matrix(matrix_key)
            
            print(f"LHS parts: {lhs_parts}")
            print(f"RHS parts: {rhs_parts}")

            # Normalize the parts for comparison
            lhs_normalized = [normalize_key(part) for part in lhs_parts]
            rhs_normalized = [normalize_key(part) for part in rhs_parts]
            
            print(f"LHS normalized: {lhs_normalized}")
            print(f"RHS normalized: {rhs_normalized}")

            # Check if all normalized keys exist in the DataFrames before proceeding
            missing_lhs = []
            missing_rhs = []
            
            for j, norm_key in enumerate(lhs_normalized):
                if norm_key not in cbl_normalized_index:
                    missing_lhs.append(f"{lhs_parts[j]} (normalized: {norm_key})")
            
            for j, norm_key in enumerate(rhs_normalized):
                if norm_key not in insurer_normalized_index:
                    missing_rhs.append(f"{rhs_parts[j]} (normalized: {norm_key})")
            
            if missing_lhs or missing_rhs:
                print(f"Skipping matrix key - missing keys:")
                if missing_lhs:
                    print(f"  Left side keys not found in cbl_df: {missing_lhs}")
                if missing_rhs:
                    print(f"  Right side keys not found in insurer_df: {missing_rhs}")
                continue
            
            # Get the actual keys from the normalized mapping
            lhs_actual_keys = [cbl_normalized_index[norm_key] for norm_key in lhs_normalized]
            rhs_actual_keys = [insurer_normalized_index[norm_key] for norm_key in rhs_normalized]
            
            print(f"LHS actual keys: {lhs_actual_keys}")
            print(f"RHS actual keys: {rhs_actual_keys}")
            
            # Get the rows using the actual keys
            lhs_rows = cbl_df_matrix_index.loc[lhs_actual_keys]
            rhs_rows = insurer_df_matrix_index.loc[rhs_actual_keys]
            
            print(f"Found {len(lhs_rows)} CBL rows and {len(rhs_rows)} insurer rows")

            # Rebuild the matrix key
            cols_cbl = ['PlacingNo', 'PolicyNo_1', 'ClientName', 'Amount']
            cols_insurer = ['PlacingNo_INSURER', 'PolicyNo_1_INSURER', 'ClientName_INSURER', 'Amount_INSURER']
            
            # Iterate through rows to build keys
            lhs_rebuild = []
            for _, row in lhs_rows.iterrows():
                rebuilt_key = build_key(row, cols_cbl)
                lhs_rebuild.append(rebuilt_key)
                print(f"Rebuilt CBL key: {rebuilt_key}")
            
            rhs_rebuild = []
            for _, row in rhs_rows.iterrows():
                rebuilt_key = build_key(row, cols_insurer)
                rhs_rebuild.append(rebuilt_key)
                print(f"Rebuilt insurer key: {rebuilt_key}")
            
            reconstructed_key = f"{','.join(lhs_rebuild)}|{','.join(rhs_rebuild)}"
            print(f"RECONSTRUCTED KEY: {reconstructed_key}")
            print(f"ORIGINAL KEY: {matrix_key}")

            # Normalize both keys for comparison
            normalized_reconstructed = normalize_key(reconstructed_key)
            normalized_original = normalize_key(matrix_key)
            
            print(f"NORMALIZED RECONSTRUCTED: {normalized_reconstructed}")
            print(f"NORMALIZED ORIGINAL: {normalized_original}")

            # Check if the reconstructed key matches the matrix key
            if normalized_reconstructed == normalized_original:
                print(f"✓ MATCH FOUND! Keys match after normalization")
                
                # Get all insurer row indices first for validation
                all_insurer_indices = []
                for rhs_key in rhs_actual_keys:
                    insurer_row_index = insurer_df[insurer_df['MatrixKey_INSURER'] == rhs_key].index[0]
                    all_insurer_indices.append(insurer_row_index)
                
                # Pre-validation: Check if insurer rows are available for exact matches
                if global_tracker:
                    can_use_all, available_indices, conflicts = global_tracker.can_use_for_exact(all_insurer_indices)
                    
                    if not can_use_all:
                        print(f"⚠️ Matrix match validation failed - some insurer rows already used: {conflicts}")
                        if not available_indices:
                            print(f"❌ Skipping matrix match - no available insurer rows")
                            continue
                        else:
                            print(f"🔧 Using only available insurer indices: {available_indices}")
                            # Filter to only available indices
                            filtered_rhs_keys = []
                            filtered_insurer_indices = []
                            for rhs_key in rhs_actual_keys:
                                insurer_idx = insurer_df[insurer_df['MatrixKey_INSURER'] == rhs_key].index[0]
                                if insurer_idx in available_indices:
                                    filtered_rhs_keys.append(rhs_key)
                                    filtered_insurer_indices.append(insurer_idx)
                            
                            rhs_actual_keys = filtered_rhs_keys
                            all_insurer_indices = filtered_insurer_indices
                            
                            if not all_insurer_indices:
                                print(f"❌ No valid insurer indices remaining after filtering")
                                continue
                
                # Mark CBL rows as exact matches
                for j, lhs_key in enumerate(lhs_actual_keys):
                    print(f"Marking CBL row as exact match: {lhs_key}")
                    
                    # Get the original row index from the DataFrame
                    cbl_row_index = cbl_df[cbl_df['MatrixKey'] == lhs_key].index[0]
                    
                    # Calculate matched amount total
                    matched_amount_total = sum(insurer_df.loc[all_insurer_indices, "Amount_Clean_INSURER"])
                    
                    # Apply match to CBL DataFrame
                    cbl_df.at[cbl_row_index, "match_status"] = "Exact Match"
                    cbl_df.at[cbl_row_index, "match_reason"] = "Matrix Key Match"
                    cbl_df.at[cbl_row_index, "matched_insurer_indices"] = all_insurer_indices
                    cbl_df.at[cbl_row_index, "matched_amtdue_total"] = matched_amount_total
                    cbl_df.at[cbl_row_index, "match_resolved_in_pass"] = "matrix"
                    
                    # Register the match in global tracker
                    if global_tracker:
                        success, final_indices, match_conflicts, affected_cbl_rows = global_tracker.mark_exact_match(
                            cbl_row_index, all_insurer_indices, cbl_df
                        )
                        
                        if not success:
                            print(f"❌ Failed to register matrix match in global tracker: {match_conflicts}")
                            # Revert the CBL match
                            cbl_df.at[cbl_row_index, "match_status"] = "No Match"
                            cbl_df.at[cbl_row_index, "match_reason"] = "Matrix match failed - conflicts"
                            cbl_df.at[cbl_row_index, "matched_insurer_indices"] = []
                            cbl_df.at[cbl_row_index, "matched_amtdue_total"] = None
                            continue
                        else:
                            print(f"✅ Matrix match registered in global tracker")
                            if affected_cbl_rows:
                                print(f"🔄 Matrix match affected {len(affected_cbl_rows)} other CBL rows: {affected_cbl_rows}")
                            
                            # Use the final indices confirmed by global tracker
                            cbl_df.at[cbl_row_index, "matched_insurer_indices"] = final_indices
                            all_insurer_indices = final_indices
                    
                    exact_matches_count += 1
                
                # Track matched insurer indices for legacy return value
                matched_insurer_indices.update(all_insurer_indices)
            else:
                print(f"✗ Keys don't match after normalization")
                
        except Exception as e:
            print(f"❌ Error processing matrix key {matrix_key}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nMatrix pass complete: {exact_matches_count} exact matches found")
    
    return cbl_df, insurer_df, matched_insurer_indices
