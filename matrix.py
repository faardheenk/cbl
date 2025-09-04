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
    # print("matrix_key --> ", matrix_key)
    lhs_str, rhs_str = matrix_key.split("|")
    # Don't split by commas - treat each side as a single key
    lhs_parts = lhs_str.split(",")
    rhs_parts = rhs_str.split(",")

    # print("lhs_parts --> ", lhs_parts)
    # print("rhs_parts --> ", rhs_parts)
    return lhs_parts, rhs_parts

def matrix_pass(cbl_df, insurer_df, matrix_keys):
    # print('indexing columns')
    cbl_df_matrix_index = cbl_df.set_index("MatrixKey")
    insurer_df_matrix_index = insurer_df.set_index("MatrixKey_INSURER")

    # print('indexing columns done')

    # Track which insurer rows have been matched
    matched_insurer_indices = set()
    exact_matches_count = 0

    # Create normalized versions of the indices for comparison
    cbl_normalized_index = {normalize_key(key): key for key in cbl_df_matrix_index.index}
    insurer_normalized_index = {normalize_key(key): key for key in insurer_df_matrix_index.index}

    for matrix_item in matrix_keys:
      matrix_key = matrix_item['matrixKey']
      print(f"\nProcessing matrix key: {matrix_key}")
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
      
      for i, norm_key in enumerate(lhs_normalized):
          if norm_key not in cbl_normalized_index:
              missing_lhs.append(f"{lhs_parts[i]} (normalized: {norm_key})")
      
      for i, norm_key in enumerate(rhs_normalized):
          if norm_key not in insurer_normalized_index:
              missing_rhs.append(f"{rhs_parts[i]} (normalized: {norm_key})")
      
      if missing_lhs or missing_rhs:
          print(f"Skipping matrix key - missing keys:")
          if missing_lhs:
              print(f"  Left side keys not found in cbl_df: {missing_lhs}")
          if missing_rhs:
              print(f"  Right side keys not found in insurer_df: {missing_rhs}")
          continue

      try:
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
          cols_cbl =  [ 'PlacingNo', 'PolicyNo_1', 'ClientName', 'Amount' ]
          cols_insurer = [ 'PlacingNo_INSURER', 'PolicyNo_1_INSURER', 'ClientName_INSURER', 'Amount_INSURER' ]
        
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
            
            # Mark CBL rows as exact matches
            for i, lhs_key in enumerate(lhs_actual_keys):
                print(f"Marking CBL row as exact match: {lhs_key}")
                
                # Get the original row index from the DataFrame
                cbl_row_index = cbl_df[cbl_df['MatrixKey'] == lhs_key].index[0]
                
                cbl_df.at[cbl_row_index, "match_status"] = "Exact Match"
                cbl_df.at[cbl_row_index, "match_reason"] = "Matrix Key Match"
                
                # Get insurer row indices
                insurer_row_indices = []
                for rhs_key in rhs_actual_keys:
                    insurer_row_index = insurer_df[insurer_df['MatrixKey'] == rhs_key].index[0]
                    insurer_row_indices.append(insurer_row_index)
                
                cbl_df.at[cbl_row_index, "matched_insurer_indices"] = insurer_row_indices
                cbl_df.at[cbl_row_index, "matched_amtdue_total"] = sum(insurer_df.loc[insurer_row_indices, "Amount_Clean_INSURER"])
                cbl_df.at[cbl_row_index, "match_resolved_in_pass"] = "matrix"
                exact_matches_count += 1
            
            # Track matched insurer indices
            for rhs_key in rhs_actual_keys:
                insurer_row_index = insurer_df[insurer_df['MatrixKey'] == rhs_key].index[0]
                matched_insurer_indices.add(insurer_row_index)
          else:
              print(f"✗ Keys don't match after normalization")
              
      except Exception as e:
          print(f"Error processing matrix key {matrix_key}: {str(e)}")
          continue

    print(f"\nMatrix pass complete: {exact_matches_count} exact matches found")
    
    return cbl_df, insurer_df, matched_insurer_indices

