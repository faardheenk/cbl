#!/usr/bin/env python3

import pandas as pd
import logging
import re
from matrix import build_key

logger = logging.getLogger(__name__)

def detect_header_row(file_path, max_rows_to_check=20, sheet_name=0):
    """
    Intelligently detect the header row in an Excel file by analyzing the structure.
    
    This function handles Excel files that may have:
    - Company logos and headers
    - Metadata (dates, addresses, etc.)
    - Multiple header rows
    - Empty rows before actual data
    
    Args:
        file_path (str): Path to the Excel file
        max_rows_to_check (int): Maximum number of rows to analyze for header detection
        sheet_name (str or int): Sheet name or index to analyze
        
    Returns:
        tuple: (header_row_index, column_names_list)
    """
    logger.info(f"🔍 Detecting header row in: {file_path} (sheet: {sheet_name})")
    
    try:
        # Read the first several rows without treating any as headers
        df_sample = pd.read_excel(file_path, sheet_name=sheet_name, header=None, nrows=max_rows_to_check)
        
        logger.info(f"Analyzing first {len(df_sample)} rows for header detection...")
        
        # Strategy 1: Look for rows with typical column name patterns
        header_candidates = []
        
        for idx, row in df_sample.iterrows():
            # Convert row to string and check for common column patterns
            row_str = ' '.join([str(cell) for cell in row if pd.notna(cell)])
            
            # Scoring system for header likelihood
            score = 0
            
            # Common header patterns (case-insensitive)
            header_patterns = [
                r'placing.*no',      # Placing No., Placing/Endorsement No.
                r'client.*name',     # Client Name
                r'policy.*no',       # Policy No.
                r'balance',          # Balance, Balance (MUR)
                r'amount',           # Amount, Net Amount
                r'premium',          # Premium, Net Premium
                r'brokerage',        # Brokerage
                r'currency|curr',    # Currency, Curr
                r'period.*insurance', # Period of Insurance
                r'insurance.*type'   # Insurance Type
            ]
            
            # Check how many header patterns match
            for pattern in header_patterns:
                if re.search(pattern, row_str, re.IGNORECASE):
                    score += 1
            
            # Additional scoring criteria
            non_null_count = row.notna().sum()
            total_cells = len(row)
            
            # Prefer rows with more non-null values
            if non_null_count >= 3:  # At least 3 columns
                score += 1
            
            if non_null_count >= 5:  # At least 5 columns
                score += 1
                
            # Prefer rows where most cells are filled
            if non_null_count / total_cells > 0.5:
                score += 1
            
            # Check for typical data vs header characteristics
            numeric_cells = 0
            text_cells = 0
            
            for cell in row:
                if pd.notna(cell):
                    try:
                        float(cell)
                        numeric_cells += 1
                    except (ValueError, TypeError):
                        text_cells += 1
            
            # Headers typically have more text than numbers
            if text_cells > numeric_cells and text_cells >= 3:
                score += 2
            
            header_candidates.append({
                'row_index': idx,
                'score': score,
                'non_null_count': non_null_count,
                'row_content': row_str[:100] + '...' if len(row_str) > 100 else row_str
            })
        
        # Sort candidates by score (descending)
        header_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Log the analysis
        logger.info("Header detection analysis:")
        for i, candidate in enumerate(header_candidates[:5]):  # Show top 5
            logger.info(f"  Row {candidate['row_index']}: Score={candidate['score']}, "
                       f"NonNull={candidate['non_null_count']}")
            logger.info(f"    Content: {candidate['row_content']}")
        
        # Select the best candidate
        if header_candidates and header_candidates[0]['score'] >= 2:
            best_candidate = header_candidates[0]
            header_row = best_candidate['row_index']
            
            # Extract column names from the identified header row
            header_df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row, nrows=1)
            column_names = list(header_df.columns)
            
            logger.info(f"✅ Detected header at row {header_row}")
            logger.info(f"📋 Columns found: {column_names}")
            
            return header_row, column_names
        else:
            # Fallback: assume first row with data
            logger.warning("⚠️ Could not confidently detect header row, using row 0")
            header_df = pd.read_excel(file_path, sheet_name=sheet_name, header=0, nrows=1)
            return 0, list(header_df.columns)
            
    except Exception as e:
        logger.error(f"❌ Error during header detection: {str(e)}")
        logger.info("🔄 Falling back to default header detection (row 0)")
        try:
            header_df = pd.read_excel(file_path, sheet_name=sheet_name, header=0, nrows=1)
            return 0, list(header_df.columns)
        except:
            return 0, []

def compare_column_structures(columns1, columns2, similarity_threshold=0.8):
    """
    Compare two column structures to determine if they're similar enough to merge.
    
    Args:
        columns1: List of column names from first sheet
        columns2: List of column names from second sheet
        similarity_threshold: Minimum similarity ratio to consider sheets mergeable
        
    Returns:
        tuple: (is_similar, similarity_score, common_columns)
    """
    # Convert to sets for comparison
    set1 = set(str(col).strip().lower() for col in columns1 if pd.notna(col))
    set2 = set(str(col).strip().lower() for col in columns2 if pd.notna(col))
    
    # Calculate similarity
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    if len(union) == 0:
        return False, 0.0, []
    
    similarity_score = len(intersection) / len(union)
    is_similar = similarity_score >= similarity_threshold
    
    # Get common columns in original case
    common_columns = []
    if is_similar:
        lower_to_original1 = {str(col).strip().lower(): col for col in columns1 if pd.notna(col)}
        common_columns = [lower_to_original1[col.lower()] for col in intersection if col.lower() in lower_to_original1]
    
    return is_similar, similarity_score, common_columns

def read_excel_with_smart_headers(file_path, **kwargs):
    """
    Read Excel file with intelligent header detection and multi-sheet support.
    
    This function:
    1. Detects if the Excel file has multiple sheets
    2. Applies smart header detection to each sheet
    3. Compares column structures between sheets
    4. Merges sheets with similar column structures
    5. Returns a single consolidated DataFrame
    
    Args:
        file_path (str): Path to Excel file
        **kwargs: Additional arguments to pass to pd.read_excel
        
    Returns:
        pandas.DataFrame: DataFrame with properly detected headers and merged sheets
    """
    logger.info(f"📖 Reading Excel file with smart header detection: {file_path}")
    
    try:
        # Get all sheet names
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        
        logger.info(f"📋 Found {len(sheet_names)} sheet(s): {sheet_names}")
        
        if len(sheet_names) == 1:
            # Single sheet - use existing logic
            return _read_single_sheet_with_smart_headers(file_path, sheet_names[0], **kwargs)
        
        # Multiple sheets - analyze and potentially merge
        sheet_data = {}
        sheet_headers = {}
        
        # Process each sheet
        for sheet_name in sheet_names:
            logger.info(f"🔍 Processing sheet: {sheet_name}")
            try:
                # Detect header row for this sheet
                header_row, column_names = detect_header_row(file_path, sheet_name=sheet_name)
                
                # Read the sheet with detected header
                df = _read_single_sheet_with_smart_headers(file_path, sheet_name, **kwargs)
                
                if len(df) > 0:  # Only include non-empty sheets
                    sheet_data[sheet_name] = df
                    sheet_headers[sheet_name] = list(df.columns)
                    logger.info(f"   ✅ {sheet_name}: {len(df)} rows, {len(df.columns)} columns")
                else:
                    logger.info(f"   ⚠️ {sheet_name}: Empty sheet, skipping")
                    
            except Exception as e:
                logger.warning(f"   ❌ {sheet_name}: Error reading sheet - {str(e)}")
                continue
        
        if not sheet_data:
            logger.error("❌ No readable sheets found!")
            return pd.DataFrame()
        
        if len(sheet_data) == 1:
            # Only one readable sheet
            return list(sheet_data.values())[0]
        
        # Check if sheets can be merged
        mergeable_groups = _group_mergeable_sheets(sheet_data, sheet_headers)
        
        if len(mergeable_groups) == 1 and len(mergeable_groups[0]) == len(sheet_data):
            # All sheets can be merged
            logger.info("🔗 All sheets have similar structure - merging into single DataFrame")
            return _merge_sheets(sheet_data, list(sheet_data.keys()))
        
        elif len(mergeable_groups) == 1 and len(mergeable_groups[0]) > 1:
            # Some sheets can be merged
            main_group = mergeable_groups[0]
            logger.info(f"🔗 Merging compatible sheets: {main_group}")
            merged_df = _merge_sheets(sheet_data, main_group)
            
            # Handle remaining sheets
            remaining_sheets = [s for s in sheet_data.keys() if s not in main_group]
            if remaining_sheets:
                logger.info(f"⚠️ Sheets with different structure found: {remaining_sheets}")
                logger.info(f"📊 Using merged data from: {main_group}")
            
            return merged_df
        
        else:
            # Sheets have different structures - use the largest one
            largest_sheet = max(sheet_data.keys(), key=lambda x: len(sheet_data[x]))
            logger.info(f"📊 Sheets have different structures - using largest sheet: {largest_sheet}")
            logger.info(f"   Available sheets: {list(sheet_data.keys())}")
            
            return sheet_data[largest_sheet]
            
    except Exception as e:
        logger.error(f"❌ Error reading Excel file: {str(e)}")
        # Fallback to single sheet reading
        logger.info("🔄 Falling back to single sheet reading...")
        return _read_single_sheet_with_smart_headers(file_path, 0, **kwargs)

def _read_single_sheet_with_smart_headers(file_path, sheet_name, **kwargs):
    """
    Read a single sheet with smart header detection.
    
    Args:
        file_path (str): Path to Excel file
        sheet_name (str or int): Sheet name or index
        **kwargs: Additional arguments to pass to pd.read_excel
        
    Returns:
        pandas.DataFrame: DataFrame with properly detected headers
    """
    # Detect the header row for this specific sheet
    header_row, column_names = detect_header_row(file_path, sheet_name=sheet_name)
    
    # Read the file with the detected header
    if header_row > 0:
        # Skip rows before the header and use the detected row as header
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row, **kwargs)
        
        # Clean up any duplicate header rows that might have been included
        # Remove rows where the first column matches the column name (duplicate headers)
        if len(df) > 0 and len(df.columns) > 0:
            first_col_name = df.columns[0]
            df = df[df.iloc[:, 0] != first_col_name]
        
    else:
        # Use regular reading
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row, **kwargs)
    
    # Remove completely empty rows
    df = df.dropna(how='all')
    
    return df

def _group_mergeable_sheets(sheet_data, sheet_headers):
    """
    Group sheets that have similar column structures and can be merged.
    
    Args:
        sheet_data (dict): Dictionary of sheet_name -> DataFrame
        sheet_headers (dict): Dictionary of sheet_name -> column_list
        
    Returns:
        list: List of lists, each containing sheet names that can be merged together
    """
    sheet_names = list(sheet_data.keys())
    groups = []
    processed = set()
    
    for i, sheet1 in enumerate(sheet_names):
        if sheet1 in processed:
            continue
            
        current_group = [sheet1]
        processed.add(sheet1)
        
        for j, sheet2 in enumerate(sheet_names[i+1:], i+1):
            if sheet2 in processed:
                continue
                
            # Compare column structures
            is_similar, similarity_score, common_columns = compare_column_structures(
                sheet_headers[sheet1], sheet_headers[sheet2]
            )
            
            if is_similar:
                current_group.append(sheet2)
                processed.add(sheet2)
                logger.info(f"   🔗 {sheet1} and {sheet2} are similar (score: {similarity_score:.2f})")
        
        if len(current_group) > 1:
            groups.append(current_group)
        elif len(current_group) == 1:
            # Single sheet group
            groups.append(current_group)
    
    return groups

def _merge_sheets(sheet_data, sheet_names_to_merge):
    """
    Merge multiple sheets into a single DataFrame.
    
    Args:
        sheet_data (dict): Dictionary of sheet_name -> DataFrame
        sheet_names_to_merge (list): List of sheet names to merge
        
    Returns:
        pandas.DataFrame: Merged DataFrame
    """
    if len(sheet_names_to_merge) == 1:
        return sheet_data[sheet_names_to_merge[0]]
    
    dfs_to_merge = []
    
    for sheet_name in sheet_names_to_merge:
        df = sheet_data[sheet_name].copy()
        # Add a column to track source sheet
        df['_source_sheet'] = sheet_name
        dfs_to_merge.append(df)
        logger.info(f"   📋 {sheet_name}: {len(df)} rows")
    
    # Concatenate all DataFrames
    merged_df = pd.concat(dfs_to_merge, ignore_index=True, sort=False)
    
    # Remove any duplicate header rows that might have been included
    # Look for rows where multiple columns contain column-name-like values
    if len(merged_df) > 0 and len(merged_df.columns) > 0:
        # Get the actual column names for comparison
        actual_columns = set(str(col).strip().lower() for col in merged_df.columns if pd.notna(col))
        
        # Find rows that look like header rows
        header_like_rows = []
        for idx, row in merged_df.iterrows():
            row_values = set(str(val).strip().lower() for val in row if pd.notna(val))
            # If more than 50% of the row values match column names, it's likely a header row
            if len(row_values.intersection(actual_columns)) > len(actual_columns) * 0.5:
                header_like_rows.append(idx)
        
        if header_like_rows:
            logger.info(f"   🧹 Removing {len(header_like_rows)} duplicate header rows")
            merged_df = merged_df.drop(header_like_rows).reset_index(drop=True)
    
    logger.info(f"✅ Merged {len(sheet_names_to_merge)} sheets into {len(merged_df)} total rows")
    logger.info(f"📋 Final columns: {list(merged_df.columns)}")
    
    return merged_df

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
