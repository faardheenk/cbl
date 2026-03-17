#!/usr/bin/env python3

import pandas as pd
import logging
import re
import io

logger = logging.getLogger(__name__)

def detect_header_row(file_content, max_rows_to_check=20, sheet_name=0):
    """
    Intelligently detect the header row in an Excel file by analyzing the structure.
    
    This function handles Excel files that may have:
    - Company logos and headers
    - Metadata (dates, addresses, etc.)
    - Multiple header rows
    - Empty rows before actual data
    
    Args:
        file_content (bytes): Excel file content as bytes
        max_rows_to_check (int): Maximum number of rows to analyze for header detection
        sheet_name (str or int): Sheet name or index to analyze
        
    Returns:
        tuple: (header_row_index, column_names_list)
    """
    import io
    
    logger.info(f"🔍 Detecting header row in memory content (sheet: {sheet_name})")
    file_source = io.BytesIO(file_content)
    
    try:
        # Read the first several rows without treating any as headers
        df_sample = pd.read_excel(file_source, sheet_name=sheet_name, header=None, nrows=max_rows_to_check)
        
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
            header_df = pd.read_excel(file_source, sheet_name=sheet_name, header=header_row, nrows=1)
            column_names = list(header_df.columns)
            
            logger.info(f"✅ Detected header at row {header_row}")
            logger.info(f"📋 Columns found: {column_names}")
            
            return header_row, column_names
        else:
            # Fallback: assume first row with data
            logger.warning("⚠️ Could not confidently detect header row, using row 0")
            header_df = pd.read_excel(file_source, sheet_name=sheet_name, header=0, nrows=1)
            return 0, list(header_df.columns)
            
    except Exception as e:
        logger.error(f"❌ Error during header detection: {str(e)}")
        logger.info("🔄 Falling back to default header detection (row 0)")
        try:
            header_df = pd.read_excel(file_source, sheet_name=sheet_name, header=0, nrows=1)
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

def read_excel_with_smart_headers(file_content, **kwargs):
    """
    Read Excel file with intelligent header detection and multi-sheet support.
    
    This function:
    1. Detects if the Excel file has multiple sheets
    2. Applies smart header detection to each sheet
    3. Compares column structures between sheets
    4. Merges sheets with similar column structures
    5. Returns a single consolidated DataFrame
    
    Args:
        file_content (bytes): Excel file content as bytes
        **kwargs: Additional arguments to pass to pd.read_excel
        
    Returns:
        pandas.DataFrame: DataFrame with properly detected headers and merged sheets
    """

    
    logger.info(f"📖 Reading Excel file from memory with smart header detection")
    file_source = io.BytesIO(file_content)
    
    try:
        # Get all sheet names
        excel_file = pd.ExcelFile(file_source)
        sheet_names = excel_file.sheet_names
        
        logger.info(f"📋 Found {len(sheet_names)} sheet(s): {sheet_names}")
        
        if len(sheet_names) == 1:
            # Single sheet - use existing logic
            return _read_single_sheet_with_smart_headers(file_content, sheet_names[0], **kwargs)
        
        # Multiple sheets - analyze and potentially merge
        sheet_data = {}
        sheet_headers = {}
        
        # Process each sheet
        for sheet_name in sheet_names:
            logger.info(f"🔍 Processing sheet: {sheet_name}")
            try:
                # Detect header row for this sheet
                header_row, column_names = detect_header_row(file_content, sheet_name=sheet_name)
                
                # Read the sheet with detected header
                df = _read_single_sheet_with_smart_headers(file_content, sheet_name, **kwargs)
                
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
        return _read_single_sheet_with_smart_headers(file_content, 0, **kwargs)

def _read_single_sheet_with_smart_headers(file_content, sheet_name, **kwargs):
    """
    Read a single sheet with smart header detection.
    
    Args:
        file_content (bytes): Excel file content as bytes
        sheet_name (str or int): Sheet name or index
        **kwargs: Additional arguments to pass to pd.read_excel
        
    Returns:
        pandas.DataFrame: DataFrame with properly detected headers
    """
    # Detect the header row for this specific sheet
    header_row, column_names = detect_header_row(file_content, sheet_name=sheet_name)
    
    # Read the file with the detected header
    import io
    file_source = io.BytesIO(file_content)
    
    if header_row > 0:
        # Skip rows before the header and use the detected row as header
        df = pd.read_excel(file_source, sheet_name=sheet_name, header=header_row, **kwargs)
        
        # Clean up any duplicate header rows that might have been included
        # Remove rows where the first column matches the column name (duplicate headers)
        if len(df) > 0 and len(df.columns) > 0:
            first_col_name = df.columns[0]
            df = df[df.iloc[:, 0] != first_col_name]
        
    else:
        # Use regular reading
        df = pd.read_excel(file_source, sheet_name=sheet_name, header=header_row, **kwargs)
    
    # Remove completely empty rows
    df = df.dropna(how='all')
    
    # Trim leading and trailing whitespaces from column names
    if len(df.columns) > 0:
        original_columns = list(df.columns)
        df.columns = [str(col).strip() if pd.notna(col) else col for col in df.columns]
        
        # Log if any column names were trimmed
        trimmed_columns = []
        for orig, new in zip(original_columns, df.columns):
            if str(orig) != str(new):
                trimmed_columns.append(f"'{orig}' -> '{new}'")
        
        if trimmed_columns:
            logger.info(f"🧹 Trimmed whitespaces from column names:")
            for trimmed in trimmed_columns:
                logger.info(f"   {trimmed}")
    
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
    
    Supports both simple key-value mappings and one-to-many mappings where a single key
    can map to multiple columns (e.g., "Details": ["PlacingNo", "PolicyNo_1"]).
    
    Args:
        cbl_columns: List of available CBL column names
        insurer_columns: List of available insurer column names
        custom_mappings: Optional custom mappings to override defaults
                        Can contain lists for one-to-many mappings
        
    Returns:
        dict: Column mappings dictionary
    """
    # Default mappings - these are the standard expected column names
    default_mappings = {
        'cbl_mappings': {
            'PlacingNo': 'PlacingNo',
            'PolicyNo': 'PolicyNo', 
            'ClientName': 'ClientName',
            # 'Amount': 'ProcessedAmount'
        },
        'insurer_mappings': {
            'PlacingNo': 'PlacingNo',
            'PolicyNo_1': 'PolicyNo_1',
            'PolicyNo_2': 'PolicyNo_2',
            'ClientName': 'ClientName', 
            # 'Amount': 'ProcessedAmount'
        }
    }
    
    # Merge with custom mappings
    mappings = default_mappings.copy()
    if custom_mappings:
        for key in ['cbl_mappings', 'insurer_mappings']:
            if key in custom_mappings:
                mappings[key].update(custom_mappings[key])
    
    def find_matching_column(target_col, available_columns):
        """Find matching column with case-insensitive, whitespace-tolerant matching."""
        target_clean = str(target_col).strip().lower()
        return next((col for col in available_columns 
                    if str(col).strip().lower() == target_clean), None)
    
    def expand_mappings(mappings_dict, available_columns):
        """Expand mappings, handling both simple and one-to-many mappings."""
        expanded = {}
        
        for source_key, target_value in mappings_dict.items():
            matching_col = find_matching_column(source_key, available_columns)
            if not matching_col:
                logger.warning(f"⚠️ No matching column found for '{source_key}' in available columns")
                continue
                
            if isinstance(target_value, list):
                # One-to-many mapping
                for i, target_col in enumerate(target_value):
                    if i == 0:
                        expanded[matching_col] = target_col
                    else:
                        expanded[f"{matching_col}_{target_col}"] = target_col
                logger.info(f"🔗 One-to-many mapping: '{source_key}' -> '{matching_col}' -> {target_value}")
            else:
                # Simple one-to-one mapping
                expanded[matching_col] = target_value
                if str(source_key) != str(matching_col):
                    logger.info(f"🔗 Simple mapping: '{source_key}' -> '{matching_col}' (case/whitespace tolerance)")
        
        return expanded
    
    # Process both CBL and insurer mappings
    result = {
        'cbl_mappings': expand_mappings(mappings['cbl_mappings'], cbl_columns),
        'insurer_mappings': expand_mappings(mappings['insurer_mappings'], insurer_columns)
    }
    
    logger.info(f"Dynamic CBL mappings: {result['cbl_mappings']}")
    logger.info(f"Dynamic insurer mappings: {result['insurer_mappings']}")
    
    return result



def preprocess(cbl_df, insurer_df, column_mappings):
    """
    Preprocess and clean the CBL and insurer dataframes.

    Args:
        cbl_df: CBL dataframe
        insurer_df: Insurer dataframe
        column_mappings: Dictionary containing column mappings (already filtered by create_dynamic_column_mappings)

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

    def handle_one_to_many_mappings(df, column_map):
        """Handle one-to-many mappings by duplicating source columns efficiently."""
        # Group mappings by original source column
        source_groups = {}
        for source_col, target_col in column_map.items():
            # Find original source column (before _TargetCol suffix)
            original_source = source_col
            if '_' in source_col:
                # Find the longest prefix that exists in the dataframe
                parts = source_col.split('_')
                for i in range(len(parts), 0, -1):
                    potential_source = '_'.join(parts[:i])
                    if potential_source in df.columns:
                        original_source = potential_source
                        break
            
            if original_source not in source_groups:
                source_groups[original_source] = []
            source_groups[original_source].append((source_col, target_col))
        
        # Create new column map with duplication
        new_column_map = {}
        for original_source, mappings in source_groups.items():
            if len(mappings) > 1 and original_source in df.columns:
                # One-to-many: duplicate the column
                targets = [target for _, target in mappings]
                logger.info(f"🔄 Duplicating '{original_source}' -> {targets}")
                
                for i, (_, target_col) in enumerate(mappings):
                    if i == 0:
                        new_column_map[original_source] = target_col
                    else:
                        temp_name = f"{original_source}_copy_{i}"
                        df[temp_name] = df[original_source]
                        new_column_map[temp_name] = target_col
            else:
                # Single mapping
                for source_key, target_col in mappings:
                    actual_source = source_key.rsplit('_', 1)[0] if '_' in source_key else source_key
                    new_column_map[actual_source] = target_col
        
        return df, new_column_map

    # Apply one-to-many handling
    cbl_df, cbl_column_map = handle_one_to_many_mappings(cbl_df, cbl_column_map)
    insurer_df, insurer_column_map = handle_one_to_many_mappings(insurer_df, insurer_column_map)

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
    
    # Log available columns after renaming
    logger.info(f"Available insurer columns after renaming: {list(insurer_df.columns)}")

    # Clean and process data dynamically based on available columns
    logger.info(f"Before data cleaning - CBL rows: {len(cbl_df)}, Insurer rows: {len(insurer_df)}")
    
    # Clean CBL columns dynamically
    if "PlacingNo" in cbl_df.columns:
        cbl_df["PlacingNo_Clean"] = cbl_df["PlacingNo"].str.upper().str.strip()
        cbl_df["PlacingNo_Clean"] = cbl_df["PlacingNo_Clean"].str.replace(pattern, '', regex=True)
    
    if "ProcessedAmount" in cbl_df.columns:
        cbl_df["ProcessedAmount_Clean"] = pd.to_numeric(cbl_df["ProcessedAmount"], errors="coerce")
        
    if "PolicyNo" in cbl_df.columns:
        # Enhanced policy number cleaning with duplicate handling
        cbl_df["PolicyNo_Clean"] = cbl_df["PolicyNo"].astype(str).str.strip()
        
        # Handle decimal removal and symbol cleaning to preserve duplicates
        def clean_policy_decimals(policy_str):
            if pd.isna(policy_str) or str(policy_str).strip() == "":
                return ""
            
            # Split on spaces first to handle multiple policy numbers
            parts = str(policy_str).strip().split()
            cleaned_parts = []
            
            for part in parts:
                # Remove .0 suffix but preserve other decimals
                if part.endswith('.0'):
                    cleaned_part = part[:-2]
                else:
                    # Split on decimal and take first part only if it looks like Excel decimal artifact
                    if '.' in part:
                        before_dot, after_dot = part.split('.', 1)
                        # If after decimal is just digits, likely Excel artifact
                        if after_dot.isdigit():
                            cleaned_part = before_dot
                        else:
                            cleaned_part = part  # Keep original
                    else:
                        cleaned_part = part
                
                # Remove ALL symbols and special characters, keep only alphanumeric
                cleaned_part = re.sub(r'[^a-zA-Z0-9]', '', cleaned_part)
                
                # Only keep meaningful parts (at least 2 characters)
                if len(cleaned_part) >= 2:
                    cleaned_parts.append(cleaned_part.upper())
            
            return ' '.join(cleaned_parts)
        
        cbl_df["PolicyNo_Clean"] = cbl_df["PolicyNo"].apply(clean_policy_decimals)
        
        # Remove common prefixes/suffixes and normalize
        cbl_df["PolicyNo_Clean"] = cbl_df["PolicyNo_Clean"].str.replace(r'^[Nn][Aa][Nn]$', '', regex=True)
        cbl_df["PolicyNo_Clean"] = cbl_df["PolicyNo_Clean"].str.replace(r'^\s*$', '', regex=True)
        cbl_df["PolicyNo_Clean"] = cbl_df["PolicyNo_Clean"].fillna("")
        
    if "ClientName" in cbl_df.columns:
        cbl_df["ClientName_Clean"] = cbl_df["ClientName"].astype(str).str.upper().str.strip()
        cbl_df["ClientName_Clean"] = cbl_df["ClientName_Clean"].str.replace(pattern, '', regex=True)

    # Clean insurer columns dynamically
    if "PlacingNo_INSURER" in insurer_df.columns:
        insurer_df["PlacingNo_Clean_INSURER"] = insurer_df["PlacingNo_INSURER"].astype(str).str.upper().str.strip()
        insurer_df["PlacingNo_Clean_INSURER"] = insurer_df["PlacingNo_Clean_INSURER"].str.replace(pattern, '', regex=True)
    
    if "PolicyNo_1_INSURER" in insurer_df.columns:
        # Enhanced policy number cleaning for insurer with duplicate handling and symbol removal
        def clean_policy_decimals(policy_str):
            if pd.isna(policy_str) or str(policy_str).strip() == "":
                return ""
            
            # Split on spaces first to handle multiple policy numbers
            parts = str(policy_str).strip().split()
            cleaned_parts = []
            
            for part in parts:
                # Remove .0 suffix but preserve other decimals
                if part.endswith('.0'):
                    cleaned_part = part[:-2]
                else:
                    # Split on decimal and take first part only if it looks like Excel decimal artifact
                    if '.' in part:
                        before_dot, after_dot = part.split('.', 1)
                        # If after decimal is just digits, likely Excel artifact
                        if after_dot.isdigit():
                            cleaned_part = before_dot
                        else:
                            cleaned_part = part  # Keep original
                    else:
                        cleaned_part = part
                
                # Remove ALL symbols and special characters, keep only alphanumeric
                cleaned_part = re.sub(r'[^a-zA-Z0-9]', '', cleaned_part)
                
                # Only keep meaningful parts (at least 2 characters)
                if len(cleaned_part) >= 2:
                    cleaned_parts.append(cleaned_part.upper())
            
            return ' '.join(cleaned_parts)
        
        insurer_df["PolicyNo_Clean_INSURER"] = insurer_df["PolicyNo_1_INSURER"].apply(clean_policy_decimals)
        
        # Remove common prefixes/suffixes and normalize
        insurer_df["PolicyNo_Clean_INSURER"] = insurer_df["PolicyNo_Clean_INSURER"].str.replace(r'^[Nn][Aa][Nn]$', '', regex=True)
        insurer_df["PolicyNo_Clean_INSURER"] = insurer_df["PolicyNo_Clean_INSURER"].str.replace(r'^\s*$', '', regex=True)
        insurer_df["PolicyNo_Clean_INSURER"] = insurer_df["PolicyNo_Clean_INSURER"].fillna("")
    
    if "ProcessedAmount_INSURER" in insurer_df.columns:
        insurer_df["ProcessedAmount_Clean_INSURER"] = pd.to_numeric(insurer_df["ProcessedAmount_INSURER"], errors="coerce")
    
    # Clean insurer client names for Pass 3 name clustering
    if "ClientName_INSURER" in insurer_df.columns:
        insurer_df["ClientName_Clean_INSURER"] = insurer_df["ClientName_INSURER"].astype(str).str.upper().str.strip()
        insurer_df["ClientName_Clean_INSURER"] = insurer_df["ClientName_Clean_INSURER"].str.replace(pattern, '', regex=True)
        logger.info("✓ Created ClientName_Clean_INSURER column for insurer name clustering")
    else:
        logger.warning("⚠ ClientName_INSURER column not found - Pass 3 name clustering may not work properly")
    
    logger.info(f"After data cleaning - CBL rows: {len(cbl_df)}, Insurer rows: {len(insurer_df)}")
    
    # Handle optional PolicyNo_2 column dynamically
    if "PolicyNo_2_INSURER" in insurer_df.columns:
        # Enhanced policy number cleaning for PolicyNo_2 with duplicate handling and symbol removal
        def clean_policy_decimals_2(policy_str):
            if pd.isna(policy_str) or str(policy_str).strip() == "":
                return ""
            
            # Split on spaces first to handle multiple policy numbers
            parts = str(policy_str).strip().split()
            cleaned_parts = []
            
            for part in parts:
                # Remove .0 suffix but preserve other decimals
                if part.endswith('.0'):
                    cleaned_part = part[:-2]
                else:
                    # Split on decimal and take first part only if it looks like Excel decimal artifact
                    if '.' in part:
                        before_dot, after_dot = part.split('.', 1)
                        # If after decimal is just digits, likely Excel artifact
                        if after_dot.isdigit():
                            cleaned_part = before_dot
                        else:
                            cleaned_part = part  # Keep original
                    else:
                        cleaned_part = part
                
                # Remove ALL symbols and special characters, keep only alphanumeric
                cleaned_part = re.sub(r'[^a-zA-Z0-9]', '', cleaned_part)
                
                # Only keep meaningful parts (at least 2 characters)
                if len(cleaned_part) >= 2:
                    cleaned_parts.append(cleaned_part.upper())
            
            return ' '.join(cleaned_parts)
        
        insurer_df["PolicyNo_2_Clean_INSURER"] = insurer_df["PolicyNo_2_INSURER"].apply(clean_policy_decimals_2)
        
        # Remove common prefixes/suffixes and normalize
        insurer_df["PolicyNo_2_Clean_INSURER"] = insurer_df["PolicyNo_2_Clean_INSURER"].str.replace(r'^[Nn][Aa][Nn]$', '', regex=True)
        insurer_df["PolicyNo_2_Clean_INSURER"] = insurer_df["PolicyNo_2_Clean_INSURER"].str.replace(r'^\s*$', '', regex=True)
        insurer_df["PolicyNo_2_Clean_INSURER"] = insurer_df["PolicyNo_2_Clean_INSURER"].fillna("")
    else:
        # Create empty PolicyNo_2 column if it doesn't exist (some matching passes may expect it)
        insurer_df["PolicyNo_2_Clean_INSURER"] = ""
        logger.info("PolicyNo_2_INSURER column not found - creating empty column for compatibility")

    logger.info(f"Preprocessing complete: {len(cbl_df)} CBL records, {len(insurer_df)} insurer records")
    return cbl_df, insurer_df


def initialize_tracking(cbl_df):
    """Initialize tracking columns for the matching process."""
    logger.info("Initializing tracking columns...")
    cbl_df["match_status"] = "No Match"
    cbl_df["match_pass"] = [[] for _ in range(len(cbl_df))]
    cbl_df["match_reason"] = ""
    cbl_df["matched_insurer_indices"] = [[] for _ in range(len(cbl_df))]
    cbl_df["matched_amtdue_total"] = None
    cbl_df["Amount Difference"] = None
    cbl_df["partial_candidates_indices"] = [[] for _ in range(len(cbl_df))]
    cbl_df["match_resolved_in_pass"] = None
    cbl_df["partial_resolved_in_pass"] = None
    return cbl_df
