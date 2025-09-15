# Dynamic Preprocessing Guide

The preprocessing function has been made more dynamic to handle different column mapping scenarios. This guide shows how to use the new flexible preprocessing capabilities.

## Key Changes

### 1. Dynamic Column Creation

- **Before**: Hardcoded required columns were always created
- **After**: Only columns that are mapped in `column_mappings` are created

### 2. Flexible Data Cleaning

- **Before**: Always tried to clean specific hardcoded columns
- **After**: Only cleans columns that exist in the dataframes

### 3. Dynamic Matrix Key Generation

- **Before**: Always used the same 4 columns for matrix keys
- **After**: Uses only available columns and can be customized

## Usage Examples

### Example 1: Standard Usage (All Columns Available)

```python
from matching.orchestrator import run_matching_process

# Standard column mappings - all columns available
column_mappings = {
    'cbl_mappings': {
        'PlacingNo': 'PlacingNo',
        'PolicyNo': 'PolicyNo',
        'ClientName': 'ClientName',
        'Amount': 'Amount'
    },
    'insurer_mappings': {
        'PlacingNo': 'PlacingNo',
        'PolicyNo_1': 'PolicyNo_1',
        'PolicyNo_2': 'PolicyNo_2',
        'ClientName': 'ClientName',
        'Amount': 'Amount'
    }
}

matrix_keys = {
    'enabled': True,
    'cbl_columns': ['PlacingNo', 'PolicyNo', 'ClientName', 'Amount'],
    'insurer_columns': ['PlacingNo_INSURER', 'PolicyNo_1_INSURER', 'ClientName_INSURER', 'Amount_INSURER']
}

results = run_matching_process(
    column_mappings=column_mappings,
    matrix_keys=matrix_keys,
    cbl_file='data.xlsx',
    insurer_file='insurer.xlsx'
)
```

### Example 2: Missing PlacingNo Columns

```python
# Scenario: Neither CBL nor insurer files have PlacingNo columns
column_mappings = {
    'cbl_mappings': {
        # 'PlacingNo': 'PlacingNo',  # Not available
        'PolicyNo': 'PolicyNo',
        'ClientName': 'ClientName',
        'Amount': 'Amount'
    },
    'insurer_mappings': {
        # 'PlacingNo': 'PlacingNo',  # Not available
        'PolicyNo_1': 'PolicyNo_1',
        'ClientName': 'ClientName',
        'Amount': 'Amount'
    }
}

matrix_keys = {
    'enabled': True,
    'cbl_columns': ['PolicyNo', 'ClientName', 'Amount'],  # No PlacingNo
    'insurer_columns': ['PolicyNo_1_INSURER', 'ClientName_INSURER', 'Amount_INSURER']
}

results = run_matching_process(
    column_mappings=column_mappings,
    matrix_keys=matrix_keys,
    cbl_file='data.xlsx',
    insurer_file='insurer.xlsx'
)
```

### Example 3: Custom Column Names

```python
# Scenario: Files have different column names that need mapping
column_mappings = {
    'cbl_mappings': {
        'Ref_Number': 'PlacingNo',      # Map 'Ref_Number' to 'PlacingNo'
        'Policy_Ref': 'PolicyNo',       # Map 'Policy_Ref' to 'PolicyNo'
        'Client': 'ClientName',         # Map 'Client' to 'ClientName'
        'Value': 'Amount'               # Map 'Value' to 'Amount'
    },
    'insurer_mappings': {
        'Reference': 'PlacingNo',       # Map 'Reference' to 'PlacingNo'
        'Policy_1': 'PolicyNo_1',       # Map 'Policy_1' to 'PolicyNo_1'
        'Customer_Name': 'ClientName',  # Map 'Customer_Name' to 'ClientName'
        'Premium': 'Amount'             # Map 'Premium' to 'Amount'
    }
}

# Matrix keys will use the mapped column names
matrix_keys = {
    'enabled': True,
    'cbl_columns': ['PlacingNo', 'PolicyNo', 'ClientName', 'Amount'],
    'insurer_columns': ['PlacingNo_INSURER', 'PolicyNo_1_INSURER', 'ClientName_INSURER', 'Amount_INSURER']
}
```

### Example 4: Minimal Columns (Only Name and Amount)

```python
# Scenario: Only client name and amount columns are available
column_mappings = {
    'cbl_mappings': {
        'ClientName': 'ClientName',
        'Amount': 'Amount'
    },
    'insurer_mappings': {
        'ClientName': 'ClientName',
        'Amount': 'Amount'
    }
}

matrix_keys = {
    'enabled': True,
    'cbl_columns': ['ClientName', 'Amount'],  # Only these columns
    'insurer_columns': ['ClientName_INSURER', 'Amount_INSURER']
}

# This will skip Pass 1 and Pass 2, only run Pass 3
results = run_matching_process(
    column_mappings=column_mappings,
    matrix_keys=matrix_keys,
    cbl_file='data.xlsx',
    insurer_file='insurer.xlsx'
)
```

### Example 5: Using Dynamic Column Mapping Helper

```python
from matching.data_processing import create_dynamic_column_mappings
import pandas as pd

# Read files to get available columns
cbl_df = pd.read_excel('cbl_data.xlsx')
insurer_df = pd.read_excel('insurer_data.xlsx')

# Create dynamic mappings based on available columns
column_mappings = create_dynamic_column_mappings(
    cbl_columns=list(cbl_df.columns),
    insurer_columns=list(insurer_df.columns),
    custom_mappings={
        'cbl_mappings': {
            'Ref_Number': 'PlacingNo',  # Custom mapping
        },
        'insurer_mappings': {
            'Reference': 'PlacingNo',   # Custom mapping
        }
    }
)

print(f"Generated mappings: {column_mappings}")
```

## What Happens When Columns Are Missing

### Data Processing Behavior

1. **Missing mapped columns**: Warning logged, empty column created
2. **Missing cleaning targets**: Cleaning step skipped for that column
3. **Missing matrix key columns**: Matrix key built with available columns only

### Matching Pass Behavior

1. **Pass 1**: Skipped if PlacingNo or Amount not available
2. **Pass 2**: Skipped if PolicyNo, ClientName, or Amount not available
3. **Pass 3**: Skipped if ClientName or Amount not available

### Example Log Output

```
WARNING - Mapped CBL column PlacingNo not found - creating with empty values
INFO - PolicyNo_2_INSURER column not found - creating empty column for compatibility
INFO - Built CBL MatrixKey using columns: ['PolicyNo', 'ClientName', 'Amount']
INFO - Built insurer MatrixKey using columns: ['PolicyNo_1_INSURER', 'ClientName_INSURER', 'Amount_INSURER']
⚠ Pass 1: Required keys (PlacingNo, Amount) not found in mappings - skipping Pass 1
✓ Pass 2: Required keys found in mappings - running Pass 2
✓ Pass 3: Required keys found in mappings - running Pass 3
```

## Benefits

### 1. **Flexibility**

- Handle files with different column structures
- No need to modify code for different data formats
- Graceful handling of missing columns

### 2. **Robustness**

- System doesn't crash when expected columns are missing
- Clear logging shows what's happening
- Matching continues with available data

### 3. **Customization**

- Easy to map custom column names
- Control which columns are used for matching
- Adapt to different business requirements

### 4. **Backward Compatibility**

- Existing configurations continue to work
- Default behavior preserved when all columns available
- No breaking changes to existing workflows

## Migration from Static to Dynamic

### Old Approach (Static)

```python
# Had to ensure files always had these exact columns
# PlacingNo, PolicyNo, ClientName, Amount (CBL)
# PlacingNo, PolicyNo_1, PolicyNo_2, ClientName, Amount (Insurer)
```

### New Approach (Dynamic)

```python
# Define what you have and how it maps
column_mappings = {
    'cbl_mappings': {
        'actual_column_name': 'standard_name'
    },
    'insurer_mappings': {
        'actual_column_name': 'standard_name'
    }
}
```

This dynamic approach makes the system much more flexible and able to handle real-world data variations!
