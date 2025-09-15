#!/usr/bin/env python3

"""
Refactored Excel Matching System

This is the new main entry point for the refactored Excel matching system.
The original functionality has been split into organized modules:

- matching/data_processing.py: Data preprocessing and validation
- matching/matching_engine.py: Core matching algorithms (pass1, pass2, pass3)
- matching/utils.py: Utility functions
- matching/output_handler.py: Output generation and data merging
- matching/orchestrator.py: Main orchestration logic

Usage:
    python match_excel_refactored.py cbl_file.xlsx insurer_file.xlsx --output result.xlsx
"""

import logging
from matching.orchestrator import run_matching_process

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Default column mappings - these can be customized per use case
# The system will now dynamically handle missing columns gracefully
DEFAULT_COLUMN_MAPPINGS = {
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

# Example: Dynamic column mappings for files with custom column names
# CUSTOM_COLUMN_MAPPINGS = {
#     'cbl_mappings': {
#         'Ref_Number': 'PlacingNo',      # Map 'Ref_Number' to 'PlacingNo'
#         'Policy_Ref': 'PolicyNo',       # Map 'Policy_Ref' to 'PolicyNo'
#         'Client': 'ClientName',         # Map 'Client' to 'ClientName'
#         'Value': 'Amount'               # Map 'Value' to 'Amount'
#     },
#     'insurer_mappings': {
#         'Reference': 'PlacingNo',       # Map 'Reference' to 'PlacingNo'
#         'Policy_1': 'PolicyNo_1',       # Map 'Policy_1' to 'PolicyNo_1'
#         'Customer_Name': 'ClientName',  # Map 'Customer_Name' to 'ClientName'
#         'Premium': 'Amount'             # Map 'Premium' to 'Amount'
#     }
# }

# Default matrix keys configuration
DEFAULT_MATRIX_KEYS = {
    'enabled': True,
    'cbl_columns': ['PlacingNo', 'PolicyNo', 'ClientName', 'Amount'],
    'insurer_columns': ['PlacingNo_INSURER', 'PolicyNo_1_INSURER', 'ClientName_INSURER', 'Amount_INSURER']
}


def main():
    """
    Main entry point for the refactored matching system.
    
    The system now features dynamic preprocessing that gracefully handles:
    - Missing columns (creates empty columns with warnings)
    - Custom column names (via column mappings)
    - Flexible matrix key generation (uses only available columns)
    - Adaptive matching passes (skips passes when required columns missing)
    """
    logger.info("=== Excel Matching System (Refactored) ===")
    
    try:
        results = run_matching_process(
            column_mappings=DEFAULT_COLUMN_MAPPINGS,
            matrix_keys=DEFAULT_MATRIX_KEYS,
            tolerance=100
        )
        
        # Print summary
        logger.info("\n=== Processing Complete ===")
        logger.info(f"Output saved to: {results['output_file']}")
        logger.info(f"CBL Exact Matches: {results['cbl_stats']['exact_matches']}")
        logger.info(f"CBL Partial Matches: {results['cbl_stats']['partial_matches']}")
        logger.info(f"CBL No Matches: {results['cbl_stats']['no_matches']}")
        logger.info(f"Insurer Match Rate: {results['insurer_stats']['exact_match_rate']:.1f}%")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise


if __name__ == "__main__":
    main()
