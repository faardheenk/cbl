#!/usr/bin/env python3

import logging
from match_excel import run_matching_process

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Run matching process on CBL Wolmar and SWAN Wolmar data files.
    """
    
    # Column mappings as specified
    column_mappings = {
        'cbl_mappings': {
            "Placing/Endorsement No.": "PlacingNo",
            "Policy No.": "PolicyNo", 
            "Client Name": "ClientName", 
            "Balance (MUR) (Net of Brokerage)": "Amount"
            # "Balance Net of Brokerage": "Amount"
        },
        'insurer_mappings': {
            "BRKREF": "PlacingNo",
            "NAME": "ClientName", 
            "DOCSER": "PolicyNo_1",
            "POLSER": "PolicyNo_2",
            "AMTDUE": "Amount",
        }
    }


    # column_mappings = {
    #     'cbl_mappings': {
    #         "Placing No.": "PlacingNo",
    #         "Policy No": "PolicyNo", 
    #         "Client Name": "ClientName", 
    #         "Net Amount": "Amount"
    #     },
    #     'insurer_mappings': {
    #         "REF": "PlacingNo",
    #         "NAME": "ClientName", 
    #         "POLNO": "PolicyNo_1",
    #         "AMTDUE": "Amount",
    #     }
    # }
    
    # Matrix keys (empty for now, will be determined by the matrix pass)
    matrix_keys = []
    
    # File paths
    # cbl_file = "data/CBL RECON REVIEW.xlsx"
    # insurer_file = "data/SWAN RECON REVIEW.xlsx"
    # output_file = "RESULT RECON REVIEW.xlsx"
    
    cbl_file = "data/CBL.xlsx"
    insurer_file = "data/SWAN.xlsx"
    output_file = "RESULT.xlsx"

    # cbl_file = "data/Demo_Input1.xls"
    # insurer_file = "data/Demo_Input2.xls"
    # output_file = "RESULT Demo.xlsx"

    # cbl_file = "data/CBL 12 SEP.xlsx"
    # insurer_file = "data/SWAN 12 SEP.xlsx"
    # output_file = "RESULT 12 SEP.xlsx"
    
    logger.info("=== Starting CBL/SWAN Recon Review Matching Process ===")
    logger.info(f"CBL File: {cbl_file}")
    logger.info(f"SWAN File: {insurer_file}")
    logger.info(f"Output File: {output_file}")
    
    try:
        # Run the matching process
        results = run_matching_process(
            column_mappings=column_mappings,
            matrix_keys=matrix_keys,
            cbl_file=cbl_file,
            insurer_file=insurer_file,
            output_file=output_file
        )
        
        logger.info("\n=== Matching Process Completed Successfully ===")
        logger.info(f"Results saved to: {results['output_file']}")
        
        # Print summary statistics
        logger.info("\n=== Summary Statistics ===")
        logger.info(f"CBL Records:")
        logger.info(f"  - Exact matches: {results['cbl_stats']['exact_matches']} (Amount: {results['cbl_stats']['exact_match_amount']:,.2f})")
        logger.info(f"  - Partial matches: {results['cbl_stats']['partial_matches']} (Amount: {results['cbl_stats']['partial_match_amount']:,.2f})")
        logger.info(f"  - No matches: {results['cbl_stats']['no_matches']} (Amount: {results['cbl_stats']['no_match_amount']:,.2f})")
        
        logger.info(f"\nSWAN Records:")
        logger.info(f"  - Total rows: {results['insurer_stats']['total_rows']}")
        logger.info(f"  - Exact match rows: {results['insurer_stats']['exact_match_rows']} ({results['insurer_stats']['exact_match_rate']:.1f}%) (Amount: {results['insurer_stats']['exact_match_amount']:,.2f})")
        logger.info(f"  - Partial match rows: {results['insurer_stats']['partial_match_rows']} ({results['insurer_stats']['partial_match_rate']:.1f}%) (Amount: {results['insurer_stats']['partial_match_amount']:,.2f})")
        logger.info(f"  - Unmatched rows: {results['insurer_stats']['unmatched_rows']} ({results['insurer_stats']['unmatched_rate']:.1f}%) (Amount: {results['insurer_stats']['unmatched_amount']:,.2f})")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during matching process: {str(e)}")
        raise

if __name__ == "__main__":
    main()
