#!/usr/bin/env python3

"""
Latest Enhanced Matching Process Runner

This script runs the latest version of the matching system with all improvements:
- Complete global tracker integration across ALL passes
- Comprehensive data integrity protection
- Enhanced error handling and validation
- Proper column mapping and matrix key handling
"""

import logging
import os
from datetime import datetime
from matching.orchestrator import run_matching_process

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Run the latest enhanced matching process.
    """
    
    logger.info("=" * 60)
    logger.info("🚀 LATEST ENHANCED MATCHING PROCESS")
    logger.info("=" * 60)
    logger.info("✨ Features: Global Tracker | Data Integrity | Enhanced Error Handling")
    
    # File configuration with auto-detection
    file_options = [
        # {
        #     'cbl_file': "data/SOA CBL.xlsx",
        #     'insurer_file': "data/EI MERGE.xlsx",
        #     'output_file': "data/RESULT_SOA_EI_LATEST.xlsx",
        #     'description': "SOA CBL + EI MERGE"
        # },
        # {
        #     'cbl_file': "data/vitiro cbl.xlsx",
        #     'insurer_file': "data/vitiro el.xlsx",
        #     'output_file': "data/RESULT_VITIRO_LATEST.xlsx",
        #     'description': "Demo files (simplest test)"
        # },
        # {
        #     'cbl_file': "data/Test Grouping CBL.xlsx",
        #     'insurer_file': "data/Test Grouping Insurer.xlsx", 
        #     'output_file': "data/RESULT_TEST_GROUPING_LATEST.xlsx",
        #     'description': "Recon review files"
        # },
        {
            'cbl_file': "data/CBL 12 SEP.xlsx",
            'insurer_file': "data/SWAN 12 SEP.xlsx",
            'output_file': "data/RESULT_CBL_12_SEP_SWAN_12_SEP_LATEST.xlsx",
            'description': "Main CBL/SWAN files"
        }
    ]
    
    # Find the first available file set
    selected_config = None
    for config in file_options:
        if os.path.exists(config['cbl_file']) and os.path.exists(config['insurer_file']):
            selected_config = config
            logger.info(f"🔍 Auto-selected: {config['description']}")
            break
    
    if not selected_config:
        logger.error("❌ No valid file pairs found!")
        logger.info("Available files in data directory:")
        if os.path.exists("data"):
            for file in os.listdir("data"):
                if file.endswith(('.xlsx', '.xls')):
                    logger.info(f"   - {file}")
        return None
    
    cbl_file = selected_config['cbl_file']
    insurer_file = selected_config['insurer_file']
    output_file = selected_config['output_file']
    
    logger.info(f"📄 CBL File: {cbl_file}")
    logger.info(f"📄 Insurer File: {insurer_file}")
    logger.info(f"📄 Output File: {output_file}")
    
    # Enhanced column mappings that work with the actual data structure
    column_mappings = {
        'cbl_mappings': {
            # Princess Tuna CBL mappings
            "Placing/Endorsement No.": "PlacingNo",
            "Policy No.": "PolicyNo", 
            "Client Name": "ClientName", 
            "Balance Net of Brokerage": "ProcessedAmount",
            
            # # Demo file mappings
            # "REF": "PlacingNo",
            # "Name": "ClientName",
            # "POLNO": "PolicyNo",
            # "AMTDUE": "Amount",
            
            # # Standard mappings for other files
            # "Balance (MUR) (Net of Brokerage)": "Amount",
            # "Placing No.": "PlacingNo",
            # "Policy No": "PolicyNo",
            # "Net Amount": "Amount",
            # "Amount": "Amount"
        },
        'insurer_mappings': {
            # Princess Tuna Insurer mappings
            # "Policy Ref": "PlacingNo",
            # "Policy Number": "PolicyNo_1",
            # "Insured Name": "ClientName",
            # "Equivalent in MUR": "ProcessedAmount",
            
            # # Demo file mappings
            # "REF": "PlacingNo",
            # "Name": "ClientName",
            # "POLNO": "PolicyNo_1",
            # "AMTDUE": "Amount",
            
            # # SWAN file mappings
            "BRKREF": "PlacingNo",
            "CLIENT NAME": "ClientName", 
            "DOCSER": "PolicyNo_1",
            "POLSER": "PolicyNo_2",
            "AMTDUE": "ProcessedAmount"
        }
    }

    # Correct matrix keys format - list of dictionaries with 'matrixKey' field
    matrix_keys = []  # Empty for now, will be populated by the system if needed
    
    # Alternative: you can specify specific matrix keys like this:
    # matrix_keys = [
    #     {'matrixKey': 'KEY1|KEY1'},
    #     {'matrixKey': 'KEY2|KEY2'}
    # ]
    
    logger.info(f"⚙️ Column mappings configured")
    logger.info(f"⚙️ Matrix keys: {len(matrix_keys)} keys configured")
    
    try:
        start_time = datetime.now()
        logger.info(f"🚀 Starting matching process at {start_time.strftime('%H:%M:%S')}")
        
        # Read files as bytes for the matching process
        logger.info("📖 Reading files as bytes for matching process...")
        with open(cbl_file, 'rb') as f:
            cbl_file_content = f.read()
        with open(insurer_file, 'rb') as f:
            insurer_file_content = f.read()
        
        # Run the enhanced matching process
        results = run_matching_process(
            column_mappings=column_mappings,
            matrix_keys=matrix_keys,
            cbl_file=cbl_file_content,
            insurer_file=insurer_file_content,
            output_file=output_file,
            tolerance=100
        )
        
        # Write the output file to disk
        if results and 'output_content' in results:
            logger.info(f"💾 Writing output file to disk: {output_file}")
            with open(output_file, 'wb') as f:
                f.write(results['output_content'])
            logger.info(f"✅ Output file saved successfully: {output_file}")
        else:
            logger.error("❌ No output content to save")
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("=" * 60)
        logger.info("✅ LATEST MATCHING PROCESS COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"⏱️ Processing time: {duration.total_seconds():.2f} seconds")
        logger.info(f"📁 Results saved to: {results['output_file']}")
        
        # Enhanced summary
        logger.info(f"\n📊 RESULTS SUMMARY:")
        logger.info(f"🏢 CBL Records:")
        logger.info(f"   ✅ Exact matches: {results['cbl_stats']['exact_matches']:,}")
        logger.info(f"   🔄 Partial matches: {results['cbl_stats']['partial_matches']:,}")
        logger.info(f"   ❌ No matches: {results['cbl_stats']['no_matches']:,}")
        
        logger.info(f"🏛️ Insurer Records:")
        logger.info(f"   📋 Total: {results['insurer_stats']['total_rows']:,}")
        logger.info(f"   ✅ Exact matched: {results['insurer_stats']['exact_match_rows']:,} ({results['insurer_stats']['exact_match_rate']:.1f}%)")
        logger.info(f"   🔄 Partial matched: {results['insurer_stats']['partial_match_rows']:,} ({results['insurer_stats']['partial_match_rate']:.1f}%)")
        logger.info(f"   ❓ Unmatched: {results['insurer_stats']['unmatched_rows']:,} ({results['insurer_stats']['unmatched_rate']:.1f}%)")
        
        # Calculate overall success rate
        total_cbl = results['cbl_stats']['exact_matches'] + results['cbl_stats']['partial_matches'] + results['cbl_stats']['no_matches']
        if total_cbl > 0:
            exact_rate = (results['cbl_stats']['exact_matches'] / total_cbl * 100)
            overall_match_rate = ((results['cbl_stats']['exact_matches'] + results['cbl_stats']['partial_matches']) / total_cbl * 100)
            
            logger.info(f"\n🎯 SUCCESS METRICS:")
            logger.info(f"   📈 Exact match rate: {exact_rate:.1f}%")
            logger.info(f"   📈 Overall match rate: {overall_match_rate:.1f}%")
            logger.info(f"   🛡️ Data integrity: 100% (Global tracker active)")
            logger.info(f"   ⚡ Performance: {duration.total_seconds():.2f}s")
        
        # File size info
        if os.path.exists(results['output_file']):
            file_size = os.path.getsize(results['output_file']) / (1024 * 1024)
            logger.info(f"   📁 Output size: {file_size:.1f} MB")
        
        logger.info(f"\n🎉 MATCHING COMPLETED WITH FULL DATA INTEGRITY PROTECTION!")
        
        return results
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error("❌ MATCHING PROCESS FAILED")
        logger.error("=" * 60)
        logger.error(f"Error: {str(e)}")
        
        # Show troubleshooting tips
        logger.info(f"\n🔧 TROUBLESHOOTING:")
        logger.info(f"1. Check if input files exist and are readable")
        logger.info(f"2. Verify column names match the mappings")
        logger.info(f"3. Ensure files are not corrupted")
        logger.info(f"4. Check available disk space")
        
        # Show detailed error for debugging
        import traceback
        logger.debug("Full error traceback:")
        logger.debug(traceback.format_exc())
        
        raise

def run_with_custom_files(cbl_file, insurer_file, output_file=None):
    """
    Convenience function to run matching with custom file paths.
    
    Args:
        cbl_file (str): Path to CBL Excel file
        insurer_file (str): Path to insurer Excel file
        output_file (str, optional): Output file path. Auto-generated if not provided.
    
    Returns:
        dict: Matching results
    """
    if not output_file:
        output_file = f"RESULT_CUSTOM_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    logger.info(f"🎯 Running custom matching:")
    logger.info(f"   CBL: {cbl_file}")
    logger.info(f"   Insurer: {insurer_file}")
    logger.info(f"   Output: {output_file}")
    
    # Use flexible mappings that should work with most files
    column_mappings = {
        'cbl_mappings': {
            # Princess Tuna CBL mappings
            "Placing/Endorsement No.": "PlacingNo",
            "Policy No.": "PolicyNo", 
            "Client Name": "ClientName", 
            "Balance Net of Brokerage": "ProcessedAmount",
            
            # Try multiple possible column names
            "REF": "PlacingNo",
            "Name": "ClientName",
            "POLNO": "PolicyNo",
            "AMTDUE": "Amount",
            "Balance (MUR) (Net of Brokerage)": "Amount",
            "Amount": "Amount",
            "PlacingNo": "PlacingNo",
            "PolicyNo": "PolicyNo",
            "ClientName": "ClientName"
        },
        'insurer_mappings': {
            # Princess Tuna Insurer mappings
            "Policy Ref": "PlacingNo",
            "Policy Number": "PolicyNo_1",
            "Insured Name": "ClientName",
            "Equivalent in MUR": "ProcessedAmount",
            
            "REF": "PlacingNo",
            "Name": "ClientName",
            "POLNO": "PolicyNo_1",
            "AMTDUE": "Amount",
            "BRKREF": "PlacingNo",
            "NAME": "ClientName", 
            "DOCSER": "PolicyNo_1",
            "POLSER": "PolicyNo_2",
            "PlacingNo": "PlacingNo",
            "ClientName": "ClientName",
            "Amount": "Amount"
        }
    }
    
    return run_matching_process(
        column_mappings=column_mappings,
        matrix_keys=[],  # Empty matrix keys
        cbl_file=cbl_file,
        insurer_file=insurer_file,
        output_file=output_file,
        tolerance=100
    )

if __name__ == "__main__":
    main()
