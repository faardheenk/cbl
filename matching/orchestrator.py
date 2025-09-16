#!/usr/bin/env python3

import pandas as pd
import argparse
import logging
import os
from matrix import matrix_pass
from .data_processing import preprocess, initialize_tracking
from .matching_engine import pass1, pass2, pass3, GlobalMatchTracker
from .output_handler import explode_and_merge

logger = logging.getLogger(__name__)


def run_matching_process(column_mappings, matrix_keys, cbl_file=None, insurer_file=None, output_file='output.xlsx', tolerance=100):
    """
    Run the matching process between CBL and insurer files.
    
    Args:
        column_mappings: Dictionary containing column mappings for CBL and insurer data
        matrix_keys: Dictionary containing matrix key configurations
        cbl_file (str, optional): Path to the CBL Excel file. If None, will be prompted.
        insurer_file (str, optional): Path to the insurer Excel file. If None, will be prompted.
        output_file (str, optional): Output Excel file name. Defaults to 'output.xlsx'.
        tolerance (int, optional): Tolerance for amount matching. Defaults to 100.
        
    Returns:
        dict: Results dictionary containing match statistics and output information
    """
    logger.info("\n=== Starting Matching Process ===")
    
    if cbl_file is None or insurer_file is None:
        parser = argparse.ArgumentParser(description='Match data between two Excel files.')
        parser.add_argument('cbl_file', help='Path to the CBL Excel file')
        parser.add_argument('insurer_file', help='Path to the insurer Excel file')
        parser.add_argument('--output', '-o', default=output_file, help='Output Excel file name')
        args = parser.parse_args()
        cbl_file = args.cbl_file
        insurer_file = args.insurer_file
        output_file = args.output

    try:
        logger.info(f"Reading files: {cbl_file} and {insurer_file}")
        
        # Debug: Read raw files first to check original row counts
        cbl_raw = pd.read_excel(cbl_file)
        insurer_raw = pd.read_excel(insurer_file)
        logger.info(f"DEBUG: Raw CBL rows: {len(cbl_raw)}, Raw insurer rows: {len(insurer_raw)}")
        
        # Read Excel files with column filtering
        cbl_df = pd.read_excel(cbl_file, usecols=lambda x: not x.startswith('Unnamed:'))
        insurer_df = pd.read_excel(insurer_file, usecols=lambda x: not x.startswith('Unnamed:'))
        
        logger.info(f"DEBUG: After column filtering - CBL rows: {len(cbl_df)}, Insurer rows: {len(insurer_df)}")

        # Get the directory of the input files
        input_dir = os.path.dirname(os.path.abspath(cbl_file))
        output_path = os.path.join(input_dir, output_file)

        # Process the data
        clean_cbl, clean_insurer = preprocess(cbl_df, insurer_df, column_mappings, matrix_keys)
        clean_cbl = initialize_tracking(clean_cbl)
        
        logger.info(f"DEBUG: After preprocessing - CBL rows: {len(clean_cbl)}, Insurer rows: {len(clean_insurer)}")

        # Initialize comprehensive global match tracker for consistent behavior across all passes
        global_tracker = GlobalMatchTracker()
        logger.info("🔧 Initialized GlobalMatchTracker for comprehensive CBL-insurer match tracking")
        
        # Run matrix pass with global tracker integration
        clean_cbl, clean_insurer, matched_insurer_indices = matrix_pass(clean_cbl, clean_insurer, matrix_keys, global_tracker)
        
        # Log matrix pass results (global tracker is already updated by matrix_pass)
        if matched_insurer_indices:
            logger.info(f"🎯 Matrix pass matched {len(matched_insurer_indices)} insurer rows with full global tracking")
        else:
            logger.info("🎯 Matrix pass completed - no matches found")

        # Run additional passes only on unmatched records
        if len(matched_insurer_indices) < len(clean_insurer):
            logger.info(f"🚀 Running additional passes with global tracking")
            
            # Helper function to check if required keys exist in column mappings
            def has_required_keys(cbl_required, insurer_required):
                cbl_mappings = column_mappings.get('cbl_mappings', {})
                insurer_mappings = column_mappings.get('insurer_mappings', {})
                
                # Check if all required CBL keys are mapped
                cbl_has_keys = all(any(target == key for target in cbl_mappings.values()) for key in cbl_required)
                
                # Check if all required insurer keys are mapped
                insurer_has_keys = all(any(target == key for target in insurer_mappings.values()) for key in insurer_required)
                
                return cbl_has_keys and insurer_has_keys
            
            # Pass 1: Requires PlacingNo and Amount
            if has_required_keys(['PlacingNo', 'Amount'], ['PlacingNo', 'Amount']):
                logger.info("✓ Pass 1: Required keys found in mappings - running Pass 1 with global tracking")
                clean_cbl = pass1(clean_cbl, clean_insurer, tolerance, global_tracker)
            else:
                logger.info("⚠ Pass 1: Required keys (PlacingNo, Amount) not found in mappings - skipping Pass 1")
            
            # Log global tracker status after Pass 1
            if global_tracker:
                logger.info(f"📊 After Pass 1: {global_tracker.get_usage_summary()}")
            
            # Pass 2: Requires PolicyNo, ClientName, and Amount (CBL) + ClientName, Amount, and at least one PolicyNo (Insurer)
            cbl_has_pass2 = has_required_keys(['PolicyNo', 'ClientName', 'Amount'], [])
            insurer_has_pass2_base = has_required_keys([], ['ClientName', 'Amount'])
            
            # Check if insurer has at least one policy number column mapped
            insurer_mappings = column_mappings.get('insurer_mappings', {})
            has_policy1 = any(target == 'PolicyNo_1' for target in insurer_mappings.values())
            has_policy2 = any(target == 'PolicyNo_2' for target in insurer_mappings.values()) 
            insurer_has_policy = has_policy1 or has_policy2
            
            if cbl_has_pass2 and insurer_has_pass2_base and insurer_has_policy:
                logger.info("✓ Pass 2: Required keys found in mappings - running Pass 2 with global tracking")
                clean_cbl = pass2(clean_cbl, clean_insurer, tolerance, 95, global_tracker)
            else:
                missing_keys = []
                if not cbl_has_pass2:
                    missing_keys.append("CBL: PolicyNo, ClientName, Amount")
                if not insurer_has_pass2_base:
                    missing_keys.append("Insurer: ClientName, Amount")
                if not insurer_has_policy:
                    missing_keys.append("Insurer: PolicyNo_1 or PolicyNo_2")
                logger.info(f"⚠ Pass 2: Required keys not found in mappings - skipping Pass 2. Missing: {'; '.join(missing_keys)}")
            
            # Log global tracker status after Pass 2
            if global_tracker:
                logger.info(f"📊 After Pass 2: {global_tracker.get_usage_summary()}")
            
            # Pass 3: Requires ClientName and Amount
            if has_required_keys(['ClientName', 'Amount'], ['ClientName', 'Amount']):
                logger.info("✓ Pass 3: Required keys found in mappings - running Pass 3 with global tracking")
                clean_cbl = pass3(clean_cbl, clean_insurer, tolerance, 95, global_tracker)
            else:
                logger.info("⚠ Pass 3: Required keys (ClientName, Amount) not found in mappings - skipping Pass 3")
            
            # Log final global tracker status
            if global_tracker:
                final_summary = global_tracker.get_usage_summary()
                logger.info(f"🎯 Final Global Tracker Summary: {final_summary}")
                logger.info(f"✅ Total unique insurer rows used: {final_summary['total_unique_insurer_used']}/{len(clean_insurer)} ({final_summary['total_unique_insurer_used']/len(clean_insurer)*100:.1f}%)")
                
        else:
            print("All records matched via matrix keys - skipping additional passes")

        # Generate output and statistics
        return _generate_output_and_statistics(clean_cbl, clean_insurer, output_path)

    except Exception as e:
        logger.error(f"\nError: {str(e)}")
        raise


def _generate_output_and_statistics(clean_cbl, clean_insurer, output_path):
    """Generate output files and calculate statistics."""
    # Track insurer indices by match type
    exact_match_insurer_indices = set()
    partial_match_insurer_indices = set()

    # Collect insurer indices from exact matches
    for indices in clean_cbl[clean_cbl["match_status"] == "Exact Match"]["matched_insurer_indices"]:
        if isinstance(indices, list):
            exact_match_insurer_indices.update(indices)
        elif pd.notna(indices):
            exact_match_insurer_indices.add(indices)
    
    # Collect insurer indices from partial matches
    for indices in clean_cbl[clean_cbl["match_status"] == "Partial Match"]["matched_insurer_indices"]:
        if isinstance(indices, list):
            partial_match_insurer_indices.update(indices)
        elif pd.notna(indices):
            partial_match_insurer_indices.add(indices)

    # Remove exact matches from partial matches to avoid double counting
    partial_match_insurer_indices = partial_match_insurer_indices - exact_match_insurer_indices

    # Calculate unmatched insurer indices BEFORE resetting index
    all_insurer_indices = set(clean_insurer.index)
    matched_insurer_indices = exact_match_insurer_indices | partial_match_insurer_indices
    unmatched_insurer_indices = all_insurer_indices - matched_insurer_indices

    # Calculate statistics BEFORE resetting index
    total_insurer_rows = len(clean_insurer)
    exact_match_insurer_count = len(exact_match_insurer_indices)
    partial_match_insurer_count = len(partial_match_insurer_indices)
    unmatched_insurer_count = len(unmatched_insurer_indices)
    
    logger.info(f"DEBUG: Statistics BEFORE reset_index:")
    logger.info(f"  - Total insurer rows: {total_insurer_rows}")
    logger.info(f"  - Exact match insurer indices: {exact_match_insurer_count}")
    logger.info(f"  - Partial match insurer indices: {partial_match_insurer_count}")
    logger.info(f"  - Unmatched insurer indices: {unmatched_insurer_count}")
    logger.info(f"  - Sum of all categories: {exact_match_insurer_count + partial_match_insurer_count + unmatched_insurer_count}")

    # Split clean_cbls
    exact_matches = clean_cbl[clean_cbl["match_status"] == "Exact Match"].copy() 
    partial_matches = clean_cbl[clean_cbl["match_status"] == "Partial Match"].copy()
    no_matches = clean_cbl[clean_cbl["match_status"] == "No Match"].copy()

    logger.info(f"DEBUG: Exact matches: {len(exact_matches)}")
    logger.info(f"DEBUG: Partial matches: {len(partial_matches)}")
    logger.info(f"DEBUG: No matches: {len(no_matches)}")

    # Explode matched_insurer_indices and merge with insurer
    exact_matches = explode_and_merge(exact_matches, clean_insurer)

    # Before exploding partials, ensure we don't carry any insurer indices that are exact-matched elsewhere
    if not partial_matches.empty:
        def _filter_partial_indices(row):
            indices = row.get("matched_insurer_indices", [])
            if not isinstance(indices, list):
                indices = [indices]
            # remove any indices that are in exact_match_insurer_indices
            filtered = [idx for idx in indices if idx not in exact_match_insurer_indices]
            return filtered

        partial_matches = partial_matches.copy()
        partial_matches["matched_insurer_indices"] = partial_matches.apply(_filter_partial_indices, axis=1)
        
        # Fix: Check for rows with no insurer data and mark them as No Match
        def _has_insurer_data(row):
            insurer_indices = row.get("matched_insurer_indices", [])
            partial_candidates = row.get("partial_candidates_indices", [])
            
            # Check if insurer_indices is not empty
            has_insurer = (
                isinstance(insurer_indices, list) and len(insurer_indices) > 0
            ) or (
                not isinstance(insurer_indices, list) and pd.notna(insurer_indices)
            )
            
            # Check if partial_candidates is not empty
            has_partial = (
                isinstance(partial_candidates, list) and len(partial_candidates) > 0
            ) or (
                not isinstance(partial_candidates, list) and pd.notna(partial_candidates)
            )
            
            return has_insurer or has_partial
        
        # Identify rows that should be No Match instead of Partial Match
        should_be_no_match = partial_matches[~partial_matches.apply(_has_insurer_data, axis=1)]
        
        if not should_be_no_match.empty:
            logger.info(f"Found {len(should_be_no_match)} partial match rows with no insurer data - marking as No Match")
            for idx in should_be_no_match.index:
                partial_matches.at[idx, "match_status"] = "No Match"
                logger.info(f"Fixed row {idx}: {partial_matches.at[idx, 'MatrixKey']} - {partial_matches.at[idx, 'match_reason']}")
            
            # Move these rows from partial_matches to no_matches
            fixed_rows = partial_matches[partial_matches["match_status"] == "No Match"].copy()
            partial_matches = partial_matches[partial_matches["match_status"] == "Partial Match"].copy()
            no_matches = pd.concat([no_matches, fixed_rows], ignore_index=False)

    partial_matches = explode_and_merge(partial_matches, clean_insurer)

    # Create unmatched insurer records
    unmatched_insurer = clean_insurer.iloc[list(unmatched_insurer_indices)].copy()

    # Update clean_cbl to reflect the post-processing fixes
    # This ensures summary statistics are calculated correctly
    if 'should_be_no_match' in locals() and not should_be_no_match.empty:
        for idx in should_be_no_match.index:
            clean_cbl.at[idx, "match_status"] = "No Match"
    
    # Create separate CBL sheets (without insurer data merged)
    exact_matches_cbl_only = clean_cbl[clean_cbl["match_status"] == "Exact Match"].copy()
    partial_matches_cbl_only = clean_cbl[clean_cbl["match_status"] == "Partial Match"].copy()
    no_matches_cbl_only = clean_cbl[clean_cbl["match_status"] == "No Match"].copy()

    # Create separate insurer sheets
    exact_match_insurer_only = clean_insurer.iloc[list(exact_match_insurer_indices)].copy()
    partial_match_insurer_only = clean_insurer.iloc[list(partial_match_insurer_indices)].copy()
    not_found_insurer_only = clean_insurer.iloc[list(unmatched_insurer_indices)].copy()

    # Write to Excel with 6 separate sheets
    try: 
        with pd.ExcelWriter(output_path) as writer:
            # CBL sheets (CBL data only)
            exact_matches_cbl_only.to_excel(writer, sheet_name="exact match cbl", index=False)
            partial_matches_cbl_only.to_excel(writer, sheet_name="partial match cbl", index=False)
            no_matches_cbl_only.to_excel(writer, sheet_name="not found cbl", index=False)
            
            # Insurer sheets (insurer data only)
            exact_match_insurer_only.to_excel(writer, sheet_name="exact match insurer", index=False)
            partial_match_insurer_only.to_excel(writer, sheet_name="partial match insurer", index=False)
            not_found_insurer_only.to_excel(writer, sheet_name="not found insurer", index=False)
            
            # Original combined sheets (for reference)
            exact_matches.to_excel(writer, sheet_name="Exact Matches", index=False)
            partial_matches.to_excel(writer, sheet_name="Partial Matches", index=False)
            no_matches.to_excel(writer, sheet_name="No Matches CBL", index=False)
            unmatched_insurer.to_excel(writer, sheet_name="No Matches Insurer", index=False)
    except Exception as e:
        logger.error(f"Error writing to Excel: {str(e)}")
        raise

    logger.info(f"✓ Results saved to: {output_path}")

    logger.info("\n=== Final Results ===")
    logger.info(f"✓ CBL Records:")
    logger.info(f"  - Total CBL rows: {len(clean_cbl)}")
    logger.info(f"  - Exact matches: {len(clean_cbl[clean_cbl['match_status'] == 'Exact Match'])}")
    logger.info(f"  - Partial matches: {len(clean_cbl[clean_cbl['match_status'] == 'Partial Match'])}")
    logger.info(f"  - No matches: {len(clean_cbl[clean_cbl['match_status'] == 'No Match'])}")
    logger.info(f"✓ Insurer Records:")
    logger.info(f"  - Total insurer rows: {total_insurer_rows}")
    logger.info(f"  - Exact match insurer rows: {exact_match_insurer_count} ({exact_match_insurer_count/total_insurer_rows*100:.1f}%)")
    logger.info(f"  - Partial match insurer rows: {partial_match_insurer_count} ({partial_match_insurer_count/total_insurer_rows*100:.1f}%)")
    logger.info(f"  - Unmatched insurer rows: {unmatched_insurer_count} ({unmatched_insurer_count/total_insurer_rows*100:.1f}%)")
    logger.info(f"✓ Results saved to: {output_path}")

   
    # Calculate amounts for different match types (ensure numeric conversion)
    cbl_exact_amount = pd.to_numeric(clean_cbl[clean_cbl['match_status'] == 'Exact Match']['Amount'], errors='coerce').sum()
    cbl_partial_amount = pd.to_numeric(clean_cbl[clean_cbl['match_status'] == 'Partial Match']['Amount'], errors='coerce').sum()
    cbl_no_match_amount = pd.to_numeric(clean_cbl[clean_cbl['match_status'] == 'No Match']['Amount'], errors='coerce').sum()
    
    # Calculate insurer amounts (ensure numeric conversion)
    exact_match_insurer_amount = pd.to_numeric(clean_insurer.iloc[list(exact_match_insurer_indices)]['Amount_INSURER'], errors='coerce').sum()
    partial_match_insurer_amount = pd.to_numeric(clean_insurer.iloc[list(partial_match_insurer_indices)]['Amount_INSURER'], errors='coerce').sum()
    unmatched_insurer_amount = pd.to_numeric(clean_insurer.iloc[list(unmatched_insurer_indices)]['Amount_INSURER'], errors='coerce').sum()
    
    return {
        'exact_matches': exact_matches,
        'partial_matches': partial_matches,
        'no_matches': no_matches,
        'unmatched_insurer': unmatched_insurer,
        'output_file': output_path,
        'cbl_stats': {
            'exact_matches': len(clean_cbl[clean_cbl['match_status'] == 'Exact Match']),
            'partial_matches': len(clean_cbl[clean_cbl['match_status'] == 'Partial Match']),
            'no_matches': len(clean_cbl[clean_cbl['match_status'] == 'No Match']),
            'exact_match_amount': cbl_exact_amount,
            'partial_match_amount': cbl_partial_amount,
            'no_match_amount': cbl_no_match_amount
        },
        'insurer_stats': {
            'total_rows': total_insurer_rows,
            'exact_match_rows': exact_match_insurer_count,
            'partial_match_rows': partial_match_insurer_count,
            'unmatched_rows': unmatched_insurer_count,
            'exact_match_rate': exact_match_insurer_count/total_insurer_rows*100,
            'partial_match_rate': partial_match_insurer_count/total_insurer_rows*100,
            'unmatched_rate': unmatched_insurer_count/total_insurer_rows*100,
            'exact_match_amount': exact_match_insurer_amount,
            'partial_match_amount': partial_match_insurer_amount,
            'unmatched_amount': unmatched_insurer_amount
        }
    }
