#!/usr/bin/env python3
import pandas as pd
import argparse
import logging
import os
from .data_processing import preprocess, initialize_tracking, read_excel_with_smart_headers
from .matching_engine import GlobalMatchTracker
from .match_history import apply_match_history, finalize_history_no_match, finalize_history_dynamic_buckets, finalize_rematch_buckets, generate_fingerprints_for_df
from .pass1 import pass1
from .pass2 import pass2
from .pass3 import pass3
from .output_handler import explode_and_merge
import io

logger = logging.getLogger(__name__)


def run_matching_process(column_mappings, cbl_file=None, insurer_file=None, output_file='output.xlsx', tolerance=50, history_file=None, dynamic_buckets=None):
    """
    Run the matching process between CBL and insurer files.

    Args:
        column_mappings: Dictionary containing column mappings for CBL and insurer data
        cbl_file (bytes, optional): CBL Excel file content as bytes. If None, will be prompted.
        insurer_file (bytes, optional): Insurer Excel file content as bytes. If None, will be prompted.
        output_file (str, optional): Output Excel file name. Defaults to 'output.xlsx'.
        tolerance (int, optional): Tolerance for amount matching. Defaults to 100.
        history_file (str or bytes, optional): Path to history.xlsx or its content as bytes.
            If provided, rows matching previous manual moves are pre-placed before passes.

    Returns:
        dict: Results dictionary containing match statistics and output file content as bytes
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

        # Read Excel files with intelligent header detection and column filtering
        logger.info("Using smart header detection for Excel files...")
        cbl_df = read_excel_with_smart_headers(cbl_file, usecols=lambda x: not str(x).startswith('Unnamed:'))
        insurer_df = read_excel_with_smart_headers(insurer_file, usecols=lambda x: not str(x).startswith('Unnamed:'))

        logger.info(f"After column filtering - CBL rows: {len(cbl_df)}, Insurer rows: {len(insurer_df)}")

        # Generate output in memory (no local file saving)
        output_filename = output_file

        # Process the data
        clean_cbl, clean_insurer = preprocess(cbl_df, insurer_df, column_mappings)
        clean_cbl = initialize_tracking(clean_cbl)

        logger.info(f"After preprocessing - CBL rows: {len(clean_cbl)}, Insurer rows: {len(clean_insurer)}")

        # ── Generate Canonical Fingerprints ───────────────────────────────
        # These fingerprints are the single source of truth for match history.
        # They are generated from preprocessed data and written to output.xlsx
        # so the frontend can read them directly instead of regenerating.
        clean_cbl["_fingerprint"] = generate_fingerprints_for_df(clean_cbl)

        # For insurer, strip _INSURER suffix before fingerprinting (same as backend replay)
        insurer_cols_stripped = {
            col: col[:-len("_INSURER")] if col.endswith("_INSURER") else col
            for col in clean_insurer.columns
        }
        fp_insurer_source = clean_insurer.rename(columns=insurer_cols_stripped)
        clean_insurer["_fingerprint_INSURER"] = generate_fingerprints_for_df(fp_insurer_source)

        logger.info(f"Generated canonical fingerprints: {len(clean_cbl)} CBL, {len(clean_insurer)} insurer")

        # Initialize comprehensive global match tracker for consistent behavior across all passes
        global_tracker = GlobalMatchTracker()

        # ── Match History Layer ──────────────────────────────────────────
        # Pre-place rows based on previous manual user moves (before any passes).
        # Rows that match history fingerprints are placed directly into their
        # target buckets; the remaining rows proceed through normal comparison.
        # Rows in rematchable buckets are stashed and left unmatched so passes
        # can try to match them against new data first.
        rematch_stash = []
        insurer_only_placements = {}
        if history_file is not None:
            logger.info(f"[HISTORY] History file provided — running match history layer")
            clean_cbl, clean_insurer, history_summary, rematch_stash, insurer_only_placements = apply_match_history(
                clean_cbl, clean_insurer, history_file, global_tracker, dynamic_buckets
            )
            logger.info(f"[HISTORY] After History Layer: {global_tracker.get_usage_summary()}")
            if rematch_stash:
                logger.info(f"[HISTORY] {sum(len(s['cbl_indices']) for s in rematch_stash)} CBL rows stashed for re-matching")
            if insurer_only_placements:
                logger.info(f"[HISTORY] {sum(len(v) for v in insurer_only_placements.values())} insurer-only rows placed into dynamic buckets")
        else:
            logger.info("[HISTORY] No history file provided — skipping match history layer")

        # ── Matching Passes ──────────────────────────────────────────────
        # Helper function to check if required keys exist in column mappings
        def has_required_keys(cbl_required, insurer_required):
            cbl_mappings = column_mappings.get('cbl_mappings', {})
            insurer_mappings = column_mappings.get('insurer_mappings', {})
            cbl_has_keys = all(any(target == key for target in cbl_mappings.values()) for key in cbl_required)
            insurer_has_keys = all(any(target == key for target in insurer_mappings.values()) for key in insurer_required)
            return cbl_has_keys and insurer_has_keys

        # Pass 1: Placing Number + Amount
        if has_required_keys(['PlacingNo', 'ProcessedAmount'], ['PlacingNo', 'ProcessedAmount']):
            logger.info("Pass 1: Running Placing Number + Amount matching")
            clean_cbl = pass1(clean_cbl, clean_insurer, tolerance, global_tracker)
        else:
            logger.info("Pass 1: Skipped — required keys (PlacingNo, ProcessedAmount) not in mappings")
        logger.info(f"After Pass 1: {global_tracker.get_usage_summary()}")

        # Pass 2: Policy Number + Amount (currently disabled)
        logger.info("Pass 2: Disabled — all unmatched records proceed to Pass 3")
        cbl_has_pass2 = has_required_keys(['PolicyNo', 'ProcessedAmount'], [])
        insurer_has_pass2_base = has_required_keys([], ['ProcessedAmount'])
        insurer_mappings = column_mappings.get('insurer_mappings', {})
        has_policy1 = any(target == 'PolicyNo_1' for target in insurer_mappings.values())
        has_policy2 = any(target == 'PolicyNo_2' for target in insurer_mappings.values())
        insurer_has_policy = has_policy1 or has_policy2

        if cbl_has_pass2 and insurer_has_pass2_base and insurer_has_policy:
            logger.info("Pass 2: Required keys found — running Policy Number + Amount matching")
            clean_cbl = pass2(clean_cbl, clean_insurer, tolerance, global_tracker)
        else:
            missing_keys = []
            if not cbl_has_pass2:
                missing_keys.append("CBL: PolicyNo, ProcessedAmount")
            if not insurer_has_pass2_base:
                missing_keys.append("Insurer: ProcessedAmount")
            if not insurer_has_policy:
                missing_keys.append("Insurer: PolicyNo_1 or PolicyNo_2")
            logger.info(f"Pass 2: Skipped — missing: {'; '.join(missing_keys)}")
        logger.info(f"After Pass 2: {global_tracker.get_usage_summary()}")

        # Pass 3: Intelligent Name Matching (Corporate Root + Fuzzy Clustering)
        if has_required_keys(['ClientName', 'ProcessedAmount'], ['ClientName', 'ProcessedAmount']):
            logger.info("Pass 3: Running Intelligent Name Matching")
            clean_cbl = pass3(clean_cbl, clean_insurer, tolerance, 95, global_tracker)
        else:
            logger.info("Pass 3: Skipped — required keys (ClientName, ProcessedAmount) not in mappings")

        final_summary = global_tracker.get_usage_summary()
        logger.info(f"Final Global Tracker: {final_summary}")
        logger.info(f"Total unique insurer rows used: {final_summary['total_unique_insurer_used']}/{len(clean_insurer)} ({final_summary['total_unique_insurer_used']/len(clean_insurer)*100:.1f}%)")

        # ── Finalize History No-Match ─────────────────────────────────
        # Convert sentinel _History_No_Match status back to "No Match"
        # now that all passes are done and won't touch these rows.
        if history_file is not None:
            clean_cbl = finalize_history_no_match(clean_cbl)
            clean_cbl = finalize_rematch_buckets(clean_cbl, rematch_stash)
            clean_cbl = finalize_history_dynamic_buckets(clean_cbl)

        # ── Assign group_id to all matched rows ─────────────────────
        # Individual 1:1 matches from pass1 have no group_id.
        # Assign one so every matched row carries a group_id for
        # frontend selection/highlighting.
        if 'group_id' in clean_cbl.columns:
            matched_mask = clean_cbl['match_status'].isin(['Exact Match', 'Partial Match'])
            no_group_mask = matched_mask & (clean_cbl['group_id'].isna() | (clean_cbl['group_id'] == ''))
            if dynamic_buckets:
                dynamic_keys = {b['BucketKey'] for b in dynamic_buckets}
                dynamic_mask = clean_cbl['match_status'].isin(dynamic_keys)
                no_group_mask = no_group_mask | (dynamic_mask & (clean_cbl['group_id'].isna() | (clean_cbl['group_id'] == '')))
            individual_counter = 0
            for idx in clean_cbl[no_group_mask].index:
                individual_counter += 1
                clean_cbl.at[idx, 'group_id'] = f"MATCH_{individual_counter}"
            if individual_counter > 0:
                logger.info(f"Assigned group_id to {individual_counter} individual matched rows without one")

        # Sort by group_id to keep grouped rows together in output
        if 'group_id' in clean_cbl.columns:
            logger.info("📋 Sorting data to group matched records together...")
            clean_cbl['_group_sort_key'] = clean_cbl['group_id'].apply(
                lambda x: (0, x) if pd.notna(x) and x is not None else (1, '')
            )
            clean_cbl = clean_cbl.sort_values('_group_sort_key').drop('_group_sort_key', axis=1)
            clean_cbl = clean_cbl.reset_index(drop=True)

        # Generate output and statistics
        return _generate_output_and_statistics(clean_cbl, clean_insurer, output_filename, dynamic_buckets, insurer_only_placements)

    except Exception as e:
        logger.error(f"\nError: {str(e)}")
        raise


def _generate_output_and_statistics(clean_cbl, clean_insurer, output_filename, dynamic_buckets=None, insurer_only_placements=None):
    """Generate output files and calculate statistics."""

    dynamic_bucket_keys = {bucket["BucketKey"] for bucket in (dynamic_buckets or [])}
    insurer_only_placements = insurer_only_placements or {}

    def _collect_insurer_indices_for_statuses(statuses):
        insurer_indices = set()
        matched_rows = clean_cbl[clean_cbl["match_status"].isin(statuses)]
        for indices in matched_rows["matched_insurer_indices"]:
            if isinstance(indices, list):
                insurer_indices.update(indices)
            elif pd.notna(indices):
                insurer_indices.add(indices)
        return insurer_indices

    # Track insurer indices by match type.
    # Dynamic bucket rows are placed rows too, so their insurer links must not
    # fall through into the No Matches Insurer sheet.
    exact_match_insurer_indices = _collect_insurer_indices_for_statuses({"Exact Match"})
    partial_match_insurer_indices = _collect_insurer_indices_for_statuses({"Partial Match"})
    dynamic_bucket_insurer_indices = _collect_insurer_indices_for_statuses(dynamic_bucket_keys)

    # Include insurer-only placements (insurer rows moved to dynamic buckets with no CBL counterpart)
    all_insurer_only = set()
    for indices in insurer_only_placements.values():
        all_insurer_only.update(indices)
    dynamic_bucket_insurer_indices = dynamic_bucket_insurer_indices | all_insurer_only

    # Remove exact matches from partial matches to avoid double counting
    partial_match_insurer_indices = partial_match_insurer_indices - exact_match_insurer_indices
    dynamic_bucket_insurer_indices = dynamic_bucket_insurer_indices - exact_match_insurer_indices - partial_match_insurer_indices

    # Calculate unmatched insurer indices BEFORE resetting index
    all_insurer_indices = set(clean_insurer.index)
    matched_insurer_indices = exact_match_insurer_indices | partial_match_insurer_indices | dynamic_bucket_insurer_indices
    unmatched_insurer_indices = all_insurer_indices - matched_insurer_indices

    # Calculate statistics BEFORE resetting index
    total_insurer_rows = len(clean_insurer)
    exact_match_insurer_count = len(exact_match_insurer_indices)
    partial_match_insurer_count = len(partial_match_insurer_indices)
    dynamic_bucket_insurer_count = len(dynamic_bucket_insurer_indices)
    unmatched_insurer_count = len(unmatched_insurer_indices)
    
    logger.info(f"DEBUG: Statistics BEFORE reset_index:")
    logger.info(f"  - Total insurer rows: {total_insurer_rows}")
    logger.info(f"  - Exact match insurer indices: {exact_match_insurer_count}")
    logger.info(f"  - Partial match insurer indices: {partial_match_insurer_count}")
    logger.info(f"  - Dynamic bucket insurer indices: {dynamic_bucket_insurer_count}")
    logger.info(f"  - Unmatched insurer indices: {unmatched_insurer_count}")
    logger.info(f"  - Sum of all categories: {exact_match_insurer_count + partial_match_insurer_count + dynamic_bucket_insurer_count + unmatched_insurer_count}")

    # Split clean_cbls by individual match_status
    logger.info("\n=== Splitting CBL records by match status ===")
    
    exact_matches = clean_cbl[clean_cbl["match_status"] == "Exact Match"].copy()
    partial_matches = clean_cbl[clean_cbl["match_status"] == "Partial Match"].copy()
    no_matches = clean_cbl[clean_cbl["match_status"] == "No Match"].copy()

    # Extract dynamic bucket rows (match_status == BucketKey after finalization)
    dynamic_bucket_dfs = {}
    if dynamic_buckets:
        for bucket in dynamic_buckets:
            key = bucket["BucketKey"]
            bucket_rows = clean_cbl[clean_cbl["match_status"] == key].copy()
            dynamic_bucket_dfs[key] = bucket_rows
            if not bucket_rows.empty:
                logger.info(f"  - Dynamic bucket '{key}': {len(bucket_rows)}")

    logger.info(f"✓ Split complete:")
    logger.info(f"  - Exact matches: {len(exact_matches)}")
    logger.info(f"  - Partial matches: {len(partial_matches)}")
    logger.info(f"  - No matches: {len(no_matches)}")

    # Explode matched_insurer_indices and merge with insurer
    exact_matches = explode_and_merge(exact_matches, clean_insurer)

    # Before exploding partials, ensure we don't carry any insurer indices that are exact-matched elsewhere
    if not partial_matches.empty:
        def _filter_partial_indices(row):
            indices = row.get("matched_insurer_indices", [])
            if not isinstance(indices, list):
                indices = [indices]
            # remove any indices that are already placed in exact or dynamic buckets
            filtered = [idx for idx in indices if idx not in exact_match_insurer_indices and idx not in dynamic_bucket_insurer_indices]
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
                logger.info(f"Fixed row {idx}: {partial_matches.at[idx, 'match_reason']}")
            
            # Move these rows from partial_matches to no_matches
            fixed_rows = partial_matches[partial_matches["match_status"] == "No Match"].copy()
            partial_matches = partial_matches[partial_matches["match_status"] == "Partial Match"].copy()
            no_matches = pd.concat([no_matches, fixed_rows], ignore_index=False)

    partial_matches = explode_and_merge(partial_matches, clean_insurer)

    # Create unmatched insurer records with robust index handling
    try:
        # Get current DataFrame indices
        current_insurer_indices = set(clean_insurer.index)
        
        # Recalculate matched indices based on current DataFrame state
        current_exact_match_indices = _collect_insurer_indices_for_statuses({"Exact Match"})
        current_partial_match_indices = _collect_insurer_indices_for_statuses({"Partial Match"})
        current_dynamic_bucket_indices = _collect_insurer_indices_for_statuses(dynamic_bucket_keys)

        # Include insurer-only placements
        for indices in insurer_only_placements.values():
            current_dynamic_bucket_indices.update(indices)

        # Remove exact matches from partial matches to avoid double counting
        current_partial_match_indices = current_partial_match_indices - current_exact_match_indices
        current_dynamic_bucket_indices = (
            current_dynamic_bucket_indices
            - current_exact_match_indices
            - current_partial_match_indices
        )

        # Calculate current matched and unmatched indices
        current_matched_indices = current_exact_match_indices | current_partial_match_indices | current_dynamic_bucket_indices
        current_unmatched_indices = current_insurer_indices - current_matched_indices
        
        logger.info(f"Index reconciliation:")
        logger.info(f"  - Original unmatched indices: {len(unmatched_insurer_indices)}")
        logger.info(f"  - Current DataFrame indices: {len(current_insurer_indices)}")
        logger.info(f"  - Current exact match indices: {len(current_exact_match_indices)}")
        logger.info(f"  - Current partial match indices: {len(current_partial_match_indices)}")
        logger.info(f"  - Current dynamic bucket indices: {len(current_dynamic_bucket_indices)}")
        logger.info(f"  - Current matched indices: {len(current_matched_indices)}")
        logger.info(f"  - Current unmatched indices: {len(current_unmatched_indices)}")
        
        if current_unmatched_indices:
            unmatched_insurer = clean_insurer.loc[list(current_unmatched_indices)].copy()
        else:
            unmatched_insurer = pd.DataFrame()
            logger.info("No unmatched insurer records found")
            
    except Exception as e:
        logger.error(f"Error accessing unmatched insurer indices: {str(e)}")
        logger.info(f"clean_insurer shape: {clean_insurer.shape}")
        logger.info(f"clean_insurer index range: {clean_insurer.index.min()} to {clean_insurer.index.max()}")
        unmatched_insurer = pd.DataFrame()

    # Update clean_cbl to reflect the post-processing fixes
    # This ensures summary statistics are calculated correctly
    if 'should_be_no_match' in locals() and not should_be_no_match.empty:
        for idx in should_be_no_match.index:
            clean_cbl.at[idx, "match_status"] = "No Match"
    
    # Sort No Match records by client name for easier review
    logger.info("\n=== Sorting No Match records by client name ===")
    if not no_matches.empty and 'ClientName' in no_matches.columns:
        no_matches = no_matches.sort_values('ClientName', ascending=True).reset_index(drop=True)
        logger.info(f"✓ Sorted {len(no_matches)} No Match CBL records by ClientName")
    
    if not unmatched_insurer.empty and 'ClientName_INSURER' in unmatched_insurer.columns:
        unmatched_insurer = unmatched_insurer.sort_values('ClientName_INSURER', ascending=True).reset_index(drop=True)
        logger.info(f"✓ Sorted {len(unmatched_insurer)} No Match Insurer records by ClientName_INSURER")

    # Write to Excel in memory with combined sheets only
    try:
        output_buffer = io.BytesIO()
        with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
            # Fixed sheets: CBL + Insurer data merged
            exact_matches.to_excel(writer, sheet_name="Exact Matches", index=False)
            partial_matches.to_excel(writer, sheet_name="Partial Matches", index=False)
            no_matches.to_excel(writer, sheet_name="No Matches CBL", index=False)
            unmatched_insurer.to_excel(writer, sheet_name="No Matches Insurer", index=False)

            # Dynamic bucket sheets
            if dynamic_buckets:
                for bucket in dynamic_buckets:
                    key = bucket["BucketKey"]
                    bucket_rows = dynamic_bucket_dfs.get(key, pd.DataFrame())
                    insurer_only_indices = insurer_only_placements.get(key, set())

                    parts = []
                    if not bucket_rows.empty:
                        parts.append(explode_and_merge(bucket_rows, clean_insurer))
                    if insurer_only_indices:
                        valid_indices = [i for i in insurer_only_indices if i in clean_insurer.index]
                        if valid_indices:
                            parts.append(clean_insurer.loc[valid_indices].copy())
                            logger.info(f"[HISTORY] Added {len(valid_indices)} insurer-only rows to bucket sheet '{key}'")

                    if parts:
                        combined = pd.concat(parts, ignore_index=True)
                        combined.to_excel(writer, sheet_name=key, index=False)
                    else:
                        pd.DataFrame(columns=exact_matches.columns).to_excel(
                            writer, sheet_name=key, index=False
                        )

                # Metadata sheet so frontend knows which sheets are dynamic buckets
                bucket_config_df = pd.DataFrame(dynamic_buckets)
                bucket_config_df.to_excel(writer, sheet_name="_BucketConfig", index=False)

        # Get the Excel file content as bytes
        output_buffer.seek(0)
        excel_content = output_buffer.getvalue()
        output_buffer.close()

    except Exception as e:
        logger.error(f"Error writing to Excel: {str(e)}")
        raise

    logger.info(f"✓ Results generated in memory: {output_filename}")

    logger.info("\n=== Final Results ===")
    logger.info(f"✓ CBL Records:")
    logger.info(f"  - Total CBL rows: {len(clean_cbl)}")
    logger.info(f"  - Exact matches: {len(clean_cbl[clean_cbl['match_status'] == 'Exact Match'])}")
    logger.info(f"  - Partial matches: {len(clean_cbl[clean_cbl['match_status'] == 'Partial Match'])}")
    for key in dynamic_bucket_keys:
        logger.info(f"  - {key}: {len(clean_cbl[clean_cbl['match_status'] == key])}")
    logger.info(f"  - No matches: {len(clean_cbl[clean_cbl['match_status'] == 'No Match'])}")
    logger.info(f"✓ Insurer Records:")
    logger.info(f"  - Total insurer rows: {total_insurer_rows}")
    logger.info(f"  - Exact match insurer rows: {exact_match_insurer_count} ({exact_match_insurer_count/total_insurer_rows*100:.1f}%)")
    logger.info(f"  - Partial match insurer rows: {partial_match_insurer_count} ({partial_match_insurer_count/total_insurer_rows*100:.1f}%)")
    logger.info(f"  - Dynamic bucket insurer rows: {dynamic_bucket_insurer_count} ({dynamic_bucket_insurer_count/total_insurer_rows*100:.1f}%)")
    logger.info(f"  - Unmatched insurer rows: {unmatched_insurer_count} ({unmatched_insurer_count/total_insurer_rows*100:.1f}%)")
    logger.info(f"✓ Results generated in memory: {output_filename}")

   
    # Calculate amounts for different match types (ensure numeric conversion)
    cbl_exact_amount = pd.to_numeric(clean_cbl[clean_cbl['match_status'] == 'Exact Match']['ProcessedAmount'], errors='coerce').sum()
    cbl_partial_amount = pd.to_numeric(clean_cbl[clean_cbl['match_status'] == 'Partial Match']['ProcessedAmount'], errors='coerce').sum()
    cbl_no_match_amount = pd.to_numeric(clean_cbl[clean_cbl['match_status'] == 'No Match']['ProcessedAmount'], errors='coerce').sum()
    
    # Calculate insurer amounts (ensure numeric conversion)
    try:
        exact_match_insurer_amount = pd.to_numeric(clean_insurer.loc[list(exact_match_insurer_indices), 'ProcessedAmount_INSURER'], errors='coerce').sum()
    except Exception as e:
        logger.error(f"Error calculating exact match insurer amount: {str(e)}")
        exact_match_insurer_amount = 0
    
    try:
        partial_match_insurer_amount = pd.to_numeric(clean_insurer.loc[list(partial_match_insurer_indices), 'ProcessedAmount_INSURER'], errors='coerce').sum()
    except Exception as e:
        logger.error(f"Error calculating partial match insurer amount: {str(e)}")
        partial_match_insurer_amount = 0

    try:
        dynamic_bucket_insurer_amount = pd.to_numeric(clean_insurer.loc[list(dynamic_bucket_insurer_indices), 'ProcessedAmount_INSURER'], errors='coerce').sum()
    except Exception as e:
        logger.error(f"Error calculating dynamic bucket insurer amount: {str(e)}")
        dynamic_bucket_insurer_amount = 0
    
    try:
        # Use the current unmatched indices instead of the original ones
        if 'current_unmatched_indices' in locals() and current_unmatched_indices:
            unmatched_insurer_amount = pd.to_numeric(clean_insurer.loc[list(current_unmatched_indices)]['ProcessedAmount_INSURER'], errors='coerce').sum()
        else:
            # Fallback to using the unmatched_insurer DataFrame if available
            if 'unmatched_insurer' in locals() and not unmatched_insurer.empty and 'ProcessedAmount_INSURER' in unmatched_insurer.columns:
                unmatched_insurer_amount = pd.to_numeric(unmatched_insurer['ProcessedAmount_INSURER'], errors='coerce').sum()
            else:
                unmatched_insurer_amount = 0
    except Exception as e:
        logger.error(f"Error calculating unmatched insurer amount: {str(e)}")
        unmatched_insurer_amount = 0
    
    return {
        'exact_matches': exact_matches,
        'partial_matches': partial_matches,
        'no_matches': no_matches,
        'unmatched_insurer': unmatched_insurer,
        'output_file': output_filename,
        'output_content': excel_content,
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
            'dynamic_bucket_rows': dynamic_bucket_insurer_count,
            'unmatched_rows': unmatched_insurer_count,
            'exact_match_rate': exact_match_insurer_count/total_insurer_rows*100,
            'partial_match_rate': partial_match_insurer_count/total_insurer_rows*100,
            'dynamic_bucket_rate': dynamic_bucket_insurer_count/total_insurer_rows*100,
            'unmatched_rate': unmatched_insurer_count/total_insurer_rows*100,
            'exact_match_amount': exact_match_insurer_amount,
            'partial_match_amount': partial_match_insurer_amount,
            'dynamic_bucket_amount': dynamic_bucket_insurer_amount,
            'unmatched_amount': unmatched_insurer_amount
        },
        'dynamic_bucket_stats': {
            bucket["BucketKey"]: len(dynamic_bucket_dfs.get(bucket["BucketKey"], pd.DataFrame()))
            for bucket in (dynamic_buckets or [])
        }
    }
