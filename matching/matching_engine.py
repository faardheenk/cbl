#!/usr/bin/env python3

import pandas as pd
import logging
from itertools import combinations
from fuzzywuzzy import fuzz
from .utils import add_pass, extract_policy_tokens

logger = logging.getLogger(__name__)


class GlobalMatchTracker:
    """
    Comprehensive tracking system to prevent row reuse across all matching passes.
    
    This class ensures data integrity by tracking which CBL and insurer rows have been used
    in different types of matches, preventing duplicate usage and ensuring 1:1 or 1:many
    relationships are properly managed.
    """
    
    def __init__(self):
        # Insurer row tracking
        self.matrix_used_insurer = set()      # Insurer rows used in matrix pass
        self.exact_used_insurer = set()       # Insurer rows used in exact matches
        self.partial_used_insurer = set()     # Insurer rows used in partial matches
        
        # CBL row tracking - prevents multiple CBL rows from claiming same insurer
        self.cbl_exact_matches = {}           # cbl_index -> insurer_indices (exact matches)
        self.cbl_partial_matches = {}         # cbl_index -> insurer_indices (partial matches)
        
        # Reverse mapping: insurer_index -> cbl_indices that claimed it
        self.insurer_to_cbl_exact = {}        # insurer_index -> cbl_index (1:1 mapping for exact)
        self.insurer_to_cbl_partial = {}      # insurer_index -> set(cbl_indices) (1:many allowed for partial)
        
    def mark_matrix_used(self, indices):
        """Mark insurer indices as used in matrix pass."""
        if isinstance(indices, (list, set)):
            self.matrix_used_insurer.update(indices)
        else:
            self.matrix_used_insurer.add(indices)
        logger.debug(f"Matrix used insurer indices: {self.matrix_used_insurer}")
    
    def mark_exact_match(self, cbl_index, insurer_indices, cbl_df=None):
        """
        Mark a CBL-insurer exact match, ensuring no conflicts and updating affected CBL rows.
        
        Args:
            cbl_index: CBL row index
            insurer_indices: List of insurer row indices
            cbl_df: CBL DataFrame to update (optional, for automatic cleanup)
            
        Returns:
            tuple: (success, available_indices, conflicts, affected_cbl_rows)
        """
        indices_set = set(insurer_indices) if isinstance(insurer_indices, (list, set)) else {insurer_indices}
        
        # Check for conflicts with existing exact matches
        conflicts = []
        for insurer_idx in indices_set:
            if insurer_idx in self.insurer_to_cbl_exact:
                existing_cbl = self.insurer_to_cbl_exact[insurer_idx]
                conflicts.append((insurer_idx, existing_cbl))
        
        if conflicts:
            logger.warning(f"CBL {cbl_index}: Exact match conflicts detected: {conflicts}")
            return False, [], conflicts, []
        
        # Track which other CBL rows will be affected
        affected_cbl_rows = set()
        
        # Remove CBL from partial matches if upgrading
        if cbl_index in self.cbl_partial_matches:
            old_partial_indices = self.cbl_partial_matches[cbl_index]
            # Remove from partial tracking
            for idx in old_partial_indices:
                if idx in self.insurer_to_cbl_partial:
                    self.insurer_to_cbl_partial[idx].discard(cbl_index)
                    if not self.insurer_to_cbl_partial[idx]:
                        del self.insurer_to_cbl_partial[idx]
            del self.cbl_partial_matches[cbl_index]
            self.partial_used_insurer -= set(old_partial_indices)
            logger.info(f"CBL {cbl_index}: Upgraded from partial to exact match")
        
        # Find other CBL rows that will lose access to these insurer indices
        for insurer_idx in indices_set:
            if insurer_idx in self.insurer_to_cbl_partial:
                # These CBL rows will lose this insurer index
                affected_cbl_rows.update(self.insurer_to_cbl_partial[insurer_idx])
                
                # Remove this insurer from all partial matches
                for affected_cbl in list(self.insurer_to_cbl_partial[insurer_idx]):
                    if affected_cbl in self.cbl_partial_matches:
                        # Remove the insurer index from this CBL's partial matches
                        current_indices = self.cbl_partial_matches[affected_cbl]
                        if insurer_idx in current_indices:
                            updated_indices = [idx for idx in current_indices if idx != insurer_idx]
                            if updated_indices:
                                self.cbl_partial_matches[affected_cbl] = updated_indices
                            else:
                                # No more partial indices - remove the CBL entirely
                                del self.cbl_partial_matches[affected_cbl]
                            
                            # Update CBL DataFrame if provided
                            if cbl_df is not None and affected_cbl in cbl_df.index:
                                current_df_indices = cbl_df.at[affected_cbl, 'matched_insurer_indices']
                                if isinstance(current_df_indices, list) and insurer_idx in current_df_indices:
                                    updated_df_indices = [idx for idx in current_df_indices if idx != insurer_idx]
                                    cbl_df.at[affected_cbl, 'matched_insurer_indices'] = updated_df_indices
                                    
                                    # Check if CBL row has no more insurer matches
                                    if not updated_df_indices:
                                        # CBL row lost all matches - convert to "No Match"
                                        cbl_df.at[affected_cbl, 'match_status'] = 'No Match'
                                        cbl_df.at[affected_cbl, 'match_reason'] = f"Lost all insurers (insurer {insurer_idx} claimed by CBL {cbl_index})"
                                        cbl_df.at[affected_cbl, 'matched_amtdue_total'] = None
                                        cbl_df.at[affected_cbl, 'partial_candidates_indices'] = []
                                        logger.info(f"CBL {affected_cbl}: Converted to 'No Match' after losing all insurer matches")
                                    else:
                                        # CBL row still has some matches - update reason
                                        current_reason = cbl_df.at[affected_cbl, 'match_reason']
                                        cbl_df.at[affected_cbl, 'match_reason'] = f"{current_reason} (Updated: insurer {insurer_idx} claimed by CBL {cbl_index})"
                                        logger.info(f"CBL {affected_cbl}: Lost insurer {insurer_idx}, still has {len(updated_df_indices)} insurer(s)")
                                    
                                    logger.info(f"CBL {affected_cbl}: Lost insurer {insurer_idx} due to exact match by CBL {cbl_index}")
                
                # Clear the reverse mapping for this insurer
                del self.insurer_to_cbl_partial[insurer_idx]
        
        # Remove affected insurer indices from partial tracking
        self.partial_used_insurer -= indices_set
        
        # Record the exact match
        self.cbl_exact_matches[cbl_index] = list(indices_set)
        self.exact_used_insurer.update(indices_set)
        
        # Update reverse mapping
        for insurer_idx in indices_set:
            self.insurer_to_cbl_exact[insurer_idx] = cbl_index
        
        logger.debug(f"CBL {cbl_index}: Exact match recorded with insurer indices: {indices_set}")
        if affected_cbl_rows:
            logger.info(f"CBL {cbl_index}: Exact match affected {len(affected_cbl_rows)} other CBL rows: {affected_cbl_rows}")
        
        return True, list(indices_set), [], list(affected_cbl_rows)
    
    def mark_partial_match(self, cbl_index, insurer_indices):
        """
        Mark a CBL-insurer partial match, allowing multiple CBL rows to share insurer rows.
        
        Args:
            cbl_index: CBL row index
            insurer_indices: List of insurer row indices
            
        Returns:
            list: Actually available indices that were marked as partial
        """
        indices_set = set(insurer_indices) if isinstance(insurer_indices, (list, set)) else {insurer_indices}
        
        # Filter out indices already used in exact or matrix matches
        available_indices = indices_set - self.exact_used_insurer - self.matrix_used_insurer
        
        if not available_indices:
            logger.warning(f"CBL {cbl_index}: No available insurer indices for partial match")
            return []
        
        # Check if this CBL row already has a partial match
        if cbl_index in self.cbl_partial_matches:
            # Update existing partial match
            old_indices = set(self.cbl_partial_matches[cbl_index])
            # Remove old mappings
            for idx in old_indices:
                if idx in self.insurer_to_cbl_partial:
                    self.insurer_to_cbl_partial[idx].discard(cbl_index)
                    if not self.insurer_to_cbl_partial[idx]:
                        del self.insurer_to_cbl_partial[idx]
            self.partial_used_insurer -= old_indices
        
        # Record the partial match
        self.cbl_partial_matches[cbl_index] = list(available_indices)
        self.partial_used_insurer.update(available_indices)
        
        # Update reverse mapping
        for insurer_idx in available_indices:
            if insurer_idx not in self.insurer_to_cbl_partial:
                self.insurer_to_cbl_partial[insurer_idx] = set()
            self.insurer_to_cbl_partial[insurer_idx].add(cbl_index)
        
        if available_indices != indices_set:
            unavailable = indices_set - available_indices
            logger.warning(f"CBL {cbl_index}: Some insurer indices already used: {unavailable}")
        
        logger.debug(f"CBL {cbl_index}: Partial match recorded with insurer indices: {available_indices}")
        return list(available_indices)
    
    def can_cbl_claim_insurer(self, cbl_index, insurer_indices, match_type='exact'):
        """
        Check if a CBL row can claim specific insurer indices.
        
        Args:
            cbl_index: CBL row index attempting to claim
            insurer_indices: List of insurer row indices to claim
            match_type: 'exact' or 'partial'
            
        Returns:
            tuple: (can_claim_all, available_indices, conflicts)
        """
        indices_set = set(insurer_indices) if isinstance(insurer_indices, (list, set)) else {insurer_indices}
        
        # Check for matrix and exact match conflicts (always blocked)
        blocked_indices = indices_set & (self.matrix_used_insurer | self.exact_used_insurer)
        
        if match_type == 'exact':
            # For exact matches, check if any insurer is already claimed by another CBL for exact match
            exact_conflicts = []
            for insurer_idx in indices_set:
                if insurer_idx in self.insurer_to_cbl_exact:
                    existing_cbl = self.insurer_to_cbl_exact[insurer_idx]
                    if existing_cbl != cbl_index:  # Different CBL already claimed it
                        exact_conflicts.append((insurer_idx, existing_cbl))
            
            if blocked_indices or exact_conflicts:
                available = indices_set - blocked_indices - {conflict[0] for conflict in exact_conflicts}
                all_conflicts = list(blocked_indices) + exact_conflicts
                return False, list(available), all_conflicts
        else:  # partial
            # For partial matches, only blocked by matrix and exact matches
            if blocked_indices:
                available = indices_set - blocked_indices
                return False, list(available), list(blocked_indices)
        
        return True, list(indices_set), []
    
    def get_insurer_claimants(self, insurer_index):
        """
        Get which CBL rows have claimed a specific insurer row.
        
        Args:
            insurer_index: Insurer row index to check
            
        Returns:
            dict: {'exact': cbl_index or None, 'partial': set of cbl_indices}
        """
        return {
            'exact': self.insurer_to_cbl_exact.get(insurer_index),
            'partial': self.insurer_to_cbl_partial.get(insurer_index, set()).copy()
        }
    
    def can_use_for_exact(self, indices):
        """
        Check if insurer indices can be used for exact match.
        
        Args:
            indices: Single index or list of indices
            
        Returns:
            tuple: (can_use_all, available_indices, unavailable_indices)
        """
        indices_set = set(indices) if isinstance(indices, (list, set)) else {indices}
        unavailable = indices_set & (self.matrix_used_insurer | self.exact_used_insurer)
        available = indices_set - unavailable
        
        return len(unavailable) == 0, list(available), list(unavailable)
    
    def can_use_for_partial(self, indices):
        """
        Check if insurer indices can be used for partial match.
        Partial matches can reuse indices that are currently in other partial matches,
        but not indices used in exact or matrix matches.
        
        Args:
            indices: Single index or list of indices
            
        Returns:
            tuple: (can_use_all, available_indices, unavailable_indices)
        """
        indices_set = set(indices) if isinstance(indices, (list, set)) else {indices}
        unavailable = indices_set & (self.matrix_used_insurer | self.exact_used_insurer)
        available = indices_set - unavailable
        
        return len(unavailable) == 0, list(available), list(unavailable)

    def get_usage_summary(self):
        """Get comprehensive summary of row usage for debugging."""
        total_cbl_with_exact = len(self.cbl_exact_matches)
        total_cbl_with_partial = len(self.cbl_partial_matches)
        
        # Count insurer rows with multiple CBL claimants (for partial matches)
        multi_claimed_insurer = sum(1 for cbl_set in self.insurer_to_cbl_partial.values() if len(cbl_set) > 1)
        
        return {
            'insurer_matrix_used': len(self.matrix_used_insurer),
            'insurer_exact_used': len(self.exact_used_insurer),
            'insurer_partial_used': len(self.partial_used_insurer),
            'total_unique_insurer_used': len(self.matrix_used_insurer | self.exact_used_insurer | self.partial_used_insurer),
            'cbl_exact_matches': total_cbl_with_exact,
            'cbl_partial_matches': total_cbl_with_partial,
            'multi_claimed_insurer_rows': multi_claimed_insurer
        }


def validate_substring_match(str1, str2, min_overlap_pct=0.8, min_length=10):
    """
    Validate substring matches with quality controls.
    
    Args:
        str1: First string (CBL placing number)
        str2: Second string (Insurer placing number)
        min_overlap_pct: Minimum overlap percentage (0.8 = 80%)
        min_length: Minimum length for both strings
    
    Returns:
        tuple: (is_valid_match, overlap_info)
    """
    # Both strings must meet minimum length
    if len(str1) < min_length or len(str2) < min_length:
        return False, f"Strings too short ({len(str1)}, {len(str2)}) < {min_length}"
    
    # Calculate overlap percentage
    if str1 in str2:
        overlap_pct = len(str1) / len(str2)
        match_type = "CBL in Insurer"
    elif str2 in str1:
        overlap_pct = len(str2) / len(str1)
        match_type = "Insurer in CBL"
    else:
        return False, "No substring relationship"
    
    # Require substantial overlap
    if overlap_pct < min_overlap_pct:
        return False, f"Low overlap: {overlap_pct:.1%} < {min_overlap_pct:.1%}"
    
    return True, f"{match_type}: {overlap_pct:.1%} overlap"


def classify_amount_match(amt1, amt2, tolerance):
    """
    Classify amount matching with business-relevant confidence levels.
    
    Args:
        amt1: CBL amount (usually negative)
        amt2: Insurer amount (usually positive) 
        tolerance: Base tolerance for exact matches
        
    Returns:
        tuple: (match_type, difference, confidence_level)
    """
    difference = abs(amt1 + amt2)
    
    if difference <= tolerance * 0.1:  # Within 10% of tolerance
        return "PERFECT_MATCH", difference, "Perfect"
    elif difference <= tolerance:  # Within tolerance
        return "EXACT_MATCH", difference, "High"
    elif difference <= tolerance * 2:  # Within 2x tolerance  
        return "CLOSE_MATCH", difference, "Medium"
    elif difference <= tolerance * 5:  # Within 5x tolerance
        return "REVIEW_REQUIRED", difference, "Low"
    elif difference <= tolerance * 10:  # Within 10x tolerance
        return "INVESTIGATION_REQUIRED", difference, "Very Low"
    else:
        return "NO_MATCH", difference, "None"


def _apply_exact_match(cbl_df, cbl_index, match_reason, insurer_indices, total_amount, fallback_indices, pass_number, global_tracker=None, confidence_level=None, amount_difference=None):
    """Apply an exact match to a CBL record."""
    # Validate indices with comprehensive global tracker if provided
    if global_tracker:
        can_claim_all, available_indices, conflicts = global_tracker.can_cbl_claim_insurer(
            cbl_index, insurer_indices, 'exact'
        )
        
        if not can_claim_all:
            logger.warning(f"Pass {pass_number} CBL {cbl_index}: Cannot claim all insurer indices. Conflicts: {conflicts}")
            if not available_indices:
                logger.error(f"Pass {pass_number} CBL {cbl_index}: No available indices for exact match - marking as No Match")
                _apply_no_match(cbl_df, cbl_index, f"{match_reason} (All indices conflicted)")
                return 0
            
            # Use only available indices
            insurer_indices = available_indices
            logger.info(f"Pass {pass_number} CBL {cbl_index}: Using available indices: {available_indices}")
        
        # Mark the CBL-insurer exact match with automatic CBL DataFrame cleanup
        success, final_indices, match_conflicts, affected_cbl_rows = global_tracker.mark_exact_match(
            cbl_index, insurer_indices, cbl_df
        )
        
        if not success:
            logger.error(f"Pass {pass_number} CBL {cbl_index}: Failed to mark exact match due to conflicts: {match_conflicts}")
            _apply_no_match(cbl_df, cbl_index, f"{match_reason} (Match conflicts)")
            return 0
        
        insurer_indices = final_indices
        
        # Log affected CBL rows for transparency
        if affected_cbl_rows:
            logger.info(f"Pass {pass_number} CBL {cbl_index}: Exact match affected {len(affected_cbl_rows)} other CBL rows: {affected_cbl_rows}")
    
    cbl_df.at[cbl_index, "match_status"] = "Exact Match"
    cbl_df.at[cbl_index, "match_reason"] = match_reason
    cbl_df.at[cbl_index, "matched_insurer_indices"] = insurer_indices
    cbl_df.at[cbl_index, "matched_amtdue_total"] = total_amount
    cbl_df.at[cbl_index, "partial_candidates_indices"] = fallback_indices or []
    cbl_df.at[cbl_index, "match_resolved_in_pass"] = pass_number
    
    # Add confidence and difference information
    if confidence_level is not None:
        cbl_df.at[cbl_index, "match_confidence"] = confidence_level
    if amount_difference is not None:
        cbl_df.at[cbl_index, "amount_difference"] = amount_difference
        
    return 1  # Return count for exact matches


def _apply_partial_match(cbl_df, cbl_index, match_reason, insurer_indices, total_amount, pass_number, global_tracker=None, confidence_level=None, amount_difference=None):
    """Apply a partial match to a CBL record."""
    # Validate and filter indices with comprehensive global tracker if provided
    if global_tracker:
        can_claim_all, available_indices, conflicts = global_tracker.can_cbl_claim_insurer(
            cbl_index, insurer_indices, 'partial'
        )
        
        if not available_indices:
            logger.warning(f"Pass {pass_number} CBL {cbl_index}: No available indices for partial match - marking as No Match")
            _apply_no_match(cbl_df, cbl_index, f"{match_reason} (All indices conflicted)")
            return 0
        
        if not can_claim_all:
            logger.info(f"Pass {pass_number} CBL {cbl_index}: Using {len(available_indices)}/{len(insurer_indices)} available indices. Conflicts: {conflicts}")
        
        # Mark the CBL-insurer partial match
        final_indices = global_tracker.mark_partial_match(cbl_index, available_indices)
        
        if not final_indices:
            logger.error(f"Pass {pass_number} CBL {cbl_index}: Failed to mark partial match")
            _apply_no_match(cbl_df, cbl_index, f"{match_reason} (Mark failed)")
            return 0
        
        # Use the indices that were successfully marked
        insurer_indices = final_indices
    
    cbl_df.at[cbl_index, "match_status"] = "Partial Match"
    cbl_df.at[cbl_index, "match_reason"] = match_reason
    cbl_df.at[cbl_index, "matched_insurer_indices"] = insurer_indices
    cbl_df.at[cbl_index, "matched_amtdue_total"] = total_amount
    cbl_df.at[cbl_index, "partial_candidates_indices"] = []
    cbl_df.at[cbl_index, "partial_resolved_in_pass"] = pass_number
    
    # Add confidence and difference information
    if confidence_level is not None:
        cbl_df.at[cbl_index, "match_confidence"] = confidence_level
    if amount_difference is not None:
        cbl_df.at[cbl_index, "amount_difference"] = amount_difference
        
    return 1  # Return count for partial matches


def _apply_no_match(cbl_df, cbl_index, match_reason):
    """Apply a no match status to a CBL record."""
    cbl_df.at[cbl_index, "match_status"] = "No Match"
    cbl_df.at[cbl_index, "match_reason"] = match_reason
    cbl_df.at[cbl_index, "matched_insurer_indices"] = []
    cbl_df.at[cbl_index, "matched_amtdue_total"] = None
    cbl_df.at[cbl_index, "partial_candidates_indices"] = []


def _handle_conflict_resolution(cbl_df, insurer_df, match, used_insurer_indices, tolerance, pass_number, global_tracker=None, fallback_rows=None):
    """
    Handle conflict resolution with fallback logic.
    
    Args:
        cbl_df: CBL dataframe
        insurer_df: Insurer dataframe (or fallback_rows for Pass 2)
        match: Match dictionary with conflict
        used_insurer_indices: Set of already used insurer indices (legacy - use global_tracker instead)
        tolerance: Tolerance for amount matching
        pass_number: Which pass is calling this function
        global_tracker: GlobalInsurerTracker instance for consistent tracking
        fallback_rows: Optional fallback rows dataframe (for Pass 2)
        
    Returns:
        tuple: (exact_matches_added, partial_matches_added)
    """
    cbl_index = match['cbl_index']
    match_type = match['match_type']
    insurer_indices = match['insurer_indices']
    
    logger.info(f"Pass {pass_number} Record {cbl_index}: Handling conflicts for {match_type} match")
    
    # Use global tracker if available, otherwise fall back to legacy logic
    if global_tracker:
        # Check availability based on match type
        if match_type in ['exact', 'combination']:
            can_use_all, available_indices, unavailable_indices = global_tracker.can_use_for_exact(insurer_indices)
        else:  # partial
            can_use_all, available_indices, unavailable_indices = global_tracker.can_use_for_partial(insurer_indices)
        
        if unavailable_indices:
            logger.info(f"Record {cbl_index}: Some indices unavailable: {unavailable_indices}")
        
        # Try fallback indices if original indices are not available
        if not available_indices and 'fallback_indices' in match and match['fallback_indices']:
            logger.info(f"Record {cbl_index}: Trying fallback indices: {match['fallback_indices']}")
            if match_type in ['exact', 'combination']:
                can_use_fallback, available_indices, _ = global_tracker.can_use_for_exact(match['fallback_indices'])
            else:
                can_use_fallback, available_indices, _ = global_tracker.can_use_for_partial(match['fallback_indices'])
            
            if available_indices:
                logger.info(f"Record {cbl_index}: Using available fallback indices: {available_indices}")
    else:
        # Legacy logic for backwards compatibility
        available_indices = []
        if 'fallback_indices' in match and match['fallback_indices']:
            available_fallback = [idx for idx in match['fallback_indices'] if idx not in used_insurer_indices]
            if available_fallback:
                logger.info(f"Record {cbl_index}: Using fallback indices {available_fallback}")
                available_indices = available_fallback
        
        if not available_indices:
            available_indices = [idx for idx in insurer_indices if idx not in used_insurer_indices]
            if available_indices:
                logger.info(f"Record {cbl_index}: Using remaining original indices {available_indices}")
    
    if not available_indices:
        # All potential indices are already used - mark as No Match
        logger.info(f"Record {cbl_index}: All potential indices used - marking as No Match")
        _apply_no_match(cbl_df, cbl_index, match['match_reason'])
        return 0, 0
    
    # Calculate amounts using the appropriate dataframe
    data_source = fallback_rows if fallback_rows is not None else insurer_df
    cbl_amount = cbl_df.at[cbl_index, "Amount_Clean"]
    available_amounts = data_source.loc[available_indices, "Amount_Clean_INSURER"]
    total_available_amount = available_amounts.sum()
    
    # Check if fallback indices create a perfect match
    if -tolerance <= (cbl_amount + total_available_amount) <= tolerance:
        # Upgrade to exact match!
        logger.info(f"Record {cbl_index}: Fallback indices upgraded to Exact Match!")
        if not global_tracker:
            used_insurer_indices.update(available_indices)
        return _apply_exact_match(cbl_df, cbl_index, f"{match['match_reason']} (Fallback Match)", 
                                 available_indices, total_available_amount, [], pass_number, global_tracker), 0
    else:
        # Apply as partial match
        logger.info(f"Record {cbl_index}: Fallback indices as Partial Match")
        if not global_tracker:
            used_insurer_indices.update(available_indices)
        return 0, _apply_partial_match(cbl_df, cbl_index, f"{match['match_reason']} (Fallback Partial)",
                                      available_indices, total_available_amount, pass_number, global_tracker)


def pass1(cbl_df, insurer_df, tolerance=100, global_tracker=None):
    """Pass 1: Matching by Placing Number and Amount."""
    logger.info("\n=== Pass 1: Matching by Placing Number and Amount ===")
    total_records = len(cbl_df)
    exact_matches = 0
    partial_matches = 0
    
    if global_tracker:
        logger.info(f"Pass 1 starting with global tracker: {global_tracker.get_usage_summary()}")
    else:
        logger.warning("Pass 1 running without global tracker - legacy mode")

    # Pre-compute string conversions for performance optimization
    logger.info("Pre-computing insurer placing strings for substring matching...")
    insurer_placing_strings = insurer_df["PlacingNo_Clean_INSURER"].astype(str)
    # Cache valid placing strings (length >= 10) to avoid repeated checks
    valid_insurer_mask = (insurer_placing_strings != 'nan') & (insurer_placing_strings.str.len() >= 10)

    # Phase 1: Collect all potential matches without applying them yet
    potential_matches = []
    
    for i, row in cbl_df.iterrows():
        if i % 100 == 0:
            logger.info(f"Progress: {i+1}/{total_records} records processed")
            
        add_pass(cbl_df, i, 1)
        
        placing = row["PlacingNo_Clean"]
        amt1 = row["Amount_Clean"]
        
        # Validate input data
        if pd.isna(placing) or placing == "" or str(placing).strip() == "":
            logger.warning(f"Record {i}: Empty or invalid placing number, skipping")
            continue
            
        if pd.isna(amt1):
            logger.warning(f"Record {i}: Invalid amount ({amt1}), skipping")
            continue

        insurer_matches = insurer_df[insurer_df["PlacingNo_Clean_INSURER"] == placing]
        
        # If no exact matches, try enhanced substring matching with quality controls
        overlap_details = {}  # Store overlap info for later use in match reasons
        if insurer_matches.empty:
            placing_str = str(placing).strip()
            
            # Only proceed if CBL placing is long enough (>= 10 chars)
            if len(placing_str) >= 10:
                logger.debug(f"Record {i}: No exact matches, trying quality-controlled substring matching for '{placing_str}'")
                qualified_indices = []
                rejected_count = 0
                
                # Check each insurer placing number with quality validation
                for idx in insurer_df.index:
                    insurer_placing = str(insurer_df.at[idx, "PlacingNo_Clean_INSURER"])
                    
                    # Skip NaN or invalid insurer placing numbers
                    if pd.isna(insurer_placing) or insurer_placing == 'nan' or not insurer_placing.strip():
                        continue
                    
                    # Validate substring match quality
                    is_valid, overlap_info = validate_substring_match(placing_str, insurer_placing.strip())
                    
                    if is_valid:
                        qualified_indices.append(idx)
                        overlap_details[idx] = overlap_info  # Store for later use in match reasons
                        logger.debug(f"Record {i}: Qualified substring match at index {idx}: {overlap_info}")
                    else:
                        rejected_count += 1
                        logger.debug(f"Record {i}: Rejected substring match at index {idx}: {overlap_info}")
                
                # Create matches dataframe from qualified indices
                insurer_matches = insurer_df.loc[qualified_indices] if qualified_indices else pd.DataFrame()
                
                if not insurer_matches.empty:
                    logger.info(f"Record {i}: Found {len(insurer_matches)} qualified substring matches (rejected {rejected_count} poor quality matches)")
                elif rejected_count > 0:
                    logger.info(f"Record {i}: No qualified substring matches found (rejected {rejected_count} poor quality matches)")
            else:
                logger.debug(f"Record {i}: Skipping substring matching - placing too short ({len(placing_str)} chars)")

        # First comparison - exact matches
        exact_match_indices = None
        exact_partial_count = 0
        insurer_indices = []  # Initialize insurer_indices outside the if block
        
        if not insurer_matches.empty:
            # Ensure unique indices to prevent duplicates in combinations
            unique_indices = []
            unique_amounts = []
            seen_indices = set()
            
            for idx, amt in zip(insurer_matches.index.tolist(), insurer_matches["Amount_Clean_INSURER"].tolist()):
                if idx not in seen_indices:
                    unique_indices.append(idx)
                    unique_amounts.append(amt)
                    seen_indices.add(idx)
            
            insurer_indices = unique_indices
            insurer_amounts = unique_amounts
            exact_partial_count = len(insurer_indices)
            
            # Check individual matches first with graduated confidence levels
            best_match = None
            for j, amt2 in zip(insurer_indices, insurer_amounts):
                if pd.notna(amt2):
                    match_type, difference, confidence = classify_amount_match(amt1, amt2, tolerance)
                    
                    if match_type in ["PERFECT_MATCH", "EXACT_MATCH"]:
                        # Auto-approve exact matches
                        # Include overlap info if this came from substring matching
                        overlap_info = overlap_details.get(j, "")
                        overlap_suffix = f" ({overlap_info})" if overlap_info else ""
                        
                        best_match = {
                            'indices': [j], 
                            'type': 'exact',
                            'confidence': confidence,
                            'difference': difference,
                            'reason': f'Placing Number{overlap_suffix} + Single Amount Match ({confidence} Confidence, Diff: ${difference:.2f})'
                        }
                        break
                    elif match_type == "CLOSE_MATCH" and best_match is None:
                        # Consider close matches if no exact match found
                        # Include overlap info if this came from substring matching
                        overlap_info = overlap_details.get(j, "")
                        overlap_suffix = f" ({overlap_info})" if overlap_info else ""
                        
                        best_match = {
                            'indices': [j],
                            'type': 'close', 
                            'confidence': confidence,
                            'difference': difference,
                            'reason': f'Placing Number{overlap_suffix} + Close Amount Match ({confidence} Confidence, Diff: ${difference:.2f})'
                        }
            
            # Set exact_match_indices based on best match found
            if best_match and best_match['type'] == 'exact':
                exact_match_indices = best_match['indices']
                exact_match_confidence = best_match['confidence']
                exact_match_difference = best_match['difference']
                exact_match_reason = best_match['reason']
            else:
                exact_match_indices = None
        
        # Second comparison - combinations (smart selection)
        combination_match_indices = None
        combination_partial_count = 0
        if not insurer_matches.empty and exact_match_indices is None:
            # Only try combinations if no exact match was found
            # Exclude any indices that were used in exact matches
            available_indices = [idx for idx in insurer_indices if idx not in (exact_match_indices or [])]
            available_amounts = [amt for idx, amt in zip(insurer_indices, insurer_amounts) if idx in available_indices]
            
            combination_partial_count = len(available_indices)
            
            # Smart selection: limit to 50 most promising items
            max_items_to_consider = 20
            target = -amt1  # We want sum(insurer_amounts) to be close to -amt1
            
            if len(available_indices) > max_items_to_consider:
                # Sort by how close each amount gets us to the target
                sorted_pairs = sorted(zip(available_indices, available_amounts), 
                                    key=lambda x: abs(x[1] - target))
                
                # Take the 50 most promising items
                limited_indices = [pair[0] for pair in sorted_pairs[:max_items_to_consider]]
                limited_amounts = [pair[1] for pair in sorted_pairs[:max_items_to_consider]]
                
                logger.info(f"Record {i}: Selected {max_items_to_consider} most promising items from {len(available_indices)} total")
                logger.info(f"Target amount: {target}, Selected amounts: {limited_amounts}")
            else:
                limited_indices = available_indices
                limited_amounts = available_amounts
            
            # Try combinations with the limited set (max 5 items per combination for business reality)
            max_combination_size = min(5, len(limited_indices))
            
            for r in range(2, max_combination_size + 1):
                for combination in combinations(zip(limited_indices, limited_amounts), r):
                    combination_indices, combination_amounts = zip(*combination)
                    
                    # Filter out NaN values and validate amounts
                    valid_amounts = [amt for amt in combination_amounts if pd.notna(amt)]
                    if len(valid_amounts) != len(combination_amounts):
                        logger.warning(f"Record {i}: Skipping combination with NaN values: {combination_amounts}")
                        continue
                    
                    total_amount = sum(valid_amounts)
                    if pd.notna(total_amount):
                        match_type, difference, confidence = classify_amount_match(amt1, total_amount, tolerance)
                        
                        if match_type in ["PERFECT_MATCH", "EXACT_MATCH"]:
                            combination_match_indices = list(combination_indices)
                            combination_match_confidence = confidence
                            combination_match_difference = difference
                            
                            # Include overlap info if any of the combination items came from substring matching
                            overlap_infos = [overlap_details.get(idx, "") for idx in combination_indices if overlap_details.get(idx, "")]
                            overlap_suffix = f" ({'; '.join(set(overlap_infos))})" if overlap_infos else ""
                            
                            combination_match_reason = f'Placing Number{overlap_suffix} + Cumulative Amount Match ({confidence} Confidence, Diff: ${difference:.2f})'
                            logger.info(f"Record {i}: Found combination match with {r} items, total: {total_amount}, confidence: {confidence}")
                            break
                        elif match_type == "CLOSE_MATCH" and combination_match_indices is None:
                            # Store close combination match as backup
                            combination_match_indices = list(combination_indices)
                            combination_match_confidence = confidence
                            combination_match_difference = difference
                            
                            # Include overlap info if any of the combination items came from substring matching
                            overlap_infos = [overlap_details.get(idx, "") for idx in combination_indices if overlap_details.get(idx, "")]
                            overlap_suffix = f" ({'; '.join(set(overlap_infos))})" if overlap_infos else ""
                            
                            combination_match_reason = f'Placing Number{overlap_suffix} + Close Cumulative Match ({confidence} Confidence, Diff: ${difference:.2f})'
                            logger.info(f"Record {i}: Found close combination match with {r} items, total: {total_amount}, confidence: {confidence}")
                            # Don't break - continue looking for exact matches
                if combination_match_indices is not None:
                    break

        # Log results for each comparison method
        logger.info(f"\nComparison results for CBL record {i}:")
        logger.info(f"Exact comparison: {1 if exact_match_indices else 0} exact matches, {exact_partial_count} partial matches")
        logger.info(f"Combination comparison: {1 if combination_match_indices else 0} exact matches, {combination_partial_count} partial matches")

        # Store potential matches for later resolution
        if exact_match_indices is not None:
            potential_matches.append({
                'cbl_index': i,
                'match_type': 'exact',
                'insurer_indices': exact_match_indices,
                'match_reason': exact_match_reason,
                'confidence_level': exact_match_confidence,
                'amount_difference': exact_match_difference,
                'fallback_indices': [idx for idx in insurer_indices if idx not in exact_match_indices]
            })
        elif combination_match_indices is not None:
            potential_matches.append({
                'cbl_index': i,
                'match_type': 'combination',
                'insurer_indices': combination_match_indices,
                'match_reason': combination_match_reason,
                'confidence_level': combination_match_confidence,
                'amount_difference': combination_match_difference,
                'fallback_indices': [idx for idx in insurer_indices if idx not in combination_match_indices]
            })
        elif not insurer_matches.empty:
            # Only create partial match if there are some valid matches (even if not perfect)
            # Use graduated classification for partial matches too
            reasonable_matches = []
            best_partial_confidence = None
            best_partial_difference = None
            
            for idx, amt in zip(insurer_indices, insurer_amounts):
                if pd.notna(amt):
                    match_type, difference, confidence = classify_amount_match(amt1, amt, tolerance)
                    
                    if match_type in ["REVIEW_REQUIRED", "INVESTIGATION_REQUIRED"]:
                        reasonable_matches.append(idx)
                        # Track the best (lowest difference) partial match for reporting
                        if best_partial_difference is None or difference < best_partial_difference:
                            best_partial_confidence = confidence
                            best_partial_difference = difference
            
            if reasonable_matches:
                # Partial match - there are some reasonable matches
                # For partial matches, fallback should be ALL other insurer indices with same placing number
                # that weren't selected as reasonable matches
                fallback_candidates = [idx for idx in insurer_indices if idx not in reasonable_matches]
                
                # Include overlap info if any reasonable matches came from substring matching
                overlap_infos = [overlap_details.get(idx, "") for idx in reasonable_matches if overlap_details.get(idx, "")]
                overlap_suffix = f" ({'; '.join(set(overlap_infos))})" if overlap_infos else ""
                
                partial_reason = f'Placing Number{overlap_suffix} Match ({best_partial_confidence} Confidence, Diff: ${best_partial_difference:.2f})'
                
                potential_matches.append({
                    'cbl_index': i,
                    'match_type': 'partial',
                    'insurer_indices': reasonable_matches,
                    'match_reason': partial_reason,
                    'confidence_level': best_partial_confidence,
                    'amount_difference': best_partial_difference,
                    'fallback_indices': fallback_candidates
                })
            # If no reasonable matches, don't create any match (will be No Match)
            # This ensures that only rows with actual reasonable matches are flagged as Partial Match

    # Phase 2: Resolve conflicts by prioritizing combination matches
    logger.info("\n=== Phase 2: Resolving conflicts and applying matches ===")
    
    # Sort potential matches: exact matches first, then combinations, then partial matches
    # Within each type, sort by number of insurer indices (larger combinations get priority)
    potential_matches.sort(key=lambda x: (
        0 if x['match_type'] == 'exact' else (1 if x['match_type'] == 'combination' else 2),
        -len(x['insurer_indices'])  # Negative for descending order (larger combinations first)
    ))
    
    # Track used insurer indices
    used_insurer_indices = set()
    
    for match in potential_matches:
        cbl_index = match['cbl_index']
        match_type = match['match_type']
        insurer_indices = match['insurer_indices']
        
        # Check if any of the insurer indices are already used
        conflicting_indices = set(insurer_indices) & used_insurer_indices
        
        if conflicting_indices:
            # Use helper function for conflict resolution
            exact_added, partial_added = _handle_conflict_resolution(
                cbl_df, insurer_df, match, used_insurer_indices, tolerance, 1, global_tracker
            )
            exact_matches += exact_added
            partial_matches += partial_added
        else:
            # Apply the match using helper functions
            if match_type in ['exact', 'combination']:
                total_amount = sum(insurer_df.loc[insurer_indices, "Amount_Clean_INSURER"])
                exact_matches += _apply_exact_match(
                    cbl_df, cbl_index, match['match_reason'], insurer_indices, 
                    total_amount, match.get('fallback_indices', []), 1, global_tracker,
                    confidence_level=match.get('confidence_level'),
                    amount_difference=match.get('amount_difference')
                )
                if not global_tracker:
                    used_insurer_indices.update(insurer_indices)
            else:  # partial
                total_amount = insurer_df.loc[insurer_indices, "Amount_Clean_INSURER"].sum()
                partial_matches += _apply_partial_match(
                    cbl_df, cbl_index, match['match_reason'], insurer_indices, total_amount, 1, global_tracker,
                    confidence_level=match.get('confidence_level'),
                    amount_difference=match.get('amount_difference')
                )
                if not global_tracker:
                    used_insurer_indices.update(insurer_indices)

    logger.info(f"✓ Pass 1 complete: {exact_matches} exact matches, {partial_matches} partial matches")
    return cbl_df


def pass2(cbl_df, insurer_df, tolerance=100, name_threshold=95, global_tracker=None):
    """Pass 2: Matching by Policy Number and Name."""
    logger.info("\n=== Pass 2: Matching by Policy Number and Name ===")
    total_records = len(cbl_df[cbl_df["match_status"].isin(["No Match", "Partial Match"])])
    exact_matches = 0
    partial_matches = 0
    processed = 0
    
    if global_tracker:
        logger.info(f"Pass 2 starting with global tracker: {global_tracker.get_usage_summary()}")
    else:
        logger.warning("Pass 2 running without global tracker - legacy mode")

    # Track which insurer rows have been used for partial matches
    partial_used_insurer_indices = set()
    
    # Get all insurer indices that have been used for partial matches in previous passes
    for indices in cbl_df[cbl_df["match_status"] == "Partial Match"]["matched_insurer_indices"]:
        if isinstance(indices, list):
            partial_used_insurer_indices.update(indices)
        elif pd.notna(indices):
            partial_used_insurer_indices.add(indices)

    fallback_index_pool = set()

    for i, row in cbl_df.iterrows():
        fallback = row.get("partial_candidates_indices")
        if isinstance(fallback, list):
            fallback_index_pool.update(fallback)

    # Filter out insurer indices that have already been used for partial matches
    available_fallback_indices = fallback_index_pool - partial_used_insurer_indices
    fallback_rows = insurer_df.loc[list(available_fallback_indices)]
    logger.info(f"Found {len(available_fallback_indices)} potential matches from Pass 1 (excluding already used partial matches)")

    # Phase 1: Collect all potential matches without applying them yet
    potential_matches = []
    
    for i, row in cbl_df[cbl_df["match_status"].isin(["No Match", "Partial Match"])].iterrows():
        processed += 1
        if processed % 50 == 0:
            logger.info(f"Progress: {processed}/{total_records} records processed")

        add_pass(cbl_df, i, 2)
        tokens = extract_policy_tokens(row["PolicyNo"])
        cbl_name = str(row["ClientName"]).upper().strip()
        cbl_amt = row["Amount_Clean"]

        matched_indices = []
        name_scores = []

        for j, insurer_row in fallback_rows.iterrows():
            insurer_name = str(insurer_row["ClientName_INSURER"]).upper().strip()
            name_score = fuzz.partial_ratio(cbl_name, insurer_name)
            
            # Check PolicyNo_1 match (only if it's not empty)
            policy_no_1_match = False
            if pd.notna(insurer_row["PolicyNo_Clean_INSURER"]) and insurer_row["PolicyNo_Clean_INSURER"]:
                policy_no_1_match = insurer_row["PolicyNo_Clean_INSURER"] in tokens
            
            # Check PolicyNo_2 match (only if it exists and is not empty)
            policy_no_2_match = False
            if "PolicyNo_2_Clean_INSURER" in insurer_row.index and pd.notna(insurer_row["PolicyNo_2_Clean_INSURER"]) and insurer_row["PolicyNo_2_Clean_INSURER"]:
                policy_no_2_match = insurer_row["PolicyNo_2_Clean_INSURER"] in tokens
            
            policy_match = policy_no_1_match or policy_no_2_match

            if policy_match and name_score >= name_threshold:
                matched_indices.append(j)
                name_scores.append(name_score)

        total_amt = fallback_rows.loc[matched_indices, "Amount_Clean_INSURER"].sum()
        highest_name_score = max(name_scores) if name_scores else 0

        if matched_indices:
            # Classify the amount match using graduated confidence levels
            match_type, difference, confidence = classify_amount_match(cbl_amt, total_amt, tolerance)
            
            # Determine if it's single or cumulative amount match
            if len(matched_indices) == 1:
                amount_match_type = 'Single Amount Match'
            else:
                amount_match_type = 'Cumulative Amount Match'
            
            if match_type in ["PERFECT_MATCH", "EXACT_MATCH"]:
                # Exact match found
                match_reason = f'Policy + Name Match (CS: {highest_name_score}%) + {amount_match_type} ({confidence} Confidence, Diff: ${difference:.2f})'
                potential_matches.append({
                    'cbl_index': i,
                    'match_type': 'exact',
                    'insurer_indices': matched_indices,
                    'match_reason': match_reason,
                    'confidence_level': confidence,
                    'amount_difference': difference,
                    'total_amount': total_amt,
                    'name_score': highest_name_score
                })
            elif match_type == "CLOSE_MATCH":
                # Close match - treat as exact but with medium confidence
                match_reason = f'Policy + Name Match (CS: {highest_name_score}%) + Close {amount_match_type} ({confidence} Confidence, Diff: ${difference:.2f})'
                potential_matches.append({
                    'cbl_index': i,
                    'match_type': 'exact',  # Still treat as exact for processing
                    'insurer_indices': matched_indices,
                    'match_reason': match_reason,
                    'confidence_level': confidence,
                    'amount_difference': difference,
                    'total_amount': total_amt,
                    'name_score': highest_name_score
                })
            elif match_type in ["REVIEW_REQUIRED", "INVESTIGATION_REQUIRED"]:
                # Partial match found
                match_reason = f'Policy + Name Match (CS: {highest_name_score}%) + {amount_match_type} ({confidence} Confidence, Diff: ${difference:.2f})'
                potential_matches.append({
                    'cbl_index': i,
                    'match_type': 'partial',
                    'insurer_indices': matched_indices,
                    'match_reason': match_reason,
                    'confidence_level': confidence,
                    'amount_difference': difference,
                    'total_amount': total_amt,
                    'name_score': highest_name_score
                })
            # If NO_MATCH, don't add to potential matches

    # Phase 2: Resolve conflicts and apply matches
    logger.info("\n=== Pass 2 Phase 2: Resolving conflicts and applying matches ===")
    
    # Sort potential matches: exact matches first, then partial matches
    # Within each type, sort by number of insurer indices (larger combinations get priority)
    potential_matches.sort(key=lambda x: (
        0 if x['match_type'] == 'exact' else 1,
        -len(x['insurer_indices'])  # Negative for descending order (larger combinations first)
    ))
    
    # Track used insurer indices
    used_insurer_indices = set()
    
    for match in potential_matches:
        cbl_index = match['cbl_index']
        match_type = match['match_type']
        insurer_indices = match['insurer_indices']
        
        # Check if any of the insurer indices are already used
        conflicting_indices = set(insurer_indices) & used_insurer_indices
        
        if conflicting_indices:
            # Use helper function for conflict resolution
            exact_added, partial_added = _handle_conflict_resolution(
                cbl_df, insurer_df, match, used_insurer_indices, tolerance, 2, global_tracker, fallback_rows
            )
            exact_matches += exact_added
            partial_matches += partial_added
        else:
            # Apply the match using helper functions
            if match_type == 'exact':
                exact_matches += _apply_exact_match(
                    cbl_df, cbl_index, match['match_reason'], insurer_indices, 
                    match['total_amount'], [], 2, global_tracker,
                    confidence_level=match.get('confidence_level'),
                    amount_difference=match.get('amount_difference')
                )
                if not global_tracker:
                    used_insurer_indices.update(insurer_indices)
                
                # Note: CBL row updates are handled automatically by GlobalMatchTracker
            else:  # partial
                total_amount = fallback_rows.loc[insurer_indices, "Amount_Clean_INSURER"].sum()
                partial_matches += _apply_partial_match(
                    cbl_df, cbl_index, match['match_reason'], insurer_indices, total_amount, 2, global_tracker,
                    confidence_level=match.get('confidence_level'),
                    amount_difference=match.get('amount_difference')
                )
                if not global_tracker:
                    used_insurer_indices.update(insurer_indices)

    logger.info(f"✓ Pass 2 complete: {exact_matches} exact matches, {partial_matches} partial matches")
    return cbl_df


def pass3(cbl_df, insurer_df, tolerance=100, fuzzy_threshold=95, global_tracker=None):
    """Pass 3: Final matching by Name and Amount (Row-by-Row + Cumulative + Improved Group Matching)."""
    logger.info("\n=== Pass 3: Final matching by Name and Amount (Row-by-Row + Cumulative + Improved Group Matching) ===")
    total_records = len(cbl_df[cbl_df["match_status"].isin(["No Match", "Partial Match"])])
    exact_matches = 0
    partial_matches = 0
    processed = 0
    
    if global_tracker:
        logger.info(f"Pass 3 starting with global tracker: {global_tracker.get_usage_summary()}")
    else:
        logger.warning("Pass 3 running without global tracker - legacy mode")

    # Use global tracker for consistent filtering, or fall back to legacy logic
    if global_tracker:
        # With global tracker, we can trust its state for filtering
        already_matched_insurer = global_tracker.exact_used_insurer | global_tracker.matrix_used_insurer
        available_insurer = insurer_df[~insurer_df.index.isin(already_matched_insurer)].copy()
        logger.info(f"Pass 3: Using global tracker - excluding {len(already_matched_insurer)} already used insurer rows")
        
        # Also get partial used insurer indices for compatibility
        partial_used_insurer = global_tracker.partial_used_insurer.copy()
    else:
        # Legacy logic for backwards compatibility
        already_matched_insurer = set()
        for indices in cbl_df[cbl_df["match_status"] == "Exact Match"]["matched_insurer_indices"]:
            if isinstance(indices, list):
                already_matched_insurer.update(indices)
            elif pd.notna(indices):
                already_matched_insurer.add(indices)
        
        # Get indices of insurer rows that have been used for partial matches
        partial_used_insurer = set()
        for indices in cbl_df[cbl_df["match_status"] == "Partial Match"]["matched_insurer_indices"]:
            if isinstance(indices, list):
                partial_used_insurer.update(indices)
            elif pd.notna(indices):
                partial_used_insurer.add(indices)

        # Filter out insurer rows already used in exact matches. We allow using rows that
        # are currently in partial matches, because step 3 may upgrade them to exact.
        available_insurer = insurer_df[~insurer_df.index.isin(already_matched_insurer)].copy()
    
    # Pre-calculate name scores for all insurer rows
    insurer_names = available_insurer["ClientName_INSURER"].fillna("").str.upper().str.strip()
    insurer_amounts = available_insurer["Amount_Clean_INSURER"]

    # Phase 1: Collect all potential matches without applying them yet
    potential_matches = []
    
    # IMPROVED GROUP MATCHING: First, try to find group matches
    logger.info("=== Phase 1: Collecting group matches ===")
    
    # Group CBL rows by name (only unmatched and partial matches)
    cbl_groups = {}
    for idx, row in cbl_df.iterrows():
        if row['match_status'] in ['No Match', 'Partial Match']:
            name = str(row['ClientName']).upper().strip()
            if name not in cbl_groups:
                cbl_groups[name] = []
            cbl_groups[name].append(idx)
    
    # Group insurer rows by name
    insurer_groups = {}
    for idx, row in available_insurer.iterrows():
        name = str(row['ClientName_INSURER']).upper().strip()
        if name not in insurer_groups:
            insurer_groups[name] = []
        insurer_groups[name].append(idx)
    
    logger.info(f"Found {len(cbl_groups)} CBL name groups and {len(insurer_groups)} insurer name groups")
    
    # Find matching groups
    group_matches = []
    
    for cbl_name, cbl_indices in cbl_groups.items():
        for insurer_name, insurer_indices in insurer_groups.items():
            # Check if names are similar enough
            name_score = fuzz.partial_ratio(cbl_name, insurer_name)
            if name_score >= fuzzy_threshold:
                # Calculate totals
                cbl_total = cbl_df.loc[cbl_indices, 'Amount_Clean'].sum()
                insurer_total = available_insurer.loc[insurer_indices, 'Amount_Clean_INSURER'].sum()
                difference = cbl_total + insurer_total

                if -tolerance <= difference <= tolerance:
                    group_matches.append({
                        'cbl_name': cbl_name,
                        'insurer_name': insurer_name,
                        'cbl_indices': cbl_indices,
                        'insurer_indices': insurer_indices,
                        'cbl_total': cbl_total,
                        'insurer_total': insurer_total,
                        'difference': difference,
                        'name_score': name_score
                    })
    
    logger.info(f"Found {len(group_matches)} potential group matches")
    
    # Debug: Log details about group matching
    logger.info(f"DEBUG: CBL groups found: {len(cbl_groups)}")
    logger.info(f"DEBUG: Insurer groups found: {len(insurer_groups)}")
    multi_cbl_groups = {k: v for k, v in cbl_groups.items() if len(v) > 1}
    multi_insurer_groups = {k: v for k, v in insurer_groups.items() if len(v) > 1}
    logger.info(f"DEBUG: CBL groups with multiple rows: {len(multi_cbl_groups)}")
    logger.info(f"DEBUG: Insurer groups with multiple rows: {len(multi_insurer_groups)}")
    
    if group_matches:
        logger.info(f"DEBUG: First few group matches:")
        for i, match in enumerate(group_matches[:3]):
            logger.info(f"  {i+1}. {match['cbl_name'][:30]}... -> {match['insurer_name'][:30]}... (Score: {match['name_score']}%)")
    else:
        logger.info("DEBUG: No group matches found - checking why...")
        if multi_cbl_groups and multi_insurer_groups:
            logger.info(f"DEBUG: Sample CBL groups: {list(multi_cbl_groups.keys())[:3]}")
            logger.info(f"DEBUG: Sample insurer groups: {list(multi_insurer_groups.keys())[:3]}")
        else:
            logger.info(f"DEBUG: No multi-row groups found - CBL: {len(multi_cbl_groups)}, Insurer: {len(multi_insurer_groups)}")
    
    # Add group matches to potential matches
    for match in group_matches:
        potential_matches.append({
            'match_type': 'group',
            'cbl_indices': match['cbl_indices'],
            'insurer_indices': match['insurer_indices'],
            'match_reason': f'Name Group Match (CS: {match["name_score"]}%)',
            'cbl_total': match['cbl_total'],
            'insurer_total': match['insurer_total'],
            'difference': match['difference'],
            'name_score': match['name_score']
        })
    
    # Now process remaining unmatched records with traditional row-by-row and cumulative matching
    logger.info("=== Phase 1: Collecting individual matches ===")
    
    for i, row in cbl_df[cbl_df["match_status"].isin(["No Match", "Partial Match"])].iterrows():
        processed += 1
        if processed % 50 == 0:
            logger.info(f"Progress: {processed}/{total_records} records processed")

        add_pass(cbl_df, i, 3)

        cbl_name = str(row["ClientName"]).upper().strip() if pd.notna(row["ClientName"]) else ""
        cbl_amt = row["Amount_Clean"]   

        # Skip if CBL name is empty
        if not cbl_name:
            continue

        # Calculate name scores for all insurer rows at once
        name_scores = insurer_names.apply(lambda x: fuzz.partial_ratio(cbl_name, x) if x else 0)
        name_matches = name_scores[name_scores >= fuzzy_threshold]
        
        if name_matches.empty:
            continue

        # Get all matching rows and their amounts
        matching_rows = available_insurer.loc[name_matches.index]
        matching_amounts = insurer_amounts.loc[name_matches.index]
        
        # Track processed rows to avoid duplicates
        processed_rows = set()
        matched_indices = []
        
        # First: Try row-by-row matching with graduated confidence
        best_individual_match = None
        for idx, amt in zip(matching_rows.index, matching_amounts):
            if idx not in processed_rows and pd.notna(amt):
                match_type, difference, confidence = classify_amount_match(cbl_amt, amt, tolerance)
                
                if match_type in ["PERFECT_MATCH", "EXACT_MATCH"]:
                    matched_indices.append(idx)
                    processed_rows.add(idx)
                elif match_type == "CLOSE_MATCH" and best_individual_match is None:
                    # Store close match as backup
                    best_individual_match = {
                        'indices': [idx],
                        'confidence': confidence,
                        'difference': difference,
                        'type': 'close_individual'
                    }
        
        # Second: Try cumulative matching with remaining unprocessed rows
        unprocessed_indices = [idx for idx in name_matches.index if idx not in processed_rows]
        best_cumulative_match = None
        
        if unprocessed_indices:
            unprocessed_amounts = matching_amounts[unprocessed_indices]
            # Filter out NaN values before summing
            valid_amounts = unprocessed_amounts[unprocessed_amounts.notna()]

            if not valid_amounts.empty:
                total_unprocessed_amount = valid_amounts.sum()
                match_type, difference, confidence = classify_amount_match(cbl_amt, total_unprocessed_amount, tolerance)
                
                if match_type in ["PERFECT_MATCH", "EXACT_MATCH"]:
                    matched_indices.extend(unprocessed_indices)
                    processed_rows.update(unprocessed_indices)
                elif match_type == "CLOSE_MATCH" and not matched_indices:
                    # Store close cumulative match as backup if no exact individual matches
                    best_cumulative_match = {
                        'indices': unprocessed_indices,
                        'confidence': confidence,
                        'difference': difference,
                        'type': 'close_cumulative'
                    }
        
        # If no exact matches found, use the best close match
        if not matched_indices:
            if best_individual_match and (not best_cumulative_match or best_individual_match['difference'] <= best_cumulative_match['difference']):
                matched_indices = best_individual_match['indices']
                match_confidence = best_individual_match['confidence']
                match_difference = best_individual_match['difference']
                close_match_type = 'individual'
            elif best_cumulative_match:
                matched_indices = best_cumulative_match['indices']
                match_confidence = best_cumulative_match['confidence']
                match_difference = best_cumulative_match['difference']
                close_match_type = 'cumulative'
        
        # Add to potential matches
        if matched_indices and cbl_df.at[i, "match_status"] != "Exact Match":
            # Get the highest name score among the matched indices
            highest_name_score = name_scores[matched_indices].max()
            
            # Calculate total amount and determine confidence
            total_amount = sum(insurer_df.loc[matched_indices, "Amount_Clean_INSURER"])
            final_match_type, final_difference, final_confidence = classify_amount_match(cbl_amt, total_amount, tolerance)
            
            # Determine match reason based on how the match was found
            if len(matched_indices) == 1:
                base_reason = "Single Amount Match"
            else:
                base_reason = "Cumulative Amount Match"
            
            # Check if this was a close match that we accepted
            if 'match_confidence' in locals():
                # This was a close match
                if close_match_type == 'individual':
                    match_reason = f"Name Match (CS: {highest_name_score}%) + Close {base_reason} ({match_confidence} Confidence, Diff: ${match_difference:.2f})"
                else:
                    match_reason = f"Name Match (CS: {highest_name_score}%) + Close {base_reason} ({match_confidence} Confidence, Diff: ${match_difference:.2f})"
                confidence_to_use = match_confidence
                difference_to_use = match_difference
            else:
                # This was an exact match
                match_reason = f"Name Match (CS: {highest_name_score}%) + {base_reason} ({final_confidence} Confidence, Diff: ${final_difference:.2f})"
                confidence_to_use = final_confidence
                difference_to_use = final_difference
            
            potential_matches.append({
                'match_type': 'exact',
                'cbl_index': i,
                'insurer_indices': matched_indices,
                'match_reason': match_reason,
                'confidence_level': confidence_to_use,
                'amount_difference': difference_to_use,
                'total_amount': total_amount,
                'name_score': highest_name_score
            })
        else:
            # If no exact matches found, mark as partial match with all similar names
            # But only if they haven't been used for partial matches before
            available_partial_indices = [idx for idx in name_matches.index if idx not in partial_used_insurer]
            
            if available_partial_indices:
                # Get the highest name score among the available partial indices
                highest_name_score = name_scores[available_partial_indices].max()
                
                # Find the best partial match by amount difference
                best_partial_difference = None
                best_partial_confidence = None
                for idx in available_partial_indices:
                    amt = insurer_amounts.loc[idx]
                    if pd.notna(amt):
                        match_type, difference, confidence = classify_amount_match(cbl_amt, amt, tolerance)
                        if match_type in ["REVIEW_REQUIRED", "INVESTIGATION_REQUIRED"]:
                            if best_partial_difference is None or difference < best_partial_difference:
                                best_partial_difference = difference
                                best_partial_confidence = confidence
                
                if best_partial_confidence:
                    match_reason = f"Name Match (CS: {highest_name_score}%) ({best_partial_confidence} Confidence, Diff: ${best_partial_difference:.2f})"
                else:
                    match_reason = f"Name Match (CS: {highest_name_score}%) (Amount mismatch)"
                    best_partial_confidence = "Very Low"
                    best_partial_difference = abs(cbl_amt + insurer_amounts.loc[available_partial_indices[0]])
                
                potential_matches.append({
                    'match_type': 'partial',
                    'cbl_index': i,
                    'insurer_indices': available_partial_indices,
                    'match_reason': match_reason,
                    'confidence_level': best_partial_confidence,
                    'amount_difference': best_partial_difference,
                    'total_amount': None,
                    'name_score': highest_name_score
                })

    # Phase 2: Resolve conflicts and apply matches
    logger.info("\n=== Phase 2: Resolving conflicts and applying matches ===")
    
    # Sort potential matches: group matches first, then exact matches, then partial matches
    # Within each type, sort by number of insurer indices (larger groups get priority)
    potential_matches.sort(key=lambda x: (
        0 if x['match_type'] == 'group' else (1 if x['match_type'] == 'exact' else 2),
        -len(x['insurer_indices'])  # Negative for descending order (larger groups first)
    ))
    
    # Track used insurer indices
    used_insurer_indices = set()
    
    for match in potential_matches:
        match_type = match['match_type']
        insurer_indices = match['insurer_indices']
        
        # Check if any of the insurer indices are already used
        conflicting_indices = set(insurer_indices) & used_insurer_indices
        
        if conflicting_indices:
            if match_type == 'group':
                # For group matches, filter out conflicted indices and apply with available ones
                available_insurer_indices = [idx for idx in insurer_indices if idx not in used_insurer_indices]
                
                if available_insurer_indices:
                    logger.info(f"Pass 3 Group Match: Applying partial group match with {len(available_insurer_indices)}/{len(insurer_indices)} available insurer indices (conflicts: {conflicting_indices})")
                    
                    # Update the match with only available indices
                    match['insurer_indices'] = available_insurer_indices
                    # Recalculate totals with available indices
                    match['insurer_total'] = available_insurer.loc[available_insurer_indices, 'Amount_Clean_INSURER'].sum()
                    match['difference'] = match['cbl_total'] + match['insurer_total']
                else:
                    logger.info(f"Pass 3 Group Match: Skipping group match - all insurer indices already used (conflicts: {conflicting_indices})")
                    continue
            else:
                # Use helper function for conflict resolution (non-group matches)
                exact_added, partial_added = _handle_conflict_resolution(
                    cbl_df, available_insurer, match, used_insurer_indices, tolerance, 3, global_tracker
                )
                exact_matches += exact_added
                partial_matches += partial_added
        else:
            # Apply the match
            if match_type == 'group':
                # Update insurer_indices to reflect any filtering from conflict resolution
                insurer_indices = match['insurer_indices']
                logger.info(f"Applying group match: {len(match['cbl_indices'])} CBL rows vs {len(insurer_indices)} insurer rows (Total: {match['cbl_total']:.2f} + {match['insurer_total']:.2f} = {match['difference']:.2f})")
                
                # For group matches, we need to distribute the insurer indices among CBL rows
                # This prevents the same insurer rows from being assigned to multiple CBL rows
                cbl_indices = match['cbl_indices']
                
                # If we have more CBL rows than insurer rows, some CBL rows will share insurer rows
                # If we have more insurer rows than CBL rows, some insurer rows will be unused
                # We'll distribute them as evenly as possible
                
                # Create a mapping of CBL indices to their assigned insurer indices
                cbl_to_insurer_mapping = {}
                
                if len(cbl_indices) <= len(insurer_indices):
                    # More insurer rows than CBL rows - distribute insurer rows among CBL rows
                    for i, cbl_idx in enumerate(cbl_indices):
                        # Each CBL row gets one insurer row
                        cbl_to_insurer_mapping[cbl_idx] = [insurer_indices[i]]
                else:
                    # More CBL rows than insurer rows - distribute insurer rows as evenly as possible
                    # Ensure every CBL row gets at least one insurer index
                    insurer_per_cbl = len(insurer_indices) // len(cbl_indices)
                    remaining_insurers = len(insurer_indices) % len(cbl_indices)
                    
                    # If insurer_per_cbl is 0, we have more CBL rows than insurer rows
                    # This creates a problematic scenario where we can't properly distribute insurers
                    # without creating duplicates. We should skip this group match to avoid issues.
                    if insurer_per_cbl == 0:
                        # Skip this group match to avoid creating duplicates
                        logger.warning(f"Skipping group match with {len(cbl_indices)} CBL rows vs {len(insurer_indices)} insurer rows to avoid duplicates")
                        continue
                    else:
                        # Normal distribution
                        insurer_idx = 0
                        for i, cbl_idx in enumerate(cbl_indices):
                            # Calculate how many insurer rows this CBL row should get
                            num_insurers = insurer_per_cbl + (1 if i < remaining_insurers else 0)
                            
                            # Assign the insurer rows
                            assigned_insurers = insurer_indices[insurer_idx:insurer_idx + num_insurers]
                            cbl_to_insurer_mapping[cbl_idx] = assigned_insurers
                            insurer_idx += num_insurers
                
                # Apply the matches with proper distribution
                for cbl_idx in cbl_indices:
                    if cbl_df.at[cbl_idx, 'match_status'] in ['No Match', 'Partial Match']:
                        assigned_insurer_indices = cbl_to_insurer_mapping[cbl_idx]
                        assigned_insurer_total = available_insurer.loc[assigned_insurer_indices, "Amount_Clean_INSURER"].sum()
                        
                        cbl_df.at[cbl_idx, 'match_status'] = 'Exact Match'
                        cbl_df.at[cbl_idx, 'match_reason'] = match['match_reason']
                        cbl_df.at[cbl_idx, 'matched_insurer_indices'] = assigned_insurer_indices
                        cbl_df.at[cbl_idx, 'matched_amtdue_total'] = assigned_insurer_total
                        cbl_df.at[cbl_idx, 'match_resolved_in_pass'] = 3
                        cbl_df.at[cbl_idx, 'partial_candidates_indices'] = []
                        exact_matches += 1
                
                # Mark insurer indices as used in global tracker
                if global_tracker:
                    # For group matches, we need to mark each CBL-insurer pair individually
                    for cbl_idx in cbl_indices:
                        assigned_insurer_indices = cbl_to_insurer_mapping[cbl_idx]
                        success, _, _, affected = global_tracker.mark_exact_match(cbl_idx, assigned_insurer_indices, cbl_df)
                        if not success:
                            logger.error(f"Pass 3 Group Match: Failed to mark exact match for CBL {cbl_idx}")
                else:
                    used_insurer_indices.update(match['insurer_indices'])
                
            elif match_type == 'exact':
                exact_matches += _apply_exact_match(
                    cbl_df, match['cbl_index'], match['match_reason'], insurer_indices, 
                    match['total_amount'], [], 3, global_tracker,
                    confidence_level=match.get('confidence_level'),
                    amount_difference=match.get('amount_difference')
                )
                if not global_tracker:
                    used_insurer_indices.update(insurer_indices)
                
                # Note: CBL row updates are handled automatically by GlobalMatchTracker
                
            else:  # partial
                total_amount = available_insurer.loc[insurer_indices, "Amount_Clean_INSURER"].sum()
                partial_matches += _apply_partial_match(
                    cbl_df, match['cbl_index'], match['match_reason'], insurer_indices, total_amount, 3, global_tracker,
                    confidence_level=match.get('confidence_level'),
                    amount_difference=match.get('amount_difference')
                )
                if not global_tracker:
                    used_insurer_indices.update(insurer_indices)

    logger.info(f"✓ Pass 3 complete: {exact_matches} exact matches, {partial_matches} partial matches (including improved group matching)")
    return cbl_df
