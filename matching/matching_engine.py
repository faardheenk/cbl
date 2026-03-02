#!/usr/bin/env python3
"""
Core matching engine: shared classes and utility functions used across all passes.

Pass implementations are in separate modules:
  - pass1.py  (Pass 1: Placing Number matching)
  - pass2.py  (Pass 2: Policy Number matching)
  - pass3.py  (Pass 3: Name-based matching)
"""

import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Set
from fuzzywuzzy import fuzz
import re
from .utils import add_pass

logger = logging.getLogger(__name__)


# ============================================================
# Helper functions used by CompanyNameMatcher
# ============================================================

def _is_compound_name(name):
    """
    Check if a name is a compound name with multiple entities.

    Compound names contain multiple companies joined by &/OR, AND/OR, etc.
    Example: "COMPANY A LTD &/OR COMPANY B LTD &/OR COMPANY C LTD"

    Args:
        name: Company name string

    Returns:
        bool: True if compound name, False otherwise
    """
    if not name:
        return False

    # Check for common compound name patterns
    compound_indicators = ['&/OR', '&OR', 'AND/OR', 'ANDOR', '&/']
    return any(indicator in name.upper() for indicator in compound_indicators)


def _extract_primary_entity(name):
    """
    Extract the primary entity from a compound name.

    For compound names, extracts the FIRST entity before the first "&/OR".
    This prevents over-clustering by focusing on the primary company.

    Args:
        name: Company name (potentially compound)

    Returns:
        str: Primary entity name
    """
    if not name or not _is_compound_name(name):
        return name

    # Split on common separators
    separators = ['&/OR', '&OR', 'AND/OR', 'ANDOR', '&/']

    name_upper = name.upper()
    for separator in separators:
        if separator in name_upper:
            # Take only the first entity (primary company)
            parts = name.split(separator)
            if parts:
                primary = parts[0].strip()
                # Remove trailing "LTD", "LIMITED" etc from the primary entity for cleaner matching
                return primary

    return name


# ============================================================
# Classes
# ============================================================

class CompanyNameMatcher:
    """
    VITIRO LTD FIX: Intelligent company name matcher that prevents over-clustering
    of companies mentioned in financial agreements.

    This class extracts primary company names from financial relationships
    (e.g., "SPICE FINANCE LTD ON LEASE TO VITIRO LTD" -> "VITIRO LTD")
    and applies intelligent penalties to prevent over-clustering.

    OPTIMIZED: Uses caching and early exits for performance on large datasets.
    """

    FINANCIAL_PATTERNS = [
        (r'(.+?)\s+ON\s+LEASE\s+TO\s+(.+?)(?:\s*$|\s+&)', 'lease_to'),
        (r'(.+?)\s+ONLEASE\s+TO\s+(.+?)(?:\s*$|\s+&)', 'onlease_to'),  # No space variant
        (r'(.+?)\s+ON\s+FINANCE\s+TO\s+(.+?)(?:\s*$|\s+&)', 'finance_to'),
        (r'(.+?)\s+ONFINANCE\s+TO\s+(.+?)(?:\s*$|\s+&)', 'onfinance_to'),  # No space variant
        (r'(.+?)\s+ON\s+FINANCE\s+LEASE\s+TO\s+(.+?)(?:\s*$|\s+&)', 'finance_lease'),
        (r'(.+?)\s+ON\s+FINANCIAL\s+LEASE\s+TO\s+(.+?)(?:\s*$|\s+&)', 'financial_lease'),
        (r'(.+?)\s+ON\s+\(FINANCE\)\s+LEASE\s+TO\s+(.+?)(?:\s*$|\s+&)', 'finance_lease_paren'),
        (r'(.+?)\s+-\s+\([^)]*FINANCE\s+LEASE[^)]*\)$', 'finance_lease_suffix'),
    ]

    def __init__(self, primary_penalty: float = 0.3, exact_match_boost: float = 2.5):
        self.primary_penalty = primary_penalty
        self.exact_match_boost = exact_match_boost
        # Cache for primary company extraction to avoid repeated regex matching
        self._primary_cache = {}

    def extract_primary_company(self, name: str) -> Tuple[str, str]:
        """Extract primary company from complex company name strings (with caching)."""
        if not name or pd.isna(name):
            return "", "unknown"

        name_upper = str(name).strip().upper()

        # Check cache first - fast path
        if name_upper in self._primary_cache:
            return self._primary_cache[name_upper]

        # COMPOUND NAME HANDLING: Extract primary entity from compound names first
        if _is_compound_name(name_upper):
            primary_entity = _extract_primary_entity(name_upper)
            result = (primary_entity, "compound")
            self._primary_cache[name_upper] = result
            return result

        # Quick check: if no financial keywords, skip regex entirely
        has_financial_keyword = any(keyword in name_upper for keyword in
                                   ['ON LEASE', 'ONLEASE', 'ON FINANCE', 'ONFINANCE',
                                    'FINANCE LEASE', 'FINANCIAL LEASE',
                                    '(FINANCE)', 'ON (FINANCE)'])

        if not has_financial_keyword:
            result = (name_upper, "direct")
            self._primary_cache[name_upper] = result
            return result

        # Only do regex matching if financial keywords found
        for pattern, relationship_type in self.FINANCIAL_PATTERNS:
            match = re.search(pattern, name_upper, re.IGNORECASE)
            if match:
                if match.lastindex and match.lastindex >= 2:
                    primary = match.group(2).strip()
                    result = (primary, relationship_type)
                    self._primary_cache[name_upper] = result
                    return result
                elif match.lastindex == 1:
                    primary = match.group(1).strip()
                    result = (primary, relationship_type)
                    self._primary_cache[name_upper] = result
                    return result

        result = (name_upper, "direct")
        self._primary_cache[name_upper] = result
        return result

    def calculate_intelligent_similarity(self, name1: str, name2: str) -> float:
        """Calculate intelligent similarity with financial relationship awareness (optimized)."""
        if not name1 or not name2 or pd.isna(name1) or pd.isna(name2):
            return 0.0

        name1_upper = str(name1).strip().upper()
        name2_upper = str(name2).strip().upper()

        # OPTIMIZATION: Quick exit for identical names
        if name1_upper == name2_upper:
            return 250.0  # Boosted exact match

        # OPTIMIZATION: Quick exit for names with no common characters
        # This catches obviously different companies early
        if not set(name1_upper) & set(name2_upper):
            return 0.0

        # OPTIMIZATION: Quick base similarity check before expensive operations
        base_similarity = fuzz.token_set_ratio(name1_upper, name2_upper)

        # OPTIMIZATION: Early exit for very low similarity - not worth further checks
        if base_similarity < 50:
            return base_similarity

        # Only do primary company extraction if base similarity is promising
        # Pass ORIGINAL names (not uppercase) to extract_primary_company for proper extraction
        primary1, rel_type1 = self.extract_primary_company(name1)
        primary2, rel_type2 = self.extract_primary_company(name2)

        # VITIRO LTD FIX: If both names have financial relationships to different companies,
        # apply penalty to prevent over-clustering
        if rel_type1 != "direct" and rel_type2 != "direct":
            if primary1 != primary2:
                base_similarity *= self.primary_penalty

        primary_similarity = fuzz.token_set_ratio(primary1, primary2)

        # Boost exact matches
        if primary1 == primary2:
            primary_similarity *= self.exact_match_boost

        return max(base_similarity, primary_similarity)


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
                                        cbl_df.at[affected_cbl, 'Amount Difference'] = None
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

    def can_use_for_partial(self, indices, allow_sharing=True):
        """
        Check if insurer indices can be used for partial match.

        Args:
            indices: Single index or list of indices
            allow_sharing: If True, partial matches can reuse indices from other partial matches (Phase 3).
                          If False, partial matches are exclusive - no sharing allowed (Phase 1 & 2).

        Returns:
            tuple: (can_use_all, available_indices, unavailable_indices)
        """
        indices_set = set(indices) if isinstance(indices, (list, set)) else {indices}

        if allow_sharing:
            # Phase 3 behavior: Can reuse partial indices, but not exact or matrix
            unavailable = indices_set & (self.matrix_used_insurer | self.exact_used_insurer)
        else:
            # Phase 1 & 2 behavior: Cannot reuse ANY already-used indices (exclusive 1:1)
            unavailable = indices_set & (self.matrix_used_insurer | self.exact_used_insurer | self.partial_used_insurer)

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


# ============================================================
# Shared utility functions
# ============================================================

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


# ============================================================
# Match application functions
# ============================================================

def _apply_no_match(cbl_df, cbl_index, match_reason):
    """Apply a no match status to a CBL record."""
    cbl_df.at[cbl_index, "match_status"] = "No Match"
    cbl_df.at[cbl_index, "match_reason"] = match_reason
    cbl_df.at[cbl_index, "matched_insurer_indices"] = []
    cbl_df.at[cbl_index, "matched_amtdue_total"] = None
    cbl_df.at[cbl_index, "Amount Difference"] = None
    cbl_df.at[cbl_index, "partial_candidates_indices"] = []


def _apply_exact_match(cbl_df, cbl_index, match_reason, insurer_indices, total_amount, fallback_indices, pass_number, global_tracker=None, confidence_level=None, amount_difference=None, skip_individual_conflict_check=False):
    """Apply an exact match to a CBL record."""
    # Validate indices with comprehensive global tracker if provided
    if global_tracker and not skip_individual_conflict_check:
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
    elif global_tracker and skip_individual_conflict_check:
        # For Pass 3 cluster matching - skip individual conflict checking
        # Mark the CBL-insurer exact match with automatic CBL DataFrame cleanup
        success, final_indices, match_conflicts, affected_cbl_rows = global_tracker.mark_exact_match(
            cbl_index, insurer_indices, cbl_df
        )

        if not success:
            logger.error(f"Pass {pass_number} CBL {cbl_index}: Failed to mark exact match due to conflicts: {match_conflicts}")
            _apply_no_match(cbl_df, cbl_index, f"{match_reason} (Match conflicts)")
            return 0

        insurer_indices = final_indices

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
        cbl_df.at[cbl_index, "Amount Difference"] = round(amount_difference, 2)

    return 1  # Return count for exact matches


def _apply_cluster_exact_match(cbl_df, cbl_index, match_reason, insurer_indices, total_amount, pass_number, global_tracker=None, confidence_level=None, amount_difference=None):
    """Apply an exact match for Pass 3 cluster matching, allowing multiple CBL rows to share insurer indices."""
    # For cluster matching, we allow multiple CBL rows to share the same insurer indices
    # This is different from regular exact matching which is 1:1

    # Mark insurer indices as used in exact matches (but allow sharing)
    if global_tracker:
        indices_set = set(insurer_indices) if isinstance(insurer_indices, (list, set)) else {insurer_indices}

        # Add to exact used insurer set
        global_tracker.exact_used_insurer.update(indices_set)

        # Track this CBL's exact match
        global_tracker.cbl_exact_matches[cbl_index] = list(indices_set)

        # For cluster matching, we allow multiple CBL rows to reference the same insurer indices
        # So we don't update the reverse mapping (insurer_to_cbl_exact) to allow sharing

        logger.debug(f"Pass {pass_number} CBL {cbl_index}: Cluster exact match recorded with insurer indices: {indices_set}")

    # Apply the match to the CBL DataFrame
    cbl_df.at[cbl_index, "match_status"] = "Exact Match"
    cbl_df.at[cbl_index, "match_reason"] = match_reason
    cbl_df.at[cbl_index, "matched_insurer_indices"] = insurer_indices
    cbl_df.at[cbl_index, "matched_amtdue_total"] = total_amount
    cbl_df.at[cbl_index, "partial_candidates_indices"] = []
    cbl_df.at[cbl_index, "match_resolved_in_pass"] = pass_number

    # Add confidence and difference information
    if confidence_level is not None:
        cbl_df.at[cbl_index, "match_confidence"] = confidence_level
    if amount_difference is not None:
        cbl_df.at[cbl_index, "Amount Difference"] = round(amount_difference, 2)


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
        cbl_df.at[cbl_index, "Amount Difference"] = round(amount_difference, 2)

    return 1  # Return count for partial matches


def deduplicate_partial_matches(cbl_df, overlap_threshold=0.8, group_by_name=True):
    """
    DEPRECATED: This function has been replaced by Pass 3 Phase 3 name grouping.

    All grouping now happens in Pass 3, and the output handler prevents data duplication
    using set() for unique insurer indices.

    Args:
        cbl_df: CBL dataframe with match results
        overlap_threshold: Unused (kept for compatibility)
        group_by_name: Unused (kept for compatibility)

    Returns:
        cbl_df: Unchanged dataframe
    """
    logger.warning("⚠️ deduplicate_partial_matches() is DEPRECATED - use Pass 3 name grouping instead")
    return cbl_df


def _handle_conflict_resolution(cbl_df, insurer_df, match, used_insurer_indices, tolerance, pass_number, global_tracker, fallback_rows=None):
    """
    Handle conflict resolution with fallback logic.

    Args:
        cbl_df: CBL dataframe
        insurer_df: Insurer dataframe (or fallback_rows for Pass 2)
        match: Match dictionary with conflict
        used_insurer_indices: Unused parameter (kept for compatibility)
        tolerance: Tolerance for amount matching
        pass_number: Which pass is calling this function
        global_tracker: GlobalMatchTracker instance for consistent tracking
        fallback_rows: Optional fallback rows dataframe (for Pass 2)

    Returns:
        tuple: (exact_matches_added, partial_matches_added)
    """
    cbl_index = match['cbl_index']
    match_type = match['match_type']
    insurer_indices = match['insurer_indices']

    logger.info(f"Pass {pass_number} Record {cbl_index}: Handling conflicts for {match_type} match")

    # Use GlobalMatchTracker for conflict resolution
    # Check availability based on match type
    # For conflict resolution, use exclusive mode (allow_sharing=False) to prevent duplication
    if match_type in ['exact', 'combination']:
        can_use_all, available_indices, unavailable_indices = global_tracker.can_use_for_exact(insurer_indices)
    else:  # partial
        can_use_all, available_indices, unavailable_indices = global_tracker.can_use_for_partial(insurer_indices, allow_sharing=False)

    if unavailable_indices:
        logger.info(f"Record {cbl_index}: Some indices unavailable: {unavailable_indices}")

    # Try fallback indices if original indices are not available
    if not available_indices and 'fallback_indices' in match and match['fallback_indices']:
        logger.info(f"Record {cbl_index}: Trying fallback indices: {match['fallback_indices']}")
        if match_type in ['exact', 'combination']:
            can_use_fallback, available_indices, _ = global_tracker.can_use_for_exact(match['fallback_indices'])
        else:
            can_use_fallback, available_indices, _ = global_tracker.can_use_for_partial(match['fallback_indices'], allow_sharing=False)

        if available_indices:
            logger.info(f"Record {cbl_index}: Using available fallback indices: {available_indices}")

    if not available_indices:
        # All potential indices are already used - mark as No Match
        logger.info(f"Record {cbl_index}: All potential indices used - marking as No Match")
        _apply_no_match(cbl_df, cbl_index, match['match_reason'])
        return 0, 0

    # Calculate amounts using the appropriate dataframe
    data_source = fallback_rows if fallback_rows is not None else insurer_df
    cbl_amount = cbl_df.at[cbl_index, "ProcessedAmount_Clean"]
    available_amounts = data_source.loc[available_indices, "ProcessedAmount_Clean_INSURER"]
    total_available_amount = available_amounts.sum()

    # Check if fallback indices create a perfect match
    if -tolerance <= (cbl_amount + total_available_amount) <= tolerance:
        # Upgrade to exact match!
        logger.info(f"Record {cbl_index}: Fallback indices upgraded to Exact Match!")
        return _apply_exact_match(cbl_df, cbl_index, f"{match['match_reason']} (Fallback Match)",
                                 available_indices, total_available_amount, [], pass_number, global_tracker), 0
    else:
        # Don't create partial match - let Phase 3 name grouping handle amount mismatches
        logger.info(f"Record {cbl_index}: Fallback indices don't match amount - marking as No Match (will be handled by Phase 3)")
        _apply_no_match(cbl_df, cbl_index, f"{match['match_reason']} (Amount mismatch)")
        return 0, 0
