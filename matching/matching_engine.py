#!/usr/bin/env python3

import pandas as pd
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from fuzzywuzzy import fuzz
import re
from itertools import combinations
from .utils import add_pass, extract_policy_tokens

logger = logging.getLogger(__name__)


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
        (r'(.+?)\s+ON\s+FINANCE\s+TO\s+(.+?)(?:\s*$|\s+&)', 'finance_to'),
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
                                   ['ON LEASE', 'ON FINANCE', 'FINANCE LEASE', 'FINANCIAL LEASE', 
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


def _has_sufficient_word_overlap(name1, name2, min_common_words=2):
    """
    Check if two names have sufficient meaningful word overlap to be clustered.
    
    This prevents over-clustering based on:
    - Single common word: "SUN LTD" vs "WOLMAR SUN HOTELS LTD" → NO MATCH
    - Proper subset: "SUN LTD" vs "SUN MARINE LTD" → NO MATCH
    - Partial overlap: "SUN RESORTS" vs "SUN HOTELS" → NO MATCH (only 50% overlap)
    
    Allows matching when:
    - High overlap: "ACME LTD" vs "ACME HOLDINGS LTD" → MATCH (100% overlap)
    - Multiple common words: "ABC XYZ LTD" vs "ABC XYZ HOLDINGS" → MATCH
    
    Args:
        name1: First company name
        name2: Second company name
        min_common_words: Minimum number of meaningful common words required (default: 2)
        
    Returns:
        tuple: (has_sufficient_overlap, common_words_count, common_words)
    """
    # Common business suffixes and words to exclude from meaningful word matching
    EXCLUDE_WORDS = {
        # Legal entity suffixes
        'LTD', 'LIMITED', 'INC', 'INCORPORATED', 'COMPANY', 'CO', 'CORP', 'CORPORATION',
        'LLC', 'PLC', 'SA', 'AG', 'GMBH', 'NV', 'BV', 'SPA', 'SRL', 'LTDA',
        # Common words
        'THE', 'AND', 'OR', 'OF', 'IN', 'AT', 'TO', 'FOR', 'WITH', 'ON',
        # Geographic and organizational descriptors
        'MAURITIUS', 'HOLDINGS', 'GROUP', 'INTERNATIONAL', 'GLOBAL',
        # Generic business activity descriptors (to prevent false clustering)
        'SERVICES', 'SERVICE', 'MANAGEMENT', 'CONSULTING', 'TRADING', 'CORPORATE',
        'FUND', 'FUNDS', 'INVESTMENT', 'INVESTMENTS', 'FINANCE', 'FINANCIAL',
        'BUSINESS', 'ENTERPRISES', 'SOLUTIONS', 'TRUST', 'TRUSTEES',
        'ADVISORS', 'ADVISORY', 'CAPITAL', 'PARTNERS', 'ASSOCIATES'
    }
    
    # Extract words from both names (remove parentheses and their content first)
    name1_cleaned = re.sub(r'\([^)]*\)', '', name1.upper()).strip()
    name2_cleaned = re.sub(r'\([^)]*\)', '', name2.upper()).strip()
    
    # Remove punctuation and split into words
    name1_cleaned = re.sub(r'[^\w\s]', ' ', name1_cleaned)  # Replace punctuation with spaces
    name2_cleaned = re.sub(r'[^\w\s]', ' ', name2_cleaned)
    
    # Split and filter out empty strings and excluded words
    words1 = set(w for w in name1_cleaned.split() if w) - EXCLUDE_WORDS
    words2 = set(w for w in name2_cleaned.split() if w) - EXCLUDE_WORDS
    
    # Find common meaningful words
    common_words = words1 & words2
    
    # Check if one name is a proper subset of the other
    # e.g., "SUN" (subset) vs "SUN MARINE" (superset)
    is_subset = (words1 < words2) or (words2 < words1)  # Proper subset (not equal)
    
    # Special case handling:
    # 1. If one name is a proper subset of the other, they should NOT match
    #    Example: "SUN LTD" should NOT match "SUN MARINE LTD"
    # 2. If BOTH names are identical after filtering, they should match
    # 3. If BOTH names are very short (1-2 words) AND equal, allow match
    
    if is_subset:
        # One name is a proper subset - require ALL words to match (subset case should fail)
        # This prevents "SUN" from matching "SUN MARINE"
        has_sufficient = (len(words1) == len(words2)) and (len(common_words) >= min_common_words)
    elif len(words1) <= 2 and len(words2) <= 2:
        # Both short AND not a subset relationship
        # Require that common words represent majority of BOTH names
        # Example: "SUN RESORTS" vs "SUN HOTELS" → both have "SUN", but only 50% overlap → fail
        # Example: "ACME LTD" vs "ACME HOLDINGS" → both have "ACME" (100% of first) → pass
        overlap_pct_1 = len(common_words) / len(words1) if len(words1) > 0 else 0
        overlap_pct_2 = len(common_words) / len(words2) if len(words2) > 0 else 0
        
        # Require at least 80% overlap in BOTH names
        has_sufficient = (overlap_pct_1 >= 0.8) and (overlap_pct_2 >= 0.8)
    else:
        # At least one name is longer than 2 words - require full min_common_words
        min_required = min_common_words
        has_sufficient = len(common_words) >= min_required
    
    return has_sufficient, len(common_words), common_words


def _build_fuzzy_name_clusters(df, name_column, fuzzy_threshold=90, prefix=""):
    """
    Build clusters of similar names using fuzzy matching.
    
    This groups records where names are similar (e.g., "ABC Ltd", "ABC Limited", "ABC (Mauritius) Ltd")
    into a single cluster, allowing partial matching across name variations.
    
    COMPOUND NAME HANDLING:
    - Detects compound names with "&/OR" patterns
    - Extracts only the PRIMARY entity (first company) for clustering
    - Prevents over-clustering of unrelated companies
    
    WORD OVERLAP VALIDATION:
    - Requires at least 2 meaningful words to match (not just common suffixes)
    - Prevents clustering of unrelated companies that share only one common word
    - Example: "SUN LTD" and "WOLMAR SUN HOTELS LTD" won't cluster (only "SUN" is common)
    
    Args:
        df: DataFrame to cluster
        name_column: Name of the column containing names to cluster
        fuzzy_threshold: Minimum similarity score (0-100) to group names together
        prefix: Prefix for logging (e.g., "CBL" or "INSURER")
        
    Returns:
        dict: {representative_name: [list of indices]} - clustered records
    """
    logger.info(f"\n=== Building Fuzzy Name Clusters for {prefix} ===")
    logger.info(f"Records to cluster: {len(df)}, Threshold: {fuzzy_threshold}%")
    
    if df.empty:
        logger.info("No records to cluster")
        return {}
    
    # Extract and normalize names - WITH COMPOUND NAME AND FINANCIAL RELATIONSHIP DETECTION
    names_with_indices = []
    compound_count = 0
    financial_relationship_count = 0
    
    # Use CompanyNameMatcher for consistent primary company extraction
    temp_matcher = CompanyNameMatcher(primary_penalty=0.3, exact_match_boost=2.5)
    
    for idx, row in df.iterrows():
        name = str(row.get(name_column, '')).upper().strip()
        if name and name != 'NAN':
            # Extract primary company using the same logic as similarity calculation
            # This handles both compound names (&/OR) AND financial relationships (ON LEASE TO)
            primary_entity, rel_type = temp_matcher.extract_primary_company(name)
            
            # Track extraction statistics
            if rel_type == "compound":
                compound_count += 1
                logger.debug(f"Compound name detected at {idx}: '{name[:80]}...' -> Primary: '{primary_entity}'")
            elif rel_type != "direct":
                financial_relationship_count += 1
                logger.debug(f"Financial relationship detected at {idx}: '{name[:80]}...' -> Primary: '{primary_entity}'")
            
            # Always use the primary entity for clustering (not the full name)
            names_with_indices.append((idx, primary_entity))
    
    if not names_with_indices:
        logger.info("No valid names found for clustering")
        return {}
    
    logger.info(f"Found {len(names_with_indices)} valid names to cluster")
    if compound_count > 0:
        logger.info(f"  ⚠️ Detected {compound_count} compound names - extracted primary entities to prevent over-clustering")
    if financial_relationship_count > 0:
        logger.info(f"  ⚠️ Detected {financial_relationship_count} financial relationships - extracted primary entities to prevent over-clustering")
    
    # Union-Find data structure for clustering
    parent = {idx: idx for idx, _ in names_with_indices}
    
    def find(x):
        """Find root parent with path compression."""
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        """Union two sets."""
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            parent[root_y] = root_x
    
    # Compare all pairs of names and union similar ones
    comparisons = 0
    unions_performed = 0
    word_overlap_rejections = 0
    
    # VITIRO LTD FIX: Use intelligent company name matcher
    # Create matcher ONCE to leverage caching across all comparisons
    matcher = CompanyNameMatcher(primary_penalty=0.3, exact_match_boost=2.5)
    
    for i, (idx1, name1) in enumerate(names_with_indices):
        for idx2, name2 in names_with_indices[i+1:]:
            comparisons += 1
            
            # VITIRO LTD FIX: Use intelligent similarity calculation
            # This prevents over-clustering of companies with financial relationships
            similarity = matcher.calculate_intelligent_similarity(name1, name2)
            
            # Stricter clustering to prevent false positives based on common words
            if similarity >= fuzzy_threshold:
                # NEW VALIDATION: Check for sufficient meaningful word overlap
                # This prevents clustering of unrelated companies that share only one common word
                # Example: "SUN LTD" and "WOLMAR SUN HOTELS LTD" won't cluster (only "SUN" is common)
                has_overlap, common_count, common_words = _has_sufficient_word_overlap(name1, name2, min_common_words=2)
                
                if not has_overlap:
                    word_overlap_rejections += 1
                    logger.debug(f"Rejected clustering due to insufficient word overlap: '{name1[:50]}' vs '{name2[:50]}' (only {common_count} common words: {common_words})")
                    continue
                
                # Additional validation: check core name similarity (without common suffixes)
                name1_core = name1.replace('LTD', '').replace('LIMITED', '').replace('INC', '').replace('COMPANY', '').replace('CORPORATION', '').strip()
                name2_core = name2.replace('LTD', '').replace('LIMITED', '').replace('INC', '').replace('COMPANY', '').replace('CORPORATION', '').strip()
                
                # Require minimum meaningful length after cleaning
                if len(name1_core) < 2 or len(name2_core) < 2:
                    # Names too short after removing suffixes - skip
                    continue
                
                # Calculate core similarity (without common words)
                core_similarity = fuzz.token_set_ratio(name1_core, name2_core)
                
                # Require BOTH overall similarity AND core similarity to prevent false positives
                # Relaxed threshold to allow more legitimate matches while still preventing common-word-only matches
                if core_similarity >= (fuzzy_threshold - 5):  # More lenient core threshold (75% for 90% overall)
                    union(idx1, idx2)
                    unions_performed += 1
                    logger.debug(f"Clustered: '{name1[:50]}' + '{name2[:50]}' (similarity: {similarity}%, common words: {common_words})")
                else:
                    # Log rejected clustering for debugging
                    if similarity >= 95:  # Only log high-similarity rejections
                        logger.debug(f"Rejected clustering despite {similarity}% similarity: '{name1[:40]}' vs '{name2[:40]}' (core similarity: {core_similarity}%)")
    
    logger.info(f"Performed {comparisons} comparisons, created {unions_performed} unions")
    if word_overlap_rejections > 0:
        logger.info(f"  ℹ️ Rejected {word_overlap_rejections} potential clusters due to insufficient word overlap (prevents over-clustering on single common words)")
    
    # Group indices by their root parent (cluster representative)
    clusters_by_root = {}
    name_by_root = {}
    
    for idx, name in names_with_indices:
        root = find(idx)
        if root not in clusters_by_root:
            clusters_by_root[root] = []
            name_by_root[root] = name  # Use the first name as representative
        clusters_by_root[root].append(idx)
    
    # Convert to final format: {representative_name: [indices]}
    name_clusters = {}
    for root, indices in clusters_by_root.items():
        representative_name = name_by_root[root]
        name_clusters[representative_name] = indices
    
    # Log cluster summary
    logger.info(f"\n📊 Clustering Results for {prefix}:")
    logger.info(f"  - Total clusters created: {len(name_clusters)}")
    logger.info(f"  - Single-record clusters: {sum(1 for v in name_clusters.values() if len(v) == 1)}")
    logger.info(f"  - Multi-record clusters: {sum(1 for v in name_clusters.values() if len(v) > 1)}")
    
    
    # Log details of multi-record clusters
    multi_clusters = {k: v for k, v in name_clusters.items() if len(v) > 1}
    if multi_clusters:
        logger.info(f"\n  Multi-record clusters details:")
        for i, (cluster_name, indices) in enumerate(list(multi_clusters.items())[:5], 1):
            logger.info(f"    {i}. '{cluster_name[:60]}...' - {len(indices)} records: {indices}")
        
        if len(multi_clusters) > 5:
            logger.info(f"    ... and {len(multi_clusters) - 5} more multi-record clusters")
    
    return name_clusters


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
        cbl_df.at[cbl_index, "amount_difference"] = amount_difference
        
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
        cbl_df.at[cbl_index, "amount_difference"] = amount_difference


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


def pass1(cbl_df, insurer_df, tolerance=100, global_tracker=None):
    """Pass 1: Matching by Placing Number and ProcessedAmount."""
    logger.info("\n=== Pass 1: Matching by Placing Number and Amount ===")
    total_records = len(cbl_df)
    exact_matches = 0
    partial_matches = 0
    
    logger.info(f"Pass 1 starting with global tracker: {global_tracker.get_usage_summary()}")

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
        amt1 = row["ProcessedAmount_Clean"]
        
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
            
            for idx, amt in zip(insurer_matches.index.tolist(), insurer_matches["ProcessedAmount_Clean_INSURER"].tolist()):
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
                        # Auto-approve exact matches only - no close matches
                        # Include overlap info if this came from substring matching
                        overlap_info = overlap_details.get(j, "")
                        overlap_suffix = f" ({overlap_info})" if overlap_info else ""
                        
                        best_match = {
                            'indices': [j], 
                            'type': 'exact',
                            'confidence': confidence,
                            'difference': difference,
                            'reason': f'Placing Number{overlap_suffix} + Single Amount Match ({confidence} Confidence, Diff: Rs{difference:.2f})'
                        }
                        break
                    # REMOVED: CLOSE_MATCH logic - let Phase 3 name grouping handle amount mismatches
            
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
                            
                            combination_match_reason = f'Placing Number{overlap_suffix} + Cumulative Amount Match ({confidence} Confidence, Diff: Rs{difference:.2f})'
                            logger.info(f"Record {i}: Found combination match with {r} items, total: {total_amount}, confidence: {confidence}")
                            break
                        # REMOVED: CLOSE_MATCH logic - let Phase 3 name grouping handle amount mismatches
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
            # Combination matches are always exact (no close matches anymore)
            potential_matches.append({
                'cbl_index': i,
                'match_type': 'combination',
                'insurer_indices': combination_match_indices,
                'match_reason': combination_match_reason,
                'confidence_level': combination_match_confidence,
                'amount_difference': combination_match_difference,
                'fallback_indices': [idx for idx in insurer_indices if idx not in combination_match_indices]
            })
        # REMOVED: Partial match logic for REVIEW_REQUIRED/INVESTIGATION_REQUIRED
        # Let Phase 3 name grouping handle all amount mismatches

    # Phase 2: Resolve conflicts by prioritizing combination matches
    logger.info("\n=== Phase 2: Resolving conflicts and applying matches ===")
    
    # Sort potential matches: exact matches first, then combinations, then partial matches
    # Within each type, sort by number of insurer indices (larger combinations get priority)
    potential_matches.sort(key=lambda x: (
        0 if x['match_type'] == 'exact' else (1 if x['match_type'] == 'combination' else 2),
        -len(x['insurer_indices'])  # Negative for descending order (larger combinations first)
    ))
    
    # Use GlobalMatchTracker for consistent tracking
    for match in potential_matches:
        cbl_index = match['cbl_index']
        match_type = match['match_type']
        insurer_indices = match['insurer_indices']
        
        # Use GlobalMatchTracker for conflict detection
        can_claim_all, available_indices, conflicts = global_tracker.can_cbl_claim_insurer(
            cbl_index, insurer_indices, 'exact' if match_type in ['exact', 'combination'] else 'partial'
        )
        
        if conflicts:
            # Use helper function for conflict resolution
            exact_added, partial_added = _handle_conflict_resolution(
                cbl_df, insurer_df, match, None, tolerance, 1, global_tracker
            )
            exact_matches += exact_added
            partial_matches += partial_added
        else:
            # Apply the match using helper functions
            # Pass 1 only creates exact matches (no partial matches)
            if match_type in ['exact', 'combination']:
                total_amount = sum(insurer_df.loc[insurer_indices, "ProcessedAmount_Clean_INSURER"])
                exact_matches += _apply_exact_match(
                    cbl_df, cbl_index, match['match_reason'], insurer_indices, 
                    total_amount, match.get('fallback_indices', []), 1, global_tracker,
                    confidence_level=match.get('confidence_level'),
                    amount_difference=match.get('amount_difference')
                )
            else:
                # This should never happen as Pass 1 only creates exact matches
                logger.error(f"Pass 1: Unexpected match_type '{match_type}' - skipping")
                continue

    logger.info(f"✓ Pass 1 complete: {exact_matches} exact matches, {partial_matches} partial matches")
    return cbl_df


def pass2(cbl_df, insurer_df, tolerance=100, global_tracker=None):
    """Pass 2: Matching by Policy Number and Amount (Name matching removed)."""
    logger.info("\n=== Pass 2: Matching by Policy Number and Amount ===")
    total_records = len(cbl_df[cbl_df["match_status"].isin(["No Match", "Partial Match"])])
    exact_matches = 0
    partial_matches = 0
    processed = 0
    
    logger.info(f"Pass 2 starting with global tracker: {global_tracker.get_usage_summary()}")

    # Use GlobalMatchTracker for consistent filtering (same logic as Pass 3)
    # For Pass 2, we exclude exact and matrix matches but allow partial matches to be upgraded
    already_matched_insurer = global_tracker.exact_used_insurer | global_tracker.matrix_used_insurer
    fallback_rows = insurer_df[~insurer_df.index.isin(already_matched_insurer)].copy()
    logger.info(f"Pass 2: Using global tracker - excluding {len(already_matched_insurer)} exact/matrix used insurer rows")
    logger.info(f"Pass 2: Available insurer rows for policy number matching: {len(fallback_rows)}")

    # Phase 1: Collect all potential matches without applying them yet
    potential_matches = []
    
    for i, row in cbl_df[cbl_df["match_status"].isin(["No Match", "Partial Match"])].iterrows():
        processed += 1
        if processed % 50 == 0:
            logger.info(f"Progress: {processed}/{total_records} records processed")

        add_pass(cbl_df, i, 2)
        tokens = extract_policy_tokens(row["PolicyNo_Clean"])
        cbl_amt = row["ProcessedAmount_Clean"]

        matched_indices = []

        for j, insurer_row in fallback_rows.iterrows():
            # Collect all available policy values for this insurer row
            insurer_policy_values = []
            
            # Add PolicyNo_1 if available
            if pd.notna(insurer_row.get("PolicyNo_Clean_INSURER")) and str(insurer_row["PolicyNo_Clean_INSURER"]).strip():
                insurer_policy_values.append(str(insurer_row["PolicyNo_Clean_INSURER"]).strip())
            
            # Add PolicyNo_2 if available
            if "PolicyNo_2_Clean_INSURER" in insurer_row.index:
                if pd.notna(insurer_row["PolicyNo_2_Clean_INSURER"]) and str(insurer_row["PolicyNo_2_Clean_INSURER"]).strip():
                    insurer_policy_values.append(str(insurer_row["PolicyNo_2_Clean_INSURER"]).strip())
            
            # Simple policy matching - check if any insurer policy is in CBL tokens
            policy_match = False
            for insurer_policy in insurer_policy_values:
                if insurer_policy in tokens:
                    policy_match = True
                    logger.debug(f"Pass 2 CBL {i}: Policy match found - '{insurer_policy}' in tokens {tokens}")
                    break

            if policy_match:
                matched_indices.append(j)

        total_amt = fallback_rows.loc[matched_indices, "ProcessedAmount_Clean_INSURER"].sum()

        if matched_indices:
            # Classify the amount match using graduated confidence levels
            match_type, difference, confidence = classify_amount_match(cbl_amt, total_amt, tolerance)
            
            # Determine if it's single or cumulative amount match
            if len(matched_indices) == 1:
                amount_match_type = 'Single Amount Match'
            else:
                amount_match_type = 'Cumulative Amount Match'
            
            # Simple policy match info
            policy_strategy_info = " (Policy Match)"
            
            if match_type in ["PERFECT_MATCH", "EXACT_MATCH"]:
                # Exact match found
                match_reason = f'Policy Number{policy_strategy_info} + {amount_match_type} ({confidence} Confidence, Diff: Rs{difference:.2f})'
                potential_matches.append({
                    'cbl_index': i,
                    'match_type': 'exact',
                    'insurer_indices': matched_indices,
                    'match_reason': match_reason,
                    'confidence_level': confidence,
                    'amount_difference': difference,
                    'total_amount': total_amt
                })
            # REMOVED: CLOSE_MATCH and REVIEW_REQUIRED/INVESTIGATION_REQUIRED logic
            # Let Phase 3 name grouping handle all amount mismatches

    # Phase 2: Resolve conflicts and apply matches
    logger.info("\n=== Pass 2 Phase 2: Resolving conflicts and applying matches ===")
    
    # Sort potential matches: exact matches first, then partial matches
    # Within each type, sort by number of insurer indices (larger combinations get priority)
    potential_matches.sort(key=lambda x: (
        0 if x['match_type'] == 'exact' else 1,
        -len(x['insurer_indices'])  # Negative for descending order (larger combinations first)
    ))
    
    # Use GlobalMatchTracker for consistent tracking
    for match in potential_matches:
        cbl_index = match['cbl_index']
        match_type = match['match_type']
        insurer_indices = match['insurer_indices']
        
        # Use GlobalMatchTracker for conflict detection
        can_claim_all, available_indices, conflicts = global_tracker.can_cbl_claim_insurer(
            cbl_index, insurer_indices, 'exact' if match_type == 'exact' else 'partial'
        )
        
        if conflicts:
            # Use helper function for conflict resolution
            exact_added, partial_added = _handle_conflict_resolution(
                cbl_df, insurer_df, match, None, tolerance, 2, global_tracker, fallback_rows
            )
            exact_matches += exact_added
            partial_matches += partial_added
        else:
            # Apply the match using helper functions
            # Pass 2 only creates exact matches (no partial matches)
            if match_type == 'exact':
                exact_matches += _apply_exact_match(
                    cbl_df, cbl_index, match['match_reason'], insurer_indices, 
                    match['total_amount'], [], 2, global_tracker,
                    confidence_level=match.get('confidence_level'),
                    amount_difference=match.get('amount_difference')
                )
            else:
                # This should never happen as Pass 2 only creates exact matches
                logger.error(f"Pass 2: Unexpected match_type '{match_type}' - skipping")
                continue

    logger.info(f"✓ Pass 2 complete: {exact_matches} exact matches, {partial_matches} partial matches")
    return cbl_df


def _merge_groups_with_overlapping_insurer_indices(cbl_df, available_insurer, global_tracker):
    """
    Merge groups that have overlapping insurer indices into single larger groups.
    
    This function identifies groups with the same matched insurer indices and merges them
    into a single group, ensuring all CBL records that share the same insurer records
    are grouped together.
    
    Args:
        cbl_df: CBL dataframe with group_id and matched_insurer_indices
        available_insurer: Available insurer dataframe
        global_tracker: GlobalMatchTracker instance
        
    Returns:
        cbl_df: Updated CBL dataframe with merged groups
    """
    logger.info("\n=== Merging Groups with Overlapping Insurer Indices ===")
    
    # Check if group_id column exists (only created when cluster matches are found)
    if 'group_id' not in cbl_df.columns:
        logger.info("No group_id column found - no groups to merge (no cluster matches found)")
        return cbl_df
    
    # Get all records that have group_id and matched_insurer_indices
    grouped_records = cbl_df[
        (cbl_df['group_id'].notna()) & 
        (cbl_df['matched_insurer_indices'].notna()) &
        (cbl_df['matched_insurer_indices'].apply(lambda x: isinstance(x, list) and len(x) > 0))
    ].copy()
    
    if grouped_records.empty:
        logger.info("No grouped records found for merging")
        return cbl_df
    
    logger.info(f"Found {len(grouped_records)} grouped records to analyze for merging")
    
    # Group records by their matched_insurer_indices
    # Convert lists to tuples for hashing (lists can't be used as dict keys)
    insurer_indices_to_groups = {}
    
    for idx, row in grouped_records.iterrows():
        insurer_indices = row['matched_insurer_indices']
        group_id = row['group_id']
        
        # Convert to sorted tuple for consistent grouping
        indices_tuple = tuple(sorted(insurer_indices))
        
        if indices_tuple not in insurer_indices_to_groups:
            insurer_indices_to_groups[indices_tuple] = []
        insurer_indices_to_groups[indices_tuple].append((group_id, idx))
    
    # Find groups that share the same insurer indices
    groups_to_merge = []
    for indices_tuple, group_records in insurer_indices_to_groups.items():
        if len(group_records) > 1:
            # Multiple groups share the same insurer indices - they should be merged
            group_ids = list(set([grp[0] for grp in group_records]))
            cbl_indices = [grp[1] for grp in group_records]
            groups_to_merge.append({
                'insurer_indices': list(indices_tuple),
                'original_group_ids': group_ids,
                'cbl_indices': cbl_indices
            })
    
    if not groups_to_merge:
        logger.info("No groups found with overlapping insurer indices - no merging needed")
        return cbl_df
    
    logger.info(f"Found {len(groups_to_merge)} sets of groups to merge")
    
    # Create new merged groups
    merged_group_counter = 0
    for merge_info in groups_to_merge:
        merged_group_counter += 1
        new_group_id = f"MERGED_GROUP_{merged_group_counter}"
        
        insurer_indices = merge_info['insurer_indices']
        original_group_ids = merge_info['original_group_ids']
        cbl_indices = merge_info['cbl_indices']
        
        logger.info(f"\n🔄 Merging Groups:")
        logger.info(f"  New Group ID: {new_group_id}")
        logger.info(f"  Original Groups: {original_group_ids}")
        logger.info(f"  Shared Insurer Indices: {insurer_indices}")
        logger.info(f"  CBL Records to Merge: {len(cbl_indices)}")
        
        # Calculate combined totals for the merged group
        cbl_total = cbl_df.loc[cbl_indices, 'ProcessedAmount_Clean'].sum()
        insurer_total = available_insurer.loc[insurer_indices, 'ProcessedAmount_Clean_INSURER'].sum()
        difference = abs(cbl_total + insurer_total)
        
        # Determine the match type for the merged group
        # If any original group was exact match, the merged group should be exact match
        has_exact_match = any(
            cbl_df.at[idx, 'match_status'] == 'Exact Match' 
            for idx in cbl_indices
        )
        
        if has_exact_match:
            match_type = "EXACT"
            confidence = "High"
        else:
            match_type = "PARTIAL"
            confidence = "Medium"
        
        logger.info(f"  Merged Group Totals:")
        logger.info(f"    CBL Total: Rs{cbl_total:.2f}")
        logger.info(f"    Insurer Total: Rs{insurer_total:.2f}")
        logger.info(f"    Difference: Rs{difference:.2f}")
        logger.info(f"    Match Type: {match_type} ({confidence} Confidence)")
        
        # Update all CBL records in the merged group
        # All records in the merged group should have the SAME match status based on cluster totals
        for cbl_idx in cbl_indices:
            # Update group_id
            cbl_df.at[cbl_idx, 'group_id'] = new_group_id
            
            # Update match reason to reflect merging (use cluster-level difference)
            original_reason = cbl_df.at[cbl_idx, 'match_reason']
            new_reason = f"{original_reason} (Merged from groups: {', '.join(original_group_ids)}, Cluster Diff: Rs{difference:.2f})"
            cbl_df.at[cbl_idx, 'match_reason'] = new_reason
            
            # Update matched_insurer_indices to include all shared indices
            cbl_df.at[cbl_idx, 'matched_insurer_indices'] = insurer_indices
            
            # Update matched_amtdue_total to reflect the full insurer total
            cbl_df.at[cbl_idx, 'matched_amtdue_total'] = insurer_total
            
            # Use cluster-level difference for all records (not individual differences)
            cbl_df.at[cbl_idx, 'amount_difference'] = difference
            
            # Apply the cluster-level match status to ALL records in the merged group
            if match_type == "EXACT":
                cbl_df.at[cbl_idx, 'match_status'] = "Exact Match"
            else:
                cbl_df.at[cbl_idx, 'match_status'] = "Partial Match"
            
            logger.info(f"  ✓ Updated CBL {cbl_idx}: {match_type} match with {len(insurer_indices)} insurer records (cluster-level decision)")
    
    logger.info(f"\n✓ Group merging complete: {len(groups_to_merge)} groups merged into {merged_group_counter} merged groups")
    return cbl_df


def _extract_corporate_root(name, max_words=2):
    """
    Extract the distinctive corporate root identifier with intelligent detection for:
    1. Parent company indicators (GROUP, HOLDINGS, etc.)
    2. Person names (MR, MRS, MS, DR, etc.)
    
    This function intelligently handles both corporate entities and individual persons
    to ensure accurate grouping without false positives.
    
    Smart Logic:
        - If starts with person title (MR, MRS, MS, DR, etc.):
          → Extract 3-4 name words (skip title) for unique identification
        - If second word is a parent company indicator (GROUP, HOLDINGS, etc.):
          → Use only first word to match subsidiaries
        - Otherwise: Use 2 words for precision
    
    Examples:
        # Person names (use 3-4 words, skip titles):
        "MRS MARIE BERTHE CHANTAL HARDY" → "MARIE BERTHE CHANTAL" (3 words)
        "MRS MARIE DESIRE CATHERINE BOYER" → "MARIE DESIRE CATHERINE" (3 words)
        "MR JOHN PAUL SMITH" → "JOHN PAUL SMITH" (3 words)
        → Result: Different people stay separate ✓
        
        # Parent company indicators (use 1 word):
        "ALTEO GROUP OF COMPANIES" → "ALTEO" (GROUP is parent indicator)
        "AXYS HOLDINGS LTD" → "AXYS" (HOLDINGS is parent indicator)
        "CIEL CORPORATE LTD" → "CIEL" (CORPORATE is parent indicator)
        
        # Specific subsidiaries (use 2 words):
        "ALTEO AGRI LIMITED" → "ALTEO AGRI"
        "ALTEO MILLING LTD" → "ALTEO MILLING"
        "ALTEO REFINERY LTD" → "ALTEO REFINERY"
        
        # Different companies with same prefix (use 2 words):
        "CITY AND BEACH HOTELS LTD" → "CITY BEACH"
        "CITY SPORT LIMITEE" → "CITY SPORT"
        "CITY BROKERS LTD" → "CITY BROKERS"
    
    Args:
        name: Company name string or person name
        max_words: Maximum number of distinctive words to extract (default: 2, but 3-4 for persons)
        
    Returns:
        str: Corporate root or person identifier (intelligently 1-4 words) or None if not found
    """
    if not name or pd.isna(name):
        return None
    
    # Person name titles - if name starts with these, treat as person (not company)
    PERSON_TITLES = {
        'MR', 'MRS', 'MS', 'MISS', 'DR', 'PROF', 'SIR', 'LADY', 'LORD',
        'MR.', 'MRS.', 'MS.', 'DR.', 'PROF.',
        'MONSIEUR', 'MADAME', 'MADEMOISELLE', 'MLLE', 'MME'
    }
    
    # Parent company indicators - if these appear as the second distinctive word,
    # use only the first word to enable subsidiary-parent matching
    PARENT_INDICATORS = {
        'GROUP', 'GROUPS', 'HOLDINGS', 'HOLDING', 'CORPORATE', 
        'COMPANIES', 'ENTERPRISES', 'INTERNATIONAL', 'GLOBAL'
    }
    
    # Common prefixes to skip (articles, etc.)
    COMMON_PREFIXES = {'THE', 'LE', 'LA', 'LES', 'DES', 'DU'}
    
    # Legal entity suffixes and common words to exclude (NOT for person names)
    EXCLUSIONS = {
        'LIMITED', 'LTD', 'INC', 'INCORPORATED', 'COMPANY', 'CO', 'CORP', 'CORPORATION',
        'LLC', 'PLC', 'SA', 'AG', 'GMBH', 'NV', 'BV', 'SPA', 'SRL', 'LTDA',
        'LIMITEE', 'LIMITADA', 'SOCIETE', 'LTEE',
        'AND', 'OR', 'OF', 'IN', 'AT', 'TO', 'FOR', 'WITH', 'ON'
    }
    
    # Clean and tokenize
    name_upper = str(name).upper().strip()
    
    # Remove parentheses and their content (e.g., "COMPANY (MAURITIUS) LTD" → "COMPANY LTD")
    name_upper = re.sub(r'\([^)]*\)', '', name_upper).strip()
    
    # Extract the LESSEE (actual insured party) from financial relationship patterns
    # "ABC BANKING LTD ON LEASE TO A & D TRANSPORT LTD" → "A & D TRANSPORT LTD" (the lessee)
    # The lessee is the actual insured party, not the lessor/financer
    financial_patterns = ['ON LEASE TO', 'ON FINANCE TO', 'ON FINANCIAL LEASE TO', 
                          'ON FINANCE LEASE TO', 'ON (FINANCE) LEASE TO']
    for pattern in financial_patterns:
        if pattern in name_upper:
            parts = name_upper.split(pattern)
            if len(parts) >= 2 and parts[1].strip():
                # Take the SECOND part (lessee/actual insured party)
                name_upper = parts[1].strip()
            else:
                # No lessee specified, fall back to first part
                name_upper = parts[0].strip()
            break
    
    # Remove compound name parts (take first entity only)
    # "ALTEO AGRI LTD &/OR ALTEO MILLING LTD" → "ALTEO AGRI LTD"
    compound_separators = ['&/OR', '&OR', 'AND/OR', 'ANDOR', '&/']
    for separator in compound_separators:
        if separator in name_upper:
            name_upper = name_upper.split(separator)[0].strip()
            break
    
    # Replace punctuation with spaces and split
    name_cleaned = re.sub(r'[^\w\s]', ' ', name_upper)
    words = name_cleaned.split()
    
    # SMART PERSON NAME DETECTION
    # Check if name starts with a person title
    is_person_name = False
    if words and words[0] in PERSON_TITLES:
        is_person_name = True
    
    if is_person_name:
        # PERSON NAME LOGIC: Extract 3-4 name words (skip title) for unique identification
        # "MRS MARIE BERTHE CHANTAL HARDY" → skip "MRS" → take "MARIE BERTHE CHANTAL" (3 words)
        person_words = []
        for word in words[1:]:  # Skip first word (title)
            # For person names, don't apply EXCLUSIONS (names can be "AND", "DE", etc.)
            # Only skip very short words (< 2 chars) like initials without periods
            if len(word) >= 2:
                person_words.append(word)
                if len(person_words) >= 3:  # Extract 3 words for person names
                    break
        
        if person_words:
            return ' '.join(person_words)
        else:
            return None
    
    # CORPORATE NAME LOGIC (non-person names)
    # Extract up to max_words distinctive words (default: 2)
    distinctive_words = []
    for word in words:
        if word not in COMMON_PREFIXES and word not in EXCLUSIONS:
            if len(word) >= 3:  # Minimum length (reduced to 3 to catch more words)
                distinctive_words.append(word)
                if len(distinctive_words) >= max_words:
                    break
    
    if not distinctive_words:
        return None
    
    # SMART LOGIC: Check if second word is a parent company indicator
    # If yes, use only first word to enable subsidiary-parent matching
    if len(distinctive_words) >= 2:
        if distinctive_words[1] in PARENT_INDICATORS:
            # Second word is a parent indicator (GROUP, HOLDINGS, etc.)
            # Use only first word so subsidiaries can match
            return distinctive_words[0]
    
    # Otherwise, return all distinctive words for precision
    return ' '.join(distinctive_words)


def _build_corporate_root_index(df, name_column, prefix="", min_occurrence=2):
    """
    Build an index of corporate roots and their associated record indices.
    
    Groups records by their corporate root identifier, filtering out roots that
    don't meet the minimum occurrence threshold (to avoid noise from unique names).
    
    Args:
        df: DataFrame to index
        name_column: Column name containing company names
        prefix: Logging prefix (e.g., "CBL" or "INSURER")
        min_occurrence: Minimum number of records required for a root to be included
        
    Returns:
        dict: {corporate_root: [list of record indices]}
    """
    logger.info(f"\n=== Building Corporate Root Index for {prefix} ===")
    
    root_index = {}
    no_root_count = 0
    
    for idx, row in df.iterrows():
        name = row.get(name_column, '')
        root = _extract_corporate_root(name)
        
        if root:
            if root not in root_index:
                root_index[root] = []
            root_index[root].append(idx)
        else:
            no_root_count += 1
    
    logger.info(f"Extracted {len(root_index)} unique corporate roots from {len(df)} records")
    if no_root_count > 0:
        logger.info(f"  ⚠️ {no_root_count} records had no extractable corporate root")
    
    # Filter: Only keep roots with minimum occurrence
    # This prevents noise from unique one-off company names
    initial_count = len(root_index)
    filtered_index = {
        root: indices 
        for root, indices in root_index.items() 
        if len(indices) >= min_occurrence
    }
    
    filtered_count = initial_count - len(filtered_index)
    if filtered_count > 0:
        logger.info(f"  ℹ️ Filtered out {filtered_count} roots with < {min_occurrence} occurrences")
    
    # Log top corporate roots for visibility
    if filtered_index:
        sorted_roots = sorted(filtered_index.items(), key=lambda x: len(x[1]), reverse=True)
        logger.info(f"\n  Top Corporate Roots in {prefix}:")
        for root, indices in sorted_roots[:10]:
            logger.info(f"    - {root}: {len(indices)} records")
        if len(sorted_roots) > 10:
            logger.info(f"    ... and {len(sorted_roots) - 10} more roots")
    
    return filtered_index


def pass4(cbl_df, insurer_df, tolerance=100, global_tracker=None):
    """
    Pass 4: Corporate Group Matching with Amount Validation.
    
    Business Rule: Group records from the same corporate family together,
    then validate amounts to classify as Exact Match or Partial Match.
    
    This pass ONLY processes CBL records with status "No Match".
    Main goal: Move records from "No Match" to "Partial Match" (or "Exact Match" if amounts align).
    
    Process:
        1. Intelligent extraction of identifiers:
           - Corporate names: Extract 1-2 distinctive words (smart parent detection)
           - Person names: Extract 3 name words (skip titles like MR, MRS, DR)
        2. Match CBL and insurer records with same identifier
        3. Calculate cumulative amounts for the entire group
        4. Classify based on amount difference:
           - If within tolerance → Exact Match
           - If beyond tolerance → Partial Match
    
    Examples with Smart Detection:
        # Corporate: Parent company indicators
        - "ALTEO AGRI LTD" → "ALTEO AGRI" (2 words)
        - "ALTEO MILLING LTD" → "ALTEO MILLING" (2 words)
        - "ALTEO GROUP OF COMPANIES" → "ALTEO" (1 word - GROUP is parent indicator)
        → Result: All ALTEO subsidiaries match ALTEO GROUP ✓
        
        # Corporate: Different companies with same prefix
        - "CITY BEACH HOTELS" → "CITY BEACH" (2 words)
        - "CITY SPORT LTD" → "CITY SPORT" (2 words)
        - "CITY BROKERS" → "CITY BROKERS" (2 words)
        → Result: CITY companies stay separate ✓
        
        # Person names: Extract 3 name words (skip titles)
        - "MRS MARIE BERTHE CHANTAL HARDY" → "MARIE BERTHE CHANTAL" (3 words)
        - "MRS MARIE DESIRE CATHERINE BOYER" → "MARIE DESIRE CATHERINE" (3 words)
        - "MRS MARIE ODETTE HARDY" → "MARIE ODETTE HARDY" (3 words)
        → Result: Different people stay separate ✓
    
    This pass handles parent-subsidiary relationships and corporate group accounts
    where the insurer may use aggregate names like "GROUP OF COMPANIES" or where
    individual subsidiaries need to be matched to consolidated accounts.
    
    Args:
        cbl_df: CBL DataFrame with match results from previous passes
        insurer_df: Insurer DataFrame
        tolerance: Amount tolerance for exact match classification (default: 100)
        global_tracker: GlobalMatchTracker instance for consistent row usage tracking
        
    Returns:
        cbl_df: Updated CBL DataFrame with corporate group matches
    """
    logger.info("\n=== Pass 4: Corporate Group Matching with Amount Validation ===")
    logger.info("Business Rule: Group by corporate root name, then validate amounts")
    logger.info(f"Amount Tolerance: Rs{tolerance} (within tolerance → Exact Match, beyond → Partial Match)")
    
    exact_matches = 0
    partial_matches = 0
    
    logger.info(f"Pass 4 starting with global tracker: {global_tracker.get_usage_summary()}")
    
    # Get only "No Match" CBL records (not partial matches)
    unmatched_cbl = cbl_df[cbl_df['match_status'] == 'No Match'].copy()
    logger.info(f"Processing {len(unmatched_cbl)} CBL records with 'No Match' status")
    
    if unmatched_cbl.empty:
        logger.info("No unmatched records to process")
        return cbl_df
    
    # Use global tracker for consistent filtering
    # Exclude exact and matrix matches but allow partial matches to be upgraded
    already_matched_insurer = global_tracker.exact_used_insurer | global_tracker.matrix_used_insurer
    available_insurer = insurer_df[~insurer_df.index.isin(already_matched_insurer)].copy()
    logger.info(f"Pass 4: Using global tracker - excluding {len(already_matched_insurer)} exact/matrix used insurer rows")
    logger.info(f"Pass 4: Available insurer rows for corporate group matching: {len(available_insurer)}")
    
    if available_insurer.empty:
        logger.info("No available insurer records to match")
        return cbl_df
    
    # Build corporate root indices
    cbl_root_index = _build_corporate_root_index(unmatched_cbl, 'ClientName', 'CBL', min_occurrence=1)
    insurer_root_index = _build_corporate_root_index(available_insurer, 'ClientName_INSURER', 'INSURER', min_occurrence=1)
    
    logger.info(f"\nFound {len(cbl_root_index)} CBL corporate groups")
    logger.info(f"Found {len(insurer_root_index)} insurer corporate groups")
    
    if not cbl_root_index or not insurer_root_index:
        logger.info("No corporate groups found for matching")
        return cbl_df
    
    # Match corporate groups by root name
    logger.info("\n=== Matching Corporate Groups ===")
    group_counter = 0
    
    for root in cbl_root_index.keys():
        if root in insurer_root_index:
            group_counter += 1
            group_id = f"CORPORATE_GROUP_{root}_{group_counter}"
            
            cbl_indices = cbl_root_index[root]
            insurer_indices = insurer_root_index[root]
            
            # Calculate group totals for amount validation
            cbl_total = cbl_df.loc[cbl_indices, 'ProcessedAmount_Clean'].sum()
            insurer_total = available_insurer.loc[insurer_indices, 'ProcessedAmount_Clean_INSURER'].sum()
            difference = abs(cbl_total + insurer_total)
            
            # Classify match based on amount difference
            match_type, _, confidence = classify_amount_match(cbl_total, insurer_total, tolerance)
            is_exact_match = match_type in ["PERFECT_MATCH", "EXACT_MATCH"]
            
            logger.info(f"\n🏢 Corporate Group Match Found:")
            logger.info(f"  Group ID: {group_id}")
            logger.info(f"  Corporate Root: {root}")
            logger.info(f"  CBL Records: {len(cbl_indices)}")
            logger.info(f"  Insurer Records: {len(insurer_indices)}")
            logger.info(f"  CBL Total: Rs{cbl_total:.2f}")
            logger.info(f"  Insurer Total: Rs{insurer_total:.2f}")
            logger.info(f"  Group Difference: Rs{difference:.2f}")
            logger.info(f"  Match Classification: {'EXACT' if is_exact_match else 'PARTIAL'} ({confidence} Confidence)")
            
            # Validate insurer indices are available based on match type
            if is_exact_match:
                can_use_all, available_indices, conflicts = global_tracker.can_use_for_exact(insurer_indices)
            else:
                # For partial matches, allow sharing
                can_use_all, available_indices, conflicts = global_tracker.can_use_for_partial(insurer_indices, allow_sharing=True)
            
            if not available_indices:
                logger.warning(f"  ⚠ No available insurer indices - skipping corporate group")
                continue
            
            if not can_use_all:
                logger.info(f"  ℹ Using {len(available_indices)}/{len(insurer_indices)} available insurer indices")
                insurer_indices = available_indices
                # Recalculate insurer total with available indices only
                insurer_total = available_insurer.loc[insurer_indices, 'ProcessedAmount_Clean_INSURER'].sum()
                difference = abs(cbl_total + insurer_total)
                # Reclassify match based on new totals
                match_type, _, confidence = classify_amount_match(cbl_total, insurer_total, tolerance)
                is_exact_match = match_type in ["PERFECT_MATCH", "EXACT_MATCH"]
                logger.info(f"  Reclassified as: {'EXACT' if is_exact_match else 'PARTIAL'} ({confidence} Confidence) after using available indices")
            
            # Apply matches to all CBL records in this corporate group
            for cbl_idx in cbl_indices:
                # Mark pass for tracking
                add_pass(cbl_df, cbl_idx, 4)
                
                # Calculate amounts for this specific record (all records get same insurer pool)
                total_insurer_amount = available_insurer.loc[insurer_indices, "ProcessedAmount_Clean_INSURER"].sum()
                cbl_amount = cbl_df.at[cbl_idx, "ProcessedAmount_Clean"]
                
                # Match reason with amount classification
                match_reason = f"Corporate Group: {root} ({len(cbl_indices)} CBL records, {len(insurer_indices)} insurer records, Group Diff: Rs{difference:.2f}, {confidence} Confidence)"
                
                # Apply match based on amount validation
                if is_exact_match:
                    # Amounts match within tolerance - mark as Exact Match
                    _apply_cluster_exact_match(
                        cbl_df, cbl_idx, match_reason, insurer_indices,
                        total_insurer_amount, 4, global_tracker,
                        confidence_level=confidence,
                        amount_difference=difference
                    )
                    exact_matches += 1
                    logger.info(f"  ✓ CBL {cbl_idx}: EXACT MATCH with {len(insurer_indices)} insurer records (CBL: Rs{cbl_amount:.2f})")
                else:
                    # Amounts don't match - mark as Partial Match (main goal of Pass 4)
                    partial_matches += _apply_partial_match(
                        cbl_df, cbl_idx, match_reason, insurer_indices,
                        total_insurer_amount, 4, global_tracker,
                        confidence_level=confidence,
                        amount_difference=difference
                    )
                    logger.info(f"  ✓ CBL {cbl_idx}: PARTIAL MATCH with {len(insurer_indices)} insurer records (CBL: Rs{cbl_amount:.2f}, Diff: Rs{difference:.2f})")
                
                # Assign group metadata
                cbl_df.at[cbl_idx, 'group_id'] = group_id
                cbl_df.at[cbl_idx, 'corporate_root'] = root
    
    logger.info(f"\n✓ Pass 4 complete: {exact_matches} exact matches, {partial_matches} partial matches in {group_counter} corporate groups")
    logger.info(f"   Main Goal Achieved: Moved {exact_matches + partial_matches} records from 'No Match' to matched status")
    return cbl_df


def pass3(cbl_df, insurer_df, tolerance=100, fuzzy_threshold=90, global_tracker=None):
    """Pass 3: Name-based Clustering and Grouping Strategy."""
    logger.info("\n=== Pass 3: Name-based Clustering and Grouping ===")
    logger.info("Strategy: Group CBL rows with similar insurer names, compare amounts to determine exact vs partial matches")
    
    exact_matches = 0
    partial_matches = 0
    
    logger.info(f"Pass 3 starting with global tracker: {global_tracker.get_usage_summary()}")
    logger.info(f"Name Clustering Threshold: {fuzzy_threshold}%")

    # Use global tracker for consistent filtering
    # For Pass 3, we exclude exact and matrix matches but allow partial matches to be upgraded
    already_matched_insurer = global_tracker.exact_used_insurer | global_tracker.matrix_used_insurer
    available_insurer = insurer_df[~insurer_df.index.isin(already_matched_insurer)].copy()
    logger.info(f"Pass 3: Using global tracker - excluding {len(already_matched_insurer)} exact/matrix used insurer rows")
    logger.info(f"Pass 3: Available insurer rows for name clustering: {len(available_insurer)}")
    
    # Get unmatched/partial CBL records
    unmatched_cbl = cbl_df[cbl_df['match_status'].isin(['No Match', 'Partial Match'])].copy()
    logger.info(f"Pass 3: Processing {len(unmatched_cbl)} CBL records with 'No Match' or 'Partial Match' status")
    
    if unmatched_cbl.empty:
        logger.info("No unmatched records to process")
        return cbl_df
    
    # Build name clusters using fuzzy matching
    logger.info("\n=== Building Name Clusters ===")
    cbl_name_clusters = _build_fuzzy_name_clusters(
        unmatched_cbl,
        name_column='ClientName',
        fuzzy_threshold=fuzzy_threshold,
        prefix="CBL"
    )
    
    insurer_name_clusters = _build_fuzzy_name_clusters(
        available_insurer,
        name_column='ClientName_INSURER',  # Use original column, not cleaned
        fuzzy_threshold=fuzzy_threshold,
        prefix="INSURER"
    )
    
    logger.info(f"Created {len(cbl_name_clusters)} CBL name clusters and {len(insurer_name_clusters)} insurer name clusters")
    
    # Match clusters together
    logger.info("\n=== Matching Clusters ===")
    group_counter = 0
    
    # Use CompanyNameMatcher for cross-cluster matching to handle compound names
    matcher = CompanyNameMatcher(primary_penalty=0.3, exact_match_boost=2.5)
    
    for cbl_cluster_name, cbl_indices in cbl_name_clusters.items():
        for insurer_cluster_name, insurer_indices in insurer_name_clusters.items():
            # Use intelligent similarity calculation for cross-cluster matching
            cluster_similarity = matcher.calculate_intelligent_similarity(cbl_cluster_name, insurer_cluster_name)
            
            # Use stricter threshold for cluster matching (90% vs 95% for within-cluster)
            # This ensures only very similar clusters are matched together
            if cluster_similarity >= 90:
                # ADDITIONAL VALIDATION: Check word overlap to prevent false positive cross-cluster matches
                # This prevents matching unrelated companies that share only one common word
                has_overlap, common_count, common_words = _has_sufficient_word_overlap(
                    cbl_cluster_name, insurer_cluster_name, min_common_words=2
                )
                
                if not has_overlap:
                    logger.debug(f"Rejected cross-cluster match despite {cluster_similarity}% similarity: "
                               f"'{cbl_cluster_name[:50]}' vs '{insurer_cluster_name[:50]}' "
                               f"(only {common_count} common words: {common_words})")
                    continue
                group_counter += 1
                group_id = f"NAME_GROUP_{group_counter}"
                
                # Calculate totals for the entire group
                cbl_total = cbl_df.loc[cbl_indices, 'ProcessedAmount_Clean'].sum()
                insurer_total = available_insurer.loc[insurer_indices, 'ProcessedAmount_Clean_INSURER'].sum()
                difference = abs(cbl_total + insurer_total)
                
                # Classify the match based on amount difference
                match_type, _, confidence = classify_amount_match(cbl_total, insurer_total, tolerance)
                is_exact_match = match_type in ["PERFECT_MATCH", "EXACT_MATCH"]
                
                logger.info(f"\n🎯 Cluster Match Found:")
                logger.info(f"  Group ID: {group_id}")
                logger.info(f"  CBL Cluster: '{cbl_cluster_name[:60]}...' ({len(cbl_indices)} records)")
                logger.info(f"  Insurer Cluster: '{insurer_cluster_name[:60]}...' ({len(insurer_indices)} records)")
                logger.info(f"  Cluster Name Similarity: {cluster_similarity}% (threshold: 90%)")
                logger.info(f"  CBL Total: Rs{cbl_total:.2f}")
                logger.info(f"  Insurer Total: Rs{insurer_total:.2f}")
                logger.info(f"  Difference: Rs{difference:.2f}")
                logger.info(f"  Match Type: {'EXACT' if is_exact_match else 'PARTIAL'} ({confidence} Confidence)")
                
                # Validate insurer indices are available
                if is_exact_match:
                    can_use_all, available_indices, conflicts = global_tracker.can_use_for_exact(insurer_indices)
                else:
                    can_use_all, available_indices, conflicts = global_tracker.can_use_for_partial(insurer_indices)
                
                if not available_indices:
                    logger.warning(f"  ⚠ No available insurer indices - skipping cluster match")
                    continue
                
                if not can_use_all:
                    logger.info(f"  ℹ Using {len(available_indices)}/{len(insurer_indices)} available insurer indices")
                    insurer_indices = available_indices
                    
                # Apply the SAME match to ALL CBL records in the cluster
                # All CBL rows get the same insurer indices and same match status
                for cbl_idx in cbl_indices:
                    # Mark pass for tracking
                    add_pass(cbl_df, cbl_idx, 3)
                    
                    # All CBL rows in the cluster get the SAME insurer indices
                    # No need to check individual conflicts since we validated at cluster level
                    usable_indices = insurer_indices
                    
                    # Calculate total amount for this specific CBL row's match
                    total_insurer_amount = available_insurer.loc[usable_indices, "ProcessedAmount_Clean_INSURER"].sum()
                    cbl_amount = cbl_df.at[cbl_idx, "ProcessedAmount_Clean"]
                    amount_diff = abs(cbl_amount + total_insurer_amount)
                    
                    # Create match reason
                    match_reason = f"Name Cluster Match (Cluster: '{cbl_cluster_name[:30]}...', Similarity: {cluster_similarity}%, Amount Diff: Rs{amount_diff:.2f})"
                    
                    # Apply the appropriate match type (same for all CBL rows in cluster)
                    if is_exact_match:
                        # For Pass 3 cluster matching, use direct assignment to allow sharing
                        _apply_cluster_exact_match(
                            cbl_df, cbl_idx, match_reason, usable_indices,
                            total_insurer_amount, 3, global_tracker,
                            confidence_level=confidence,
                            amount_difference=amount_diff
                        )
                        exact_matches += 1
                        logger.info(f"  ✓ CBL {cbl_idx}: EXACT match with {len(usable_indices)} insurer records")
                    else:
                        # Mark as partial match
                        partial_matches += _apply_partial_match(
                            cbl_df, cbl_idx, match_reason, usable_indices,
                            total_insurer_amount, 3, global_tracker,
                            confidence_level=confidence,
                            amount_difference=amount_diff
                        )
                        logger.info(f"  ✓ CBL {cbl_idx}: PARTIAL match with {len(usable_indices)} insurer records")
                    
                    # Assign group_id for output organization
                    cbl_df.at[cbl_idx, 'group_id'] = group_id
    
    # NEW: Merge groups with overlapping insurer indices
    cbl_df = _merge_groups_with_overlapping_insurer_indices(cbl_df, available_insurer, global_tracker)
                        
    logger.info(f"\n✓ Pass 3 complete: {exact_matches} exact matches, {partial_matches} partial matches in {group_counter} name groups")
    return cbl_df
