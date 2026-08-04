#!/usr/bin/env python3

import pandas as pd
import logging
import re
from typing import Dict, List, Tuple, Optional, Set
from itertools import combinations
from fuzzywuzzy import fuzz
from .utils import add_pass
from .matching_engine import (
    CompanyNameMatcher,
    GlobalMatchTracker,
    classify_amount_match,
    validate_substring_match,
    _apply_cluster_exact_match,
    _apply_partial_match,
    _apply_no_match,
)

logger = logging.getLogger(__name__)


def _has_sufficient_word_overlap(name1, name2, min_common_words=2):
    """
    Check if two names have sufficient meaningful word overlap to be clustered.

    This prevents over-clustering based on:
    - Single common word: "SUN LTD" vs "WOLMAR SUN HOTELS LTD" → NO MATCH
    - Proper subset: "SUN LTD" vs "SUN MARINE LTD" → NO MATCH
    - Partial overlap: "SUN RESORTS" vs "SUN HOTELS" → NO MATCH (only 50% overlap)
    - Different primary identifiers: "SHA TRAVEL TOURS" vs "TAJ TRAVEL TOURS" → NO MATCH

    Allows matching when:
    - High overlap: "ACME LTD" vs "ACME HOLDINGS LTD" → MATCH (100% overlap)
    - Multiple common words: "ABC XYZ LTD" vs "ABC XYZ HOLDINGS" → MATCH
    - Typo tolerance: "BLU CONSTRUCTION" vs "BLU CONSTRUCTON" → MATCH (fuzzy word match)

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

    # Also keep ordered list to check first word (primary company identifier)
    words1_list = [w for w in name1_cleaned.split() if w and w not in EXCLUDE_WORDS]
    words2_list = [w for w in name2_cleaned.split() if w and w not in EXCLUDE_WORDS]

    # Find common meaningful words (exact matches)
    common_words = words1 & words2

    # FUZZY WORD MATCHING: Find additional matches for typos
    # e.g., "CONSTRUCTION" vs "CONSTRUCTON" (1 char difference)
    # Only check words that didn't match exactly
    unmatched_words1 = words1 - common_words
    unmatched_words2 = words2 - common_words

    fuzzy_matched_words = set()
    FUZZY_WORD_THRESHOLD = 85  # 85% similarity for individual words

    for w1 in unmatched_words1:
        for w2 in unmatched_words2:
            # Only fuzzy match words of similar length (avoid "A" matching "APPLE")
            if abs(len(w1) - len(w2)) <= 2 and len(w1) >= 4 and len(w2) >= 4:
                # Use simple ratio for individual words (faster than token_set)
                similarity = fuzz.ratio(w1, w2)
                if similarity >= FUZZY_WORD_THRESHOLD:
                    # Add the word from name1 to common words (arbitrary choice)
                    fuzzy_matched_words.add(w1)
                    logger.debug(f"Fuzzy word match: '{w1}' ≈ '{w2}' ({similarity}%)")
                    break  # Each word only matches once

    # Combine exact and fuzzy matches
    common_words = common_words | fuzzy_matched_words

    # CRITICAL CHECK: First distinctive word (primary identifier) must match
    # This prevents "SHA TRAVEL TOURS" from matching "TAJ TRAVEL TOURS"
    # The first word is typically the actual company name/identifier
    if words1_list and words2_list:
        first_word_1 = words1_list[0]
        first_word_2 = words2_list[0]

        # If first words are different and both are substantive (3+ chars), check fuzzy
        if first_word_1 != first_word_2 and len(first_word_1) >= 3 and len(first_word_2) >= 3:
            # Allow if first words are very similar (typo tolerance)
            first_word_similarity = fuzz.ratio(first_word_1, first_word_2)
            if first_word_similarity < FUZZY_WORD_THRESHOLD:
                return False, len(common_words), common_words

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
    # "MCB LEASING LIMITED ONLEASE TO ECOBAT LTD" → "ECOBAT LTD" (the lessee)
    # The lessee is the actual insured party, not the lessor/financer
    financial_patterns = ['ON LEASE TO', 'ONLEASE TO', 'ON FINANCE TO', 'ONFINANCE TO',
                          'ON FINANCIAL LEASE TO', 'ON FINANCE LEASE TO', 'ON (FINANCE) LEASE TO']
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
    # "MR JOHN SMITH AND MRS JANE SMITH" → "MR JOHN SMITH" (first person)
    # "MR JOHN SMITH OR MRS JANE SMITH" → "MR JOHN SMITH" (first person)
    compound_separators = ['&/OR', '&OR', 'AND/OR', 'ANDOR', '&/', ' AND ', ' OR ']
    for separator in compound_separators:
        if separator in name_upper:
            name_upper = name_upper.split(separator)[0].strip()
            break

    # ORGANIZATIONAL PREFIX HANDLING: Skip common organizational prefixes to extract actual identifier
    # These are multi-word legal/organizational prefixes where the actual identifier comes AFTER
    # Examples:
    #   "SYNDICAT DES COPROPRIETAIRES DE LES TERRASSES DU BARACHOIS" → Extract "LES TERRASSES DU BARACHOIS"
    #   "LE SYNDICAT DES COPROPRIETAIRES DU CENTRE FINANCIER DU NORD" → Extract "CENTRE FINANCIER DU NORD"
    ORGANIZATIONAL_PREFIXES = [
        'LE SYNDICAT DES COPROPRIETAIRES',
        'SYNDICAT DES COPROPRIETAIRES',
        'LA SYNDICAT DES COPROPRIETAIRES',
        'SYNDICATE OF CO OWNERS',
        'THE SYNDICATE OF CO OWNERS',
    ]

    # Check if name starts with any organizational prefix
    for org_prefix in ORGANIZATIONAL_PREFIXES:
        if name_upper.startswith(org_prefix):
            # Extract everything AFTER the prefix
            remaining = name_upper[len(org_prefix):].strip()

            # Check for connecting words (DE, DU, DE LA, DE L', etc.) and skip them
            # "DE LES TERRASSES DU BARACHOIS" → "LES TERRASSES DU BARACHOIS"
            # "DU CENTRE FINANCIER DU NORD" → "CENTRE FINANCIER DU NORD"
            connecting_patterns = ['DE L\'', 'DE LA', 'DE LES', 'DES', 'DE', 'DU', 'OF THE', 'OF']
            for connector in connecting_patterns:
                if remaining.startswith(connector + ' '):
                    remaining = remaining[len(connector):].strip()
                    break

            if remaining:
                name_upper = remaining
                logger.debug(f"Organizational prefix detected: '{org_prefix}' → Extracted: '{name_upper}'")
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


def _get_primary_corporate_root(name, max_words=2):
    """
    Get the PRIMARY (first) corporate root from a name.

    For compound names like "KASA GROUP OF COMPANIES - REY AND LENFERNA LTD &/OR ...",
    this returns only the PRIMARY entity's root (e.g., "KASA"), not secondary
    entities that appear after separators.

    This is used for GROUP ASSIGNMENT: a CBL row should only be assigned to
    a group that matches its PRIMARY root, not secondary subsidiaries.

    Examples:
        "KASA GROUP OF COMPANIES - REY AND LENFERNA LTD &/OR CEAL LTEE"
        → "KASA" (primary entity)

        "GALEA GROUP OF COMPANIES - REY AND LENFERNA LTD &/OR KASA CORPORATE"
        → "GALEA" (primary entity, not KASA or REY)

        "REY & LENFERNA LTD"
        → "REY LENFERNA" (simple name)

    Args:
        name: Company name string
        max_words: Max distinctive words (default: 2)

    Returns:
        str: The PRIMARY corporate root, or empty string if none
    """
    if not name or pd.isna(name):
        return ""

    name_upper = str(name).upper().strip()

    # First, extract the PRIMARY entity (before any group indicator like " - ")
    # e.g., "KASA GROUP OF COMPANIES - REY..." → "KASA GROUP OF COMPANIES"
    primary_indicators = [' - ', ' – ', ' — ']
    for indicator in primary_indicators:
        if indicator in name_upper:
            name_upper = name_upper.split(indicator)[0].strip()
            break

    # Now extract the corporate root from the primary entity
    # This handles compound primary entities like "PALCO LTD &/OR TBA LTEE"
    # by returning only the FIRST root
    roots = _extract_all_corporate_roots(name_upper, max_words)
    return roots[0] if roots else ""


def _extract_all_corporate_roots(name, max_words=2):
    """
    Extract ALL corporate roots from a name (handles compound names).

    For compound names like "PALCO LTD &/OR TBA LTEE &/OR AIRSTREAM LTD",
    this extracts roots for EACH entity: ["PALCO", "TBA", "AIRSTREAM"]

    This enables matching regardless of entity order between CBL and insurer data.

    For simple names (no compound separators), returns a single-element list
    with the same result as _extract_corporate_root().

    Examples:
        # Compound name - extracts ALL entity roots:
        "PALCO WATERPROOFING LTD &/OR TBA (MAURICE) LTEE &/OR AIRSTREAM LTD"
        → ["PALCO WATERPROOFING", "TBA MAURICE", "AIRSTREAM"]

        # Same entities, different order - produces same roots (order may vary):
        "TBA (MAURICE) LTEE &/OR PALCO WATERPROOFING LTD &/OR AIRSTREAM LTD"
        → ["TBA MAURICE", "PALCO WATERPROOFING", "AIRSTREAM"]

        # Simple name - single root (existing behavior):
        "ACME HOLDINGS LTD"
        → ["ACME"]

        # Filters out generic terms:
        "PALCO LTD &/OR SUBSIDIARIES &/OR AFFILIATED COMPANIES"
        → ["PALCO"]  (SUBSIDIARIES and AFFILIATED COMPANIES are skipped)

    Args:
        name: Company name string (potentially compound)
        max_words: Max distinctive words per entity (default: 2)

    Returns:
        list: List of corporate roots (1 for simple names, multiple for compound)
    """
    if not name or pd.isna(name):
        return []

    name_upper = str(name).upper().strip()

    # Check if this is a compound name
    compound_separators = ['&/OR', '&OR', 'AND/OR', 'ANDOR', '&/']
    is_compound = any(sep in name_upper for sep in compound_separators)

    if not is_compound:
        # Simple name - return single root (existing behavior)
        root = _extract_corporate_root(name, max_words)
        return [root] if root else []

    # Compound name - extract roots for ALL entities
    roots = []

    # Generic terms that don't represent actual companies - skip these
    GENERIC_TERMS = {
        'SUBSIDIARIES', 'SUBSIDIARY', 'ASSOCIATES', 'ASSOCIATED COMPANIES',
        'AFFILIATED COMPANIES', 'AFFILIATES', 'RELATED COMPANIES',
        'SISTER COMPANIES', 'HOLDING COMPANIES', 'PARENT COMPANY',
        'TBA', 'TO BE ADVISED', 'TO BE ANNOUNCED', 'TO BE CONFIRMED'
    }

    # Split on compound separators progressively
    entities = [name_upper]
    for separator in compound_separators:
        new_entities = []
        for entity in entities:
            parts = entity.split(separator)
            new_entities.extend(parts)
        entities = new_entities

    # Extract root from each entity
    compound_entity_count = 0
    for entity in entities:
        entity = entity.strip()

        # Skip empty strings
        if not entity:
            continue

        # Skip generic terms that don't represent actual companies
        if entity in GENERIC_TERMS:
            logger.debug(f"Skipping generic term in compound name: '{entity}'")
            continue

        compound_entity_count += 1
        root = _extract_corporate_root(entity, max_words)

        if root and root not in roots:  # Avoid duplicates
            roots.append(root)
            logger.debug(f"Compound entity '{entity[:40]}...' → root '{root}'")

    if len(roots) > 1:
        logger.debug(f"Compound name extracted {len(roots)} roots from {compound_entity_count} entities: {roots[:5]}{'...' if len(roots) > 5 else ''}")

    return roots


def _build_corporate_root_index(df, name_column, prefix="", min_occurrence=2, return_primary_map=False):
    """
    Build an index of corporate roots and their associated record indices.

    Groups records by their corporate root identifier, filtering out roots that
    don't meet the minimum occurrence threshold (to avoid noise from unique names).

    COMPOUND NAME SUPPORT:
    For compound names (e.g., "PALCO LTD &/OR TBA LTEE &/OR AIRSTREAM LTD"),
    the record is indexed under ALL entity roots, not just the first.
    This enables matching regardless of entity order between CBL and insurer.

    Example:
        CBL: "PALCO LTD &/OR TBA LTEE" → indexed under ["PALCO", "TBA"]
        Insurer: "TBA LTEE &/OR PALCO LTD" → indexed under ["TBA", "PALCO"]
        → They'll match on EITHER root!

    Args:
        df: DataFrame to index
        name_column: Column name containing company names
        prefix: Logging prefix (e.g., "CBL" or "INSURER")
        min_occurrence: Minimum number of records required for a root to be included
        return_primary_map: If True, also returns a map of idx → primary_root

    Returns:
        dict: {corporate_root: [list of record indices]}
        (optional) dict: {idx: primary_root} if return_primary_map=True
    """
    logger.info(f"\n=== Building Corporate Root Index for {prefix} ===")

    root_index = {}
    primary_root_map = {}  # idx → primary root (for group assignment)
    no_root_count = 0
    compound_name_count = 0
    multi_indexed_count = 0

    for idx, row in df.iterrows():
        name = row.get(name_column, '')

        # Use multi-root extraction for compound name support
        roots = _extract_all_corporate_roots(name)

        # Also track the PRIMARY root (for group assignment decisions)
        primary_root = _get_primary_corporate_root(name)
        if primary_root:
            primary_root_map[idx] = primary_root

        if roots:
            # Track compound names for logging
            if len(roots) > 1:
                compound_name_count += 1
                multi_indexed_count += len(roots)

            # Index this record under ALL extracted roots
            for root in roots:
                if root not in root_index:
                    root_index[root] = []
                # Avoid duplicate indices under same root
                if idx not in root_index[root]:
                    root_index[root].append(idx)
        else:
            no_root_count += 1

    logger.info(f"Extracted {len(root_index)} unique corporate roots from {len(df)} records")
    if compound_name_count > 0:
        logger.info(f"  📋 {compound_name_count} compound names detected → created {multi_indexed_count} additional index entries")
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

    if return_primary_map:
        return filtered_index, primary_root_map
    return filtered_index


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
    # Exclude history-pre-placed rows — their insurer indices are not in available_insurer
    base_mask = (
        (cbl_df['group_id'].notna()) &
        (cbl_df['matched_insurer_indices'].notna()) &
        (cbl_df['matched_insurer_indices'].apply(lambda x: isinstance(x, list) and len(x) > 0))
    )
    if 'match_resolved_in_pass' in cbl_df.columns:
        base_mask = base_mask & (~cbl_df['match_resolved_in_pass'].isin(['history', 'matrix']))
    grouped_records = cbl_df[base_mask].copy()

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
            new_reason = f"{original_reason} (Merged from groups: {', '.join(original_group_ids)})"
            cbl_df.at[cbl_idx, 'match_reason'] = new_reason

            # Update matched_insurer_indices to include all shared indices
            cbl_df.at[cbl_idx, 'matched_insurer_indices'] = insurer_indices

            # Update matched_amtdue_total to reflect the full insurer total
            cbl_df.at[cbl_idx, 'matched_amtdue_total'] = insurer_total

            # Use cluster-level difference for all records (not individual differences)
            cbl_df.at[cbl_idx, 'Amount Difference'] = round(difference, 2)

            # Apply the cluster-level match status to ALL records in the merged group
            if match_type == "EXACT":
                cbl_df.at[cbl_idx, 'match_status'] = "Exact Match"
            else:
                cbl_df.at[cbl_idx, 'match_status'] = "Partial Match"

            logger.info(f"  ✓ Updated CBL {cbl_idx}: {match_type} match with {len(insurer_indices)} insurer records (cluster-level decision)")

    logger.info(f"\n✓ Group merging complete: {len(groups_to_merge)} groups merged into {merged_group_counter} merged groups")
    return cbl_df


def _line_by_line_name_matching(cbl_df, remaining_cbl, available_insurer, insurer_df,
                                global_tracker, tolerance=50, name_threshold=85,
                                group_counter=0):
    """
    Line-by-line name + amount matching for remaining unmatched CBL records.

    For each unmatched CBL row, finds the best insurer row(s) by:
      1. Client name similarity (>= name_threshold)
      2. Amount validation (single match or combination of 2-5 insurer rows)

    This is more precise than group-based matching because it establishes
    actual 1:1 or 1:few relationships between CBL and insurer rows.

    Args:
        cbl_df: Full CBL DataFrame (modified in-place)
        remaining_cbl: Subset of unmatched CBL rows to process
        available_insurer: Insurer rows not yet used in exact matches
        insurer_df: Full insurer DataFrame (for amount lookups)
        global_tracker: GlobalMatchTracker instance
        tolerance: Amount tolerance for exact match classification
        name_threshold: Minimum similarity score for name matching (default: 85)
        group_counter: Starting group counter for group_id assignment

    Returns:
        tuple: (exact_match_count, set of matched CBL indices)
    """
    matcher = CompanyNameMatcher(primary_penalty=0.3, exact_match_boost=2.5)

    # Pre-extract primary names for all available insurer rows
    insurer_primary_names = {}
    for idx in available_insurer.index:
        name = str(available_insurer.at[idx, 'ClientName_INSURER']).upper().strip()
        if name and name != 'NAN':
            primary, _ = matcher.extract_primary_company(name)
            insurer_primary_names[idx] = primary

    logger.info(f"Pre-extracted {len(insurer_primary_names)} insurer primary names")

    # Collect all potential matches, then apply greedily by quality
    all_potential = []

    for cbl_idx in remaining_cbl.index:
        cbl_name = str(cbl_df.at[cbl_idx, 'ClientName']).upper().strip()
        if not cbl_name or cbl_name == 'NAN':
            continue

        cbl_primary, _ = matcher.extract_primary_company(cbl_name)
        cbl_amt = cbl_df.at[cbl_idx, 'ProcessedAmount_Clean']
        if pd.isna(cbl_amt):
            continue

        # Find insurer rows with similar names
        name_matches = []
        for ins_idx, ins_primary in insurer_primary_names.items():
            similarity = matcher.calculate_intelligent_similarity(cbl_primary, ins_primary)
            if similarity >= name_threshold:
                ins_amt = available_insurer.at[ins_idx, 'ProcessedAmount_Clean_INSURER']
                if pd.notna(ins_amt):
                    name_matches.append({
                        'ins_idx': ins_idx,
                        'ins_primary': ins_primary,
                        'ins_amt': ins_amt,
                        'similarity': similarity,
                    })

        if not name_matches:
            continue

        # Try single match first
        for nm in name_matches:
            match_type, difference, confidence = classify_amount_match(cbl_amt, nm['ins_amt'], tolerance)
            if match_type in ["PERFECT_MATCH", "EXACT_MATCH"]:
                all_potential.append({
                    'cbl_idx': cbl_idx,
                    'insurer_indices': [nm['ins_idx']],
                    'difference': difference,
                    'confidence': confidence,
                    'similarity': nm['similarity'],
                    'cbl_primary': cbl_primary,
                    'ins_primary': nm['ins_primary'],
                    'cbl_amt': cbl_amt,
                    'match_type': 'single',
                })

        # Try combination matches (2-5 items from top 20 most promising)
        if len(name_matches) >= 2:
            target = -cbl_amt
            sorted_nms = sorted(name_matches, key=lambda x: abs(x['ins_amt'] - target))
            limited = sorted_nms[:20]
            max_combo = min(5, len(limited))

            for r in range(2, max_combo + 1):
                found = False
                for combo in combinations(limited, r):
                    total_amt = sum(c['ins_amt'] for c in combo)
                    match_type, difference, confidence = classify_amount_match(cbl_amt, total_amt, tolerance)
                    if match_type in ["PERFECT_MATCH", "EXACT_MATCH"]:
                        avg_sim = sum(c['similarity'] for c in combo) / len(combo)
                        all_potential.append({
                            'cbl_idx': cbl_idx,
                            'insurer_indices': [c['ins_idx'] for c in combo],
                            'difference': difference,
                            'confidence': confidence,
                            'similarity': avg_sim,
                            'cbl_primary': cbl_primary,
                            'ins_primary': combo[0]['ins_primary'],
                            'cbl_amt': cbl_amt,
                            'match_type': 'combination',
                        })
                        found = True
                        break
                if found:
                    break

    logger.info(f"Found {len(all_potential)} potential line-by-line matches")

    # Sort by: highest similarity first, then lowest amount difference
    all_potential.sort(key=lambda x: (-x['similarity'], x['difference']))

    # Apply matches greedily (best quality first)
    matched_cbl = set()
    used_insurer = set()
    exact_count = 0
    local_group_counter = group_counter

    for match in all_potential:
        cbl_idx = match['cbl_idx']
        if cbl_idx in matched_cbl:
            continue

        insurer_indices = match['insurer_indices']

        # Check insurer rows not already used in this phase
        if any(idx in used_insurer for idx in insurer_indices):
            continue

        # Check insurer availability via global tracker
        can_use, available_idx, conflicts = global_tracker.can_use_for_exact(insurer_indices)
        if not available_idx:
            continue
        insurer_indices = available_idx

        # Apply the match using cluster-style (consistent with other Pass 3 phases)
        add_pass(cbl_df, cbl_idx, 3)
        total_amount = insurer_df.loc[insurer_indices, 'ProcessedAmount_Clean_INSURER'].sum()

        match_reason = (
            f"Name Match (Line-by-Line): "
            f"{match['cbl_primary'][:40]} ~ {match['ins_primary'][:40]} "
            f"({match['similarity']:.0f}% sim, {match['confidence']})"
        )

        _apply_cluster_exact_match(
            cbl_df, cbl_idx, match_reason, insurer_indices,
            total_amount, 3, global_tracker,
            confidence_level=match['confidence'],
            amount_difference=match['difference']
        )

        exact_count += 1
        matched_cbl.add(cbl_idx)
        for idx in insurer_indices:
            used_insurer.add(idx)

        local_group_counter += 1
        cbl_df.at[cbl_idx, 'group_id'] = f"NAME_GROUP_{local_group_counter}_LBL"

        logger.info(
            f"  ✓ CBL {cbl_idx}: {match['match_type']} match "
            f"'{match['cbl_primary'][:30]}' ~ '{match['ins_primary'][:30]}' "
            f"({match['similarity']:.0f}%, diff={match['difference']:.2f}, "
            f"insurer={insurer_indices})"
        )

    return exact_count, matched_cbl


def _placing_based_regrouping(cbl_df, insurer_df, global_tracker, tolerance=50):
    """
    Regroup partial matches by placing number.

    For each partial match group, check if CBL and insurer rows share placing numbers.
    If a placing-based sub-group's amounts balance within tolerance, upgrade to exact match.

    Args:
        cbl_df: Full CBL DataFrame (modified in-place)
        insurer_df: Full insurer DataFrame
        global_tracker: GlobalMatchTracker instance
        tolerance: Amount tolerance for exact match classification

    Returns:
        tuple: (upgraded_count, new_group_counter_offset)
    """
    partial_mask = cbl_df['match_status'] == 'Partial Match'
    partial_rows = cbl_df[partial_mask].copy()

    if partial_rows.empty:
        logger.info("No partial matches to regroup by placing number")
        return 0, 0

    logger.info(f"Analyzing {len(partial_rows)} partial match rows for placing-based regrouping")

    # Check if PlacingNo columns exist
    has_placing_cbl = 'PlacingNo_Clean' in cbl_df.columns
    has_placing_ins = 'PlacingNo_Clean_INSURER' in insurer_df.columns

    if not has_placing_cbl or not has_placing_ins:
        logger.info("PlacingNo columns not available — skipping placing-based regrouping")
        return 0, 0

    # Group partial match rows by group_id
    groups = {}
    for idx, row in partial_rows.iterrows():
        gid = row.get('group_id')
        if pd.notna(gid) and gid:
            groups.setdefault(gid, []).append(idx)

    logger.info(f"Found {len(groups)} partial match groups to analyze")

    upgraded_count = 0
    new_groups_created = 0

    for group_id, cbl_indices in groups.items():
        if len(cbl_indices) < 1:
            continue

        # Get the insurer indices for this group (all rows in the group share the same insurer indices)
        insurer_indices = cbl_df.at[cbl_indices[0], 'matched_insurer_indices']
        if not isinstance(insurer_indices, list) or not insurer_indices:
            continue

        # Build placing-number sub-groups
        # Map: placing_no -> {cbl_indices: [], insurer_indices: []}
        placing_subgroups = {}

        for cbl_idx in cbl_indices:
            placing = str(cbl_df.at[cbl_idx, 'PlacingNo_Clean']).strip()
            if placing and placing != 'nan' and placing != 'NAN':
                placing_subgroups.setdefault(placing, {'cbl': [], 'insurer': []})
                placing_subgroups[placing]['cbl'].append(cbl_idx)

        # Match insurer rows to placing sub-groups
        for ins_idx in insurer_indices:
            if ins_idx not in insurer_df.index:
                continue
            ins_placing = str(insurer_df.at[ins_idx, 'PlacingNo_Clean_INSURER']).strip()
            if ins_placing and ins_placing != 'nan' and ins_placing != 'NAN':
                # Check exact match first
                if ins_placing in placing_subgroups:
                    placing_subgroups[ins_placing]['insurer'].append(ins_idx)
                else:
                    # Check substring overlap match
                    for placing_key in placing_subgroups:
                        is_valid, _ = validate_substring_match(placing_key, ins_placing)
                        if is_valid:
                            placing_subgroups[placing_key]['insurer'].append(ins_idx)
                            break

        # Check if any placing sub-group can be upgraded to exact
        for placing_no, subgroup in placing_subgroups.items():
            sub_cbl = subgroup['cbl']
            sub_ins = subgroup['insurer']

            if not sub_cbl or not sub_ins:
                continue

            # Skip if this sub-group is the entire original group (no benefit to regrouping)
            if set(sub_cbl) == set(cbl_indices) and set(sub_ins) == set(insurer_indices):
                continue

            # Calculate amounts for the sub-group
            cbl_total = cbl_df.loc[sub_cbl, 'ProcessedAmount_Clean'].sum()
            ins_total = insurer_df.loc[sub_ins, 'ProcessedAmount_Clean_INSURER'].sum()
            difference = abs(cbl_total + ins_total)

            match_type, _, confidence = classify_amount_match(cbl_total, ins_total, tolerance)
            is_exact = match_type in ["PERFECT_MATCH", "EXACT_MATCH"]

            if is_exact:
                # Check insurer availability
                can_use, available_idx, _ = global_tracker.can_use_for_exact(sub_ins)
                if not available_idx:
                    continue

                sub_ins = available_idx
                ins_total = insurer_df.loc[sub_ins, 'ProcessedAmount_Clean_INSURER'].sum()
                difference = abs(cbl_total + ins_total)
                match_type, _, confidence = classify_amount_match(cbl_total, ins_total, tolerance)

                if match_type not in ["PERFECT_MATCH", "EXACT_MATCH"]:
                    continue

                new_groups_created += 1
                new_gid = f"{group_id}_PLACING_{new_groups_created}"

                logger.info(f"\n  Placing sub-group upgrade: {group_id} → {new_gid}")
                logger.info(f"    Placing No: {placing_no}")
                logger.info(f"    CBL rows: {sub_cbl}, Insurer rows: {sub_ins}")
                logger.info(f"    CBL Total: Rs{cbl_total:.2f}, Insurer Total: Rs{ins_total:.2f}")
                logger.info(f"    Difference: Rs{difference:.2f} → EXACT ({confidence})")

                for cbl_idx in sub_cbl:
                    match_reason = (
                        f"Placing Regroup: {placing_no} "
                        f"({len(sub_cbl)} CBL, {len(sub_ins)} insurer, {confidence})"
                    )

                    _apply_cluster_exact_match(
                        cbl_df, cbl_idx, match_reason, sub_ins,
                        ins_total, 3, global_tracker,
                        confidence_level=confidence,
                        amount_difference=difference
                    )

                    cbl_df.at[cbl_idx, 'group_id'] = new_gid
                    upgraded_count += 1

                    logger.info(f"    ✓ CBL {cbl_idx}: Upgraded to EXACT via placing regroup")

                # Remove upgraded insurer indices from remaining CBL rows in the original group
                remaining_cbl_in_group = [idx for idx in cbl_indices if idx not in sub_cbl]
                if remaining_cbl_in_group:
                    for cbl_idx in remaining_cbl_in_group:
                        current_ins = cbl_df.at[cbl_idx, 'matched_insurer_indices']
                        if isinstance(current_ins, list):
                            updated_ins = [i for i in current_ins if i not in sub_ins]
                            cbl_df.at[cbl_idx, 'matched_insurer_indices'] = updated_ins

                            # Recalculate amounts for remaining group
                            if updated_ins:
                                new_ins_total = insurer_df.loc[updated_ins, 'ProcessedAmount_Clean_INSURER'].sum()
                                cbl_df.at[cbl_idx, 'matched_amtdue_total'] = new_ins_total
                                cbl_amt = cbl_df.at[cbl_idx, 'ProcessedAmount_Clean']
                                new_diff = abs(cbl_df.loc[remaining_cbl_in_group, 'ProcessedAmount_Clean'].sum() + new_ins_total)
                                cbl_df.at[cbl_idx, 'Amount Difference'] = round(new_diff, 2)

    return upgraded_count, new_groups_created


def pass3(cbl_df, insurer_df, tolerance=50, fuzzy_threshold=85, global_tracker=None):
    """
    Pass 3: Intelligent Name Matching with Corporate Root & Fuzzy Clustering.

    UNIFIED APPROACH: Combines corporate root extraction with fuzzy clustering
    for comprehensive name-based matching with amount validation.

    Business Rule: Group records by name similarity, then validate amounts to classify
    as Exact Match or Partial Match.

    This pass processes CBL records with status "No Match" or "Partial Match".
    Main goal: Move records from "No Match" to matched status, or upgrade "Partial Match".

    Two-Phase Matching Strategy:
        Phase 1 - Exact Corporate Root Matching (Fast & Precise):
            1. Extract corporate roots using intelligent detection:
               - Corporate names: 1-2 distinctive words (smart parent detection)
               - Person names: 3 name words (skip titles like MR, MRS, DR)
               - Financial relationships: Extract lessee (ONLEASE TO, ON LEASE TO)
               - Organizational prefixes: Skip prefixes to extract property/building name
            2. Match CBL and insurer records with identical roots
            3. Group by exact root match

        Phase 2 - Fuzzy Clustering Fallback (Catches Variations & Typos):
            1. For remaining unmatched records, build fuzzy name clusters (similarity >= threshold)
            2. Match clusters across CBL and insurer using intelligent similarity
            3. Validate with first-word matching to prevent false positives

        Phase 3 - Amount Validation & Classification:
            1. Calculate cumulative amounts for each group
            2. Classify based on amount difference:
               - Within tolerance → Exact Match
               - Beyond tolerance → Partial Match

    Examples with Smart Detection:
        # Corporate: Parent company indicators
        - "ALTEO AGRI LTD" → "ALTEO AGRI" (2 words)
        - "ALTEO MILLING LTD" → "ALTEO MILLING" (2 words)
        - "ALTEO GROUP OF COMPANIES" → "ALTEO" (1 word - GROUP is parent indicator)
        → Result: All ALTEO subsidiaries match ALTEO GROUP ✓

        # Different companies with common words (prevented by first-word check)
        - "SHA TRAVEL TOURS" → First word "SHA"
        - "TAJ TRAVEL TOURS" → First word "TAJ"
        → Result: Different companies stay separate ✓

        # Financial relationships (extract lessee)
        - "MCB LEASING ONLEASE TO ECOBAT" → Extract "ECOBAT" (the lessee)
        - "MCB LEASING LIMITED" → Extract "MCB LEASING"
        → Result: ECOBAT grouped separately from MCB ✓

        # Person names: Extract 3 name words (skip titles)
        - "MRS MARIE BERTHE CHANTAL HARDY" → "MARIE BERTHE CHANTAL" (3 words)
        - "MRS MARIE DESIRE CATHERINE BOYER" → "MARIE DESIRE CATHERINE" (3 words)
        → Result: Different people stay separate ✓

        # Organizational prefixes: Extract property/building name
        - "SYNDICAT DES COPROPRIETAIRES DE LES TERRASSES DU BARACHOIS" → "LES TERRASSES"
        - "LE SYNDICAT DES COPROPRIETAIRES DU CENTRE FINANCIER DU NORD" → "CENTRE FINANCIER"
        - "SYNDICAT DES COPROPRIETAIRES LA LUXURY PALMERAIE" → "LUXURY PALMERAIE"
        → Result: Different syndicates stay separate by property name ✓

        # Fuzzy variations (caught by Phase 2)
        - "ACME LIMITED" vs "ACME LTD" → 95% similarity → Match ✓
        - "CITY BROKERS (MAURITIUS)" vs "CITY BROKERS LTD" → 90% similarity → Match ✓

    Args:
        cbl_df: CBL DataFrame with match results from previous passes
        insurer_df: Insurer DataFrame
        tolerance: Amount tolerance for exact match classification (default: 100)
        fuzzy_threshold: Minimum similarity for fuzzy clustering (default: 85)
        global_tracker: GlobalMatchTracker instance for consistent row usage tracking

    Returns:
        cbl_df: Updated CBL DataFrame with name-based matches
    """
    logger.info("\n=== Pass 3: Intelligent Name Matching (Corporate Root + Fuzzy Clustering) ===")
    logger.info("Strategy: Phase 1 - Exact corporate root matching | Phase 2 - Fuzzy clustering fallback")
    logger.info(f"Amount Tolerance: Rs{tolerance} (within → Exact Match, beyond → Partial Match)")
    logger.info(f"Fuzzy Threshold: {fuzzy_threshold}%")

    exact_matches = 0
    partial_matches = 0

    logger.info(f"Pass 3 starting with global tracker: {global_tracker.get_usage_summary()}")

    # Get unmatched or partial match CBL records
    # Exclude history-pre-placed rows — user manual placements are authoritative
    status_mask = cbl_df['match_status'].isin(['No Match', 'Partial Match'])
    if 'match_resolved_in_pass' in cbl_df.columns:
        status_mask = status_mask & (~cbl_df['match_resolved_in_pass'].isin(['history', 'matrix']))
    unmatched_cbl = cbl_df[status_mask].copy()
    logger.info(f"Processing {len(unmatched_cbl)} CBL records with 'No Match' or 'Partial Match' status")

    if unmatched_cbl.empty:
        logger.info("No unmatched records to process")
        return cbl_df

    # Use global tracker for consistent filtering
    already_matched_insurer = global_tracker.exact_used_insurer | global_tracker.matrix_used_insurer
    available_insurer = insurer_df[~insurer_df.index.isin(already_matched_insurer)].copy()
    logger.info(f"Pass 3: Using global tracker - excluding {len(already_matched_insurer)} exact/matrix used insurer rows")
    logger.info(f"Pass 3: Available insurer rows for name matching: {len(available_insurer)}")

    if available_insurer.empty:
        logger.info("No available insurer records to match")
        return cbl_df

    # ========== PHASE 1: EXACT CORPORATE ROOT MATCHING ==========
    logger.info("\n=== Phase 1: Exact Corporate Root Matching ===")

    # Build corporate root indices
    # For CBL: also get primary_root_map to control group assignment
    cbl_root_index, cbl_primary_root_map = _build_corporate_root_index(
        unmatched_cbl, 'ClientName', 'CBL', min_occurrence=1, return_primary_map=True
    )
    insurer_root_index = _build_corporate_root_index(available_insurer, 'ClientName_INSURER', 'INSURER', min_occurrence=1)

    logger.info(f"Found {len(cbl_root_index)} CBL corporate roots")
    logger.info(f"Found {len(insurer_root_index)} insurer corporate roots")
    logger.info(f"Tracked {len(cbl_primary_root_map)} CBL primary roots for group assignment")

    # Track which CBL indices were matched in Phase 1
    phase1_matched_cbl = set()
    group_counter = 0

    if cbl_root_index and insurer_root_index:
        logger.info("\n--- Pre-grouping: Consolidating CBL records using PRIMARY ROOT assignment ---")
        logger.info("  📌 Key change: CBL rows are only added to groups that match their PRIMARY root")
        logger.info("     e.g., 'KASA GROUP... - REY...' → assigned to KASA group only (not REY)")

        # STEP 1: Collect all root matches (without applying yet)
        # KEY CHANGE: Only add a CBL to a group if the group's root matches CBL's PRIMARY root
        # This prevents compound names like "KASA GROUP... - REY..." from being assigned to REY group

        # Map: insurer_indices_tuple -> {cbl_indices: set, roots: list}
        insurer_to_cbl_groups = {}

        # Track skipped assignments for logging
        skipped_secondary_assignments = 0

        for root in cbl_root_index.keys():
            if root in insurer_root_index:
                insurer_indices = tuple(sorted(insurer_root_index[root]))  # Tuple for hashing

                if insurer_indices not in insurer_to_cbl_groups:
                    insurer_to_cbl_groups[insurer_indices] = {
                        'cbl_indices': set(),
                        'roots': []
                    }

                # KEY FIX: Only add CBL indices where this root is the PRIMARY root
                # This ensures "KASA GROUP... - REY..." goes to KASA group, not REY group
                for cbl_idx in cbl_root_index[root]:
                    primary_root = cbl_primary_root_map.get(cbl_idx, "")

                    # Check if this root matches the CBL's primary root
                    if primary_root == root:
                        # Direct match - add to group
                        insurer_to_cbl_groups[insurer_indices]['cbl_indices'].add(cbl_idx)
                    elif not primary_root:
                        # No primary root extracted (edge case) - add to group
                        insurer_to_cbl_groups[insurer_indices]['cbl_indices'].add(cbl_idx)
                    else:
                        # This root is a SECONDARY root for this CBL (e.g., REY in "KASA... - REY...")
                        # Skip - this CBL should be assigned via its PRIMARY root
                        skipped_secondary_assignments += 1
                        logger.debug(f"Skipping secondary assignment: CBL {cbl_idx} matched on '{root}' but primary is '{primary_root}'")

                if root not in insurer_to_cbl_groups[insurer_indices]['roots']:
                    insurer_to_cbl_groups[insurer_indices]['roots'].append(root)

        # Log pre-grouping results
        if insurer_to_cbl_groups:
            multi_cbl_groups = sum(1 for g in insurer_to_cbl_groups.values() if len(g['cbl_indices']) > 1)
            multi_root_groups = sum(1 for g in insurer_to_cbl_groups.values() if len(g['roots']) > 1)
            logger.info(f"  📊 Pre-grouping Results:")
            logger.info(f"     - Total insurer groups: {len(insurer_to_cbl_groups)}")
            logger.info(f"     - Groups with multiple CBL records: {multi_cbl_groups}")
            logger.info(f"     - Groups matched via multiple roots: {multi_root_groups}")
            if skipped_secondary_assignments > 0:
                logger.info(f"     - ✅ Skipped {skipped_secondary_assignments} secondary root assignments (prevented cross-group duplication)")

            # Log groups that would have caused duplication
            for insurer_indices, group_data in insurer_to_cbl_groups.items():
                if len(group_data['roots']) > 1:
                    logger.info(f"     ⚠️ Insurer indices {list(insurer_indices)[:3]}... matched via {len(group_data['roots'])} roots: {group_data['roots'][:5]}")
                    logger.info(f"        → Contains {len(group_data['cbl_indices'])} CBL records (primary-root filtered)")

        # Remove empty groups (all CBL rows filtered out due to primary root logic)
        insurer_to_cbl_groups = {
            k: v for k, v in insurer_to_cbl_groups.items()
            if len(v['cbl_indices']) > 0
        }

        logger.info("\n--- Matching Consolidated Groups ---")

        # STEP 2: Apply matches at the consolidated group level (not per-root)
        # Sort by number of CBL rows for consistent ordering
        sorted_groups = sorted(
            insurer_to_cbl_groups.items(),
            key=lambda x: len(x[1]['cbl_indices']),
            reverse=True  # Largest groups first
        )

        logger.info(f"  📋 Processing {len(sorted_groups)} groups (primary-root filtered)")

        for insurer_indices_tuple, group_data in sorted_groups:
            insurer_indices = list(insurer_indices_tuple)
            cbl_indices = list(group_data['cbl_indices'])
            matched_roots = group_data['roots']

            group_counter += 1
            group_id = f"NAME_GROUP_{group_counter}_ROOT"

            # Calculate group totals for amount validation
            cbl_total = cbl_df.loc[cbl_indices, 'ProcessedAmount_Clean'].sum()
            insurer_total = available_insurer.loc[insurer_indices, 'ProcessedAmount_Clean_INSURER'].sum()
            difference = abs(cbl_total + insurer_total)

            # Classify match based on amount difference
            match_type, _, confidence = classify_amount_match(cbl_total, insurer_total, tolerance)
            is_exact_match = match_type in ["PERFECT_MATCH", "EXACT_MATCH"]

            # Format roots for display
            roots_display = ', '.join(matched_roots[:3]) + ('...' if len(matched_roots) > 3 else '')

            logger.info(f"\n🏢 Corporate Root Match Found:")
            logger.info(f"  Group ID: {group_id}")
            logger.info(f"  Corporate Root(s): {roots_display} ({len(matched_roots)} total)")
            logger.info(f"  CBL Records: {len(cbl_indices)}")
            logger.info(f"  Insurer Records: {len(insurer_indices)}")
            logger.info(f"  CBL Total: Rs{cbl_total:.2f}")
            logger.info(f"  Insurer Total: Rs{insurer_total:.2f}")
            logger.info(f"  Difference: Rs{difference:.2f}")
            logger.info(f"  Classification: {'EXACT' if is_exact_match else 'PARTIAL'} ({confidence} Confidence)")

            # Validate insurer indices availability
            if is_exact_match:
                can_use_all, available_indices, conflicts = global_tracker.can_use_for_exact(insurer_indices)
            else:
                # Use allow_sharing=False to prevent duplication across groups
                can_use_all, available_indices, conflicts = global_tracker.can_use_for_partial(insurer_indices, allow_sharing=False)

            if not available_indices:
                logger.warning(f"  ⚠ No available insurer indices - skipping group")
                continue

            if not can_use_all:
                logger.info(f"  ℹ Using {len(available_indices)}/{len(insurer_indices)} available indices")
                insurer_indices = available_indices
                # Recalculate with available indices
                insurer_total = available_insurer.loc[insurer_indices, 'ProcessedAmount_Clean_INSURER'].sum()
                difference = abs(cbl_total + insurer_total)
                match_type, _, confidence = classify_amount_match(cbl_total, insurer_total, tolerance)
                is_exact_match = match_type in ["PERFECT_MATCH", "EXACT_MATCH"]
                logger.info(f"  Reclassified: {'EXACT' if is_exact_match else 'PARTIAL'} ({confidence} Confidence)")

            # Apply matches to all CBL records in this consolidated group
            # IMPORTANT: Filter out CBL rows already matched to another group
            # This prevents overwriting when a CBL row has multiple roots (e.g., compound names)
            unmatched_cbl_in_group = [idx for idx in cbl_indices if idx not in phase1_matched_cbl]
            skipped_count = len(cbl_indices) - len(unmatched_cbl_in_group)

            if skipped_count > 0:
                logger.info(f"  ℹ️ {skipped_count} CBL row(s) already matched to another group - skipping")

                # Recalculate totals with only unmatched CBL rows
                if unmatched_cbl_in_group:
                    cbl_total = cbl_df.loc[unmatched_cbl_in_group, 'ProcessedAmount_Clean'].sum()
                    difference = abs(cbl_total + insurer_total)
                    match_type, _, confidence = classify_amount_match(cbl_total, insurer_total, tolerance)
                    is_exact_match = match_type in ["PERFECT_MATCH", "EXACT_MATCH"]
                    logger.info(f"  Recalculated with {len(unmatched_cbl_in_group)} CBL rows: Diff Rs{difference:.2f}, {'EXACT' if is_exact_match else 'PARTIAL'}")

            if not unmatched_cbl_in_group:
                logger.warning(f"  ⚠ No unmatched CBL rows remaining in this group - skipping entirely")
                continue

            for cbl_idx in unmatched_cbl_in_group:
                add_pass(cbl_df, cbl_idx, 3)
                phase1_matched_cbl.add(cbl_idx)

                total_insurer_amount = available_insurer.loc[insurer_indices, "ProcessedAmount_Clean_INSURER"].sum()
                cbl_amount = cbl_df.at[cbl_idx, "ProcessedAmount_Clean"]

                # Include all matched roots in the reason
                match_reason = f"Corporate Root: {roots_display} ({len(unmatched_cbl_in_group)} CBL, {len(insurer_indices)} insurer, {confidence})"

                if is_exact_match:
                    _apply_cluster_exact_match(
                        cbl_df, cbl_idx, match_reason, insurer_indices,
                        total_insurer_amount, 3, global_tracker,
                        confidence_level=confidence,
                        amount_difference=difference
                    )
                    exact_matches += 1
                    logger.info(f"  ✓ CBL {cbl_idx}: EXACT (CBL: Rs{cbl_amount:.2f})")
                else:
                    partial_matches += _apply_partial_match(
                        cbl_df, cbl_idx, match_reason, insurer_indices,
                        total_insurer_amount, 3, global_tracker,
                        confidence_level=confidence,
                        amount_difference=difference
                    )
                    logger.info(f"  ✓ CBL {cbl_idx}: PARTIAL (CBL: Rs{cbl_amount:.2f})")

                cbl_df.at[cbl_idx, 'group_id'] = group_id
                cbl_df.at[cbl_idx, 'corporate_root'] = roots_display

        logger.info(f"\n✓ Phase 1 Complete: {len(phase1_matched_cbl)} CBL records matched by corporate root")
    else:
        logger.info("No corporate root matches found")

    # ========== PHASE 2: FUZZY CLUSTERING FALLBACK ==========
    logger.info("\n=== Phase 2: Fuzzy Clustering Fallback (for remaining records) ===")

    # Track which CBL indices were matched in Phase 2
    phase2_matched_cbl = set()

    # Get CBL records that weren't matched in Phase 1
    remaining_cbl = unmatched_cbl[~unmatched_cbl.index.isin(phase1_matched_cbl)].copy()
    logger.info(f"Processing {len(remaining_cbl)} remaining CBL records with fuzzy clustering")

    if not remaining_cbl.empty:
        # Build fuzzy name clusters
        cbl_name_clusters = _build_fuzzy_name_clusters(
            remaining_cbl,
            name_column='ClientName',
            fuzzy_threshold=fuzzy_threshold,
            prefix="CBL"
        )

        insurer_name_clusters = _build_fuzzy_name_clusters(
            available_insurer,
            name_column='ClientName_INSURER',
            fuzzy_threshold=fuzzy_threshold,
            prefix="INSURER"
        )

        logger.info(f"Created {len(cbl_name_clusters)} CBL clusters, {len(insurer_name_clusters)} insurer clusters")

        if cbl_name_clusters and insurer_name_clusters:
            logger.info("\n--- Matching Fuzzy Clusters ---")

            # Use CompanyNameMatcher for cross-cluster matching
            matcher = CompanyNameMatcher(primary_penalty=0.3, exact_match_boost=2.5)

            for cbl_cluster_name, cbl_indices in cbl_name_clusters.items():
                for insurer_cluster_name, insurer_indices in insurer_name_clusters.items():
                    # Use intelligent similarity calculation
                    cluster_similarity = matcher.calculate_intelligent_similarity(cbl_cluster_name, insurer_cluster_name)

                    if cluster_similarity >= fuzzy_threshold:
                        # VALIDATION: Check word overlap to prevent false positives
                        has_overlap, common_count, common_words = _has_sufficient_word_overlap(
                            cbl_cluster_name, insurer_cluster_name, min_common_words=2
                        )

                        if not has_overlap:
                            logger.debug(f"Rejected cross-cluster match: '{cbl_cluster_name[:40]}' vs '{insurer_cluster_name[:40]}' "
                                       f"(only {common_count} common words: {common_words})")
                            continue

                        group_counter += 1
                        group_id = f"NAME_GROUP_{group_counter}_FUZZY"

                        # Calculate totals
                        cbl_total = cbl_df.loc[cbl_indices, 'ProcessedAmount_Clean'].sum()
                        insurer_total = available_insurer.loc[insurer_indices, 'ProcessedAmount_Clean_INSURER'].sum()
                        difference = abs(cbl_total + insurer_total)

                        # Classify match
                        match_type, _, confidence = classify_amount_match(cbl_total, insurer_total, tolerance)
                        is_exact_match = match_type in ["PERFECT_MATCH", "EXACT_MATCH"]

                        logger.info(f"\n🎯 Fuzzy Cluster Match Found:")
                        logger.info(f"  Group ID: {group_id}")
                        logger.info(f"  CBL Cluster: '{cbl_cluster_name[:50]}...' ({len(cbl_indices)} records)")
                        logger.info(f"  Insurer Cluster: '{insurer_cluster_name[:50]}...' ({len(insurer_indices)} records)")
                        logger.info(f"  Similarity: {cluster_similarity}%")
                        logger.info(f"  CBL Total: Rs{cbl_total:.2f}")
                        logger.info(f"  Insurer Total: Rs{insurer_total:.2f}")
                        logger.info(f"  Difference: Rs{difference:.2f}")
                        logger.info(f"  Classification: {'EXACT' if is_exact_match else 'PARTIAL'} ({confidence} Confidence)")

                        # Validate availability
                        if is_exact_match:
                            can_use_all, available_indices, conflicts = global_tracker.can_use_for_exact(insurer_indices)
                        else:
                            can_use_all, available_indices, conflicts = global_tracker.can_use_for_partial(insurer_indices, allow_sharing=False)

                        if not available_indices:
                            logger.warning(f"  ⚠ No available insurer indices - skipping cluster")
                            continue

                        if not can_use_all:
                            logger.info(f"  ℹ Using {len(available_indices)}/{len(insurer_indices)} available indices")
                            insurer_indices = available_indices

                        # Apply matches
                        for cbl_idx in cbl_indices:
                            add_pass(cbl_df, cbl_idx, 3)
                            phase2_matched_cbl.add(cbl_idx)  # Track Phase 2 matches

                            total_insurer_amount = available_insurer.loc[insurer_indices, "ProcessedAmount_Clean_INSURER"].sum()
                            cbl_amount = cbl_df.at[cbl_idx, "ProcessedAmount_Clean"]
                            amount_diff = abs(cbl_amount + total_insurer_amount)

                            match_reason = f"Fuzzy Cluster: '{cbl_cluster_name[:30]}...' (Sim: {cluster_similarity}%)"

                            if is_exact_match:
                                _apply_cluster_exact_match(
                                    cbl_df, cbl_idx, match_reason, insurer_indices,
                                    total_insurer_amount, 3, global_tracker,
                                    confidence_level=confidence,
                                    amount_difference=amount_diff
                                )
                                exact_matches += 1
                                logger.info(f"  ✓ CBL {cbl_idx}: EXACT (CBL: Rs{cbl_amount:.2f})")
                            else:
                                partial_matches += _apply_partial_match(
                                    cbl_df, cbl_idx, match_reason, insurer_indices,
                                    total_insurer_amount, 3, global_tracker,
                                    confidence_level=confidence,
                                    amount_difference=amount_diff
                                )
                                logger.info(f"  ✓ CBL {cbl_idx}: PARTIAL (CBL: Rs{cbl_amount:.2f})")

                            cbl_df.at[cbl_idx, 'group_id'] = group_id

            logger.info(f"\n✓ Phase 2 Complete: {len(phase2_matched_cbl)} records matched via fuzzy clustering")
        else:
            logger.info("No fuzzy clusters created")
    else:
        logger.info("No remaining records for fuzzy clustering")

    # ========== PHASE 3: SECONDARY ROOT LOOSE CAPTURE ==========
    logger.info("\n=== Phase 3: Secondary Root Loose Capture ===")
    logger.info("Purpose: Match remaining CBL rows using secondary roots from compound names (&/OR entities)")

    # Track Phase 3 matches
    phase3_matched_cbl = set()
    phase3_matches = 0

    # Get CBL records still unmatched after Phase 1 & 2
    already_matched = phase1_matched_cbl | phase2_matched_cbl
    remaining_for_phase3 = unmatched_cbl[~unmatched_cbl.index.isin(already_matched)].copy()

    logger.info(f"Processing {len(remaining_for_phase3)} remaining unmatched CBL records")

    if not remaining_for_phase3.empty and cbl_root_index and insurer_root_index:
        # For each remaining unmatched CBL row
        for cbl_idx in remaining_for_phase3.index:
            cbl_name = remaining_for_phase3.at[cbl_idx, 'ClientName']

            # Get ALL roots for this CBL (including secondary roots from &/OR entities)
            all_roots = _extract_all_corporate_roots(cbl_name)
            primary_root = cbl_primary_root_map.get(cbl_idx, "")

            # Get secondary roots (exclude primary - already tried in Phase 1)
            secondary_roots = [r for r in all_roots if r != primary_root]

            if not secondary_roots:
                continue  # No secondary roots to try

            logger.debug(f"CBL {cbl_idx}: Primary='{primary_root}', trying {len(secondary_roots)} secondary roots")

            # Try each secondary root
            matched_via_secondary = False
            for sec_root in secondary_roots:
                if sec_root not in insurer_root_index:
                    continue  # This secondary root doesn't match any insurer

                insurer_indices = insurer_root_index[sec_root]

                # Validate insurer availability
                can_use_all, available_indices, conflicts = global_tracker.can_use_for_partial(
                    insurer_indices, allow_sharing=False
                )

                if not available_indices:
                    logger.debug(f"  Secondary root '{sec_root}': No available insurer indices")
                    continue

                # Found a match via secondary root!
                insurer_indices = available_indices

                # Calculate amounts
                cbl_amount = cbl_df.at[cbl_idx, 'ProcessedAmount_Clean']
                insurer_total = available_insurer.loc[insurer_indices, 'ProcessedAmount_Clean_INSURER'].sum()
                difference = abs(cbl_amount + insurer_total)

                # Classify match - secondary root matches are always PARTIAL (lower confidence)
                match_type, _, confidence = classify_amount_match(cbl_amount, insurer_total, tolerance)

                # For secondary root matches, cap confidence at MEDIUM
                if confidence == "High":
                    confidence = "Medium"

                group_counter += 1
                group_id = f"NAME_GROUP_{group_counter}_SECONDARY"

                logger.info(f"\n🔗 Secondary Root Match Found:")
                logger.info(f"  CBL {cbl_idx}: '{cbl_name[:60]}...'")
                logger.info(f"  Primary Root: '{primary_root}' (no insurer match)")
                logger.info(f"  Secondary Root: '{sec_root}' → MATCHED!")
                logger.info(f"  Insurer Records: {len(insurer_indices)}")
                logger.info(f"  CBL Amount: Rs{cbl_amount:.2f}")
                logger.info(f"  Insurer Total: Rs{insurer_total:.2f}")
                logger.info(f"  Difference: Rs{difference:.2f}")
                logger.info(f"  Classification: PARTIAL ({confidence} Confidence - Secondary Root)")

                # Apply partial match (secondary root matches are always partial for safety)
                add_pass(cbl_df, cbl_idx, 3)

                match_reason = f"Secondary Root: '{sec_root}' (loose capture, Primary: '{primary_root}')"

                partial_matches += _apply_partial_match(
                    cbl_df, cbl_idx, match_reason, insurer_indices,
                    insurer_total, 3, global_tracker,
                    confidence_level=confidence,
                    amount_difference=difference
                )

                cbl_df.at[cbl_idx, 'group_id'] = group_id
                cbl_df.at[cbl_idx, 'corporate_root'] = f"{primary_root} → {sec_root} (secondary)"

                phase3_matched_cbl.add(cbl_idx)
                phase3_matches += 1
                matched_via_secondary = True

                logger.info(f"  ✓ CBL {cbl_idx}: PARTIAL via secondary root")
                break  # Only match once per CBL row

            if not matched_via_secondary:
                logger.debug(f"CBL {cbl_idx}: No secondary root matched any insurer")

        logger.info(f"\n✓ Phase 3 Complete: {phase3_matches} matches via secondary roots")
    else:
        logger.info("No remaining records for secondary root matching")

    # ========== PHASE 4: LINE-BY-LINE NAME + AMOUNT MATCHING ==========
    logger.info("\n=== Phase 4: Line-by-Line Name + Amount Matching ===")
    logger.info("Purpose: Match individual CBL rows to specific insurer rows by name similarity + amount validation")

    phase4_matched_cbl = set()
    phase4_matches = 0

    # Get CBL records still unmatched after Phases 1-3
    already_matched_phases = phase1_matched_cbl | phase2_matched_cbl | phase3_matched_cbl
    remaining_for_phase4 = unmatched_cbl[~unmatched_cbl.index.isin(already_matched_phases)].copy()

    # Refresh available insurer rows (some may have been used in Phases 1-3)
    already_used_insurer = global_tracker.exact_used_insurer | global_tracker.matrix_used_insurer
    available_insurer_p4 = insurer_df[~insurer_df.index.isin(already_used_insurer)].copy()

    logger.info(f"Processing {len(remaining_for_phase4)} remaining CBL records against {len(available_insurer_p4)} available insurer rows")

    if not remaining_for_phase4.empty and not available_insurer_p4.empty:
        lbl_exact, lbl_matched = _line_by_line_name_matching(
            cbl_df, remaining_for_phase4, available_insurer_p4, insurer_df,
            global_tracker, tolerance, name_threshold=85, group_counter=group_counter
        )
        exact_matches += lbl_exact
        phase4_matched_cbl = lbl_matched
        phase4_matches = len(lbl_matched)
        group_counter += phase4_matches

        logger.info(f"\n✓ Phase 4 Complete: {phase4_matches} matches via line-by-line name matching")
    else:
        logger.info("No remaining records for line-by-line matching")

    # ========== PHASE 5: PLACING-BASED REGROUPING OF PARTIAL MATCHES ==========
    logger.info("\n=== Phase 5: Placing-Based Regrouping of Partial Matches ===")
    logger.info("Purpose: Sub-divide partial match groups by placing number to find exact sub-matches")

    phase5_upgraded, phase5_new_groups = _placing_based_regrouping(
        cbl_df, insurer_df, global_tracker, tolerance
    )
    exact_matches += phase5_upgraded

    logger.info(f"\n✓ Phase 5 Complete: {phase5_upgraded} partial matches upgraded to exact via placing regrouping ({phase5_new_groups} new sub-groups)")

    # ========== PHASE 6: MERGE GROUPS WITH OVERLAPPING INSURER INDICES ==========
    cbl_df = _merge_groups_with_overlapping_insurer_indices(cbl_df, available_insurer, global_tracker)

    logger.info(f"\n✓ Pass 3 Complete: {exact_matches} exact matches, {partial_matches} partial matches in {group_counter} name groups")
    logger.info(f"   Phase 1 (Corporate Root - Primary): Matched {len(phase1_matched_cbl)} records")
    logger.info(f"   Phase 2 (Fuzzy Clustering): Matched {len(phase2_matched_cbl)} records")
    logger.info(f"   Phase 3 (Secondary Root - Loose): Matched {len(phase3_matched_cbl)} records")
    logger.info(f"   Phase 4 (Line-by-Line Name + Amount): Matched {len(phase4_matched_cbl)} records")
    logger.info(f"   Phase 5 (Placing-Based Regrouping): Upgraded {phase5_upgraded} partial → exact")
    return cbl_df
