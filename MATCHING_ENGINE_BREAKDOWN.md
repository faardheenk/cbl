# Matching Engine Breakdown: Detailed Explanation

This document provides a comprehensive breakdown of the `matching_engine.py` codebase, explaining how each matching pass works in detail.

## Table of Contents
1. [Overview](#overview)
2. [Core Components](#core-components)
3. [Pass 1: Placing Number + Amount Matching](#pass-1-placing-number--amount-matching)
4. [Pass 2: Policy Number + Amount Matching](#pass-2-policy-number--amount-matching)
5. [Pass 3: Name-Based Matching](#pass-3-name-based-matching)
6. [Helper Functions](#helper-functions)

---

## Overview

The matching engine uses a **multi-pass approach** to match CBL (City Brokers Limited) records with Insurer records. Each pass progressively refines matches using different criteria:

- **Pass 1**: Exact matches using Placing Numbers + Amounts
- **Pass 2**: Matches using Policy Numbers + Amounts  
- **Pass 3**: Intelligent name-based matching with corporate root extraction and fuzzy clustering

The system uses a `GlobalMatchTracker` to ensure **data integrity** - preventing row reuse and managing 1:1 or 1:many relationships correctly.

---

## Core Components

### 1. GlobalMatchTracker

**Purpose**: Prevents row reuse across all matching passes, ensuring data integrity.

**Key Features**:
- Tracks which insurer rows are used in matrix, exact, and partial matches
- Prevents multiple CBL rows from claiming the same insurer row (for exact matches)
- Allows multiple CBL rows to share insurer rows (for partial matches in Pass 3)
- Automatically updates CBL DataFrame when conflicts occur

**Data Structures**:
- `matrix_used_insurer`: Insurer rows used in matrix pass
- `exact_used_insurer`: Insurer rows used in exact matches
- `partial_used_insurer`: Insurer rows used in partial matches
- `cbl_exact_matches`: Maps CBL index → insurer indices (exact matches)
- `cbl_partial_matches`: Maps CBL index → insurer indices (partial matches)
- `insurer_to_cbl_exact`: Reverse mapping (1:1 for exact matches)
- `insurer_to_cbl_partial`: Reverse mapping (1:many for partial matches)

### 2. CompanyNameMatcher

**Purpose**: Intelligent company name matching that prevents over-clustering.

**Key Features**:
- Extracts primary company names from financial relationships
  - Example: "SPICE FINANCE LTD ON LEASE TO VITIRO LTD" → extracts "VITIRO LTD"
- Handles compound names with "&/OR" patterns
- Applies intelligent penalties to prevent false matches
- Uses caching for performance optimization

**Methods**:
- `extract_primary_company()`: Extracts primary entity from complex names
- `calculate_intelligent_similarity()`: Calculates similarity with relationship awareness

---

## Pass 1: Placing Number + Amount Matching

**Purpose**: Match CBL records to Insurer records using Placing Numbers and ProcessedAmounts.

**Input**: All CBL records (no filtering)

**Strategy**: Two-phase approach
1. **Phase 1**: Collect all potential matches without applying them
2. **Phase 2**: Resolve conflicts and apply matches (prioritizing combination matches)

### Phase 1: Match Collection

For each CBL record:

1. **Extract Placing Number and Amount**
   - `placing = row["PlacingNo_Clean"]`
   - `amt1 = row["ProcessedAmount_Clean"]`

2. **Find Insurer Matches by Placing Number**
   - First try **exact match**: `insurer_df[insurer_df["PlacingNo_Clean_INSURER"] == placing]`
   - If no exact match, try **substring matching** (quality-controlled):
     - Only if CBL placing number is ≥ 10 characters
     - Validates substring matches using `validate_substring_match()`
     - Requires ≥80% overlap and minimum length of 10 chars
     - Example: "ABC123456789" matches "ABC1234567890" (substring)

3. **Individual Amount Matching**
   - For each matched insurer record, check if amount matches:
     - Uses `classify_amount_match()` to determine match type
     - Only accepts `PERFECT_MATCH` or `EXACT_MATCH` (within tolerance)
     - Creates exact match if found

4. **Combination Amount Matching** (if no individual match)
   - **Purpose**: When a single insurer record doesn't match, try combining multiple insurer records
   - **Smart Selection Process**:
     1. If there are more than 20 matched insurer records, limit to the 20 most promising
     2. Sort insurer records by how close each individual amount is to the target
     3. Target: `-amt1` (CBL amounts are negative, insurer amounts are positive)
        - Example: If CBL amount is -1000, target is 1000
        - We want: `sum(insurer_amounts) ≈ 1000`
     4. Select the 20 insurer records whose individual amounts are closest to 1000
   - **Combination Testing**:
     - Tries all combinations of 2, 3, 4, or 5 records from the selected 20
     - For each combination, calculates: `total_amount = sum(combination_amounts)`
     - Checks if `abs(amt1 + total_amount) ≤ tolerance`
     - Example: CBL has -1000, tries combinations:
       - 600 + 400 = 1000 → `abs(-1000 + 1000) = 0` → PERFECT_MATCH ✓
       - 700 + 250 = 950 → `abs(-1000 + 950) = 50` → EXACT_MATCH (if tolerance ≥ 50) ✓
       - 500 + 300 = 800 → `abs(-1000 + 800) = 200` → No match (if tolerance < 200) ✗
   - **Why Smart Selection?**: Prevents combinatorial explosion
     - Without limiting: 100 records → 4,950 combinations of 2, 161,700 combinations of 3, etc.
     - With limiting: 20 records → 190 combinations of 2, 1,140 combinations of 3, etc.
   - **Only accepts**: `PERFECT_MATCH` or `EXACT_MATCH` (within tolerance)

5. **Store Potential Matches**
   - Stores matches in `potential_matches` list with metadata:
     - Match type (exact or combination)
     - Insurer indices
     - Match reason
     - Confidence level
     - Amount difference

### Phase 2: Conflict Resolution & Application

1. **Sort Potential Matches**
   - Priority order:
     1. Exact matches first
     2. Combination matches second
     3. Within each type, larger combinations get priority

2. **Apply Matches**
   - For each potential match:
     - Check availability using `GlobalMatchTracker.can_cbl_claim_insurer()`
     - If conflicts exist, use `_handle_conflict_resolution()`
     - If no conflicts, apply using `_apply_exact_match()`
     - Updates CBL DataFrame with match status, reason, indices, etc.

**Output**: CBL records marked as "Exact Match" or remain "No Match"

**Key Characteristics**:
- Only creates exact matches (no partial matches)
- Handles substring matching for placing numbers
- Smart combination selection prevents combinatorial explosion
- Conflict resolution ensures data integrity

---

## Pass 2: Policy Number + Amount Matching

**Purpose**: Match remaining CBL records using Policy Numbers and Amounts.

**Input**: CBL records with status "No Match" or "Partial Match"

**Strategy**: Two-phase approach (similar to Pass 1)

### Phase 1: Match Collection

For each unmatched CBL record:

1. **Extract Policy Tokens and Amount**
   - `tokens = extract_policy_tokens(row["PolicyNo_Clean"])`
   - `cbl_amt = row["ProcessedAmount_Clean"]`

2. **Filter Available Insurer Records**
   - Uses `GlobalMatchTracker` to exclude already-used insurer rows
   - Only considers insurer rows NOT in `exact_used_insurer` or `matrix_used_insurer`
   - Allows upgrading partial matches to exact matches

3. **Policy Number Matching**
   - For each available insurer record:
     - Collects all policy values:
       - `PolicyNo_Clean_INSURER`
       - `PolicyNo_2_Clean_INSURER` (if exists)
     - Checks if any insurer policy number is in CBL tokens
     - Example: CBL has "POL123", insurer has "POL123" → match

4. **Amount Validation**
   - Sums amounts from all matched insurer records
   - Uses `classify_amount_match()` to classify
   - Only accepts `PERFECT_MATCH` or `EXACT_MATCH`
   - Creates match reason with confidence level

5. **Store Potential Matches**
   - Similar structure to Pass 1

### Phase 2: Conflict Resolution & Application

Same as Pass 1:
- Sort by match type and size
- Resolve conflicts
- Apply matches using `_apply_exact_match()`

**Output**: Additional CBL records marked as "Exact Match"

**Key Characteristics**:
- Only processes unmatched CBL records
- Handles multiple policy number fields per insurer record
- Simple token-based matching (no fuzzy matching)
- Only creates exact matches

---

## Pass 3: Name-Based Matching

**Purpose**: Intelligent name-based matching using corporate root extraction and fuzzy clustering.

**Input**: CBL records with status "No Match" or "Partial Match"

**Strategy**: Four-phase approach

### Phase 1: Exact Corporate Root Matching

**Goal**: Match records with identical corporate roots (fast & precise)

1. **Build Corporate Root Indices**
   - For CBL: `_build_corporate_root_index(unmatched_cbl, 'ClientName', 'CBL')`
   - For Insurer: `_build_corporate_root_index(available_insurer, 'ClientName_INSURER', 'INSURER')`
   - Also builds `cbl_primary_root_map` to track primary roots for each CBL record

2. **Corporate Root Extraction Logic**
   - Uses `_extract_corporate_root()` with intelligent detection:
     - **Corporate names**: 1-2 distinctive words
       - Example: "ALTEO AGRI LTD" → "ALTEO AGRI"
       - Example: "ALTEO GROUP OF COMPANIES" → "ALTEO" (GROUP is parent indicator)
     - **Person names**: 3 name words (skip titles like MR, MRS, DR)
       - Example: "MRS MARIE BERTHE CHANTAL HARDY" → "MARIE BERTHE CHANTAL"
     - **Financial relationships**: Extract lessee
       - Example: "MCB LEASING ONLEASE TO ECOBAT" → "ECOBAT"
     - **Organizational prefixes**: Skip prefixes to extract property name
       - Example: "SYNDICAT DES COPROPRIETAIRES DE LES TERRASSES" → "LES TERRASSES"

3. **Pre-Grouping: Consolidate CBL Records**
   - **KEY FEATURE**: Only adds CBL to a group if the group's root matches CBL's PRIMARY root
   - Prevents compound names like "KASA GROUP... - REY..." from being assigned to REY group
   - Example: "KASA GROUP... - REY..." → assigned to KASA group only (not REY)
   - Tracks skipped secondary assignments

4. **Match Groups**
   - For each matched root:
     - Groups CBL records with same root
     - Groups insurer records with same root
     - Calculates group totals:
       - `cbl_total = sum(CBL amounts)`
       - `insurer_total = sum(insurer amounts)`
     - Classifies match:
       - `difference = abs(cbl_total + insurer_total)`
       - Within tolerance → EXACT MATCH
       - Beyond tolerance → PARTIAL MATCH

5. **Apply Matches**
   - Validates insurer availability using `GlobalMatchTracker`
   - For exact matches: Uses `_apply_cluster_exact_match()` (allows sharing)
   - For partial matches: Uses `_apply_partial_match()`
   - Assigns `group_id` and `corporate_root` to CBL records

**Output**: CBL records matched via corporate root extraction

### Phase 2: Fuzzy Clustering Fallback

**Goal**: Catch name variations and typos using fuzzy matching

1. **Get Remaining Records**
   - CBL records not matched in Phase 1
   - Insurer records still available

2. **Build Fuzzy Name Clusters**
   - Uses `_build_fuzzy_name_clusters()` for both CBL and Insurer
   - **Clustering Process**:
     - Extracts primary company names (handles compound names & financial relationships)
     - Uses Union-Find data structure for clustering
     - Compares all pairs using `CompanyNameMatcher.calculate_intelligent_similarity()`
     - Requires similarity ≥ `fuzzy_threshold` (default: 85%)
     - **Validation**: Checks `_has_sufficient_word_overlap()` (requires ≥2 common words)
     - Prevents false positives from single common words
     - Example: "ACME LTD" vs "ACME LIMITED" → clusters (95% similarity, 1 common word "ACME")
     - Example: "SUN LTD" vs "WOLMAR SUN HOTELS LTD" → doesn't cluster (only 1 common word)

3. **Match Clusters**
   - For each CBL cluster, find matching insurer cluster:
     - Uses `CompanyNameMatcher` for cross-cluster similarity
     - Requires similarity ≥ `fuzzy_threshold`
     - Validates word overlap (≥2 common words)
     - Calculates group totals and classifies match

4. **Apply Matches**
   - Similar to Phase 1
   - Assigns `group_id` with "FUZZY" suffix

**Output**: Additional CBL records matched via fuzzy clustering

### Phase 3: Secondary Root Loose Capture

**Goal**: Match remaining CBL records using secondary roots from compound names

1. **Get Remaining Records**
   - CBL records not matched in Phase 1 or 2

2. **Extract Secondary Roots**
   - For each remaining CBL record:
     - Gets ALL roots using `_extract_all_corporate_roots()`
     - Excludes primary root (already tried in Phase 1)
     - Example: "KASA GROUP... - REY..." → primary="KASA", secondary="REY"

3. **Try Secondary Root Matching**
   - For each secondary root:
     - Checks if root exists in insurer root index
     - Validates insurer availability (allows sharing for partial matches)
     - Calculates amounts and classifies match
     - **Always creates PARTIAL match** (lower confidence, capped at "Medium")

4. **Apply Matches**
   - Uses `_apply_partial_match()` with secondary root reason
   - Assigns `group_id` with "SECONDARY" suffix

**Output**: Additional CBL records matched via secondary roots

### Phase 4: Merge Groups with Overlapping Insurer Indices

**Goal**: Merge groups that share the same insurer indices

1. **Identify Overlapping Groups**
   - Finds groups with identical `matched_insurer_indices`
   - Example: Group A and Group B both use insurer indices [1, 2, 3]

2. **Merge Groups**
   - Creates new merged group ID
   - Combines CBL records from all overlapping groups
   - Recalculates totals with combined CBL records
   - Updates match status based on combined totals
   - Updates all CBL records in merged group

**Output**: Consolidated groups with shared insurer indices

**Key Characteristics**:
- Four-phase approach: exact root → fuzzy → secondary root → merge
- Intelligent corporate root extraction handles various name formats
- Fuzzy clustering catches typos and variations
- Secondary root matching handles compound names
- Allows multiple CBL rows to share insurer rows (cluster matching)
- Comprehensive conflict resolution

---

## Helper Functions

### Amount Classification

**`classify_amount_match(amt1, amt2, tolerance)`**
- Classifies amount matching with confidence levels
- Returns: `(match_type, difference, confidence_level)`
- Match types:
  - `PERFECT_MATCH`: Within 10% of tolerance
  - `EXACT_MATCH`: Within tolerance
  - `CLOSE_MATCH`: Within 2x tolerance (not used in Pass 1/2)
  - `REVIEW_REQUIRED`: Within 5x tolerance (not used)
  - `INVESTIGATION_REQUIRED`: Within 10x tolerance (not used)
  - `NO_MATCH`: Beyond 10x tolerance

### Substring Validation

**`validate_substring_match(str1, str2, min_overlap_pct=0.8, min_length=10)`**
- Validates substring matches with quality controls
- Requires ≥80% overlap and minimum length of 10 chars
- Returns: `(is_valid_match, overlap_info)`

### Corporate Root Extraction

**`_extract_corporate_root(name, max_words=2)`**
- Extracts distinctive corporate root identifier
- Handles:
  - Person names (3 words, skip titles)
  - Parent company indicators (1 word)
  - Financial relationships (extract lessee)
  - Organizational prefixes (extract property name)

**`_get_primary_corporate_root(name, max_words=2)`**
- Gets the PRIMARY root (first entity in compound names)

**`_extract_all_corporate_roots(name, max_words=2)`**
- Extracts ALL roots from compound names (for secondary root matching)

### Word Overlap Validation

**`_has_sufficient_word_overlap(name1, name2, min_common_words=2)`**
- Checks if two names have sufficient meaningful word overlap
- Excludes common words (LTD, LIMITED, GROUP, etc.)
- Requires ≥2 meaningful common words
- Validates first word match (primary identifier)
- Prevents false clustering from single common words

### Match Application Functions

**`_apply_exact_match(...)`**
- Applies exact match to CBL record
- Validates indices with GlobalMatchTracker
- Updates CBL DataFrame with match details
- Returns: 1 if successful, 0 otherwise

**`_apply_cluster_exact_match(...)`**
- Applies exact match for Pass 3 cluster matching
- Allows multiple CBL rows to share insurer indices
- Used in Pass 3 Phase 1 & 2

**`_apply_partial_match(...)`**
- Applies partial match to CBL record
- Validates indices with GlobalMatchTracker
- Updates CBL DataFrame
- Returns: 1 if successful, 0 otherwise

**`_apply_no_match(...)`**
- Marks CBL record as "No Match"
- Clears match-related fields

### Conflict Resolution

**`_handle_conflict_resolution(...)`**
- Handles conflicts when multiple CBL rows claim same insurer rows
- Uses GlobalMatchTracker to check availability
- Applies fallback logic if not all indices available
- Returns: `(exact_matches_added, partial_matches_added)`

---

## Summary

The matching engine uses a **progressive refinement strategy**:

1. **Pass 1**: Exact matches using placing numbers + amounts (highest confidence)
2. **Pass 2**: Exact matches using policy numbers + amounts (medium-high confidence)
3. **Pass 3**: Name-based matching with multiple phases:
   - Phase 1: Exact corporate root matching (fast & precise)
   - Phase 2: Fuzzy clustering fallback (catches variations)
   - Phase 3: Secondary root loose capture (handles compound names)
   - Phase 4: Merge overlapping groups (consolidation)

Each pass processes records that weren't matched in previous passes, ensuring comprehensive coverage while maintaining data integrity through the GlobalMatchTracker system.
