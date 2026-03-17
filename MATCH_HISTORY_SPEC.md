# Match History — Implementation Specification

This document explains how the match history system works in the city-broker frontend, and what the backend (pandas) codebase needs to do to integrate with it.

## Overview

When a user manually moves rows between buckets (exact match, partial match, not found) in the reconciliation UI, we record a **fingerprint** of each moved row and persist it to an Excel file (`history.xlsx`). The goal: when **new files are uploaded for the same insurer**, the processing logic should read this history, regenerate fingerprints from the new data, and **automatically place matching rows into their previously assigned buckets** — skipping them from the comparison algorithm entirely.

---

## 1. Where History Is Stored

```
SharePoint Document Library: Matrix
Path: Matrix/{INSURER_NAME}/history.xlsx
```

- One `history.xlsx` per insurer, stored at the insurer folder level.
- The file accumulates across all reconciliation runs — new entries are appended, old entries are preserved.
- The insurer name is always **UPPERCASE and trimmed** (e.g., `"ALLIANZ"`, `"AXA XL"`).

---

## 2. history.xlsx Structure

The file contains a single sheet named **`MatchHistory`** with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `CblFingerprints` | String (JSON array) | `["fingerprint1", "fingerprint2", ...]` — fingerprints of all CBL rows in this group move |
| `InsurerFingerprints` | String (JSON array) | `["fingerprint1", "fingerprint2", ...]` — fingerprints of all insurer rows in this group move |
| `FromBucket` | String | The bucket the rows were moved **from**: `"exact"`, `"partial"`, or `"no-match"` |
| `TargetBucket` | String | The bucket the rows were moved **to**: `"exact"`, `"partial"`, or `"no-match"` |
| `Timestamp` | String (ISO 8601) | When the move happened, e.g., `"2026-03-11T14:30:00.000Z"` |

### Example row in history.xlsx:

| CblFingerprints | InsurerFingerprints | FromBucket | TargetBucket | Timestamp |
|---|---|---|---|---|
| `["5000\|ABC Corp\|PL-001\|POL-12345\|5000\|POL-12345"]` | `["2500\|ABC Corp\|PL-001\|POL-12345-A\|2500\|POL-12345-A", "2500\|ABC Corp\|PL-001\|POL-12345-B\|2500\|POL-12345-B"]` | partial | exact | 2026-03-11T14:30:00.000Z |

Each row represents **one group move** — all the CBL rows and insurer rows the user selected and moved together in a single action.

---

## 3. How Fingerprints Are Constructed

A fingerprint uniquely identifies a row by its **business data only** (not metadata).

### Algorithm:

```
1. Get all column names (keys) from the row
2. Remove metadata columns (see exclusion list below)
3. Sort remaining column names ALPHABETICALLY (ascending, case-sensitive)
4. For each column (in sorted order), convert its value to a string:
   - null, undefined, None, NaN → empty string ""
   - everything else → str(value)
5. Join all values with the pipe character "|"
6. Return the resulting string
```

### Pandas implementation:

```python
FINGERPRINT_EXCLUDE_COLUMNS = {
    "idx",
    "match_condition",
    "match_group",
    "matched_insurer_indices",
    "Placing No.",
    "PlacingNo_Clean",
    "Amount_Clean",
    "match_status",
    "match_pass",
    "matched_amtdue_total",
    "partial_candidates_indices",
    "match_resolved_in_pass",
    "partial_resolved_in_pass",
    "string matching_INSURER",
    "PlacingNo_Clean_INSURER",
    "PolicyNo_1_Clean_INSURER",
    "PolicyNo_2_Clean_INSURER",
    "Amount_Clean_INSURER",
    "string matching",
    "PolicyNo_1_Clean",
    "PolicyNo_2_Clean",
    "PolicyNo_Clean",
    "MatrixKey",
    "ProcessedAmount_Clean",
    "ClientName_Clean",
    "match_reason",
    "group_id",
    "corporate_root",
    "match_confidence",
}


def generate_fingerprint(row: pd.Series) -> str:
    """Generate a fingerprint for a row by concatenating all non-metadata
    column values, sorted alphabetically by column name."""
    keys = sorted([k for k in row.index if k not in FINGERPRINT_EXCLUDE_COLUMNS])
    values = []
    for k in keys:
        val = row[k]
        if pd.isna(val) or val is None:
            values.append("")
        elif isinstance(val, float) and val == int(val):
            # Match JavaScript's String(5000.0) → "5000"
            values.append(str(int(val)))
        else:
            values.append(str(val))
    return "|".join(values)


def generate_fingerprints_for_df(df: pd.DataFrame) -> pd.Series:
    """Generate fingerprints for all rows in a DataFrame."""
    return df.apply(generate_fingerprint, axis=1)
```

### Concrete example:

Given a CBL row (after column mapping/normalization):
```json
{
  "idx": "PM-3",
  "ProcessedPolicyNumber": "POL-12345",
  "ProcessedAmount": 5000,
  "ClientName": "ABC Corp",
  "PlacingNo": "PL-001",
  "PolicyNo_1": "POL-12345",
  "Amount": 5000,
  "matched_insurer_indices": "[0, 1]",
  "match_condition": "partial",
  "match_group": 1,
  "PlacingNo_Clean": "PL001"
}
```

Step 1 — All keys: `idx, ProcessedPolicyNumber, ProcessedAmount, ClientName, PlacingNo, PolicyNo_1, Amount, matched_insurer_indices, match_condition, match_group, PlacingNo_Clean`

Step 2 — After removing excluded: `ProcessedPolicyNumber, ProcessedAmount, ClientName, PlacingNo, PolicyNo_1, Amount`

Step 3 — Sorted alphabetically: `Amount, ClientName, PlacingNo, PolicyNo_1, ProcessedAmount, ProcessedPolicyNumber`

Step 4 — Values as strings: `"5000", "ABC Corp", "PL-001", "POL-12345", "5000", "POL-12345"`

Step 5 — Joined with `|`: **`"5000|ABC Corp|PL-001|POL-12345|5000|POL-12345"`**

### Critical rules:
- Column names are **case-sensitive** — `"Amount"` and `"amount"` are different keys.
- Sorting is standard **lexicographic sort** (`sorted()` in Python).
- `None`, `NaN`, `pd.NA` → empty string `""`.
- Numbers: JavaScript's `String(5000.0)` produces `"5000"` (not `"5000.0"`). Python must match this — use `str(int(x)) if x == int(x) else str(x)` for floats.
- The pipe `|` separator is literal — it is NOT escaped if a value contains `|`.
- **Both CBL and insurer rows use the exact same algorithm.** The insurer row has the `_INSURER` suffix stripped from column names before fingerprinting.

---

## 4. How the Data Reaches This State

Understanding the data pipeline is critical for reproducing fingerprints correctly.

### The output.xlsx file (produced by the comparison engine)

The comparison engine outputs an Excel file with 4 sheets:
- **Exact Matches** — matched CBL+insurer rows side by side. Insurer columns have `_INSURER` suffix (e.g., `Amount_INSURER`, `PolicyNo_1_INSURER`).
- **Partial Matches** — same format as Exact Matches.
- **No Matches CBL** — unmatched CBL rows only (no `_INSURER` columns).
- **No Matches Insurer** — unmatched insurer rows only (all columns have `_INSURER` suffix).

### How the frontend splits the data before fingerprinting

For Exact/Partial sheets (which contain both sides merged):
1. Each row is split into a CBL object and an insurer object.
2. CBL gets all columns that do NOT end with `_INSURER`.
3. Insurer gets all columns that DO end with `_INSURER`, **with the suffix stripped**.
4. Both get an `idx` field added (e.g., `"EM-0"`, `"PM-3"`).

For No Match sheets:
- No Match CBL: rows used as-is, `idx` = `"NM-0"`, `"NM-1"`, etc.
- No Match Insurer: `_INSURER` suffix is stripped from column names, `idx` = `"NM-0"`, etc.

**The fingerprint is generated AFTER this splitting**, meaning:
- CBL fingerprints use column names WITHOUT `_INSURER` suffix.
- Insurer fingerprints ALSO use column names WITHOUT `_INSURER` suffix (suffix was already stripped).
- Both sides share the same column names (e.g., `Amount`, `PolicyNo_1`, `ClientName`).

---

## 5. What the Backend Needs to Do (Pandas)

When 2 new Excel files are uploaded for an insurer, the processing logic should:

### Step 1: Read history.xlsx

```python
import pandas as pd
import json

def read_match_history(history_path: str) -> list[dict]:
    """Read match history entries from history.xlsx."""
    try:
        df = pd.read_excel(history_path, sheet_name="MatchHistory")
    except (FileNotFoundError, ValueError):
        return []  # No history yet

    entries = []
    for _, row in df.iterrows():
        entries.append({
            "cbl_fingerprints": json.loads(row["CblFingerprints"]),
            "insurer_fingerprints": json.loads(row["InsurerFingerprints"]),
            "from_bucket": row["FromBucket"],
            "target_bucket": row["TargetBucket"],
            "timestamp": row["Timestamp"],
        })
    return entries
```

### Step 2: Generate fingerprints from the new uploaded files

After reading and normalizing the 2 uploaded files (applying column mappings so column names match the output.xlsx format), generate fingerprints:

```python
# CBL DataFrame (file 1) — columns already mapped to standard names
cbl_df["_fingerprint"] = generate_fingerprints_for_df(cbl_df)

# Insurer DataFrame (file 2) — columns already mapped to standard names
# IMPORTANT: column names must NOT have _INSURER suffix at this point
insurer_df["_fingerprint"] = generate_fingerprints_for_df(insurer_df)

# Build lookup maps
cbl_fp_map = {}
for idx, row in cbl_df.iterrows():
    fp = row["_fingerprint"]
    if fp not in cbl_fp_map:
        cbl_fp_map[fp] = []
    cbl_fp_map[fp].append(idx)

insurer_fp_map = {}
for idx, row in insurer_df.iterrows():
    fp = row["_fingerprint"]
    if fp not in insurer_fp_map:
        insurer_fp_map[fp] = []
    insurer_fp_map[fp].append(idx)
```

### Step 3: Match against history and pre-place rows

```python
history = read_match_history(history_path)

# Track which row indices have been claimed by history
claimed_cbl_indices = set()
claimed_insurer_indices = set()

# Pre-placed results
pre_placed = {
    "exact":    {"cbl_indices": [], "insurer_indices": []},
    "partial":  {"cbl_indices": [], "insurer_indices": []},
    "no-match": {"cbl_indices": [], "insurer_indices": []},
}

for entry in history:
    target = entry["target_bucket"]

    # Find CBL rows matching this history entry
    for fp in entry["cbl_fingerprints"]:
        if fp in cbl_fp_map:
            for idx in cbl_fp_map[fp]:
                if idx not in claimed_cbl_indices:
                    pre_placed[target]["cbl_indices"].append(idx)
                    claimed_cbl_indices.add(idx)
                    break  # One match per fingerprint

    # Find insurer rows matching this history entry
    for fp in entry["insurer_fingerprints"]:
        if fp in insurer_fp_map:
            for idx in insurer_fp_map[fp]:
                if idx not in claimed_insurer_indices:
                    pre_placed[target]["insurer_indices"].append(idx)
                    claimed_insurer_indices.add(idx)
                    break  # One match per fingerprint
```

### Step 4: Split into pre-placed and remaining

```python
# Extract pre-placed DataFrames
exact_cbl_preplaced = cbl_df.loc[pre_placed["exact"]["cbl_indices"]]
exact_insurer_preplaced = insurer_df.loc[pre_placed["exact"]["insurer_indices"]]
partial_cbl_preplaced = cbl_df.loc[pre_placed["partial"]["cbl_indices"]]
partial_insurer_preplaced = insurer_df.loc[pre_placed["partial"]["insurer_indices"]]
nomatch_cbl_preplaced = cbl_df.loc[pre_placed["no-match"]["cbl_indices"]]
nomatch_insurer_preplaced = insurer_df.loc[pre_placed["no-match"]["insurer_indices"]]

# Remaining rows that need normal comparison
remaining_cbl = cbl_df.drop(index=claimed_cbl_indices)
remaining_insurer = insurer_df.drop(index=claimed_insurer_indices)
```

### Step 5: Run comparison algorithm on remaining rows only

```python
comparison_result = run_comparison(remaining_cbl, remaining_insurer)
```

### Step 6: Merge pre-placed rows with comparison results

```python
final_exact_cbl = pd.concat([exact_cbl_preplaced, comparison_result.exact_cbl])
final_exact_insurer = pd.concat([exact_insurer_preplaced, comparison_result.exact_insurer])
final_partial_cbl = pd.concat([partial_cbl_preplaced, comparison_result.partial_cbl])
final_partial_insurer = pd.concat([partial_insurer_preplaced, comparison_result.partial_insurer])
final_nomatch_cbl = pd.concat([nomatch_cbl_preplaced, comparison_result.nomatch_cbl])
final_nomatch_insurer = pd.concat([nomatch_insurer_preplaced, comparison_result.nomatch_insurer])
```

### Step 7: Write output.xlsx as usual

Write the merged results into output.xlsx with the standard 4 sheets. Remember to add `_INSURER` suffix back to insurer columns in the Exact/Partial sheets. The frontend will load this file and display the pre-placed rows already in their correct buckets.

---

## 6. Important Edge Cases

1. **Row data changed slightly**: If even one column value differs (e.g., amount changed from 5000 to 5001), the fingerprint will be different and the row will NOT be auto-placed. It goes through normal comparison. This is by design.

2. **Duplicate fingerprints**: If two rows in the new file produce the same fingerprint (identical data), only one will be matched to history. The other goes through normal comparison. Use the `claimed_*_indices` sets to prevent double-claiming.

3. **History file doesn't exist**: If `Matrix/{INSURER}/history.xlsx` doesn't exist, skip the preprocessing entirely and run normal comparison. This is the case for the first-ever reconciliation for an insurer.

4. **Column name consistency**: The fingerprint depends on column names being identical between the frontend and backend. The backend must use the **post-mapping, normalized column names** — the same column names that appear in the output.xlsx (without `_INSURER` suffix).

5. **Number formatting**: JavaScript's `String(5000.0)` produces `"5000"` (not `"5000.0"`). In pandas, a float `5000.0` would produce `"5000.0"` via `str()`. You must match JavaScript behavior:
   ```python
   if isinstance(val, float) and val == int(val):
       return str(int(val))  # 5000.0 → "5000"
   ```

6. **Empty strings vs null**: In the frontend, empty strings and null are treated differently. `null`/`undefined` → `""`. An actual empty string `""` stays as `""`. In pandas, `NaN` should be treated as `""` (empty string).

---

## 7. Full Flow Diagram

```
┌──────────────────────────────────────────────────────────┐
│ USER UPLOADS 2 NEW FILES FOR INSURER "ALLIANZ"           │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│ STEP 1: Read Matrix/ALLIANZ/history.xlsx                 │
│         Parse the MatchHistory sheet                     │
│         Get list of history entries with fingerprints     │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│ STEP 2: Read & normalize both uploaded files             │
│         Apply column mappings so names match output.xlsx  │
│         Generate fingerprints for ALL rows in both files  │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│ STEP 3: Match fingerprints against history               │
│         For each history entry, check if the CBL and     │
│         insurer fingerprints exist in the new data       │
│         If yes → mark as pre-placed in target bucket     │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│ STEP 4: Remove pre-placed rows from comparison pool      │
│         Only remaining (unmatched) rows go through       │
│         the normal comparison algorithm                  │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│ STEP 5: Run comparison algorithm on remaining rows       │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│ STEP 6: Merge pre-placed + comparison results            │
│         Write output.xlsx with all 4 sheets              │
│         Upload to SharePoint                             │
└──────────────────────────────────────────────────────────┘
```
