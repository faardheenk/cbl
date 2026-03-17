# Dynamic Buckets — Implementation Plan

## Overview

Extend the reconciliation system to support **dynamic buckets** beyond the 3 fixed ones (Exact Matches, Partial Matches, No Matches). Dynamic buckets are defined per insurance company in a SharePoint list and allow users to manually categorize matched CBL+insurer pairs into custom categories (e.g., "Mise en Demeure", "Disputed", "Write-Off").

---

## SharePoint List Schema: "Buckets"

| Column | Type | Description |
|--------|------|-------------|
| `InsuranceCompany` | Single line of text | Insurer name (e.g., "EAGLE INSURANCE LTD") |
| `BucketName` | Single line of text | Display name shown in UI (e.g., "Mise en D'Meurre") |
| `BucketKey` | Single line of text | Excel-safe key used as sheet name + history key (e.g., "Mise en DMeurre") |

**BucketKey rules** (Excel sheet name constraints):
- Max 31 characters
- No `\ / ? * : [ ]`
- Cannot start or end with `'`
- Auto-generated from `BucketName` by stripping invalid characters

---

## Backend Changes

### 1. Fetch dynamic buckets from SharePoint

**File:** `sharepoint_dynamic.py`

Add a new function:

```python
def get_dynamic_buckets(insurer_name: str) -> list[dict]:
    """
    Fetch dynamic bucket definitions from SharePoint 'Buckets' list
    for the given insurer.

    Returns:
        list of {"BucketName": str, "BucketKey": str}
    """
```

- Query the "Buckets" list filtered by `InsuranceCompany == insurer_name`
- Return list of `{BucketName, BucketKey}` dicts
- Return empty list if no dynamic buckets configured

### 2. Pass dynamic buckets into `run_matching_process()`

**File:** `matching/orchestrator.py`

Update `run_matching_process()` signature:

```python
def run_matching_process(
    column_mappings,
    cbl_file=None,
    insurer_file=None,
    output_file='output.xlsx',
    tolerance=50,
    history_file=None,
    dynamic_buckets=None,  # NEW — list of {"BucketName": str, "BucketKey": str}
):
```

- `dynamic_buckets` is passed through to `apply_match_history()` and `_generate_output_and_statistics()`

### 3. Match history layer: support dynamic bucket targets

**File:** `matching/match_history.py`

**Current code (line 227):**
```python
if target not in ("exact", "partial", "no-match"):
    logger.warning(f"Unknown target bucket '{target}' — skipping")
    continue
```

**Change to:**
```python
# Build set of all valid bucket keys
valid_targets = {"exact", "partial", "no-match"}
if dynamic_buckets:
    valid_targets.update(b["BucketKey"] for b in dynamic_buckets)

if target not in valid_targets:
    logger.warning(f"Unknown target bucket '{target}' — skipping")
    continue
```

**Dynamic bucket pre-placement logic:**

Dynamic bucket targets should be treated like `exact`/`partial` targets since they contain matched CBL+insurer pairs:

```python
if target in ("exact", "partial") or target in dynamic_bucket_keys:
    # Same logic — set match_status, link insurer indices, lock in GlobalTracker
    # For dynamic buckets, use a dedicated match_status value:
    match_status = {
        "exact": "Exact Match",
        "partial": "Partial Match",
    }.get(target, f"_DynamicBucket_{target}")  # sentinel for dynamic buckets
    ...
```

Dynamic bucket rows get a sentinel `match_status` (like `_History_No_Match`) so passes skip them. After all passes complete, the sentinel is converted to the actual `BucketKey` value.

**Update `apply_match_history()` signature:**
```python
def apply_match_history(cbl_df, insurer_df, history_source, global_tracker=None, dynamic_buckets=None):
```

**Add a finalization function:**
```python
def finalize_history_dynamic_buckets(cbl_df):
    """Convert _DynamicBucket_* sentinels back to their BucketKey values after all passes."""
    mask = cbl_df["match_status"].str.startswith("_DynamicBucket_", na=False)
    if mask.any():
        cbl_df.loc[mask, "match_status"] = cbl_df.loc[mask, "match_status"].str.replace("_DynamicBucket_", "", n=1)
    return cbl_df
```

### 4. Output generation: add dynamic bucket sheets

**File:** `matching/orchestrator.py` — `_generate_output_and_statistics()`

**Current code writes 4 fixed sheets:**
```python
exact_matches.to_excel(writer, sheet_name="Exact Matches", index=False)
partial_matches.to_excel(writer, sheet_name="Partial Matches", index=False)
no_matches.to_excel(writer, sheet_name="No Matches CBL", index=False)
unmatched_insurer.to_excel(writer, sheet_name="No Matches Insurer", index=False)
```

**Add after the fixed sheets:**
```python
# Write dynamic bucket sheets
if dynamic_buckets:
    for bucket in dynamic_buckets:
        bucket_key = bucket["BucketKey"]
        bucket_rows = clean_cbl[clean_cbl["match_status"] == bucket_key].copy()
        if not bucket_rows.empty:
            # Explode and merge with insurer data (same as exact/partial)
            bucket_merged = explode_and_merge(bucket_rows, clean_insurer)
            bucket_merged.to_excel(writer, sheet_name=bucket_key, index=False)
        else:
            # Write empty sheet with headers so frontend knows the bucket exists
            pd.DataFrame(columns=exact_matches.columns).to_excel(
                writer, sheet_name=bucket_key, index=False
            )
```

**Update `_generate_output_and_statistics()` signature:**
```python
def _generate_output_and_statistics(clean_cbl, clean_insurer, output_filename, dynamic_buckets=None):
```

### 5. Include dynamic bucket metadata in output

Add a hidden metadata sheet `_BucketConfig` to `output.xlsx` so the frontend knows which sheets are dynamic buckets and their display names:

```python
if dynamic_buckets:
    bucket_config_df = pd.DataFrame(dynamic_buckets)  # BucketName, BucketKey columns
    bucket_config_df.to_excel(writer, sheet_name="_BucketConfig", index=False)
```

### 6. Finalization order in orchestrator

After all passes complete, before output generation:

```python
# Finalize history sentinels
if history_file is not None:
    clean_cbl = finalize_history_no_match(clean_cbl)
    clean_cbl = finalize_history_dynamic_buckets(clean_cbl)
```

### 7. Update statistics

Dynamic bucket rows should be excluded from the standard exact/partial/no-match counts and reported separately:

```python
# In results dict, add:
'dynamic_bucket_stats': {
    bucket["BucketKey"]: len(clean_cbl[clean_cbl["match_status"] == bucket["BucketKey"]])
    for bucket in (dynamic_buckets or [])
}
```

---

## Frontend Changes

### 1. Read dynamic bucket configuration from output.xlsx

When loading `output.xlsx`:

- Check for `_BucketConfig` sheet
- If present, parse it to get the list of dynamic buckets: `{BucketName, BucketKey}`
- Each `BucketKey` corresponds to an additional sheet in the workbook

### 2. Load dynamic bucket sheets

After loading the 3 fixed sheets (Exact Matches, Partial Matches, No Matches CBL, No Matches Insurer):

- For each bucket in `_BucketConfig`, read the sheet named `BucketKey`
- Split merged rows into CBL + insurer objects (same logic as Exact/Partial)
- Store in state alongside the fixed buckets

### 3. Display dynamic buckets in UI

- Render dynamic bucket tabs/sections alongside the 3 fixed ones
- Use `BucketName` for display labels (human-readable, may contain special chars)
- Use `BucketKey` internally for state management

### 4. Support moving rows to/from dynamic buckets

The existing move-row flow should work with dynamic buckets:

- User selects CBL + insurer rows
- User picks a target bucket (now includes dynamic buckets in the dropdown/picker)
- Frontend reads `_fingerprint` / `_fingerprint_INSURER` from the row objects
- Frontend saves to `history.xlsx` with:
  - `FromBucket`: source bucket key (e.g., `"partial"`, `"exact"`, or a dynamic `BucketKey`)
  - `TargetBucket`: destination bucket key (e.g., `"mise_en_dmeurre"`)
  - `CblFingerprints`: JSON array of `_fingerprint` values
  - `InsurerFingerprints`: JSON array of `_fingerprint_INSURER` values
  - `Timestamp`: ISO 8601

### 5. Fetch dynamic buckets from SharePoint (if frontend needs them independently)

If the frontend needs to know available buckets before loading an output file (e.g., to show bucket options when no output exists yet):

- Fetch from SharePoint "Buckets" list filtered by the current insurer
- This is only needed if the frontend creates buckets or shows them before output.xlsx is loaded
- Otherwise, the `_BucketConfig` sheet in output.xlsx is sufficient

### 6. Exclude `_BucketConfig` sheet from data display

The `_BucketConfig` sheet is metadata, not reconciliation data. Do not render it as a data tab.

---

## Data Flow

```
SharePoint "Buckets" list
    │
    ▼
Backend fetches dynamic buckets for current insurer
    │
    ▼
Backend runs matching:
    1. Preprocess + generate canonical fingerprints
    2. Match history layer (supports dynamic bucket targets)
    3. Pass 1, 2, 3 (skip all history-resolved rows)
    4. Finalize sentinels → actual bucket keys
    5. Write output.xlsx:
       - Exact Matches (fixed)
       - Partial Matches (fixed)
       - No Matches CBL (fixed)
       - No Matches Insurer (fixed)
       - {BucketKey} sheets (dynamic, one per bucket)
       - _BucketConfig (metadata)
    │
    ▼
Frontend loads output.xlsx
    - Reads _BucketConfig → knows which sheets are dynamic buckets
    - Loads all sheets, splits merged rows
    - Displays fixed + dynamic buckets
    │
    ▼
User moves rows between any buckets (fixed or dynamic)
    │
    ▼
Frontend saves to history.xlsx
    - TargetBucket = BucketKey (works for both fixed and dynamic)
    - Reads _fingerprint / _fingerprint_INSURER from row objects
    │
    ▼
Next run: backend reads history.xlsx
    - Pre-places rows into their target buckets (fixed or dynamic)
    - Passes skip all history-resolved rows
```

---

## Files to Modify

### Backend

| File | Change |
|------|--------|
| `sharepoint_dynamic.py` | Add `get_dynamic_buckets(insurer_name)` |
| `matching/orchestrator.py` | Accept `dynamic_buckets` param, pass to history + output, add dynamic sheets, finalize sentinels |
| `matching/match_history.py` | Accept dynamic bucket keys as valid targets, add `finalize_history_dynamic_buckets()` |
| `run_latest_matching.py` | Fetch dynamic buckets, pass to `run_matching_process()` |

### Frontend

| Area | Change |
|------|--------|
| Output loading | Read `_BucketConfig` sheet, load dynamic bucket sheets |
| Bucket state | Add dynamic buckets to state alongside fixed buckets |
| UI | Render dynamic bucket tabs, add to move-row target picker |
| History save | Use `BucketKey` as `TargetBucket` for dynamic buckets |
| Column exclusion | Exclude `_BucketConfig` sheet from data display |
