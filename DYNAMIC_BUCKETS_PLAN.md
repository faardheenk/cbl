# Dynamic Buckets — Implementation Plan

## Overview

Extend the reconciliation system to support **dynamic buckets** beyond the 3 fixed ones (Exact Matches, Partial Matches, No Matches). Dynamic buckets are defined globally in a SharePoint list and apply to all insurance companies. They allow users to manually categorize matched CBL+insurer pairs into custom categories (e.g., "Mise en Demeure", "Disputed", "Write-Off").

---

## SharePoint List Schema: "Buckets"

| Column | Type | Description |
|--------|------|-------------|
| `BucketName` | Single line of text | Display name shown in UI (e.g., "Mise en D'Meurre") |
| `BucketKey` | Single line of text | Excel-safe key used as sheet name + history key (e.g., "Mise en DMeurre") |

**BucketKey rules** (Excel sheet name constraints):
- Max 31 characters
- No `\ / ? * : [ ]`
- Cannot start or end with `'`
- Auto-generated from `BucketName` by stripping invalid characters

---

## Backend Changes (DONE)

### 1. Fetch dynamic buckets from SharePoint

**File:** `sharepoint_dynamic.py` — `get_dynamic_buckets()`

- Fetches all rows from the "Buckets" list (no insurer filter — global)
- Returns `list[{"BucketName": str, "BucketKey": str}]`
- Returns empty list on error

### 2. Pass dynamic buckets into `run_matching_process()`

**File:** `matching/orchestrator.py`

- `run_matching_process()` accepts `dynamic_buckets=None` parameter
- Passed through to `apply_match_history()` and `_generate_output_and_statistics()`

### 3. Match history layer: support dynamic bucket targets

**File:** `matching/match_history.py`

- `apply_match_history()` accepts `dynamic_buckets` parameter
- Valid targets = `{"exact", "partial", "no-match"}` + all `BucketKey` values from dynamic buckets
- Dynamic bucket rows get sentinel `match_status = "_DynamicBucket_{BucketKey}"` so passes skip them
- `match_resolved_in_pass = "history"` is set so all passes skip these rows
- Insurer indices registered in GlobalTracker so passes don't reuse them

### 4. All passes skip history-resolved rows

**Files:** `matching/pass1.py`, `matching/pass2.py`, `matching/pass3.py`

- Pass 1: skips rows where `match_resolved_in_pass == "history"`
- Pass 2: same skip (already filtered to No Match/Partial Match, now also checks history)
- Pass 3: excludes history rows from `unmatched_cbl` selection + excludes from group merging

### 5. Finalization after passes

**File:** `matching/orchestrator.py`

After all passes complete, before output generation:
- `finalize_history_no_match()` — converts `_History_No_Match` sentinel → `"No Match"`
- `finalize_history_dynamic_buckets()` — converts `_DynamicBucket_{key}` sentinel → `{key}`

### 6. Output generation: dynamic bucket sheets + metadata

**File:** `matching/orchestrator.py` — `_generate_output_and_statistics()`

Output sheets in order:
1. `Exact Matches` (fixed)
2. `Partial Matches` (fixed)
3. `No Matches CBL` (fixed)
4. `No Matches Insurer` (fixed)
5. One sheet per dynamic bucket (sheet name = `BucketKey`)
   - Contains exploded CBL+insurer merged rows (same format as Exact/Partial)
   - Empty sheets written with headers if no rows, so frontend knows the bucket exists
6. `_BucketConfig` (metadata sheet with `BucketName` + `BucketKey` columns)

### 7. Statistics

Return dict includes `dynamic_bucket_stats`:
```python
'dynamic_bucket_stats': {
    "BucketKey1": <row_count>,
    "BucketKey2": <row_count>,
    ...
}
```

---

## Frontend Changes (TODO)

### 1. Read `_BucketConfig` metadata sheet from output.xlsx

When loading `output.xlsx`:

- Check if a sheet named `_BucketConfig` exists in the workbook
- If present, parse it to get the list of dynamic buckets:
  ```
  [{ BucketName: "Mise en D'Meurre", BucketKey: "Mise en DMeurre" }, ...]
  ```
- Store this config in state — it drives everything below

### 2. Load dynamic bucket sheet data

After loading the 4 fixed sheets:

- For each entry in `_BucketConfig`, read the sheet named by `BucketKey`
- Each sheet has the same column structure as Exact Matches / Partial Matches (merged CBL + insurer columns)
- Split each row into CBL object + insurer object using the same logic as Exact/Partial:
  - CBL columns = columns NOT ending in `_INSURER`
  - Insurer columns = columns ending in `_INSURER` (strip suffix)
- Store rows in state keyed by `BucketKey`

### 3. Display dynamic bucket tabs in UI

- Render a tab/section for each dynamic bucket alongside the fixed tabs
- Use `BucketName` as the display label (human-readable)
- Use `BucketKey` as the internal identifier
- Dynamic bucket tabs should show the same table format as Exact/Partial (CBL rows with linked insurer rows)
- Show row count per bucket in the tab label

### 4. Add dynamic buckets to the move-row target picker

The existing move-row UI (dropdown/dialog where user picks a destination bucket) needs to include dynamic buckets:

- Current options: `exact`, `partial`, `no-match`
- New options: all `BucketKey` values from `_BucketConfig`, displayed with their `BucketName`
- When a user moves rows TO a dynamic bucket, the same flow applies:
  1. Read `_fingerprint` from CBL row objects (already present in the data)
  2. Read `_fingerprint_INSURER` from insurer row objects (already present in the data)
  3. Save to `history.xlsx` with `TargetBucket` = the dynamic `BucketKey`

### 5. Support moving rows FROM dynamic buckets

Users should also be able to move rows out of a dynamic bucket back to a fixed bucket (or another dynamic bucket):

- `FromBucket` in history.xlsx = the source dynamic `BucketKey`
- `TargetBucket` = destination bucket key
- Same fingerprint read + save logic

### 6. Exclude `_BucketConfig` from data display

The `_BucketConfig` sheet is metadata. Do NOT render it as a data tab or include it in any row counts.

### 7. No separate SharePoint fetch needed

The frontend does NOT need to independently fetch from the SharePoint "Buckets" list. All bucket information is embedded in `output.xlsx` via the `_BucketConfig` sheet. The backend handles the SharePoint fetch at processing time.

---

## Data Flow

```
SharePoint "Buckets" list (global, all insurers)
    |
    v
Backend fetches dynamic buckets (no insurer filter)
    |
    v
Backend runs matching:
    1. Preprocess + generate canonical fingerprints
    2. Match history layer (supports dynamic bucket targets)
       - Rows matching history -> pre-placed with sentinel status
       - Insurer indices locked in GlobalTracker
    3. Pass 1, 2, 3 (skip all history-resolved rows)
    4. Finalize sentinels -> actual bucket keys
    5. Write output.xlsx:
       - Exact Matches (fixed)
       - Partial Matches (fixed)
       - No Matches CBL (fixed)
       - No Matches Insurer (fixed)
       - {BucketKey} sheets (dynamic, one per bucket)
       - _BucketConfig (metadata)
    |
    v
Frontend loads output.xlsx
    - Reads _BucketConfig -> knows which sheets are dynamic buckets
    - Loads all sheets, splits merged rows into CBL + insurer
    - Displays fixed + dynamic bucket tabs
    |
    v
User moves rows between any buckets (fixed or dynamic)
    |
    v
Frontend saves to history.xlsx
    - TargetBucket = BucketKey (works for both fixed and dynamic)
    - Reads _fingerprint / _fingerprint_INSURER directly from row objects
    |
    v
Next run: backend reads history.xlsx
    - Pre-places rows into their target buckets (fixed or dynamic)
    - Passes skip all history-resolved rows
    - Output includes rows in their correct sheets
```

---

## Files Modified

### Backend (DONE)

| File | Change |
|------|--------|
| `sharepoint_dynamic.py` | Added `get_dynamic_buckets()` (global, no insurer filter) |
| `matching/orchestrator.py` | Accepts `dynamic_buckets`, passes to history + output, writes dynamic sheets + `_BucketConfig` |
| `matching/match_history.py` | Accepts dynamic bucket keys as valid targets, sentinel prefix `_DynamicBucket_`, `finalize_history_dynamic_buckets()` |
| `matching/pass1.py` | Skips rows with `match_resolved_in_pass == "history"` |
| `matching/pass2.py` | Same skip |
| `matching/pass3.py` | Same skip in CBL selection + group merge function |

### Frontend (TODO)

| Area | Change |
|------|--------|
| Output loading | Read `_BucketConfig` sheet, load dynamic bucket sheets, split merged rows |
| Bucket state | Store dynamic bucket rows in state keyed by `BucketKey` |
| UI tabs | Render dynamic bucket tabs with `BucketName` labels |
| Move-row picker | Add dynamic buckets to destination options |
| History save | Use `BucketKey` as `TargetBucket` / `FromBucket` for dynamic buckets |
| Sheet exclusion | Hide `_BucketConfig` sheet from data display |
