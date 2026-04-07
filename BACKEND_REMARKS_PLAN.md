# Backend Plan: Remarks Column Implementation

## Overview

The frontend reconciliation app is adding an optional "Remarks" feature that allows users to annotate rows with free-text notes during reconciliation. The backend (Python) needs to generate and preserve a `Remarks` column in its output so the frontend can display and populate it.

---

## What the Frontend Expects

The frontend reads an Excel file (output.xlsx) from SharePoint with these sheets:

| Sheet | Column format |
|-------|---------------|
| **Exact Matches** | CBL columns + Insurer columns with `_INSURER` suffix (merged rows) |
| **Partial Matches** | CBL columns + Insurer columns with `_INSURER` suffix (merged rows) |
| **No Matches CBL** | CBL columns only (no suffix) |
| **No Matches Insurer** | Insurer columns with `_INSURER` suffix |
| **Dynamic bucket sheets** | Same format as Exact/Partial Matches (merged rows) |
| **_BucketConfig** | Bucket metadata |

The frontend splits merged sheets by suffix: columns without `_INSURER` go to the CBL table, columns ending in `_INSURER` go to the Insurer table (with suffix stripped for display).

---

## Required Changes

### 1. Add `Remarks` Column to CBL Data

Add a `Remarks` column to all CBL-side output. Initialize to empty string `""`.

**Affected sheets:**
- **Exact Matches** — add `Remarks` column (CBL side, no suffix)
- **Partial Matches** — add `Remarks` column (CBL side, no suffix)
- **No Matches CBL** — add `Remarks` column
- **Dynamic bucket sheets** — add `Remarks` column (CBL side, no suffix)

### 2. Add `Remarks_INSURER` Column to Insurer Data

Add a `Remarks_INSURER` column to all Insurer-side output. Initialize to empty string `""`.

**Affected sheets:**
- **Exact Matches** — add `Remarks_INSURER` column (Insurer side, with suffix)
- **Partial Matches** — add `Remarks_INSURER` column (Insurer side, with suffix)
- **No Matches Insurer** — add `Remarks_INSURER` column (already has suffix convention)
- **Dynamic bucket sheets** — add `Remarks_INSURER` column (Insurer side, with suffix)

### 3. Column Placement

Place `Remarks` as the **last CBL column** (before any `_INSURER` columns in merged sheets) and `Remarks_INSURER` as the **last Insurer column** in each sheet.

Example column order for Exact Matches sheet:
```
PolicyNo | ClientName | ProcessedAmount | ... | Remarks | PolicyNo_INSURER | ... | Remarks_INSURER
```

### 4. Preserve Existing Remarks on Re-processing

When the backend re-processes data that was previously saved by the frontend (e.g., re-running matching on an existing output.xlsx), it must:

1. Check if `Remarks` / `Remarks_INSURER` columns already exist in the input data
2. If they do, **preserve the existing values** — do not overwrite with empty strings
3. If they don't exist, initialize to empty string `""`

This is critical because users may have added remarks via the frontend, and re-processing should not erase them.

### 5. Exclude from Matching Logic

The `Remarks` column is purely informational and must **NOT** be used in:
- Fingerprint generation
- Match scoring / confidence calculation
- Any comparison or grouping logic
- Amount calculations

Treat it the same as `match_status`, `match_pass`, etc. — a metadata column that passes through untouched.

### 6. Exclude from Fingerprint

If the backend generates `_fingerprint` values, ensure `Remarks` and `Remarks_INSURER` are excluded from the fingerprint computation, similar to how `match_status`, `group_id`, etc. are excluded.

The frontend already excludes these backend tracking columns from its own fingerprint logic (see `FINGERPRINT_EXCLUDE_COLUMNS` in the frontend code). The backend should mirror this for `Remarks`.

---

## Column Summary

| Column Name | Side | Sheets | Initial Value | Notes |
|-------------|------|--------|---------------|-------|
| `Remarks` | CBL | Exact Matches, Partial Matches, No Matches CBL, Dynamic buckets | `""` | Last CBL column |
| `Remarks_INSURER` | Insurer | Exact Matches, Partial Matches, No Matches Insurer, Dynamic buckets | `""` | Last Insurer column |

---

## No Validation Required

- No length limits
- No format constraints
- No business rules
- The column is free-text, managed entirely by the frontend UI
- The backend's only job is to ensure the column exists and is preserved

---

## Testing Checklist

1. **Fresh run (no existing output):** Verify `Remarks` and `Remarks_INSURER` columns appear in all sheets, initialized to `""`
2. **Re-processing (existing output with remarks):** Verify existing remark values are preserved after re-running the backend
3. **Fingerprint integrity:** Verify that adding/changing `Remarks` values does not change `_fingerprint` output
4. **Match results unchanged:** Verify that the presence of the `Remarks` column does not affect match results, scores, or bucket assignments
