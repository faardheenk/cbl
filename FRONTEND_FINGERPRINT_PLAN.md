# Frontend Fingerprint Alignment — Implementation Plan

The backend now defines the **authoritative fingerprint exclude list** based on columns it actually creates. The frontend must adopt the same list so fingerprints match between both sides.

---

## The Problem

The frontend currently has its own exclude list that mixes UI-only columns (`idx`, `string matching`, `match_condition`, `match_group`) with backend columns. The backend has no knowledge of those UI columns, and some entries in the frontend list don't correspond to real backend columns (e.g. `Amount_Clean`, `Placing No.`, `PolicyNo_1_Clean`). This means the two sides could produce different fingerprints for the same row.

---

## How Fingerprinting Works

Both sides must produce **identical fingerprints** for the same business data. The algorithm:

```
1. Get all column names from the row
2. Remove columns in the exclude list
3. Sort remaining column names alphabetically (case-sensitive)
4. Convert each value to string:
   - null / undefined / NaN  →  ""
   - float where value == int(value)  →  String(int)   (5000.0 → "5000")
   - everything else  →  String(value)
5. Join with "|"
```

---

## Backend's Authoritative Exclude List

These are the **only** columns the backend creates during preprocessing, tracking, and matching passes. The frontend must exclude **at minimum** every column in this list.

```
# Cleaned / computed columns (from preprocessing)
PlacingNo_Clean
PolicyNo_Clean
PolicyNo_2_Clean
ProcessedAmount_Clean
ClientName_Clean
MatrixKey

# Match state columns (from tracking initialization)
match_status
match_pass
match_reason
matched_insurer_indices
matched_amtdue_total
Amount Difference
partial_candidates_indices
match_resolved_in_pass
partial_resolved_in_pass

# Pass-generated columns (set during matching)
group_id
corporate_root
match_confidence

# Internal / temporary
_source_sheet
_fingerprint
```

**Total: 20 columns.**

---

## What the Frontend Needs to Change

### 1. Replace the exclude list

The frontend's current `FINGERPRINT_EXCLUDE_COLUMNS` (or equivalent) should be replaced with:

```typescript
/**
 * Backend-created columns to exclude from fingerprinting.
 * This list is defined by the backend — do NOT add/remove entries
 * without updating the backend's match_history.py in sync.
 */
const BACKEND_EXCLUDE_COLUMNS = new Set([
  // Cleaned / computed columns (from preprocessing)
  "PlacingNo_Clean",
  "PolicyNo_Clean",
  "PolicyNo_2_Clean",
  "ProcessedAmount_Clean",
  "ClientName_Clean",
  "MatrixKey",

  // Match state columns (from tracking initialization)
  "match_status",
  "match_pass",
  "match_reason",
  "matched_insurer_indices",
  "matched_amtdue_total",
  "Amount Difference",
  "partial_candidates_indices",
  "match_resolved_in_pass",
  "partial_resolved_in_pass",

  // Pass-generated columns (set during matching)
  "group_id",
  "corporate_root",
  "match_confidence",

  // Internal / temporary
  "_source_sheet",
  "_fingerprint",
]);

/**
 * Frontend-only columns to also exclude.
 * These are added by the UI and never exist in the backend.
 */
const FRONTEND_EXCLUDE_COLUMNS = new Set([
  "idx",
  // Add any other UI-only columns here
]);

/** Combined exclude set used when generating fingerprints. */
const FINGERPRINT_EXCLUDE_COLUMNS = new Set([
  ...BACKEND_EXCLUDE_COLUMNS,
  ...FRONTEND_EXCLUDE_COLUMNS,
]);
```

### 2. Remove columns that don't exist on either side

The following entries in the current frontend exclude list are **not real columns** — they should be removed:

| Column | Why it should be removed |
|---|---|
| `Placing No.` | Pre-mapping column name; after mapping it becomes `PlacingNo` which is business data (INCLUDED in fingerprint) |
| `Amount_Clean` | Does not exist in backend; the actual column is `ProcessedAmount_Clean` |
| `PolicyNo_1_Clean` | Does not exist in backend; the insurer clean column is `PolicyNo_Clean` (derived from PolicyNo_1) |
| `PolicyNo_1_Clean_INSURER` | Has _INSURER suffix — after stripping it becomes `PolicyNo_Clean` which is already excluded |
| `PolicyNo_2_Clean_INSURER` | After stripping becomes `PolicyNo_2_Clean` which is already excluded |
| `PlacingNo_Clean_INSURER` | After stripping becomes `PlacingNo_Clean` which is already excluded |
| `Amount_Clean_INSURER` | Does not exist in backend |
| `string matching` | Not a backend column |
| `string matching_INSURER` | Not a backend column |
| `match_condition` | Not a backend column |
| `match_group` | Not a backend column |

These phantom entries are harmless (excluding a non-existent column is a no-op), but they add confusion. Cleaning them up makes the list trustworthy.

### 3. Keep the fingerprint algorithm identical

The frontend's fingerprint generation function should remain:

```typescript
function generateFingerprint(row: Record<string, unknown>): string {
  const keys = Object.keys(row)
    .filter(k => !FINGERPRINT_EXCLUDE_COLUMNS.has(k))
    .sort(); // lexicographic, case-sensitive

  return keys
    .map(k => {
      const val = row[k];
      if (val === null || val === undefined) return "";
      return String(val); // JS String(5000.0) already produces "5000"
    })
    .join("|");
}
```

No algorithm changes needed — only the exclude list changes.

### 4. Keep the _INSURER stripping logic

When fingerprinting insurer rows from the Exact/Partial sheets or No Matches Insurer sheet:

1. Take all columns ending with `_INSURER`
2. Strip the `_INSURER` suffix
3. Then apply the exclude list and fingerprint

This is already how the frontend works. No change needed here.

---

## How to Verify

After making the frontend changes, verify that fingerprints match by:

1. Run the backend on a test file pair → produces `output.xlsx`
2. Frontend reads `output.xlsx`, splits rows, generates fingerprints
3. Backend generates fingerprints from the same input files (before passes)
4. Compare: for every CBL/insurer row, the frontend fingerprint should equal the backend fingerprint

A simple test: add logging on both sides to print `[column_names_used] → fingerprint` for the first few rows and compare.

---

## Summary of Changes

| Side | What to do |
|---|---|
| **Backend** | Already done — `match_history.py` has the authoritative 20-column exclude list |
| **Frontend** | Replace exclude list with `BACKEND_EXCLUDE_COLUMNS` + `FRONTEND_EXCLUDE_COLUMNS` (just `idx`) |
| **Frontend** | Remove phantom columns that don't exist on either side |
| **Frontend** | No algorithm changes — sort, pipe-join, String() coercion all stay the same |
