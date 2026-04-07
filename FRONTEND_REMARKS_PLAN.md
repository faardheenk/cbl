# Frontend Plan: Remarks Preservation via History

## Context

The backend is adding `Remarks` and `Remarks_INSURER` columns to all output sheets. These are free-text annotation columns that users can edit in the reconciliation UI.

The problem: when the backend re-processes data (fresh CBL + insurer uploads), any remarks the user previously entered would be lost because the fresh files don't contain them.

**Solution:** Use the existing `history.xlsx` mechanism to preserve remarks across re-runs.

---

## How History Works Today

When a user manually moves rows between buckets, the frontend writes to `history.xlsx` with this format:

| Column | Type | Description |
|--------|------|-------------|
| `CblFingerprints` | JSON array of strings | Fingerprints of moved CBL rows |
| `InsurerFingerprints` | JSON array of strings | Fingerprints of moved insurer rows |
| `FromBucket` | string | Source bucket key |
| `TargetBucket` | string | Target bucket key |
| `Timestamp` | string | When the move happened |

The backend reads this during `apply_match_history()`, matches fingerprints to new data, and pre-places rows into their correct buckets.

---

## What the Frontend Needs to Add

### 1. New Columns in History Entries

Add two optional JSON columns to each history entry:

| Column | Type | Description |
|--------|------|-------------|
| `CblRemarks` | JSON array of strings | Remarks for each CBL fingerprint (parallel array) |
| `InsurerRemarks` | JSON array of strings | Remarks for each insurer fingerprint (parallel array) |

These arrays are **parallel** to `CblFingerprints` and `InsurerFingerprints` — same length, same order. Each element is the remark text for the corresponding fingerprint.

### 2. Example

If a user has 2 CBL rows and 1 insurer row in a bucket, and has added remarks to the first CBL row and the insurer row:

```json
{
  "CblFingerprints": ["fp_cbl_1", "fp_cbl_2"],
  "InsurerFingerprints": ["fp_ins_1"],
  "CblRemarks": ["Need to verify amount", ""],
  "InsurerRemarks": ["Confirmed with underwriter"],
  "FromBucket": "partial",
  "TargetBucket": "exact",
  "Timestamp": "2026-04-07T10:00:00Z"
}
```

### 3. When to Write Remarks to History

Write remarks into history entries in the same flow where you currently write bucket moves. Specifically:

- When saving/persisting history entries, include the current `Remarks` value for each CBL row and the current `Remarks_INSURER` value for each insurer row
- If a row has no remark, use empty string `""`
- The arrays must maintain 1:1 correspondence with the fingerprint arrays

### 4. Handling Rows Without Remarks

- If no remarks exist for any row in an entry, you can either:
  - Omit `CblRemarks` / `InsurerRemarks` entirely (backend treats missing as all-empty)
  - Include arrays of empty strings

Both approaches work — the backend handles missing columns gracefully.

---

## What the Backend Will Do

The backend will handle this in `apply_match_history()`:

1. Read `CblRemarks` and `InsurerRemarks` from each history entry (if present)
2. After matching a fingerprint to a row in the new data, set the `Remarks` / `Remarks_INSURER` value from the parallel array
3. If the columns are missing from history, all remarks default to `""`

This means:
- **Old history files without remarks columns** continue to work (backwards compatible)
- **New history files with remarks** will have remarks restored automatically

---

## Important: Fingerprints Must NOT Include Remarks

The backend excludes `Remarks` from fingerprint generation (`FINGERPRINT_EXCLUDE_COLUMNS`).

The frontend must also exclude `Remarks` and `Remarks_INSURER` from its fingerprint logic (add to `FINGERPRINT_EXCLUDE_COLUMNS` in the frontend code). This ensures:

- A row's fingerprint is the same whether or not it has a remark
- History matching works correctly across re-runs
- Adding/editing a remark doesn't change a row's identity

---

## Column Details in output.xlsx

After the backend changes, output.xlsx sheets will have:

| Sheet | New Columns |
|-------|------------|
| Exact Matches | `Remarks` (last CBL col), `Remarks_INSURER` (last insurer col) |
| Partial Matches | `Remarks` (last CBL col), `Remarks_INSURER` (last insurer col) |
| No Matches CBL | `Remarks` (last CBL col) |
| No Matches Insurer | `Remarks_INSURER` (last insurer col) |
| Dynamic bucket sheets | `Remarks` (last CBL col), `Remarks_INSURER` (last insurer col) |

The frontend should:
- Display the `Remarks` column as an editable text field in the CBL table
- Display the `Remarks_INSURER` column as an editable text field in the insurer table (strip `_INSURER` suffix for display, as with other insurer columns)
- Treat both as free-text, no validation needed

---

## Summary of Frontend Changes

1. **Add `Remarks` / `Remarks_INSURER` to `FINGERPRINT_EXCLUDE_COLUMNS`** in frontend fingerprint logic
2. **Write `CblRemarks` / `InsurerRemarks`** into history.xlsx entries (parallel arrays to fingerprints)
3. **Display remarks columns** as editable text fields in both CBL and insurer tables
4. **Read remarks from output.xlsx** — they'll be present in all sheets after backend update
