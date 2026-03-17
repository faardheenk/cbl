# Frontend Fingerprint Fix — Final Solution

## Problem (Resolved)

The frontend and backend generated different fingerprints for the same CBL rows because they used different source data:
- Backend fingerprinted from **raw preprocessed input** (integer values like `816`, ` - ` strings)
- Frontend fingerprinted from **output.xlsx** (transformed values like `815.92`, `0`)

This mismatch is unfixable from the frontend side alone — the output.xlsx values are genuinely different from the raw input values.

---

## Solution: Backend-Owned Canonical Fingerprints

The backend now writes **canonical fingerprint columns** into `output.xlsx`. The frontend should read these directly instead of regenerating fingerprints.

### New columns in output.xlsx

| Column | Present In | Description |
|--------|-----------|-------------|
| `_fingerprint` | Exact Matches, Partial Matches, No Matches CBL | Canonical CBL row fingerprint |
| `_fingerprint_INSURER` | Exact Matches, Partial Matches | Canonical insurer row fingerprint |

These are generated from the preprocessed input data — the same source the backend uses for match history replay. They are guaranteed to match.

---

## What the Frontend Must Change

### 1. Use `_fingerprint` / `_fingerprint_INSURER` directly when saving history

When the user moves rows between buckets and the frontend saves to `history.xlsx`:

```typescript
// BEFORE (broken — regenerates fingerprint from transformed output data):
cblFingerprints: selectedCblRows.map(row => generateFingerprint(row))
insurerFingerprints: selectedInsurerRows.map(row => generateInsurerFingerprint(row))

// AFTER (correct — reads the backend's canonical fingerprint):
cblFingerprints: selectedCblRows.map(row => row["_fingerprint"])
insurerFingerprints: selectedInsurerRows.map(row => row["_fingerprint_INSURER"])
```

That's it. No fingerprint generation needed on the frontend for match history.

### 2. Keep `_fingerprint` and `_fingerprint_INSURER` in the exclude list

These columns should remain excluded from any other processing. They are already in the backend's exclude list, and the frontend should exclude them too (they're internal metadata, not business data):

```typescript
const FINGERPRINT_EXCLUDE_COLUMNS = new Set([
  // ... existing entries ...
  "_fingerprint",
  // _fingerprint_INSURER becomes _fingerprint after _INSURER stripping,
  // so it's already covered
]);
```

### 3. Handle the `_fingerprint` column for insurer rows

For insurer rows in the Exact/Partial sheets, the fingerprint column is named `_fingerprint_INSURER` (with the suffix). When saving to history, use this column name directly:

```typescript
// For CBL rows (from any sheet):
const cblFp = row["_fingerprint"];

// For insurer rows (from Exact/Partial merged sheets):
const insurerFp = row["_fingerprint_INSURER"];

// For insurer rows (from No Matches Insurer sheet, if columns already stripped):
const insurerFp = row["_fingerprint"];
```

### 4. Remove `generateFingerprint` / `generateInsurerFingerprint` from the history save path

The functions in `src/utils/matchHistory.ts` (`generateFingerprint`, `normalizeRowForFingerprint`, `formatFingerprintValue`) can remain in the codebase if they're used elsewhere, but they should **not** be called when building the `CblFingerprints` / `InsurerFingerprints` arrays for `history.xlsx`.

### 5. Fix `amount_difference` exclusion (separate bug)

The frontend exclude list has `"Amount Difference"` but the actual column in output.xlsx is `amount_difference` (lowercase). This is a separate bug that caused the extra 21st fingerprint part. Since the frontend no longer regenerates fingerprints for history, this bug is bypassed — but it should still be fixed if `generateFingerprint` is used anywhere else.

---

## How It Works End-to-End

```
1. User uploads CBL + insurer files
2. Backend preprocesses data, generates canonical _fingerprint columns
3. Backend runs matching passes
4. Backend writes output.xlsx (includes _fingerprint and _fingerprint_INSURER)
5. Frontend loads output.xlsx, displays rows in buckets
6. User moves rows between buckets
7. Frontend reads _fingerprint / _fingerprint_INSURER from row objects
8. Frontend saves those values directly into history.xlsx
9. Next run: backend reads history.xlsx, regenerates fingerprints from new input
10. Backend matches history fingerprints against new fingerprints → pre-places rows
```

Steps 7-8 are the key change: the frontend stores the backend's canonical fingerprints instead of regenerating its own.

---

## Verification

After implementing the frontend change:

1. Run the backend on test files → produces output.xlsx with `_fingerprint` columns
2. Frontend loads output.xlsx, user moves rows
3. Frontend saves history.xlsx with fingerprints read from `_fingerprint` columns
4. Run `python test_history_local.py` → should show `[MATCH]` for all fingerprints

---

## Summary

| What Changed | Where | Details |
|-------------|-------|---------|
| Backend writes canonical fingerprints | `matching/orchestrator.py` | Generates `_fingerprint` and `_fingerprint_INSURER` after preprocessing |
| Backend history replay unchanged | `matching/match_history.py` | Uses same canonical fingerprints for matching |
| Frontend reads fingerprints directly | `src/utils/matchHistory.ts` + save flow | Reads `_fingerprint` / `_fingerprint_INSURER` instead of regenerating |
