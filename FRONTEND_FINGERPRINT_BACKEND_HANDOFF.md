# Frontend Fingerprint Handoff

## Purpose

This note explains what was changed on the frontend to align match-history fingerprints with the backend implementation in `matching/match_history.py`.

It is intended for the backend agent to review/confirm parity.

## Files Changed

- `src/utils/matchHistory.ts`
- `src/webparts/reconciliation/components/Reconciliation.tsx`

## What Was Changed

### 1. Fingerprint exclude list was aligned to backend

The frontend fingerprint generator now excludes the same backend-generated metadata columns described in the fingerprint fix spec, plus the frontend/output-only fields:

- `MatrixKey`
- `idx`

The active exclude list now contains:

```ts
[
  "PlacingNo_Clean",
  "PolicyNo_Clean",
  "PolicyNo_2_Clean",
  "ProcessedAmount_Clean",
  "ClientName_Clean",
  "match_status",
  "match_pass",
  "match_reason",
  "matched_insurer_indices",
  "matched_amtdue_total",
  "Amount Difference",
  "partial_candidates_indices",
  "match_resolved_in_pass",
  "partial_resolved_in_pass",
  "group_id",
  "corporate_root",
  "match_confidence",
  "_source_sheet",
  "_fingerprint",
  "MatrixKey",
  "idx",
]
```

## 2. Frontend value formatting now mirrors backend rules

The fingerprint formatter in `src/utils/matchHistory.ts` now applies these rules:

- `null` -> `""`
- `undefined` -> `""`
- `""` -> `""`
- `NaN` -> `""`
- `Date` -> `DD/MM/YYYY`
- integer-valued numbers like `5000.0` -> `"5000"`
- all other values -> `String(value)`

Keys are still:

- filtered by the exclude list
- sorted lexicographically with default JS `.sort()`
- joined with `"|"`

## 3. Insurer fingerprint generation now strips `_INSURER`

Frontend match-history saving previously called the generic fingerprint function for both CBL and insurer rows.

That has been changed so that:

- CBL rows still use `generateFingerprint(row)`
- insurer rows now use `generateInsurerFingerprint(row)`

`generateInsurerFingerprint()` normalizes keys by stripping `_INSURER` before applying the exclude list and before building the fingerprint string.

This was added to match backend behavior.

## 4. Match-history save path was updated

In `src/webparts/reconciliation/components/Reconciliation.tsx`, the history entry now does:

```ts
cblFingerprints: nonBlankCBL.map((row) => generateFingerprint(row)),
insurerFingerprints: nonBlankInsurer.map((row) =>
  generateInsurerFingerprint(row),
),
```

## Important Implementation Notes

- The frontend fingerprint code uses the row objects already loaded from `output.xlsx`.
- No additional computed values were intentionally introduced into the fingerprint path.
- This change was focused on fingerprint generation for match-history persistence, not on altering the backend preprocessing pipeline.

## What Has Not Been Fully Verified Yet

The code changes were implemented and checked for local TypeScript/lint issues in the edited files, but a full end-to-end parity verification has not yet been completed.

The backend agent should still verify:

1. CBL fingerprint part counts now match backend output.
2. CBL fingerprint values match position-by-position.
3. Insurer fingerprint values still match after `_INSURER` stripping.
4. No backend-only normalization exists that the frontend still does not mirror.

## Suggested Backend Validation

Please compare a few frontend-generated fingerprints against backend-generated fingerprints from the same `output.xlsx` source rows.

Recommended checks:

1. Compare the first 3 CBL fingerprints.
2. Compare the first 3 insurer fingerprints.
3. If any mismatch remains, split both strings by `"|"` and inspect the first differing position.

## Summary

Frontend changes completed:

- aligned fingerprint exclude columns
- added backend-style value formatting
- added insurer `_INSURER` key stripping
- updated match-history save flow to use the insurer-specific fingerprint function

If parity is still failing after this, the most likely remaining issue is not the fingerprint algorithm itself, but a mismatch in the row values being fingerprinted on one side versus the other.
