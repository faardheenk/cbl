# Match History Fingerprint Problem And Proposed Solution

## Context

We investigated why insurer match-history replay works but CBL replay does not.

After comparing:

- raw source files
- `output.xlsx`
- `history.xlsx`
- frontend fingerprint generation flow

the issue is now clear.

## Confirmed Problem

The frontend saves match-history fingerprints from the rows it loads from `output.xlsx`.

The backend validates/replays match history using fingerprints generated from its backend preprocessing/raw business-row representation.

Those two row representations are not the same for CBL.

As a result, the frontend and backend generate different fingerprints for the same logical CBL row.

## What We Confirmed

### 1. Frontend history is generated from `output.xlsx`

Frontend flow:

1. Load `output.xlsx`
2. Parse `Exact Matches` / `Partial Matches`
3. Split merged rows into CBL and insurer row objects
4. When user moves rows between buckets, generate fingerprints from those in-memory row objects
5. Save those fingerprints into `history.xlsx`

So the frontend is not fingerprinting the original uploaded CBL file directly at move time.

### 2. `output.xlsx` already contains transformed CBL values

Example from `Partial Matches` row 2 in `output.xlsx`:

- `Premiun = 5439.45`
- `Brokerage = 815.92`
- `Policy Fees + FSC = 19.04`
- `Compensation Fees = 0`
- `Net Premium = 4642.57`
- `Pending Brokerage = 0.159999999999968`
- `ProcessedAmount = -4642.153243845825`
- `amount_difference = 392974.6069436371`

Those exact transformed values appear in the saved CBL fingerprint in `history.xlsx`.

This means the frontend is behaving consistently with the workbook it loaded.

### 3. Raw CBL source values are different

For the same logical row in the raw CBL workbook, values are business/raw-style values such as:

- `Premiun = 5439`
- `Brokerage = 816`
- `Policy Fees + FSC = 19`
- `Compensation Fees = " - "`
- `Net Premium = 4643`
- `Pending Brokerage = 0`
- `Month Closed = 2025-04-28`

So backend-side fingerprint generation and frontend-side fingerprint generation are not using the same source representation.

### 4. There is also a frontend exclusion mismatch

The extra 21st fingerprint part comes from `amount_difference`, which exists in `output.xlsx`.

Frontend currently excludes `Amount Difference`, but the actual column in `output.xlsx` is `amount_difference`.

That is a real frontend bug and should still be fixed.

## Root Cause

The system currently has two different fingerprint sources:

- backend-generated fingerprint logic based on backend preprocessing/raw values
- frontend-generated fingerprint logic based on transformed `output.xlsx` rows

For insurer rows, the two representations happen to align closely enough.

For CBL rows, they do not.

## Recommended Solution

## Preferred Fix: Canonical Fingerprint Owned By Backend

The backend should compute the fingerprint once and expose it as the authoritative fingerprint for each row.

Recommended approach:

1. Backend computes canonical fingerprint for each CBL row and insurer row.
2. Backend writes that fingerprint into `output.xlsx`.
3. Frontend reads and stores that fingerprint directly when saving history.
4. Backend replay logic compares against that same stored canonical fingerprint.

This avoids frontend/backend divergence caused by:

- transformed numeric values
- Excel serial date formatting
- output-only columns
- future formatting drift

## Suggested Shape

Add explicit columns to `output.xlsx`, for example:

- `_fingerprint` for CBL rows
- `_fingerprint_INSURER` for insurer rows

or similarly named canonical fields that both sides agree on.

Then the frontend should use those fields directly instead of rebuilding fingerprints from visible row data.

## Why This Is The Best Fix

It removes the need for the frontend to perfectly reconstruct backend preprocessing rules from display/output data.

Today, even if we patch frontend formatting and exclusions, the frontend still only sees transformed CBL output values, not the original backend fingerprint input representation.

So a frontend-only fix is fragile.

## Short-Term Frontend Fixes Still Needed

Even with the canonical backend fix, the frontend should still clean up obvious mismatches:

1. Exclude `amount_difference` from fingerprint generation.
2. Handle Excel date cells consistently.
3. Avoid relying on reconstructed business fingerprints when canonical fingerprint fields are available.

## Decision Summary

### Frontend-only fix?

Not sufficient for a durable solution.

It can reduce mismatch, but it cannot guarantee parity if the frontend fingerprints transformed output rows while backend fingerprints preprocessed/raw rows.

### Backend-only fix?

Mostly yes, if backend provides canonical fingerprint values in `output.xlsx` and frontend consumes them directly.

### Best solution?

Both:

- backend provides canonical fingerprint fields
- frontend switches to storing those canonical fields
- frontend also fixes the `amount_difference` exclusion bug during transition

## Request For Backend Agent

Please evaluate the feasibility of:

1. writing canonical row fingerprint fields into `output.xlsx`
2. using those same canonical fields for history replay/pre-placement
3. keeping the current backend fingerprint function as the single source of truth

If needed, the frontend can then be updated to stop reconstructing CBL fingerprints from output row values and instead persist the backend-provided canonical fingerprint verbatim.
