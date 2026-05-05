# Regroup Match History Backend Question

## Purpose

This note documents how row regrouping currently works from the frontend point of view, how it is persisted into `history.xlsx`, and the key backend question we need answered:

**Does the backend reprocessing flow support regroup-specific match history, or does it only support moving rows into target buckets?**

This distinction matters because regrouping is not just a bucket move. It means selected rows must be merged into an existing target group, using that target group's grouping relationship.

## Existing Match History Model

The current match history model is based on two concepts:

1. `history.xlsx` stores the action history.
2. Fingerprints identify the rows involved in each action.

Fingerprints are row identity. They are generated from business row data and exclude frontend/backend metadata such as:

- `idx`
- `match_group`
- `match_condition`
- `matched_insurer_indices`
- `group_id`
- `_fingerprint`
- `_fingerprint_INSURER`
- remarks and other output-only fields

This means the backend can reprocess fresh Excel files, regenerate canonical fingerprints, read `history.xlsx`, and locate the same logical rows again.

The documented backend flow is:

```text
New Excel files uploaded
↓
Backend preprocesses rows
↓
Backend generates canonical fingerprints
↓
Backend reads Matrix/{INSURER}/history.xlsx
↓
Backend matches history fingerprints against fresh rows
↓
Matched rows are pre-placed into TargetBucket
↓
Matched rows are skipped by normal matching passes
↓
Backend writes output.xlsx
```

This works well for normal manual moves, where the intent is:

```text
Move these CBL/insurer rows from bucket A to bucket B.
```

## Why Regrouping Is Different

Regrouping means:

```text
Take selected rows from one bucket and merge them into an existing target group in another bucket.
```

That is more specific than a normal move.

For regrouping, the system needs to know:

- Which rows moved.
- Which bucket they moved from.
- Which bucket they moved to.
- Which existing target group they were merged into.
- Which target CBL rows defined that group.
- Which target insurer rows defined that group.
- Which one-sided rows became orphaned and moved to no-match.

The frontend currently captures this using regroup-specific fields in match history:

```text
ActionType = "regroup"
CblFingerprints
InsurerFingerprints
TargetCblFingerprints
TargetInsurerFingerprints
OrphanedCblFingerprints
OrphanedInsurerFingerprints
FromBucket
TargetBucket
Timestamp
```

For a normal bucket move, `CblFingerprints` and `InsurerFingerprints` plus `TargetBucket` are enough.

For regrouping, `TargetCblFingerprints` and `TargetInsurerFingerprints` are essential because they identify the existing group that the moved rows should be merged into.

## Frontend Regroup Behavior

When the user regroups rows in the UI, the frontend does the following:

1. User pins a regroup target row.
2. User selects rows from another bucket.
3. Frontend identifies the full target group using the target row.
4. Frontend removes selected rows from the source bucket.
5. Frontend removes the old target group from the target bucket.
6. Frontend merges:
   - existing target group rows
   - selected moved rows
7. Frontend assigns the moved rows to the target group metadata:
   - `match_group`
   - `match_condition`
   - `group_id`
8. Frontend recomputes row relationship data such as `matched_insurer_indices`.
9. Frontend saves a regroup history entry into `history.xlsx`.

So in the UI, regrouping is already treated as a group merge, not merely as a bucket transfer.

## Current Risk

The docs confirm that backend `apply_match_history()` reads history entries and pre-places rows into their target buckets using fingerprints.

However, we need to confirm whether backend `apply_match_history()` understands regroup-specific fields:

- `ActionType`
- `TargetCblFingerprints`
- `TargetInsurerFingerprints`
- `OrphanedCblFingerprints`
- `OrphanedInsurerFingerprints`

If the backend ignores these fields, then during reprocessing it may only do this:

```text
Put moved rows into TargetBucket.
```

But it may not do this:

```text
Merge moved rows into the specific target group identified by TargetCblFingerprints and TargetInsurerFingerprints.
```

That would mean the backend output could preserve bucket placement but lose the exact regrouped structure.

## Backend Question

Does the backend currently support `ActionType = "regroup"` in `apply_match_history()`?

More specifically:

1. Does backend reprocessing read `ActionType` from `history.xlsx`?
2. If `ActionType` is `"regroup"`, does it read:
   - `TargetCblFingerprints`
   - `TargetInsurerFingerprints`
   - `OrphanedCblFingerprints`
   - `OrphanedInsurerFingerprints`
3. Does it locate both the moved rows and the target group rows by fingerprint?
4. Does it remove the target group rows from their original position before rebuilding the merged group?
5. Does it append or place the merged regrouped block as one group in the target bucket?
6. Does it assign a consistent group identifier to all regrouped CBL and insurer rows?
7. Does it recompute `matched_insurer_indices` or equivalent relationship metadata for the regrouped block?
8. Does it handle one-sided regroup selections by moving orphaned rows to `no-match`?
9. Does it support regrouping into dynamic buckets as well as fixed buckets?

## Expected Backend Behavior If Supported

If backend-side regroup support exists, reprocessing should behave like this:

```text
Backend reads fresh uploaded Excel files
↓
Backend generates canonical fingerprints
↓
Backend reads history.xlsx
↓
For normal move entries:
  pre-place rows into TargetBucket
↓
For regroup entries:
  find moved rows by CblFingerprints / InsurerFingerprints
  find target group rows by TargetCblFingerprints / TargetInsurerFingerprints
  merge target group rows + moved rows
  place merged block into TargetBucket as one group
  move orphaned rows to no-match
  recompute grouping/relationship metadata
↓
Backend skips history-resolved rows during matching passes
↓
Backend writes output.xlsx with regrouped rows already grouped
```

## If Backend Does Not Support This Yet

If backend history replay only supports bucket placement, then regroup persistence is only partially supported.

The backend can still place the moved rows into the target bucket, but it will not necessarily preserve the UI regroup relationship.

In that case, we have two options:

1. **Frontend replay only**
   - Backend writes output normally after bucket pre-placement.
   - Frontend loads `output.xlsx`.
   - Frontend reads regroup history.
   - Frontend reapplies regrouping in memory.
   - This makes the UI look correct, but backend-generated `output.xlsx` is not fully regroup-aware before frontend loads it.

2. **Backend regroup support**
   - Backend extends `apply_match_history()` to handle `ActionType = "regroup"`.
   - Backend uses target fingerprints to rebuild the target group.
   - Backend writes `output.xlsx` with regrouped rows already grouped.
   - This is the stronger and more consistent option if reports, exports, or downstream processes depend on backend output.

## Recommendation

We should ask the backend owner/agent to confirm the current behavior before changing frontend regroup metadata.

The key answer needed is:

```text
When reprocessing Excel files, does backend apply_match_history() only pre-place rows into TargetBucket,
or does it also rebuild regrouped target groups using ActionType = "regroup" and target fingerprints?
```

If the answer is "bucket placement only", then backend changes are needed for true regroup persistence during reprocessing.

If the answer is "regroup is supported", then the frontend should keep writing regroup-specific history fields and make sure any new frontend-only metadata does not enter fingerprints or saved Excel output.
