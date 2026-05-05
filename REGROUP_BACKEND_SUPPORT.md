# Backend Regroup Support — Implementation Summary

## Status

Backend `apply_match_history()` now fully supports `ActionType = "regroup"`.

No frontend changes are needed if the frontend is already writing these fields to `history.xlsx`. If it is not yet writing them, this document specifies the expected contract.

## History Entry Contract

The backend reads the following columns from the `MatchHistory` sheet in `history.xlsx`.

### Existing fields (unchanged)

| Column | Type | Description |
|---|---|---|
| `CblFingerprints` | JSON array of strings | Fingerprints of the CBL rows being moved |
| `InsurerFingerprints` | JSON array of strings | Fingerprints of the insurer rows being moved |
| `CblRemarks` | JSON array of strings | Remarks parallel to `CblFingerprints` |
| `InsurerRemarks` | JSON array of strings | Remarks parallel to `InsurerFingerprints` |
| `FromBucket` | string | Source bucket key |
| `TargetBucket` | string | Destination bucket key (`exact`, `partial`, `no-match`, or a dynamic bucket key) |
| `Timestamp` | string/datetime | When the action occurred |

### New fields (regroup-specific)

| Column | Type | Required for regroup | Description |
|---|---|---|---|
| `ActionType` | string | Yes | `"move"` (default) or `"regroup"` |
| `TargetCblFingerprints` | JSON array of strings | Yes | Fingerprints of the CBL rows in the existing target group being merged into |
| `TargetInsurerFingerprints` | JSON array of strings | Yes | Fingerprints of the insurer rows in the existing target group being merged into |
| `OrphanedCblFingerprints` | JSON array of strings | Optional | Fingerprints of CBL rows that became one-sided after the regroup and should go to no-match |
| `OrphanedInsurerFingerprints` | JSON array of strings | Optional | Fingerprints of insurer rows that became one-sided after the regroup and should go to no-match |

## Backward Compatibility

- If `ActionType` is missing or empty, the backend defaults to `"move"`. Existing history entries continue to work exactly as before.
- The regroup-specific columns (`TargetCblFingerprints`, etc.) are only read when `ActionType == "regroup"`. They are ignored for move entries.

## What the Backend Does During Regroup Replay

When processing a regroup history entry, the backend:

1. Locates the **moved rows** using `CblFingerprints` / `InsurerFingerprints`.
2. Locates the **target group rows** using `TargetCblFingerprints` / `TargetInsurerFingerprints`.
3. Locates any **orphaned rows** using `OrphanedCblFingerprints` / `OrphanedInsurerFingerprints`.
4. Merges moved + target rows into a single group:
   - All merged CBL rows receive the same `group_id` (`HISTORY_REGROUP_N`).
   - All merged CBL rows receive a combined `matched_insurer_indices` list covering both moved and target insurer rows.
   - All merged rows are marked with `match_pass = ["history"]` and `match_reason = "History regroup (bucket)"`.
5. Places orphaned CBL rows into no-match.
6. Registers all involved insurer indices in `GlobalTracker` so matching passes do not reuse them.

## What the Frontend Needs to Write

When a user performs a regroup action, the frontend should write a single history entry with:

```json
{
  "ActionType": "regroup",
  "CblFingerprints": ["<moved CBL row fingerprints>"],
  "InsurerFingerprints": ["<moved insurer row fingerprints>"],
  "TargetCblFingerprints": ["<existing target group CBL fingerprints>"],
  "TargetInsurerFingerprints": ["<existing target group insurer fingerprints>"],
  "OrphanedCblFingerprints": ["<orphaned CBL fingerprints, if any>"],
  "OrphanedInsurerFingerprints": ["<orphaned insurer fingerprints, if any>"],
  "FromBucket": "source-bucket-key",
  "TargetBucket": "target-bucket-key",
  "Timestamp": "2026-04-29T12:00:00Z"
}
```

### Definitions

- **Moved rows** (`CblFingerprints` / `InsurerFingerprints`): The rows the user selected and dragged/moved from the source bucket.
- **Target group rows** (`TargetCblFingerprints` / `TargetInsurerFingerprints`): The rows that already existed in the target group that the moved rows are being merged into. These are the rows identified by the pinned regroup target row's group.
- **Orphaned rows** (`OrphanedCblFingerprints` / `OrphanedInsurerFingerprints`): Rows from the source group that became one-sided after the selected rows were removed. For example, if a user moves all CBL rows out of a group but leaves insurer rows behind, those insurer rows are orphaned. These should be empty arrays if no orphans were created.

## Dynamic Bucket Support

Regrouping into dynamic buckets is supported. Set `TargetBucket` to the dynamic bucket's `BucketKey` value. The backend uses the same sentinel mechanism (`_DynamicBucket_` prefix) for both move and regroup entries targeting dynamic buckets.

## Verification

The backend logs regroup operations with the `[HISTORY]` prefix. During reprocessing, the logs will show:

```
[HISTORY] Entry #N (regroup -> target-bucket): CBL X/Y matched, Insurer X/Y matched
[HISTORY] Entry #N regroup: target CBL=A, target Insurer=B, orphaned CBL=C, orphaned Insurer=D
[HISTORY] Regroup-placed CBL idx=I -> Match Status | group=HISTORY_REGROUP_N
```

If any fingerprints fail to match during reprocessing, they are logged individually with `NOT FOUND` so mismatches can be diagnosed.
