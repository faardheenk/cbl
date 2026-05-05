# Regroup Backend Follow-Up Questions

## Purpose

`REGROUP_BACKEND_SUPPORT.md` says backend `apply_match_history()` now supports `ActionType = "regroup"`.

Before the frontend relies on that behavior, we need clarification on a few details where backend behavior must match the frontend regroup model.

The key concern is this:

```text
Regroup is not only "move these rows to TargetBucket".
Regroup is "merge these moved rows into this existing target group".
```

That means the backend must preserve the same relationship between CBL rows, insurer rows, group IDs, orphaned rows, and `matched_insurer_indices`.

## 1. Are Orphaned Insurer Rows Also Moved To No-Match?

### Why This Matters

`REGROUP_BACKEND_SUPPORT.md` says:

```text
Places orphaned CBL rows into no-match.
```

But frontend regroup can create **orphaned CBL rows** and **orphaned insurer rows**.

The frontend writes both fields:

```json
{
  "OrphanedCblFingerprints": ["cbl_orphan_1"],
  "OrphanedInsurerFingerprints": ["insurer_orphan_1"]
}
```

### Example

Before regroup, source group in `partial`:

```text
CBL side                Insurer side
------------------------------------------------
CBL A                  Insurer A
CBL B                  Insurer B
```

User selects only:

```text
CBL A
```

and regroups it into an existing target group.

Now `Insurer A` may be left behind without its CBL partner. The frontend treats that as an orphaned insurer row and moves it to `no-match`.

Expected history entry:

```json
{
  "ActionType": "regroup",
  "CblFingerprints": ["cbl_a"],
  "InsurerFingerprints": [],
  "TargetCblFingerprints": ["target_cbl_1"],
  "TargetInsurerFingerprints": ["target_ins_1"],
  "OrphanedCblFingerprints": [],
  "OrphanedInsurerFingerprints": ["insurer_a"],
  "FromBucket": "partial",
  "TargetBucket": "exact"
}
```

### Question

Does backend regroup replay move **both** `OrphanedCblFingerprints` and `OrphanedInsurerFingerprints` to `no-match`?

Expected answer:

```text
Yes, orphaned CBL rows go to No Matches CBL, and orphaned insurer rows go to No Matches Insurer.
```

## 2. Is `group_id` Assigned To Both CBL And Insurer Rows?

### Why This Matters

Frontend regroup assigns target group identity to both sides:

```text
CBL rows receive target group_id
Insurer rows receive target group_id
```

Some frontend logic also checks insurer `group_id` when resolving target ranges and highlighting regroup targets.

### Example

Target group before regroup:

```text
CBL side                         Insurer side
----------------------------------------------------------
Target CBL 1 group_id=G100       Target Ins 1 group_id=G100
```

Moved rows:

```text
Moved CBL 1 group_id=G200        Moved Ins 1 group_id=G200
```

After regroup, expected:

```text
CBL side                         Insurer side
----------------------------------------------------------
Target CBL 1 group_id=G100       Target Ins 1 group_id=G100
Moved CBL 1  group_id=G100       Moved Ins 1  group_id=G100
```

If backend creates a new history group ID, that is also fine, but it should still be consistent on both sides:

```text
CBL side                                      Insurer side
----------------------------------------------------------------------
Target CBL 1 group_id=HISTORY_REGROUP_1       Target Ins 1 group_id=HISTORY_REGROUP_1
Moved CBL 1  group_id=HISTORY_REGROUP_1       Moved Ins 1  group_id=HISTORY_REGROUP_1
```

### Question

Does backend assign the regroup `group_id` to **both** merged CBL rows and merged insurer rows?

Expected answer:

```text
Yes, all rows in the regrouped block, on both CBL and insurer sides, receive the same regroup group_id.
```

## 3. How Exactly Is `matched_insurer_indices` Recomputed?

### Why This Matters

The frontend currently relies on `matched_insurer_indices` for auto-selection.

In frontend regroup, `matched_insurer_indices` is recomputed positionally after blank row equalization.

It does **not always mean** every CBL row gets the same list of all insurer rows in the whole group.

### Frontend-Style Example

Suppose the merged regrouped block looks like this:

```text
Row position    CBL side             Insurer side
---------------------------------------------------------
0               CBL A                Insurer A
1               blank                Insurer B
2               CBL B                Insurer C
```

Frontend positional repair produces:

```text
CBL A matched_insurer_indices = [0, 1]
CBL B matched_insurer_indices = [0]
```

Meaning:

```text
CBL A owns rows 0..1 on insurer side
CBL B owns row 2 on insurer side
```

But `REGROUP_BACKEND_SUPPORT.md` says:

```text
All merged CBL rows receive a combined matched_insurer_indices list covering both moved and target insurer rows.
```

That could imply:

```text
CBL A matched_insurer_indices = [0, 1, 2]
CBL B matched_insurer_indices = [0, 1, 2]
```

That is different from frontend behavior and could make auto-selection select too many insurer rows.

### Question

Which algorithm does backend use for regrouped `matched_insurer_indices`?

Expected answer if backend should match frontend:

```text
Backend recomputes matched_insurer_indices positionally from the final parallel CBL/insurer row layout.
Each non-blank CBL row owns the insurer span from its row position until the next non-blank CBL row.
Blank CBL rows do not receive matched_insurer_indices.
```

If backend intentionally uses a different algorithm, we need the exact expected frontend behavior so auto-selection can be aligned.

## 4. Does Backend Equalize CBL And Insurer Row Counts With Blank Spacer Rows?

### Why This Matters

Frontend regroup keeps CBL and insurer arrays parallel. If one side has fewer real rows, frontend inserts blank spacer rows.

This matters because `matched_insurer_indices` depends on row position.

### Example

Merged real rows:

```text
CBL rows:     CBL A
Insurer rows: Insurer A, Insurer B
```

Frontend equalizes this as:

```text
Row position    CBL side             Insurer side
---------------------------------------------------------
0               CBL A                Insurer A
1               blank                Insurer B
```

Then:

```text
CBL A matched_insurer_indices = [0, 1]
```

If backend does not add the blank spacer row, output layout may become ambiguous:

```text
Row position    CBL side             Insurer side
---------------------------------------------------------
0               CBL A                Insurer A
?               ???                  Insurer B
```

### Question

During regroup replay, does backend equalize the merged CBL and insurer block with blank spacer rows before writing `output.xlsx`?

Expected answer:

```text
Yes, backend writes the regrouped block in the same parallel layout the frontend expects:
same row count on both sides, blank spacer rows where needed.
```

## 5. Does Backend Remove The Original Target Group Before Appending The Merged Group?

### Why This Matters

Frontend regroup removes the old target group from the target bucket, merges it with moved rows, then appends the rebuilt group.

If backend does not remove the original target group first, target rows could appear twice.

### Example

Before regroup:

```text
Exact bucket:
  Target CBL 1 / Target Ins 1

Partial bucket:
  Moved CBL 1 / Moved Ins 1
```

After regroup, expected exact bucket:

```text
Exact bucket:
  Target CBL 1 / Target Ins 1
  Moved CBL 1  / Moved Ins 1
```

Wrong result if target group is not removed first:

```text
Exact bucket:
  Target CBL 1 / Target Ins 1
  Target CBL 1 / Target Ins 1
  Moved CBL 1  / Moved Ins 1
```

### Question

Does backend remove the original target rows identified by `TargetCblFingerprints` / `TargetInsurerFingerprints` before writing the rebuilt regrouped block?

Expected answer:

```text
Yes, target group rows are claimed/removed first, then emitted once as part of the merged regrouped block.
```

## 6. Does This Work For Dynamic Buckets Too?

### Why This Matters

The frontend supports regrouping into fixed buckets and dynamic buckets.

`TargetBucket` may be:

```text
exact
partial
no-match
some_dynamic_bucket_key
```

### Example

```json
{
  "ActionType": "regroup",
  "CblFingerprints": ["cbl_moved_1"],
  "InsurerFingerprints": ["ins_moved_1"],
  "TargetCblFingerprints": ["cbl_target_1"],
  "TargetInsurerFingerprints": ["ins_target_1"],
  "FromBucket": "partial",
  "TargetBucket": "mise_en_demeure"
}
```

### Question

Does backend regroup replay support dynamic bucket target keys exactly the same way normal move history does?

Expected answer:

```text
Yes, regroup TargetBucket can be any valid dynamic BucketKey, and the regrouped block is written to that dynamic bucket sheet.
```

## Summary Of Questions For Backend

Please confirm:

1. Are orphaned insurer rows moved to `No Matches Insurer`, not only orphaned CBL rows?
2. Is regroup `group_id` assigned to both CBL and insurer rows?
3. Is `matched_insurer_indices` recomputed using the frontend positional span algorithm, or a different algorithm?
4. Are blank spacer rows inserted so CBL and insurer sides stay parallel?
5. Are original target group rows removed before the rebuilt merged group is written?
6. Does the same behavior work for dynamic bucket targets?

## Desired Contract

The safest contract is:

```text
Backend regroup replay should produce an output.xlsx layout that the frontend can load without needing to replay regroup history again.
```

That means:

- rows are already in the correct target bucket;
- target and moved rows appear once;
- the regrouped block is contiguous;
- both sides share a consistent group ID;
- orphaned rows are in no-match;
- row counts are equalized with blank spacers;
- `matched_insurer_indices` supports the same auto-selection behavior as frontend regroup.
