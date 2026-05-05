# Frontend Auto-Selection Simplification — Backend Caveats

## Purpose

The backend proposal in `FRONTEND_AUTO_SELECTION_PROPOSAL.md` is a good direction:

```text
Assign group_id to both CBL rows and insurer rows.
Frontend can use group_id for auto-selection and highlighting.
```

This would remove a lot of fragile frontend logic that currently reverse-engineers relationships from `matched_insurer_indices` and row positions.

Before we refactor the frontend, we need backend confirmation on the caveats below.

## Current Frontend Workflow

### 1. CBL Click Auto-Selects Insurer Rows

When the user clicks a CBL row, the frontend currently determines insurer rows through `matched_insurer_indices`.

Current flow:

```text
User clicks CBL row
↓
Frontend reads CBL row.matched_insurer_indices
↓
Frontend parses it as a JSON array
↓
Frontend converts the matched span into insurer row IDs
↓
Frontend marks those insurer rows as auto-selected
```

The main helper is:

```text
src/utils/rowMapping.ts
getTargetInsurerRowIdsForCblRow()
```

That helper does not directly receive insurer rows. It uses the CBL row ID pattern and the parsed `matched_insurer_indices` length to infer the parallel insurer row IDs.

### 2. Partial CBL Group Selection Uses `group_id`

There is already some `group_id`-based selection, but it is deliberately restricted.

Current behavior:

```text
If fileType is CBL
and section is partial
and row has group_id
and auto-select is enabled
then select other non-blank CBL rows with the same group_id.
```

This is restricted to `partial` because exact/dynamic buckets may carry backend `group_id` values that were not previously safe as frontend UI group identifiers.

### 3. Insurer Rows Are Mostly Selected By CBL Mapping

Current insurer auto-selection is driven from CBL rows.

The frontend does not currently treat an insurer row's own `group_id` as the source of truth for finding related CBL rows.

Current flow:

```text
CBL row selected
↓
matched_insurer_indices determines insurer rows
↓
insurer rows become external/auto-selected rows
```

### 4. Regroup Target Highlighting Uses Mixed Logic

Regroup target highlighting currently uses a helper that combines:

- row position;
- CBL `matched_insurer_indices`;
- CBL `group_id`;
- insurer `group_id` if present.

Current flow:

```text
Pinned regroup target row
↓
Find target row position
↓
If target is insurer row, walk backward to find anchor CBL row
↓
Use matched_insurer_indices and/or group_id to find group range
↓
Highlight CBL and insurer rows in that parallel range
```

This logic exists because insurer rows historically did not have reliable group identity.

### 5. Regroup Repairs Relationships

After a regroup action, the frontend currently does several things:

```text
Remove selected rows from source bucket
↓
Remove old target group from target bucket
↓
Merge target rows + moved rows
↓
Assign target match_group / match_condition / group_id to moved rows
↓
Equalize CBL and insurer row counts with blank spacer rows
↓
Recompute matched_insurer_indices positionally
↓
Append rebuilt group to target bucket
↓
Move one-sided orphaned rows to no-match
```

The most fragile part is recomputing `matched_insurer_indices`.

The current frontend algorithm is positional:

```text
Each non-blank CBL row owns the insurer span from its row position
until the next non-blank CBL row.
```

Example:

```text
Row position    CBL side             Insurer side
---------------------------------------------------------
0               CBL A                Insurer A
1               blank                Insurer B
2               CBL B                Insurer C
```

Frontend repairs this as:

```text
CBL A matched_insurer_indices = [0, 1]
CBL B matched_insurer_indices = [0]
```

## What The Backend Proposal Would Improve

If backend gives every matched row a reliable `group_id` on both sides, frontend selection can become:

```text
User clicks any row with group_id = G123
↓
Select all CBL rows with group_id = G123
↓
Select all insurer rows with group_id = G123
```

This would simplify:

- CBL-to-insurer auto-selection;
- insurer-to-CBL reverse selection;
- group highlighting;
- regroup target range detection;
- dynamic bucket support;
- history/regroup replay display.

## Caveat 1: Is `group_id` A Safe UI Group Identifier In Every Matched Bucket?

Previously, the frontend avoided using `group_id` for bulk selection outside `partial` because exact rows could share backend group IDs in ways that were unsafe for UI selection.

### Example Risk

If backend outputs:

```text
Exact bucket:
CBL A / Insurer A / group_id = G1
CBL B / Insurer B / group_id = G1
```

but these are actually two unrelated exact matches, then clicking CBL A would incorrectly select CBL B and Insurer B.

### Backend Question

Can the frontend now treat `group_id` as a unique logical UI group identifier in:

- exact bucket;
- partial bucket;
- dynamic buckets;
- history pre-placed move groups;
- history pre-placed regroup groups?

Expected contract:

```text
Rows should share group_id only when the UI should select/highlight them together.
Unrelated matched groups must never share group_id in the same bucket.
```

## Caveat 2: Do Insurer Rows Always Have The Same Group ID As Their Related CBL Rows?

The simplification only works if insurer rows carry the same logical `group_id` as their CBL group.

### Example

Expected backend output:

```text
Row position    CBL side                  Insurer side
----------------------------------------------------------------
0               CBL A group_id=G100       Insurer A group_id=G100
1               CBL B group_id=G100       Insurer B group_id=G100
```

### Backend Question

Can the frontend rely on insurer rows having `group_id` populated for all matched buckets?

Expected contract:

```text
Every non-blank insurer row in a matched group has the same group_id as the related CBL rows.
No-match insurer rows have empty/null group_id.
```

## Caveat 3: Do We Still Need Per-CBL Insurer Ownership?

`group_id` tells us group membership. It does not tell us per-CBL ownership inside a group.

### Example

One group may contain:

```text
CBL A
CBL B
Insurer X
Insurer Y
Insurer Z
```

With only `group_id`, the frontend knows:

```text
All five rows belong together.
```

But it does not know:

```text
CBL A owns Insurer X and Insurer Y.
CBL B owns Insurer Z.
```

### Backend Question

Does the frontend need to preserve per-CBL insurer ownership for any workflow?

Known possible workflows:

- auto-selecting only a specific CBL row's insurer span;
- manually deselecting part of a group;
- moving only part of a group;
- amount-difference display by CBL row;
- regrouping only selected rows from a larger group.

Expected answer if group-level behavior is enough:

```text
Frontend can treat group_id as the unit of selection/highlighting.
Clicking any row in the group selects/highlights the full group.
```

If per-CBL ownership is still needed, then `matched_insurer_indices` or another explicit relationship field must remain available.

## Caveat 4: Does Backend Preserve Parallel Layout And Blank Spacer Rows?

Even if selection uses `group_id`, the frontend still displays and saves matched sheets as parallel CBL/insurer arrays.

Blank spacer rows may still be needed when one side has more rows than the other.

### Example

One CBL row matched to two insurer rows:

```text
Row position    CBL side             Insurer side
---------------------------------------------------------
0               CBL A                Insurer A
1               blank                Insurer B
```

This layout is important for display and Excel output.

### Backend Question

Will backend output continue to preserve a parallel row layout with blank spacer rows where needed?

Expected contract:

```text
Matched bucket sheets remain positionally aligned.
If a group has more rows on one side, blank spacer rows are emitted on the shorter side.
```

## Caveat 5: What Happens To `matched_insurer_indices`?

The proposal says `matched_insurer_indices` will continue to exist on CBL rows.

Frontend can stop using it for group-level selection, but it may still need it as a fallback for:

- older output files;
- debugging;
- per-CBL ownership;
- migration period while backend rollout is incomplete.

### Backend Question

Will `matched_insurer_indices` continue to be populated consistently after:

- normal matching;
- history move replay;
- history regroup replay;
- dynamic bucket placement?

Expected contract:

```text
matched_insurer_indices remains available and consistent, but group_id is the preferred UI selection/highlighting field.
```

## Caveat 6: How Should No-Match Rows Be Marked?

The proposal says no-match rows should have no `group_id`.

### Backend Question

Can the frontend rely on no-match rows always having empty/null `group_id` on both sides?

Expected contract:

```text
No Matches CBL and No Matches Insurer rows have no group_id.
The frontend should not auto-select unrelated no-match rows together.
```

## Recommended Frontend Migration

If backend confirms the above contracts, the frontend should migrate in phases.

### Phase 1: Prefer `group_id`, Keep Fallbacks

Use `group_id` for group-level selection/highlighting when both sides have reliable group IDs.

Keep `matched_insurer_indices` fallback for:

- old output files;
- rows missing insurer-side group ID;
- per-CBL ownership behavior;
- debugging mismatches.

### Phase 2: Simplify Regroup Target Highlighting

Replace mixed `matched_insurer_indices` + row-position logic with:

```text
Find target group_id
↓
Highlight all CBL and insurer rows in the target bucket with same group_id
```

Only fall back to positional logic when `group_id` is missing.

### Phase 3: Revisit Regroup Repair Logic

Do not immediately delete blank row equalization or layout repair.

First confirm backend output and frontend save/export still require parallel arrays.

Then decide whether frontend regroup can become:

```text
Set moved rows to target group_id
Rebuild local display block
Avoid using matched_insurer_indices for selection
```

instead of:

```text
Recompute matched_insurer_indices as the main relationship source.
```

## Summary Questions For Backend

Please confirm:

1. Is `group_id` now a safe UI group identifier in exact, partial, dynamic, move-history, and regroup-history output?
2. Do all non-blank insurer rows in matched buckets receive the same `group_id` as their related CBL rows?
3. Should frontend selection/highlighting be group-level only, or do we still need per-CBL insurer ownership?
4. Will backend preserve parallel row layout and blank spacer rows?
5. Will `matched_insurer_indices` remain populated consistently during the migration?
6. Are no-match rows guaranteed to have empty/null `group_id`?

## Desired Contract

The desired frontend contract is:

```text
For selection and highlighting:
  group_id is the source of truth.

For layout and backwards compatibility:
  matched_insurer_indices and blank spacer rows may still exist.

For no-match:
  group_id is empty/null, so no automatic group selection happens.
```

If backend can guarantee this, then the frontend refactor is a good idea and should significantly reduce the risk around auto-selection and regroup highlighting.
