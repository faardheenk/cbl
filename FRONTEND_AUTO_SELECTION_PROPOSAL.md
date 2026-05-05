# Frontend Auto-Selection / Highlighting — Proposed Backend Simplification

## Purpose

The backend is proposing a change that would significantly simplify frontend auto-selection, highlighting, and group relationship logic.

Before implementing, we need the frontend agent to describe its current approach so we can confirm this change removes the right complexity.

## What We Need From Frontend

Please describe the current frontend implementation for the following:

### 1. Auto-Selection

When a user clicks a CBL row, how does the frontend determine which insurer rows to auto-select (and vice versa)?

Specific questions:

- Does the frontend read `matched_insurer_indices` from the row and select those insurer rows directly?
- Does the frontend do positional span repair (e.g. CBL row A "owns" insurer rows 0–1, CBL row B "owns" insurer row 2)?
- Does the frontend use `group_id`, `match_group`, `match_condition`, or some combination?
- Does the approach differ between exact match, partial match, and dynamic bucket groups?

### 2. Highlighting

When a group is displayed in the table, how does the frontend determine which rows share the same visual group highlight?

Specific questions:

- Is highlighting driven by `group_id`, by `matched_insurer_indices`, by row adjacency, or by something else?
- Are insurer rows highlighted based on their own metadata, or inferred from the CBL side?
- Does the frontend assign its own in-memory group identifiers after loading `output.xlsx`?

### 3. Regroup — Group Relationship Repair

After a user regroups rows, what does the frontend do to update group relationships?

Specific questions:

- Does the frontend recompute `matched_insurer_indices` for the merged group?
- Does the frontend reassign `group_id` / `match_group` / `match_condition` on moved rows?
- Does the frontend do blank row equalization (insert spacer rows so CBL and insurer counts match)?
- Does the frontend detect and move orphaned rows to no-match?
- How many of these steps are driven by positional layout vs. explicit metadata?

### 4. Pain Points

What parts of the current auto-selection / highlighting / regroup logic are:

- Most fragile or error-prone?
- Hardest to maintain when new features are added?
- Most dependent on backend output format staying exactly right?

## Proposed Backend Change

### The Problem

Today the backend only assigns `group_id` to CBL rows. Insurer rows have no explicit group identity in the backend output.

This means the frontend must reverse-engineer group membership from `matched_insurer_indices` — a flat list of insurer DataFrame indices that was designed for output layout, not for frontend group logic.

That forces the frontend into complex recomputation:

```
Parse matched_insurer_indices
→ figure out which insurer rows belong to which CBL row
→ positional span repair
→ blank row equalization
→ orphan detection
→ re-derive group identity after every move/regroup
```

This is fragile because `matched_insurer_indices` is an indirect relationship. It tells you what rows were zipped together in the output, not what rows logically belong to the same group.

### The Solution

The backend will assign `group_id` to **both CBL and insurer rows**.

Every row in the backend output (`output.xlsx`) will carry a `group_id` value:

- All CBL rows in a group share the same `group_id`.
- All insurer rows in that same group share the same `group_id`.
- Unmatched rows (no-match) have no `group_id` (null/empty).

This applies to:

- Exact match groups
- Partial match groups
- Dynamic bucket groups
- History pre-placed groups (move and regroup)

### What Changes For The Frontend

| Task | Before (matched_insurer_indices) | After (group_id on both sides) |
|---|---|---|
| Auto-select on click | Parse `matched_insurer_indices`, compute positional spans | Read `group_id` → select all rows with same `group_id` |
| Highlight group | Derive from `matched_insurer_indices` or row adjacency | Same `group_id` = same highlight |
| Click insurer row → find CBL | Reverse-lookup which CBL rows reference this insurer index | Read insurer `group_id` → find CBL rows with same `group_id` |
| Regroup | Recompute `matched_insurer_indices`, reassign metadata, equalize rows, detect orphans | Set target `group_id` on moved rows |
| Orphan detection | Walk both sides, compare membership lists | Any `group_id` with rows on only one side = orphan |
| Move to no-match | Clear `matched_insurer_indices`, recompute remaining group | Clear `group_id` |

### What Does NOT Change

- `matched_insurer_indices` continues to exist on CBL rows. The backend still uses it internally for output layout (`explode_and_merge`) and amount difference computation.
- The output sheet structure stays the same (Exact Matches, Partial Matches, No Matches CBL, No Matches Insurer, dynamic bucket sheets).
- Fingerprints, history format, and all existing fields remain unchanged.
- The frontend can still read `matched_insurer_indices` if needed for any purpose, but it no longer needs to for selection or highlighting.

### Example Output

Before (only CBL has group_id):

```
Exact Matches sheet:
Row  | ClientName | group_id      | matched_insurer_indices | ... | ClientName_INSURER | ...
0    | Client A   | G_001         | [0, 1]                 | ... | Insurer X          | ...
1    |            |               |                        | ... | Insurer Y          | ...
2    | Client B   | G_001         | [0, 1]                 | ... |                    | ...
```

After (both sides have group_id):

```
Exact Matches sheet:
Row  | ClientName | group_id      | matched_insurer_indices | ... | ClientName_INSURER | group_id_INSURER | ...
0    | Client A   | G_001         | [0, 1]                 | ... | Insurer X          | G_001            | ...
1    |            |               |                        | ... | Insurer Y          | G_001            | ...
2    | Client B   | G_001         | [0, 1]                 | ... |                    |                  | ...
```

Frontend auto-selection with new model:

```
User clicks "Client A" (group_id = G_001)
→ select all rows where group_id == G_001 OR group_id_INSURER == G_001
→ done
```

## Questions For Frontend

1. Does this proposed change actually remove the complexity you are currently dealing with?
2. Are there any frontend workflows that depend on per-CBL-row insurer ownership (i.e. knowing that CBL A specifically owns Insurer X but not Insurer Y within the same group)?
3. Is there any reason the frontend would still need positional span repair if `group_id` is available on both sides?
4. Are there other frontend pain points around group logic that this proposal does not address?

## Next Steps

Once the frontend confirms:

- Which parts of the current logic this would replace
- Whether per-row ownership within a group is needed (or group-level selection is sufficient)
- Any edge cases we should be aware of

The backend will implement `group_id` assignment on insurer rows across all matching passes and history replay.
