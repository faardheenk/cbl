# Backend group_id Changes — Implementation Complete

## Summary

The backend now assigns `group_id` to every matched row in `output.xlsx`, on both CBL and insurer sides. The frontend can use `group_id` as the single source of truth for auto-selection and highlighting.

## What Changed

### 1. Every matched row now has a `group_id` (`orchestrator.py`)

Previously, individual 1:1 matches from pass1 had no `group_id`. Only multi-row groups from pass3 and history replay had one.

Now, after all passes complete, the backend assigns a unique `group_id` (`MATCH_1`, `MATCH_2`, ...) to every matched CBL row that doesn't already have one. This covers:

- Individual exact matches (1 CBL → 1 or more insurer rows)
- Individual partial matches
- Individual dynamic bucket matches

Rows that already have a `group_id` from pass3 or history replay keep their existing value. No existing group IDs are changed.

### 2. `group_id` propagates to all rows in the combined output (`output_handler.py`)

Previously, when a group had more insurer rows than CBL rows (or vice versa), the spacer rows on the shorter side had `None` for `group_id`. This meant insurer-only rows in the output lost group identity.

Now, every row in a zipped output block carries the same `group_id`:

- **Group matches** (`_process_group_match`): The `group_id` is extracted from the group's CBL rows and applied to every row in the block, including insurer-only spacer rows.
- **Individual matches** (`_process_individual_match`): When a single CBL row matches multiple insurer rows, subsequent insurer-only rows (where CBL data is blank) now carry the same `group_id` as the first row.

## Output Format

Before:

```
Row | ClientName | group_id | ... | ClientName_INSURER | ...
0   | Client A   | G_001    | ... | Insurer X          | ...
1   |            |          | ... | Insurer Y          | ...
2   | Client B   | G_001    | ... |                    | ...
3   | Client C   |          | ... | Insurer Z          | ...
```

After:

```
Row | ClientName | group_id | ... | ClientName_INSURER | ...
0   | Client A   | G_001    | ... | Insurer X          | ...
1   |            | G_001    | ... | Insurer Y          | ...
2   | Client B   | G_001    | ... |                    | ...
3   | Client C   | MATCH_1  | ... | Insurer Z          | ...
```

Row 1 now carries `G_001` (was blank). Row 3 now carries `MATCH_1` (was blank).

## Frontend Auto-Selection — New Approach

The frontend can now replace all `matched_insurer_indices`-based selection logic with:

```
User clicks any row
→ read row.group_id
→ if group_id is null/empty: no auto-selection (this is a no-match row)
→ if group_id has a value: select all rows in the current bucket where group_id matches
→ done
```

This works for:

- CBL → insurer selection (click CBL row, find all rows with same `group_id`)
- Insurer → CBL selection (click insurer-only row, find all rows with same `group_id`)
- Full group highlighting (all rows with same `group_id` get the same highlight)

## Frontend Regroup — Simplified

With `group_id` on every row, regroup target detection becomes:

```
User pins a regroup target row
→ read target row.group_id
→ find all rows in the target bucket with same group_id
→ that is the target group
→ done
```

After regroup, updating group identity becomes:

```
Set moved rows' group_id to the target group's group_id
→ done
```

The frontend no longer needs to:

- Parse `matched_insurer_indices` to find insurer rows
- Do positional span repair
- Walk backward from insurer rows to find anchor CBL rows
- Recompute `matched_insurer_indices` after merging

## What Did NOT Change

- **`matched_insurer_indices`** — still populated on all CBL rows, unchanged. The backend still uses it internally for output layout and statistics. The frontend can stop relying on it for selection.
- **Output sheet structure** — same sheets (Exact Matches, Partial Matches, No Matches CBL, No Matches Insurer, dynamic bucket sheets).
- **Blank spacer rows** — still emitted when one side has more rows. The only difference is they now carry `group_id`.
- **Fingerprints** — `group_id` was already in the `FINGERPRINT_EXCLUDE_COLUMNS` set, so this change does not affect fingerprint generation.
- **History format** — no changes to `history.xlsx` columns or format.
- **Matching pass logic** — pass1, pass2, pass3 are unchanged. The new `group_id` assignment happens after all passes complete.

## `group_id` Values By Source

| Source | Format | Example |
|---|---|---|
| Pass 3 — corporate root | `NAME_GROUP_N_ROOT` | `NAME_GROUP_1_ROOT` |
| Pass 3 — fuzzy clustering | `NAME_GROUP_N_FUZZY` | `NAME_GROUP_5_FUZZY` |
| Pass 3 — secondary root | `NAME_GROUP_N_SECONDARY` | `NAME_GROUP_3_SECONDARY` |
| Pass 3 — label match | `NAME_GROUP_N_LBL` | `NAME_GROUP_2_LBL` |
| Pass 3 — merge overlapping | `MERGED_GROUP_N` | `MERGED_GROUP_1` |
| History — move | `HISTORY_N` | `HISTORY_0` |
| History — regroup | `HISTORY_REGROUP_N` | `HISTORY_REGROUP_0` |
| Individual match (new) | `MATCH_N` | `MATCH_1` |

The frontend does not need to parse or interpret these values. They are opaque identifiers — the only operation is equality comparison.

## No-Match Rows

No-match rows (both CBL and insurer) have `null`/empty `group_id`. The frontend should not auto-select unrelated no-match rows together.

## Migration

There is no backward compatibility concern. The backend reprocesses fresh Excel files on each run. After this change, every new `output.xlsx` will have `group_id` on all matched rows. There are no old output files to support.

## Files Modified

- `matching/orchestrator.py` — assigns `group_id` to individual matched rows after passes complete
- `matching/output_handler.py` — propagates `group_id` to spacer rows in `_process_group_match` and `_process_individual_match`
