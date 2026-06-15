# Backend Guide: Reading Split Match History Entries

## Context

The frontend saves match history to `history.xlsx` (stored in `Matrix/{insurer}/` on SharePoint). Each row represents a user action (move/regroup) and stores fingerprint arrays as JSON strings in Excel cells.

We hit an **XLSX cell character limit of 32,767 characters**. The fingerprint arrays, when JSON-serialized, can exceed this limit for actions involving many rows or rows with many columns.

## Chosen Solution: Array-Level Row Splitting

When a fingerprint array's JSON string exceeds the cell limit, the frontend will split the **array elements** across multiple rows in the Excel sheet. A new `ActionType` value (`"continuation"`) marks rows that belong to the previous entry.

## How `history.xlsx` Changes

### Before (current format)

Every row in the `MatchHistory` sheet is a standalone entry:

| ActionType | CblFingerprints | InsurerFingerprints | FromBucket | TargetBucket | Timestamp | ...other columns |
|---|---|---|---|---|---|---|
| move | `["fp1","fp2","fp3","fp4"]` | `["fpA","fpB"]` | no_match | exact | 2026-05-26T... | ... |
| regroup | `["fp5"]` | `["fpC"]` | partial | exact | 2026-05-26T... | ... |

### After (new format with splitting)

If a cell would exceed ~32,000 characters, the array is split across rows. The first row keeps the original `ActionType`. Continuation rows use `"continuation"`:

| ActionType | CblFingerprints | InsurerFingerprints | FromBucket | TargetBucket | Timestamp | ...other columns |
|---|---|---|---|---|---|---|
| move | `["fp1","fp2"]` | `["fpA","fpB"]` | no_match | exact | 2026-05-26T... | ... |
| continuation | `["fp3","fp4"]` | `[]` | no_match | exact | 2026-05-26T... | ... |
| regroup | `["fp5"]` | `["fpC"]` | partial | exact | 2026-05-26T... | ... |

Key points:
- Only the overflowing column(s) spill into continuation rows. Columns that fit stay in the first row.
- Non-fingerprint fields (`FromBucket`, `TargetBucket`, `Timestamp`) are repeated on continuation rows for consistency, but the first row's values are authoritative.
- A single entry can have multiple continuation rows if the data is very large.
- Entries that fit in one row remain unchanged (no `"continuation"` rows). Existing history files are fully backward compatible.

## How to Read and Reassemble on the Backend

When parsing `history.xlsx`, the backend should:

1. Read all rows from the `MatchHistory` sheet in order.
2. Iterate through rows sequentially.
3. When `ActionType` is **not** `"continuation"` — this is a **new entry**. Start building a new `MatchHistoryEntry`.
4. When `ActionType` is `"continuation"` — this is a **continuation of the previous entry**. Concatenate each fingerprint array field onto the previous entry's corresponding array.

### Pseudocode

```python
def read_match_history(rows):
    entries = []
    current_entry = None

    for row in rows:
        if row["ActionType"] != "continuation":
            # New entry — save the previous one (if any) and start fresh
            if current_entry is not None:
                entries.append(current_entry)

            current_entry = {
                "action_type": row["ActionType"],
                "cbl_fingerprints": json.loads(row["CblFingerprints"] or "[]"),
                "insurer_fingerprints": json.loads(row["InsurerFingerprints"] or "[]"),
                "target_cbl_fingerprints": json.loads(row["TargetCblFingerprints"] or "[]"),
                "target_insurer_fingerprints": json.loads(row["TargetInsurerFingerprints"] or "[]"),
                "orphaned_cbl_fingerprints": json.loads(row["OrphanedCblFingerprints"] or "[]"),
                "orphaned_insurer_fingerprints": json.loads(row["OrphanedInsurerFingerprints"] or "[]"),
                "cbl_remarks": json.loads(row["CblRemarks"] or "[]"),
                "insurer_remarks": json.loads(row["InsurerRemarks"] or "[]"),
                "from_bucket": row["FromBucket"],
                "target_bucket": row["TargetBucket"],
                "timestamp": row["Timestamp"],
            }
        else:
            # Continuation row — merge arrays into current entry
            if current_entry is None:
                continue  # orphaned continuation row, skip

            current_entry["cbl_fingerprints"] += json.loads(row["CblFingerprints"] or "[]")
            current_entry["insurer_fingerprints"] += json.loads(row["InsurerFingerprints"] or "[]")
            current_entry["target_cbl_fingerprints"] += json.loads(row["TargetCblFingerprints"] or "[]")
            current_entry["target_insurer_fingerprints"] += json.loads(row["TargetInsurerFingerprints"] or "[]")
            current_entry["orphaned_cbl_fingerprints"] += json.loads(row["OrphanedCblFingerprints"] or "[]")
            current_entry["orphaned_insurer_fingerprints"] += json.loads(row["OrphanedInsurerFingerprints"] or "[]")
            current_entry["cbl_remarks"] += json.loads(row["CblRemarks"] or "[]")
            current_entry["insurer_remarks"] += json.loads(row["InsurerRemarks"] or "[]")

    # Don't forget the last entry
    if current_entry is not None:
        entries.append(current_entry)

    return entries
```

## Fingerprint Array Fields That Can Be Split

All of these columns store JSON arrays of strings and are subject to splitting:

| Column | Description |
|---|---|
| `CblFingerprints` | Fingerprints of CBL rows involved in the action |
| `InsurerFingerprints` | Fingerprints of insurer rows involved in the action |
| `TargetCblFingerprints` | CBL fingerprints in the target bucket (regroup actions) |
| `TargetInsurerFingerprints` | Insurer fingerprints in the target bucket (regroup actions) |
| `OrphanedCblFingerprints` | CBL rows orphaned by the action |
| `OrphanedInsurerFingerprints` | Insurer rows orphaned by the action |
| `CblRemarks` | User remarks on CBL rows |
| `InsurerRemarks` | User remarks on insurer rows |

## What Does NOT Change

- The fingerprint values themselves remain raw `|`-separated strings (e.g., `"ACME|POL-123|5000.00"`). No hashing or truncation.
- Non-fingerprint columns (`FromBucket`, `TargetBucket`, `Timestamp`, `ActionType`) keep the same format.
- Rows that don't exceed the limit are written exactly as before — no splitting, no `"continuation"`.
- The file location remains `Matrix/{INSURER_NAME}/history.xlsx`.

## Backward Compatibility

- Old `history.xlsx` files (without continuation rows) will continue to work with no changes — the read logic simply never encounters `"continuation"` rows.
- If the backend currently reads `history.xlsx` and does not filter by `ActionType`, continuation rows would appear as separate entries with `ActionType = "continuation"`. The backend should be updated to merge these using the logic above.
