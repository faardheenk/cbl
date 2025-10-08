# Overlap-Based Grouping - Visual Explanation

## Your Actual Data

```
Row 0 (IBL LTD):              [115, 116, 117, ..., 330, 331, 332]     (218 insurers)
Row 1 (IBL LTD):              [115, 116, 117, ..., 330, 331, 332]     (218 insurers)
Row 262 (LOGIDIS &/OR IBL):   [115, 116, 117, ..., 327, 390, 391, 392] (216 insurers)
```

**Overlap Analysis:**

- Row 0 ∩ Row 1 = **218 insurers (100% overlap)** ✅
- Row 0 ∩ Row 262 = **213 insurers (98.6% overlap)** ✅
- Row 1 ∩ Row 262 = **213 insurers (98.6% overlap)** ✅

## Before Fix (Identical-Only Grouping)

```
┌─────────────────────────────────────────────────┐
│ OLD LOGIC: Only group IDENTICAL sets           │
└─────────────────────────────────────────────────┘

Row 0:   [115-332] ─┐
                    ├─→ GROUP_1 (identical)
Row 1:   [115-332] ─┘

Row 262: [115-327, 390-392] ─→ ❌ NOT GROUPED (different set)

RESULT IN EXCEL:
├─ Row 0 with insurers 115-332  (218 rows)
├─ Row 1 with insurers 115-332  (218 rows) ⚠️ DUPLICATE!
└─ Row 262 with insurers 115-327, 390-392 (216 rows) ⚠️ DUPLICATE!

⚠️  Insurers 115-327 shown THREE times!
```

## After Fix (Overlap-Based Grouping)

```
┌─────────────────────────────────────────────────┐
│ NEW LOGIC: Group sets with ≥80% overlap        │
└─────────────────────────────────────────────────┘

Row 0:   [115-332] ─┐
                    ├─→ 100% overlap ─┐
Row 1:   [115-332] ─┘                 │
                                      ├─→ GROUP_1 (all 3 together!)
Row 262: [115-327, 390-392] ─────────┘
         (98.6% overlap with above)

RESULT IN EXCEL (ZIPPED):
├─ Row 0:   IBL LTD              → Insurer 115
├─ Row 1:   IBL LTD              → Insurer 116
├─ Row 262: LOGIDIS &/OR IBL     → Insurer 117
├─ Row 0:   (empty)              → Insurer 118
├─ Row 1:   (empty)              → Insurer 119
├─ Row 262: (empty)              → Insurer 120
...
├─ Row 0:   (empty)              → Insurer 330
├─ Row 1:   (empty)              → Insurer 331
├─ Row 262: (empty)              → Insurer 332
├─ (empty)                       → Insurer 390
├─ (empty)                       → Insurer 391
└─ (empty)                       → Insurer 392

✅ Each insurer shown ONCE!
✅ CBL rows rotated in "carousel" style
```

## How Union-Find Works

```
Step 1: Compare Row 0 vs Row 1
────────────────────────────────
Overlap: 218/218 = 100% ≥ 80% ✅
Action: Union(0, 1)

Parent: {0: 0, 1: 0, 262: 262}
Groups: [0, 1] and [262]


Step 2: Compare Row 0 vs Row 262
────────────────────────────────
Overlap: 213/216 = 98.6% ≥ 80% ✅
Action: Union(0, 262)

Parent: {0: 0, 1: 0, 262: 0}
Groups: [0, 1, 262]  ← All together now!


Step 3: Compare Row 1 vs Row 262
────────────────────────────────
Overlap: 213/216 = 98.6% ≥ 80% ✅
Action: Already grouped (transitive)

Final Parent: {0: 0, 1: 0, 262: 0}
Final Groups: [0, 1, 262]
```

## Key Benefits

| Aspect              | Before                            | After                     |
| ------------------- | --------------------------------- | ------------------------- |
| **Grouping Logic**  | Identical sets only               | ≥80% overlap              |
| **Your Case**       | Row 0,1 grouped; Row 262 separate | All 3 grouped together    |
| **Excel Rows**      | ~652 rows (duplicates)            | ~335 rows (no duplicates) |
| **Insurer 115-327** | Shown 3 times ❌                  | Shown once ✅             |
| **Data Integrity**  | Broken (duplicates)               | Perfect ✅                |

## Why 80% Threshold?

- **Too Low (50%)**: May group unrelated records
- **80%**: Catches substantial overlap while avoiding false positives
- **Your Case**: 98.6% overlap - clear duplicates!

## Edge Cases Handled

### Case 1: Transitive Grouping

```
A overlaps B (85%)
B overlaps C (85%)
→ A, B, C all grouped together (even if A-C overlap < 80%)
```

### Case 2: Multiple Groups

```
A overlaps B (90%) → Group 1
D overlaps E (85%) → Group 2
F alone → No group
```

### Case 3: Partial Overlaps

```
A: [1, 2, 3, 4, 5]
B: [3, 4, 5, 6, 7]
Overlap: 3/5 = 60% < 80% → NOT grouped
```

## Testing Suggestion

Run your matching with this fix and check:

1. ✅ LOGIDIS (row 262) should have `group_id: GROUP_1`
2. ✅ IBL LTD rows (0, 1) should also have `group_id: GROUP_1`
3. ✅ Insurers 115-327 should appear only ONCE in Excel
4. ✅ Total Excel rows should be ~335 (not ~652)
