# Pass 3: Name-based Clustering and Grouping - Step-by-Step Analysis

## 🎯 **Overview**

Pass 3 is the final matching pass that uses **fuzzy name matching** to group similar company names and create matches based on name similarity and amount compatibility.

---

## 📋 **Step 1: Initial Setup and Filtering**

### **1.1 Filter Available Records**

```python
# Only process CBL records that are still unmatched
unmatched_cbl = cbl_df[cbl_df['match_status'].isin(['No Match', 'Partial Match'])]

# Only use insurer records that haven't been used in Pass 1 or Pass 2
already_matched_insurer = global_tracker.exact_used_insurer | global_tracker.matrix_used_insurer
available_insurer = insurer_df[~insurer_df.index.isin(already_matched_insurer)]
```

**Purpose:** Ensure we only work with records that haven't been matched in previous passes.

---

## 🏗️ **Step 2: Build Name Clusters (Fuzzy Matching)**

### **2.1 CBL Name Clustering**

```python
cbl_name_clusters = _build_fuzzy_name_clusters(
    unmatched_cbl,
    name_column='ClientName',
    fuzzy_threshold=90,  # 90% similarity required
    prefix="CBL"
)
```

**What happens:**

- Takes all unmatched CBL records
- Groups them by **similar company names** using fuzzy matching
- **Example:** "ABC Ltd", "ABC Limited", "ABC (Mauritius) Ltd" → **1 cluster**

### **2.2 Insurer Name Clustering**

```python
insurer_name_clusters = _build_fuzzy_name_clusters(
    available_insurer,
    name_column='ClientName_INSURER',
    fuzzy_threshold=90,
    prefix="INSURER"
)
```

**What happens:**

- Takes all available insurer records
- Groups them by **similar company names** using fuzzy matching
- **Example:** "XYZ Corp", "XYZ Corporation", "XYZ Ltd" → **1 cluster**

### **2.3 Cluster Structure**

```python
# Result format:
cbl_name_clusters = {
    'ABC Ltd': [cbl_index_1, cbl_index_2, cbl_index_3],  # 3 CBL records with similar names
    'XYZ Corp': [cbl_index_4, cbl_index_5],               # 2 CBL records with similar names
    'DEF Ltd': [cbl_index_6]                               # 1 CBL record (no similar names)
}

insurer_name_clusters = {
    'ABC Limited': [insurer_index_1, insurer_index_2],    # 2 insurer records with similar names
    'XYZ Corporation': [insurer_index_3],                 # 1 insurer record
    'DEF Ltd': [insurer_index_4, insurer_index_5]         # 2 insurer records
}
```

---

## 🔗 **Step 3: Match Clusters Together**

### **3.1 Cluster-to-Cluster Matching**

```python
for cbl_cluster_name, cbl_indices in cbl_name_clusters.items():
    for insurer_cluster_name, insurer_indices in insurer_name_clusters.items():
        # Check if cluster names are similar
        cluster_similarity = fuzz.token_set_ratio(cbl_cluster_name, insurer_cluster_name)

        if cluster_similarity >= fuzzy_threshold:  # 90%
            # Create a NAME_GROUP
```

**What happens:**

- Compares **every CBL cluster** with **every insurer cluster**
- If cluster names are similar (≥90%), create a **NAME_GROUP**
- **Example:** "ABC Ltd" cluster matches "ABC Limited" cluster → **NAME_GROUP_1**

### **3.2 Name Group Creation**

```python
group_counter += 1
group_id = f"NAME_GROUP_{group_counter}"  # NAME_GROUP_1, NAME_GROUP_2, etc.

# Calculate totals for the entire group
cbl_total = cbl_df.loc[cbl_indices, 'ProcessedAmount_Clean'].sum()
insurer_total = available_insurer.loc[insurer_indices, 'ProcessedAmount_Clean_INSURER'].sum()
difference = abs(cbl_total + insurer_total)
```

**Example:**

- **NAME_GROUP_1:**
  - CBL Cluster: "ABC Ltd" (3 records, total: -Rs50,000)
  - Insurer Cluster: "ABC Limited" (2 records, total: +Rs49,900)
  - Difference: Rs100 → **EXACT MATCH**

---

## 🎯 **Step 4: Apply Matches to Individual Records**

### **4.1 Cluster-Level Decision**

```python
# All CBL records in the cluster get the SAME match status
for cbl_idx in cbl_indices:
    if is_exact_match:
        _apply_cluster_exact_match(...)  # All get "Exact Match"
    else:
        _apply_partial_match(...)        # All get "Partial Match"
```

**Key Point:** **All CBL records in a cluster get the same match status** based on the cluster totals.

### **4.2 Individual Record Processing**

```python
# Each CBL record gets matched to ALL insurer records in the cluster
for cbl_idx in cbl_indices:
    # This CBL record gets matched to ALL insurer records
    usable_indices = insurer_indices  # All insurer records in the cluster

    # Calculate amount difference for this specific CBL record
    total_insurer_amount = available_insurer.loc[usable_indices, "ProcessedAmount_Clean_INSURER"].sum()
    cbl_amount = cbl_df.at[cbl_idx, "ProcessedAmount_Clean"]
    amount_diff = abs(cbl_amount + total_insurer_amount)
```

**Example:**

- **NAME_GROUP_1** has 3 CBL records and 2 insurer records
- **All 3 CBL records** get matched to **both insurer records**
- Each CBL record gets the same match status (Exact/Partial)

---

## 🔄 **Step 5: Group Merging (Overlapping Insurer Indices)**

### **5.1 The Merging Problem**

After Step 4, you might have:

- **NAME_GROUP_1:** CBL records [A, B] matched to insurer records [X, Y]
- **NAME_GROUP_2:** CBL records [C, D] matched to insurer records [Y, Z]

**Problem:** Both groups use insurer record Y → **Conflict!**

### **5.2 Merging Logic**

```python
def _merge_groups_with_overlapping_insurer_indices(cbl_df, available_insurer, global_tracker):
    # Find groups that share the same insurer indices
    for indices_tuple, group_records in insurer_indices_to_groups.items():
        if len(group_records) > 1:
            # Multiple groups share the same insurer indices - merge them
            groups_to_merge.append({
                'insurer_indices': list(indices_tuple),
                'original_group_ids': group_ids,
                'cbl_indices': cbl_indices
            })
```

**What happens:**

- Groups with **overlapping insurer indices** get merged
- **NAME_GROUP_1** and **NAME_GROUP_2** → **MERGED_GROUP_1**
- All CBL records from both groups get the same insurer records

### **5.3 Merged Group Processing**

```python
# Calculate combined totals for the merged group
cbl_total = cbl_df.loc[cbl_indices, 'ProcessedAmount_Clean'].sum()
insurer_total = available_insurer.loc[insurer_indices, 'ProcessedAmount_Clean_INSURER'].sum()
difference = abs(cbl_total + insurer_total)

# All records in the merged group get the SAME match status
if match_type == "EXACT":
    cbl_df.at[cbl_idx, 'match_status'] = "Exact Match"
else:
    cbl_df.at[cbl_idx, 'match_status'] = "Partial Match"
```

---

## 📊 **Final Result Structure**

### **Name Groups Created:**

- **NAME_GROUP_1:** CBL cluster "ABC Ltd" ↔ Insurer cluster "ABC Limited"
- **NAME_GROUP_2:** CBL cluster "XYZ Corp" ↔ Insurer cluster "XYZ Corporation"
- **NAME_GROUP_3:** CBL cluster "DEF Ltd" ↔ Insurer cluster "DEF Ltd"

### **Merged Groups Created:**

- **MERGED_GROUP_1:** NAME_GROUP_1 + NAME_GROUP_2 (overlapping insurer indices)
- **MERGED_GROUP_2:** NAME_GROUP_3 (no overlaps, stays as NAME_GROUP_3)

### **Final Match Status:**

- All CBL records in each group get the **same match status**
- All CBL records in each group get matched to the **same insurer records**
- Match status is determined by **cluster-level amount totals**

---

## 🎯 **Key Concepts Summary**

1. **Name Clusters:** Groups of records with similar company names (fuzzy matching)
2. **Name Groups:** Matches between CBL clusters and insurer clusters
3. **Merged Groups:** Groups that share insurer records get merged together
4. **Cluster-Level Decisions:** All records in a cluster get the same match status
5. **Shared Insurer Records:** Multiple CBL records can share the same insurer records

This approach allows for **flexible matching** where similar company names can be grouped together and matched based on overall cluster compatibility rather than individual record matching.
