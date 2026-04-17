import pandas as pd

"""
Missing columns: []

Non-binary values in flags:

Mutation count distribution:
mutation_count
1    56
2    12
3     2
Name: count, dtype: int64

Compound mutation mismatch:
0

Remaining bool columns:
Index([], dtype='object')

Duplicate rows:
0
"""


df = pd.read_csv("egfr_mutation_updated.csv")

# 1. Required mutation columns
required_cols = [
    "has_exon19del","has_L858R","has_L861Q","has_G719X",
    "has_exon20_alteration","has_other_mutation",
    "mutation_count","is_compound_mutation","is_egfr_hotspot"
]

print("Missing columns:", [c for c in required_cols if c not in df.columns])


# 2. Check all flags are 0/1
flag_cols = [c for c in df.columns if c.startswith("has_")] + ["is_compound_mutation","is_egfr_hotspot"]

print("\nNon-binary values in flags:")
for c in flag_cols:
    bad = df[~df[c].isin([0,1])]
    if len(bad) > 0:
        print(c, "❌")


# 3. mutation_count sanity
print("\nMutation count distribution:")
print(df["mutation_count"].value_counts())


# 4. compound mutation correctness
print("\nCompound mutation mismatch:")
print(((df["mutation_count"] > 1).astype(int) != df["is_compound_mutation"]).sum())


# 5. IC50/KD check (if present)
if "binding_type_IC50" in df.columns:
    print("\nIC50/KD overlap:")
    print(((df["binding_type_IC50"] == 1) & (df["binding_type_KD"] == 1)).sum())


# 6. Boolean leftovers
print("\nRemaining bool columns:")
print(df.select_dtypes(include="bool").columns)


# 7. Duplicates
print("\nDuplicate rows:")
print(df.duplicated().sum())