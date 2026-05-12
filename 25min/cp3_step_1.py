import pandas as pd
import re

"""
Updated from mutation types

What this script does:
1. Parses the `mutation` column into structured 0/1 flags.
2. Adds `mutation_count`, `is_compound_mutation`, and `is_egfr_hotspot`.
3. Converts True/False-style columns to numeric 0/1.
4. Normalizes `binding_type_IC50` and `binding_type_KD` if present.
5. Removes rows where IC50 and KD are both 1.
6. Removes exact duplicates and, when possible, duplicate patient-ligand-mutation-assay rows.
"""

INPUT_FILE = "egfr_mutation_updated.csv"
OUTPUT_FILE = "egfr_mutation_updated.csv"

MUTATION_FLAG_COLS = [
    "has_exon19del",
    "has_L858R",
    "has_L861Q",
    "has_G719X",
    "has_exon20_alteration",
    "has_other_mutation",
    "is_compound_mutation",
    "is_egfr_hotspot",
]


def split_mutations(val):
    if pd.isna(val):
        return []
    s = str(val).strip()
    if s == "" or s.lower() in {"unknown", "nan", "none"}:
        return []
    return [p for p in re.split(r"[;,/]+|\s+", s) if p]


def token_has_exon19del(tok):
    t = tok.upper()
    return (
        (
            "DEL" in t
            and any(x in t for x in ["E746", "A750", "L747", "T751", "S752", "I759", "EXON19", "19DEL"])
        )
        or (
            "DELINS" in t
            and any(x in t for x in ["T751", "I759", "E746", "A750", "L747", "S752"])
        )
    )


def token_has_l858r(tok):
    return "L858R" in tok.upper()


def token_has_l861q(tok):
    return "L861Q" in tok.upper()


def token_has_g719x(tok):
    t = tok.upper()
    return re.search(r"G719[A-Z*]", t) is not None or "G719" in t


def token_has_exon20(tok):
    t = tok.upper()
    exon20_markers = [
        "A763", "Y764", "V765", "M766", "A767", "S768", "V769", "D770",
        "N771", "P772", "H773", "V774", "C775", "R776", "L777"
    ]
    return (
        "EXON20" in t
        or ("INS" in t and any(m in t for m in exon20_markers))
        or any(m in t for m in exon20_markers)
    )


def extract_features(mutation_str):
    toks = split_mutations(mutation_str)
    feats = {
        "has_exon19del": 0,
        "has_L858R": 0,
        "has_L861Q": 0,
        "has_G719X": 0,
        "has_exon20_alteration": 0,
        "has_other_mutation": 0,
        "mutation_count": len(toks),
    }
    for tok in toks:
        matched = False
        if token_has_exon19del(tok):
            feats["has_exon19del"] = 1
            matched = True
        if token_has_l858r(tok):
            feats["has_L858R"] = 1
            matched = True
        if token_has_l861q(tok):
            feats["has_L861Q"] = 1
            matched = True
        if token_has_g719x(tok):
            feats["has_G719X"] = 1
            matched = True
        if token_has_exon20(tok):
            feats["has_exon20_alteration"] = 1
            matched = True
        if not matched:
            feats["has_other_mutation"] = 1
    return pd.Series(feats)


def normalize_binary_series(series):
    mapping = {
        True: 1,
        False: 0,
        "True": 1,
        "False": 0,
        "TRUE": 1,
        "FALSE": 0,
        "true": 1,
        "false": 0,
        "T": 1,
        "F": 0,
        "Y": 1,
        "N": 0,
        "Yes": 1,
        "No": 0,
        "yes": 1,
        "no": 0,
        1: 1,
        0: 0,
        "1": 1,
        "0": 0,
    }
    mapped = series.map(mapping)
    return mapped


def main():
    df = pd.read_csv(INPUT_FILE).copy()

    # Ensure mutation column exists so feature engineering can run safely.
    if "mutation" not in df.columns:
        df["mutation"] = ""

    # Remove old feature columns if they already exist, then rebuild them cleanly.
    existing_feature_cols = [c for c in [*MUTATION_FLAG_COLS, "mutation_count"] if c in df.columns]
    if existing_feature_cols:
        df = df.drop(columns=existing_feature_cols)

    feature_df = df["mutation"].apply(extract_features)
    df = pd.concat([df, feature_df], axis=1)

    df["is_compound_mutation"] = (df["mutation_count"] > 1).astype(int)
    df["is_egfr_hotspot"] = (
        (df["has_exon19del"] == 1)
        | (df["has_L858R"] == 1)
        | (df["has_L861Q"] == 1)
        | (df["has_G719X"] == 1)
    ).astype(int)

    # Convert actual bool dtypes to 0/1.
    for col in df.columns:
        if pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype(int)

    # Convert assay columns to numeric 0/1 if present.
    assay_cols = [c for c in ["binding_type_IC50", "binding_type_KD"] if c in df.columns]
    for col in assay_cols:
        mapped = normalize_binary_series(df[col])
        df[col] = pd.to_numeric(mapped.fillna(df[col]), errors="coerce")
        df[col] = df[col].fillna(0).astype(int)

    # Convert mutation flags to 0/1 explicitly.
    for col in [c for c in MUTATION_FLAG_COLS if c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        df[col] = df[col].clip(0, 1)

    # Convert obvious one-hot / dummy columns to numeric if they are True/False-like.
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in object_cols:
        normalized = normalize_binary_series(df[col])
        if normalized.notna().any() and normalized.notna().sum() == df[col].notna().sum():
            df[col] = normalized.fillna(0).astype(int)

    # Remove invalid rows where both assay columns are 1.
    invalid_assay_rows = 0
    if set(["binding_type_IC50", "binding_type_KD"]).issubset(df.columns):
        invalid_mask = (df["binding_type_IC50"] == 1) & (df["binding_type_KD"] == 1)
        invalid_assay_rows = int(invalid_mask.sum())
        df = df.loc[~invalid_mask].copy()

    # Remove exact duplicates first.
    exact_dupes = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    # Then remove duplicate patient-ligand-mutation-assay combinations when available.
    dedup_priority = [
        "patient_id",
        "pref_name",
        "mutation",
        "binding_type_IC50",
        "binding_type_KD",
    ]
    dedup_cols = [c for c in dedup_priority if c in df.columns]
    combo_dupes_removed = 0
    if len(dedup_cols) >= 3:
        before = len(df)
        df = df.drop_duplicates(subset=dedup_cols).copy()
        combo_dupes_removed = before - len(df)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved {OUTPUT_FILE} with {len(df)} rows.")
    if invalid_assay_rows:
        print(f"Removed {invalid_assay_rows} invalid rows where both IC50 and KD were 1.")
    if exact_dupes:
        print(f"Removed {exact_dupes} exact duplicate rows.")
    if combo_dupes_removed:
        print(f"Removed {combo_dupes_removed} duplicate patient-ligand-mutation-assay rows.")

    summary_cols = [
        "has_exon19del", "has_L858R", "has_L861Q", "has_G719X",
        "has_exon20_alteration", "has_other_mutation",
        "mutation_count", "is_compound_mutation", "is_egfr_hotspot"
    ]
    summary_cols = [c for c in summary_cols if c in df.columns]
    if summary_cols:
        print(df[summary_cols].sum(numeric_only=True))

    if set(["binding_type_IC50", "binding_type_KD"]).issubset(df.columns):
        print("\nFinal assay check:")
        print("Rows with IC50=1 and KD=1:", int(((df["binding_type_IC50"] == 1) & (df["binding_type_KD"] == 1)).sum()))


if __name__ == "__main__":
    main()
