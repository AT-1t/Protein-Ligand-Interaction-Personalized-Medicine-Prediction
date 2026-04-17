
import pandas as pd
import re

"""
Saved egfr_mutation_updated.csv with 70 rows.
has_exon19del            23
has_L858R                23
has_L861Q                 3
has_G719X                 3
has_exon20_alteration     5
has_other_mutation       24
mutation_count           86
is_compound_mutation     14
is_egfr_hotspot          55
dtype: int64
"""


INPUT_FILE = "mutation.csv"
OUTPUT_FILE = "egfr_mutation_updated.csv"


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
        ("DEL" in t and any(x in t for x in ["E746","A750","L747","T751","S752","I759","EXON19","19DEL"]))
        or ("DELINS" in t and any(x in t for x in ["T751","I759","E746","A750","L747","S752"]))
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
    exon20_markers = ["A763","Y764","V765","M766","A767","S768","V769","D770","N771","P772","H773","V774","C775","R776","L777"]
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

def main():
    df = pd.read_csv(INPUT_FILE).copy()
    feature_df = df["mutation"].apply(extract_features)
    df = pd.concat([df, feature_df], axis=1)
    df["is_compound_mutation"] = (df["mutation_count"] > 1).astype(int)
    df["is_egfr_hotspot"] = (
        (df["has_exon19del"] == 1)
        | (df["has_L858R"] == 1)
        | (df["has_L861Q"] == 1)
        | (df["has_G719X"] == 1)
        | (df["has_exon20_alteration"] == 1)
    ).astype(int)

    for col in df.columns:
        if pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype(int)

    df = df.drop_duplicates()
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved {OUTPUT_FILE} with {len(df)} rows.")
    print(df[[
        "has_exon19del", "has_L858R", "has_L861Q", "has_G719X",
        "has_exon20_alteration", "has_other_mutation",
        "mutation_count", "is_compound_mutation", "is_egfr_hotspot"
    ]].sum(numeric_only=True))

if __name__ == "__main__":
    main()
