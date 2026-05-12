# =========
# Checkpoint-2, Step-1
# Sequence Feature Extraction
# Extract amino acid composition and physicochemical properties
# =========

import os
import warnings
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============
# Directory Setup
# ============
Data_directory = "Checkpoint_2_data"
os.makedirs(Data_directory, exist_ok=True)

Checkpoint_2_step_1_input_path = os.path.join(
    "New_checkpoint_1_data_here", "checkpoint_1_step_2_alignment_features_data.csv"
)
Checkpoint_2_step_1_output_path = os.path.join(
    Data_directory, "checkpoint_2_step_1_sequence_feature_data.csv"
)

# ============
# Amino Acid Properties
# ============
# Standard 20 amino acids
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# Physicochemical property values for each amino acid
# Sources: Kyte-Doolittle hydrophobicity, molecular weight, pKa values
AA_PROPERTIES = {
    "A": {
        "hydrophobicity": 1.8,
        "molecular_weight": 89.1,
        "polarity": 0,
        "charge": 0,
        "volume": 88.6,
    },
    "C": {
        "hydrophobicity": 2.5,
        "molecular_weight": 121.2,
        "polarity": 0,
        "charge": 0,
        "volume": 108.5,
    },
    "D": {
        "hydrophobicity": -3.5,
        "molecular_weight": 133.1,
        "polarity": 1,
        "charge": -1,
        "volume": 111.1,
    },
    "E": {
        "hydrophobicity": -3.5,
        "molecular_weight": 147.1,
        "polarity": 1,
        "charge": -1,
        "volume": 138.4,
    },
    "F": {
        "hydrophobicity": 2.8,
        "molecular_weight": 165.2,
        "polarity": 0,
        "charge": 0,
        "volume": 189.9,
    },
    "G": {
        "hydrophobicity": -0.4,
        "molecular_weight": 75.1,
        "polarity": 0,
        "charge": 0,
        "volume": 60.1,
    },
    "H": {
        "hydrophobicity": -3.2,
        "molecular_weight": 155.2,
        "polarity": 1,
        "charge": 0.5,
        "volume": 153.2,
    },
    "I": {
        "hydrophobicity": 4.5,
        "molecular_weight": 131.2,
        "polarity": 0,
        "charge": 0,
        "volume": 166.7,
    },
    "K": {
        "hydrophobicity": -3.9,
        "molecular_weight": 146.2,
        "polarity": 1,
        "charge": 1,
        "volume": 168.6,
    },
    "L": {
        "hydrophobicity": 3.8,
        "molecular_weight": 131.2,
        "polarity": 0,
        "charge": 0,
        "volume": 166.7,
    },
    "M": {
        "hydrophobicity": 1.9,
        "molecular_weight": 149.2,
        "polarity": 0,
        "charge": 0,
        "volume": 162.9,
    },
    "N": {
        "hydrophobicity": -3.5,
        "molecular_weight": 132.1,
        "polarity": 1,
        "charge": 0,
        "volume": 114.1,
    },
    "P": {
        "hydrophobicity": -1.6,
        "molecular_weight": 115.1,
        "polarity": 0,
        "charge": 0,
        "volume": 112.7,
    },
    "Q": {
        "hydrophobicity": -3.5,
        "molecular_weight": 146.2,
        "polarity": 1,
        "charge": 0,
        "volume": 143.8,
    },
    "R": {
        "hydrophobicity": -4.5,
        "molecular_weight": 174.2,
        "polarity": 1,
        "charge": 1,
        "volume": 173.4,
    },
    "S": {
        "hydrophobicity": -0.8,
        "molecular_weight": 105.1,
        "polarity": 1,
        "charge": 0,
        "volume": 89.0,
    },
    "T": {
        "hydrophobicity": -0.7,
        "molecular_weight": 119.1,
        "polarity": 1,
        "charge": 0,
        "volume": 116.1,
    },
    "V": {
        "hydrophobicity": 4.2,
        "molecular_weight": 117.1,
        "polarity": 0,
        "charge": 0,
        "volume": 140.0,
    },
    "W": {
        "hydrophobicity": -0.9,
        "molecular_weight": 204.2,
        "polarity": 0,
        "charge": 0,
        "volume": 227.8,
    },
    "Y": {
        "hydrophobicity": -1.3,
        "molecular_weight": 181.2,
        "polarity": 1,
        "charge": 0,
        "volume": 193.6,
    },
}


# ============
# Sequence Cleaning
# ============
def clean_sequence(seq):
    """Clean and standardize amino acid sequence."""
    if pd.isna(seq):
        return None
    seq = str(seq).strip().upper()
    # Keep only standard amino acids
    seq = "".join([c for c in seq if c in AMINO_ACIDS])
    if seq == "":
        return None
    return seq


# ============
# Feature Extraction Functions
# ============
def get_aa_composition(seq):
    """Calculate amino acid composition (frequency of each AA)."""
    if seq is None or len(seq) == 0:
        return {aa: 0.0 for aa in AMINO_ACIDS}

    total = len(seq)
    counts = Counter(seq)
    composition = {aa: counts.get(aa, 0) / total for aa in AMINO_ACIDS}
    return composition


def get_dipeptide_composition(seq):
    """Calculate dipeptide composition (frequency of AA pairs)."""
    if seq is None or len(seq) < 2:
        return {}

    dipeptides = []
    for i in range(len(seq) - 1):
        dipeptides.append(seq[i : i + 2])

    total = len(dipeptides)
    counts = Counter(dipeptides)

    # Return top 20 most common dipeptides
    top_dipeptides = counts.most_common(20)
    return {f"dipep_{dp}": count / total for dp, count in top_dipeptides}


def get_physicochemical_properties(seq):
    """Calculate physicochemical properties from sequence."""
    if seq is None or len(seq) == 0:
        return {
            "mean_hydrophobicity": np.nan,
            "mean_molecular_weight": np.nan,
            "mean_polarity": np.nan,
            "net_charge": np.nan,
            "mean_volume": np.nan,
            "hydrophobic_ratio": np.nan,
            "polar_ratio": np.nan,
            "charged_ratio": np.nan,
        }

    hydrophobicity_values = []
    molecular_weight_values = []
    polarity_values = []
    charge_values = []
    volume_values = []

    hydrophobic_count = 0
    polar_count = 0
    charged_count = 0

    for aa in seq:
        if aa in AA_PROPERTIES:
            props = AA_PROPERTIES[aa]
            hydrophobicity_values.append(props["hydrophobicity"])
            molecular_weight_values.append(props["molecular_weight"])
            polarity_values.append(props["polarity"])
            charge_values.append(props["charge"])
            volume_values.append(props["volume"])

            # Count categories
            if props["hydrophobicity"] > 0:
                hydrophobic_count += 1
            if props["polarity"] == 1:
                polar_count += 1
            if props["charge"] != 0:
                charged_count += 1

    total = len(seq)
    return {
        "mean_hydrophobicity": (
            np.mean(hydrophobicity_values) if hydrophobicity_values else np.nan
        ),
        "mean_molecular_weight": (
            np.mean(molecular_weight_values) if molecular_weight_values else np.nan
        ),
        "mean_polarity": np.mean(polarity_values) if polarity_values else np.nan,
        "net_charge": sum(charge_values) if charge_values else np.nan,
        "mean_volume": np.mean(volume_values) if volume_values else np.nan,
        "hydrophobic_ratio": hydrophobic_count / total if total > 0 else np.nan,
        "polar_ratio": polar_count / total if total > 0 else np.nan,
        "charged_ratio": charged_count / total if total > 0 else np.nan,
    }


def get_sequence_length_features(seq):
    """Calculate sequence length based features."""
    if seq is None:
        return {"seq_length": 0}
    return {"seq_length": len(seq)}


def extract_all_features(seq):
    """Extract all features for a single sequence."""
    features = {}

    # Length features
    features.update(get_sequence_length_features(seq))

    # Amino acid composition
    aa_comp = get_aa_composition(seq)
    for aa, freq in aa_comp.items():
        features[f"aa_{aa}"] = freq

    # Physicochemical properties
    features.update(get_physicochemical_properties(seq))

    return features


# ============
# Main Processing
# ============
def process_sequences(df):
    """Process all sequences and extract features."""
    print("Cleaning sequences...")
    df["sequence_clean"] = df["sequence"].apply(clean_sequence)

    # Remove rows with invalid sequences
    valid_mask = df["sequence_clean"].notna()
    df = df[valid_mask].copy()

    print(f"Valid sequences: {len(df)}")

    # Extract features for each unique sequence (optimization)
    unique_seqs = df["sequence_clean"].unique()
    print(f"Unique sequences to process: {len(unique_seqs)}")

    feature_map = {}
    for seq in tqdm(unique_seqs, desc="Extracting features"):
        feature_map[seq] = extract_all_features(seq)

    # Build feature dataframe
    feature_rows = []
    for seq in tqdm(df["sequence_clean"], desc="Building feature table"):
        feature_rows.append(feature_map[seq])

    feature_df = pd.DataFrame(feature_rows)

    # Combine with original dataframe
    df = df.reset_index(drop=True)
    result_df = pd.concat([df, feature_df], axis=1)

    return result_df


# ============
# Main
# ============
def main():
    if not os.path.exists(Checkpoint_2_step_1_input_path):
        print(
            "Checkpoint 2 step 1 input file not found:", Checkpoint_2_step_1_input_path
        )
        return

    df = pd.read_csv(Checkpoint_2_step_1_input_path, low_memory=False)
    print("Input loaded, shape:", df.shape)

    # Check required columns
    required_cols = ["sequence", "target_name"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print("Sample target names:")
    print(df["target_name"].dropna().head(10).tolist())

    print("Sample sequences:")
    print(df["sequence"].dropna().head(3).tolist())

    # Process sequences
    result_df = process_sequences(df)

    # Save output
    result_df.to_csv(Checkpoint_2_step_1_output_path, index=False)
    print("Checkpoint 2 step 1 output saved:", Checkpoint_2_step_1_output_path)
    print("Final shape:", result_df.shape)


if __name__ == "__main__":
    main()
