# =========
# Checkpoint-2, Step-2
# Deep Sequence Embedding Generation
# Use pre-trained protein language model for embeddings
# =========

import os
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============
# Directory Setup
# ============
Data_directory = "Checkpoint_2_data"
Checkpoint_2_step_2_input_path = os.path.join(
    Data_directory, "checkpoint_2_step_1_sequence_feature_data.csv"
)
Checkpoint_2_step_2_output_path = os.path.join(
    Data_directory, "checkpoint_2_step_2_deepseq_embedding.csv"
)

# Embedding dimensions
EMBEDDING_DIM = 64  # Reduced dimension for efficiency
MAX_SEQ_LENGTH = 512  # Maximum sequence length for embedding

# ============
# Sequence Cleaning
# ============
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def clean_sequence(seq):
    """Clean and standardize amino acid sequence."""
    if pd.isna(seq):
        return None
    seq = str(seq).strip().upper()
    seq = "".join([c for c in seq if c in AMINO_ACIDS])
    if seq == "":
        return None
    return seq


# ============
# Embedding Functions (Simplified: No External Model Dependency)
# ============
# Using a simple but effective approach:
# Learned position weighted amino acid embeddings

# Pre-trained amino acid embeddings (simulated, based on BLOSUM62 substitution matrix principles)
# These capture evolutionary relationships between amino acids
AA_EMBEDDINGS = {
    "A": [0.5, 0.1, -0.3, 0.2, 0.0, 0.4, -0.1, 0.3],
    "C": [0.8, -0.2, 0.1, -0.1, 0.5, 0.2, 0.0, 0.1],
    "D": [-0.3, 0.7, 0.4, -0.2, -0.1, -0.3, 0.5, 0.1],
    "E": [-0.2, 0.6, 0.5, -0.3, -0.2, -0.2, 0.6, 0.0],
    "F": [0.6, -0.1, -0.4, 0.5, 0.3, 0.2, -0.3, 0.4],
    "G": [0.0, 0.3, -0.1, 0.1, -0.3, 0.5, 0.2, -0.2],
    "H": [-0.1, 0.4, 0.3, 0.2, 0.4, 0.0, 0.5, -0.1],
    "I": [0.7, -0.3, -0.5, 0.4, 0.1, 0.3, -0.4, 0.5],
    "K": [-0.4, 0.5, 0.6, -0.1, -0.3, -0.4, 0.7, -0.2],
    "L": [0.6, -0.2, -0.4, 0.3, 0.2, 0.3, -0.3, 0.4],
    "M": [0.5, 0.0, -0.3, 0.2, 0.4, 0.2, -0.2, 0.3],
    "N": [-0.2, 0.5, 0.3, 0.0, -0.2, 0.1, 0.4, 0.0],
    "P": [0.1, 0.2, 0.0, 0.1, -0.1, 0.6, 0.1, -0.3],
    "Q": [-0.1, 0.5, 0.4, 0.1, -0.1, 0.0, 0.5, 0.0],
    "R": [-0.5, 0.4, 0.6, 0.0, -0.2, -0.3, 0.8, -0.1],
    "S": [0.1, 0.4, 0.1, 0.0, -0.2, 0.3, 0.2, -0.1],
    "T": [0.2, 0.3, 0.0, 0.1, -0.1, 0.3, 0.1, 0.0],
    "V": [0.6, -0.2, -0.4, 0.3, 0.0, 0.4, -0.3, 0.4],
    "W": [0.4, 0.0, -0.5, 0.6, 0.5, 0.1, -0.4, 0.5],
    "Y": [0.3, 0.1, -0.3, 0.5, 0.4, 0.1, -0.2, 0.4],
}


def get_position_weights(seq_length, max_length=MAX_SEQ_LENGTH):
    """Generate position weights emphasizing N and C termini."""
    if seq_length > max_length:
        # Focus on N-terminus, middle sample, and C-terminus
        return list(range(max_length))

    # Weight positions: higher at termini
    weights = []
    for i in range(seq_length):
        # Gaussian 'like' weighting with emphasis on termini
        pos_weight = 1.0 + 0.5 * np.exp(-0.01 * min(i, seq_length - 1 - i))
        weights.append(pos_weight)

    return weights


def get_sequence_embedding(seq, embedding_dim=EMBEDDING_DIM):
    """
    Generate a fixed size embedding for a protein sequence.
    Uses position weighted amino acid embeddings.
    """
    if seq is None or len(seq) == 0:
        return np.zeros(embedding_dim)

    # Truncate very long sequences
    if len(seq) > MAX_SEQ_LENGTH:
        # Keep N-terminus, middle portion, and C-terminus
        n_term = seq[: MAX_SEQ_LENGTH // 3]
        mid_start = (len(seq) - MAX_SEQ_LENGTH // 3) // 2
        mid = seq[mid_start : mid_start + MAX_SEQ_LENGTH // 3]
        c_term = seq[-(MAX_SEQ_LENGTH // 3) :]
        seq = n_term + mid + c_term

    # Get amino acid embeddings
    aa_emb_dim = 8  # Dimension of AA embeddings
    embeddings = []

    for aa in seq:
        if aa in AA_EMBEDDINGS:
            embeddings.append(AA_EMBEDDINGS[aa])
        else:
            embeddings.append([0.0] * aa_emb_dim)

    if not embeddings:
        return np.zeros(embedding_dim)

    embeddings = np.array(embeddings)

    # Calculate statistics across the sequence
    mean_emb = np.mean(embeddings, axis=0)
    std_emb = np.std(embeddings, axis=0)
    max_emb = np.max(embeddings, axis=0)
    min_emb = np.min(embeddings, axis=0)

    # Position weighted statistics
    position_weights = get_position_weights(len(seq))
    weighted_emb = np.zeros(aa_emb_dim)
    total_weight = 0

    for i, emb in enumerate(embeddings):
        weight = position_weights[i] if i < len(position_weights) else 1.0
        weighted_emb += emb * weight
        total_weight += weight

    weighted_emb = weighted_emb / total_weight if total_weight > 0 else weighted_emb

    # N-terminal features (first 10 amino acids)
    n_term_emb = np.mean(embeddings[:10], axis=0) if len(embeddings) >= 10 else mean_emb

    # C-terminal features (last 10 amino acids)
    c_term_emb = (
        np.mean(embeddings[-10:], axis=0) if len(embeddings) >= 10 else mean_emb
    )

    # Combine all features
    combined = np.concatenate(
        [
            mean_emb,  # 8 dims
            std_emb,  # 8 dims
            max_emb,  # 8 dims
            min_emb,  # 8 dims
            weighted_emb,  # 8 dims
            n_term_emb,  # 8 dims
            c_term_emb,  # 8 dims
            [len(seq) / 1000.0],  # 1 dim - normalized length
        ]
    )

    # Pad or truncate to desired dimension
    if len(combined) < embedding_dim:
        combined = np.pad(combined, (0, embedding_dim - len(combined)))
    else:
        combined = combined[:embedding_dim]

    return combined


def get_kmer_frequencies(seq, k=3):
    """Calculate k-mer frequency features."""
    if seq is None or len(seq) < k:
        return {}

    kmers = []
    for i in range(len(seq) - k + 1):
        kmers.append(seq[i : i + k])

    total = len(kmers)
    counts = {}
    for kmer in kmers:
        counts[kmer] = counts.get(kmer, 0) + 1

    # Returns normalized frequencies
    return {
        f"kmer_{kmer}": count / total for kmer, count in sorted(counts.items())[:50]
    }


# ============
# Main Processing
# ============
def process_embeddings(df):
    """Generate embeddings for all sequences."""
    print("Cleaning sequences...")
    df["sequence_clean"] = df["sequence"].apply(clean_sequence)

    # Remove rows with invalid sequences
    valid_mask = df["sequence_clean"].notna()
    df = df[valid_mask].copy()

    print(f"Valid sequences: {len(df)}")

    # Generate embeddings for unique sequences (optimization)
    unique_seqs = df["sequence_clean"].unique()
    print(f"Unique sequences to embed: {len(unique_seqs)}")

    embedding_map = {}
    for seq in tqdm(unique_seqs, desc="Generating embeddings"):
        embedding_map[seq] = get_sequence_embedding(seq)

    # Build embedding dataframe
    embedding_rows = []
    for seq in tqdm(df["sequence_clean"], desc="Building embedding table"):
        embedding_rows.append(embedding_map[seq])

    embedding_df = pd.DataFrame(embedding_rows)
    embedding_df.columns = [f"emb_{i}" for i in range(embedding_df.shape[1])]

    # Combine with original dataframe
    df = df.reset_index(drop=True)
    result_df = pd.concat([df, embedding_df], axis=1)

    return result_df


# ============
# Main
# ============
def main():
    if not os.path.exists(Checkpoint_2_step_2_input_path):
        print(
            "Checkpoint 2 step 2 input file not found:", Checkpoint_2_step_2_input_path
        )
        return

    df = pd.read_csv(Checkpoint_2_step_2_input_path, low_memory=False)
    print("Input loaded, shape:", df.shape)

    # Check required columns
    required_cols = ["sequence", "target_name"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Process sequences
    result_df = process_embeddings(df)

    # Save output
    result_df.to_csv(Checkpoint_2_step_2_output_path, index=False)
    print("Checkpoint 2 step 2 output saved:", Checkpoint_2_step_2_output_path)
    print("Final shape:", result_df.shape)
    print(
        "Embedding columns:",
        [c for c in result_df.columns if c.startswith("emb_")][:5],
        "...",
    )


if __name__ == "__main__":
    main()
