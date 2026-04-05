import numpy as np
import pandas as pd
from pathlib import Path

data_dir = Path('/kaggle/input/cafa6-protein-embeddings-esm2')

protein_ids = pd.read_csv(data_dir / "protein_ids.csv")["protein_id"].tolist()
embeddings = np.load(data_dir / "protein_embeddings.npy")


print(f"Loaded {len(protein_ids)} embeddings of dimension {embeddings.shape[1]}")


pid_emb_dict = {}
for pid, emb in zip(protein_ids, embeddings):
    pid_emb_dict[str(pid)] = emb