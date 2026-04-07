# ==========
# Checkpoint-1 Step-2 
# Sequence alignment features Needleman Winsch
# Use kinase-family referecence to find kinase family realtion
# we are going to keep all rows splitting is going to happen in step4

import os
import re
import warnings
import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ==== Directory set up the same directory for all four steps with changing input and output filenames

Data_directory = "New_checkpoint_1_data_here"
Checkpoint_1_step_1_input_path = os.path.join(Data_directory,"checkpoint_1_step_1_kinase_filtered_data.csv")
Checkpoint_1_step_2_output_path =os.path.join(Data_directory,"checkpoint_1_step_2_alignment_features_data.csv")
Reference_seq_output = os.path.join(Data_directory,"checkpoint_1_kinase_reference_sequence_data.csv" )

EGFR_UNIPROT_ID = "P00533"
Min_family_size = 5 #removes kinase family have less than 5 memebers
max_reference_families = 25 #removes kinase family have more than 25 memebers

# ===
# Helper functions
# =====

def sequence_cleanup(seq):
    if pd.isna(seq): #checks if the value's missing NaN,None,missing
        return None
    seq = str(seq).strip().upper()
    if seq == "":
        return None
    #this keeps only letters and removes spaces or weird characters
    seq = "".join([c for c in seq if c.isalpha()]) #isalpha only keeps letter no numbers, spcaes or symbols
    if seq == "":
        return None
    return seq
def family_name_extraction(target_name):
    if pd.isna(target_name):
        return "unknown kinase family" #can be treated as a category later.keeps the row assigned a category 
    target_name =str(target_name).strip().lower()
    #this is going to remove generic wording, 
    #it only keeps kinase rleative infromation (regex)
    target_name = re.sub(r"[\(\)\[\],;/]+", " ", target_name)
    target_name = re.sub(r"\s+", " ", target_name).strip()

    remove_words = {
        "protein", "serine", "threonine", "tyrosine", "receptor", "human",
        "fragment", "domain", "catalytic", "chain", "precursor", "isoform",
        "mutant", "activated", "kinase"
    }
    #In bigingdb almost every entry contains the name above they don't help selecting families.
    tokens = [ t for t in target_name.split() if t not in remove_words]
    if not tokens:
        return "unknown_kinase_family"
    #keeps 1-3 tokens as an approximate family label
    fam_tokens = tokens[:3]
    fam_name = "_".join(fam_tokens)

    if fam_name == "":
        return "unknown_kinase_family"
    return fam_name
def get_fam_ref_seq(df):
    if "target_name" not in df.columns:
        raise ValueError("Columns 'target_name' not found in step-1 data.")
    if "sequence" not in df.columns:
        raise ValueError("Columns'sequence' not founf in step-1 data.")
    
    works_df =df.copy()
    works_df["sequence"] = works_df["sequence"].apply(sequence_cleanup)
    #delete empty sequences data
    works_df = works_df.dropna(subset=["sequence"])
   
    if works_df.empty:
        raise ValueError("No valid sequences found in step-1 data.")
    
    works_df["kinase_family"] = works_df["target_name"].apply(family_name_extraction)

    #Exclude EGFR rows from the reference panel so that step-2 is not egfr centered 
    #I had to debug this part step-1 labeling is going to help at step-4

    if "uniprot_id" in works_df.columns:
        non_egfr_df = works_df[
            ~works_df["uniprot_id"].astype(str).str.upper().str.contains(EGFR_UNIPROT_ID, na=False)
        ].copy()
    else:
        non_egfr_df = works_df.copy()
    if non_egfr_df.empty:
        raise ValueError("Non non-EGFR kinase rows availabe for family reference built")
    
    #rset index add an index
    fam_counts = (
        non_egfr_df["kinase_family"]
        .value_counts()
        .reset_index()
    )
    # Min size =5 max size 25 is on the global reference in begining of the code
    fam_counts.columns = ["kinase_family", "family_size"]
    valid_fam = fam_counts[fam_counts["family_size"] >= Min_family_size].copy()

    if valid_fam.empty: #is empty?
        #if no family is big enough take the top(head) max reference =25 cases
        valid_fam = fam_counts.head(max_reference_families).copy()
    else:
        valid_fam = valid_fam.head(max_reference_families).copy() #from valid cases take top N
    
    def build_reference_row(fam):
        fam_df = non_egfr_df[non_egfr_df["kinase_family"] ==fam].copy()
        if fam_df.empty:
            return None
        fam_df["seq_len"] = fam_df["sequence"].str.len()
        ref_row = fam_df.sort_values("seq_len", ascending=False).iloc[0]


        return {
            "kinase_family": fam,
            "reference_sequence": ref_row["sequence"],
            "reference_length": len(ref_row["sequence"]),
            "family_size": int(( non_egfr_df["kinase_family"] == fam).sum()),
            "reference_uniprot_id": ref_row["uniprot_id"] if "uniprot_id" in ref_row.index else np.nan,
            "reference_target_name" :ref_row["target_name"] if "target_name" in ref_row.index else np.nan,

        }
    
    reference_rows = list(
        filter(None, map(build_reference_row, valid_fam["kinase_family"].tolist()))
    )

    reference_data_df = pd.DataFrame(reference_rows)

    if reference_data_df.empty:
        raise ValueError("No kinase family reference sequences could be built")
    return reference_data_df                                

"""
    for fam in valid_fam["kinase_family"].tolist():
        fam_df = non_egfr_df[non_egfr_df["kinase_family"] == fam].copy()

        if fam_df.empty:
            continue
        #it chooses the longest clean sequence in each family reference
        fam_df["seq_len"] = fam_df["sequence"].str.len()
        ref_row = fam_df.sort_values("seq_len", ascending=False).iloc[0]"""


   
def aligner_build():
    aligner = PairwiseAligner()
    #global allignment from beginig to end not just best matching parts.
    aligner.mode = "global"

    #Alignment scoring
    aligner.match_score = 1.0
    aligner.mismatch_score = 0.0
    aligner.open_gap_score = -1  #penalty score for starting a new gap
    aligner.extend_gap_score = -0.5  #smaller penalty for couning an existing gap
    
    return aligner

#Calculates base pair allignment and exact nidex by index similarity
def execute_alignment_features(reference_sequence, query_sequence, aligner):
    query_sequence = sequence_cleanup(query_sequence)
    
    if query_sequence is None:
        return np.nan, np.nan, np.nan
    
    reference_len =len(reference_sequence) #known seq from the datbase
    query_len =len(query_sequence) #seq of interest 
    
    if reference_len == 0 or query_len == 0:
        return np.nan, np.nan, np.nan
    
    align_raw_score = aligner.score(reference_sequence, query_sequence)

    #It will normalize the max possible self score of the longer sequence length
    #Common bioinformatics sequnce comparison normalization
    denominator =max(reference_len, query_len)
    normalized_score = align_raw_score/denominator if denominator > 0 else np.nan

    #this is for the same position similarity 
    min_len = min(reference_len, query_len)
    #counts how many exact base pair mathces witht the same index. ie. Adenine in index 11 in both seq.
    #sum loops through both seq together adds 1 everythime it finds identical characters.
    same_count = sum(a == b for a,b in zip(reference_sequence[:min_len], query_sequence[:min_len]))
    #fraction of matching positions 
    prefix_similarity = same_count/ min_len if min_len > 0 else np.nan

    return align_raw_score, normalized_score, prefix_similarity

def execute_fam_panel_features(query_sequence, reference_data_df,aligner):
    query_sequence = sequence_cleanup(query_sequence)

    if query_sequence is None:
        return {
            "closest_kinase_family": np.nan,
            "needleman_wunsch_score":np.nan,
            "normalized_nw_score":np.nan,
            "sequence_similarity":np.nan,
            "mean_nw_score":np.nan,
            "mean_normalized_nw_score":np.nan,
            "mean_sequence_similarity":np.nan,
            "family_reference_count":np.nan
        }
    
    def compute_score(row):
        align_raw_score, normalized_score, prefix_similarity =execute_alignment_features(
            row["reference_sequence"], query_sequence, aligner

        )
        return {
            "kinase_family": row["kinase_family"],
            "raw_score": align_raw_score,
            "normalized_score": normalized_score,
            "prefix_similarity": prefix_similarity,
        }
        
    scores_list_df = pd.DataFrame(
        list(map(compute_score, reference_data_df.to_dict("records")))
    ).dropna()
  

    if scores_list_df.empty:
        return {
            "closest_kinase_family": np.nan,
            "needleman_wunsch_score": np.nan,
            "normalized_nw_score":np.nan,
            "sequence_similarity":np.nan,
            "mean_nw_score":np.nan,
            "mean_normalized_nw_score":np.nan,
            "mean_sequence_similarity":np.nan,
            "family_reference_count":np.nan
        }
    #provides the best scores in highest order
    best_sc_row = scores_list_df.sort_values(
        ["normalized_score", "prefix_similarity", "raw_score"],
        ascending=False
    ).iloc[0]
    #it provides the best mathc similarity
    return{
        "closest_kinase_family": best_sc_row["kinase_family"],
        "needleman_wunsch_score": best_sc_row["raw_score"],
        "normalized_nw_score":best_sc_row["normalized_score"],
        "sequence_similarity":best_sc_row["prefix_similarity"],
        "mean_nw_score":scores_list_df["raw_score"].mean(),
        "mean_normalized_nw_score":scores_list_df["normalized_score"].mean(),
        "mean_sequence_similarity":scores_list_df["prefix_similarity"].mean(),
        "family_reference_count":len(scores_list_df)
    }
#Normilized new score needleman scscled by sequence length
#mean new score and mean normalzied new score  the avg raw and nromalized alignment scores across all reference seq 
#so overall similarity  trends.



# =======
# MAIN 
# =======

def main():
    os.makedirs(Data_directory, exist_ok=True)
    
    if not os.path.exists(Checkpoint_1_step_1_input_path):
        print("Step-1 file is not found", Checkpoint_1_step_1_input_path)
        return
    
    df = pd.read_csv(Checkpoint_1_step_1_input_path)
    print("Loaded Step-1 Data Shape:", df.shape)

    if df.empty:
        print("Error Step-1 input is empty")
        return 
    
    required_cols =["sequence", "target_name"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in Step-1 file{missing_cols}")
    
    reference_data_df = get_fam_ref_seq(df)
    reference_data_df.to_csv(Reference_seq_output, index=False)

    print("Reference kinase families used:", reference_data_df.shape[0])
    print("Sample reference families:", reference_data_df["kinase_family"].head(10).tolist())

    aligner = aligner_build()
    #Clean up the sequence scores
    df["sequence"] = df["sequence"].apply(sequence_cleanup)

    #Executes once per unique sequence to speed up processing
    unique_sequences = df["sequence"].dropna().unique().tolist()
    print("Unique sequences to score:", len(unique_sequences))

    feature_map = {} #this getting created with seq with closest kinase fam,needleamn score,seq similairty --> follow along the code.
    
    #tqdm is gong to show loading progress bar in the terminal
    for seq in tqdm(unique_sequences, desc="Computing alignment features"):
        feature_map[seq] = execute_fam_panel_features(seq, reference_data_df, aligner)
    #it maps each seq to its closest family from feature map and if seq mising assign a missing value.
    df ["closest_kinase_family"] = df["sequence"].map(
        lambda s: feature_map[s] ["closest_kinase_family"] if pd.notna(s) else np.nan
    )
    df ["needleman_wunsch_score"] = df["sequence"].map(
        lambda s: feature_map[s] ["needleman_wunsch_score"] if pd.notna(s) else np.nan
    )
    df ["normalized_nw_score"] = df["sequence"].map(
        lambda s: feature_map[s] ["normalized_nw_score"] if pd.notna(s) else np.nan
    )
    df ["sequence_similarity"] = df["sequence"].map(
        lambda s: feature_map[s] ["sequence_similarity"] if pd.notna(s) else np.nan
    )
    df ["mean_nw_score"] = df["sequence"].map(
        lambda s: feature_map[s] ["mean_nw_score"] if pd.notna(s) else np.nan
    )
    df ["mean_normalized_nw_score"] = df["sequence"].map(
        lambda s: feature_map[s] ["mean_normalized_nw_score"] if pd.notna(s) else np.nan
    )
    df ["mean_sequence_similarity"] = df["sequence"].map(
        lambda s: feature_map[s] ["mean_sequence_similarity"] if pd.notna(s) else np.nan
    )
    df ["family_reference_count"] = df["sequence"].map(
        lambda s: feature_map[s] ["family_reference_count"] if pd.notna(s) else np.nan
    )

    df.to_csv(Checkpoint_1_step_2_output_path, index=False)
    print("Checkpoint 1 step_2_output_path saved in:", Checkpoint_1_step_2_output_path)
    print("Final Checkpoint-1 Step-2 shape:", df.shape)

if __name__ == "__main__":
    main()

    
    

    

