#Checkpoint-2 Step 3
"""
Checkpoint 2 Step 3
Feature Engineering and Local Alignments

This step combines multiple types of protein features:
- sequence embeddings
- physicochemical properties
- global alighment scores
- local alignment scores using Smith Waterman

Local alignment features capture converd motifs between proteins and known kinase
families, improving classification performance.

The processed dataset is then saved and used in Step 4 - model training

"""
#importing the necessary libraries
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
import umap
#from Bio import pairwise2
from Bio.Align import PairwiseAligner

#Setting up directory, input and output
Data_directory = "Checkpoint_2_data"
Checkpoint_2_step_3_input_path = os.path.join(Data_directory,"checkpoint_2_step_2_deepseq_embedding.csv") #input data  is output data from checkpoint 2 step 2
Checkpoint_2_step_3_output_path = os.path.join(Data_directory, "checkpoint_2_step_3_family_classification.csv") #output data
Checkpoint_2_step_3_family_dend_plot = os.path.join(Data_directory, "checkpoint_2_step_3_family_tree_plot.png") #plot of family trees for visualization
Checkpoint_2_step_3_protein_dend_plot = os.path.join(Data_directory, "checkpoint_2_step_3_protein_subset.png")
Checkpoint_2_step_3_umap = os.path.join(Data_directory, "checkpoint_2_step_3_umap.png")


AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def clean_sequence(seq):
    """
    Cleans a protein sequence so step 3 is only using standard amino acid letters.
    This is mostly done in step 2, but this function serves as a check to prevent inconsistent sequences.

    Parameters:
        seq: raw sequence imput from dataframe column
    
    Returns:
        str or None: clean uppercase amino-acid sequence, or None if nothing valid is present after.
    """

    #returns None is the input is missing using pandas verification
    if pd.isna(seq):
        return None
    
    #converting the sequence to uppercase string with all whitespace stripped and only valid AA letters
    seq = "".join(c for c in str(seq).strip().upper() if c in AMINO_ACIDS)

    #return the cleaned sequence if non-empty; otherwise it will return none
    return seq if seq else None


def family_reference_sequences(df, max_refs_per_family=1):
    """
    Building a small reference panel per kinase family using cleaned sequences.
    This produces the reference sequences per family so that each query can be compared
    agaisnt those references for Smith-Waterman.

    Parameters:
        df(pd.Dataframe): Step 2 output, in this case
        max_refs_per_family (int): this is the max number of reference sequences to keep per family
    
    Returns:
        dict: mapping from family name to a short list of cleaned reference sequences.
    """
    #returns an empty dict if the cleaned sequence column is unavailable
    if "sequence_clean" not in df.columns:
        return {}
    
    #keeping only the rows that have both a family label and a cleaned sequence candidate
    ref_df = df.dropna(subset=["closest_kinase_family", "sequence_clean"]).copy()

    #re-clean the sequence column as a safety step
    ref_df["sequence_clean"] = ref_df["sequence_clean"].map(clean_sequence)

    #drop the rows thaat became empty after cleaning.
    ref_df = ref_df.dropna(subset=["sequence_clean"]).copy()

    #returning an empty dict if there is no valid reference rows remaining
    if ref_df.empty:
        return {}
    
    #avoiding repeated references when possible!
    #removing duplicate reference rows to that the reference panel is small and less repetative - might need to add check to this
    cols = ["uniprot_id", "sequence_clean"] if "uniprot_id" in ref_df.columns else ["sequence_clean"]
    ref_df = ref_df.drop_duplicates(subset=cols)

    #build one family-to-reference mapping using only the first few rows per family.
    ref_map = {str(family): fam_df["sequence_clean"].astype(str).head(max_refs_per_family).tolist() 
               for family, fam_df in ref_df.groupby("closest_kinase_family") if not fam_df.empty}
    
    #returns the completed reference map
    return ref_map



LOCAL_ALIGNER = PairwiseAligner()
LOCAL_ALIGNER.mode = "local"
LOCAL_ALIGNER.match_score = 2
LOCAL_ALIGNER.mismatch_score = -1
LOCAL_ALIGNER.open_gap_score = -1
LOCAL_ALIGNER.extend_gap_score = -1

def computing_local_alignment_feats_seq(seq, reference_map, hit_threshold=40):
    """
    For summarizing local alighment scores for one sequence vs family references

    Parameters:
        seq (str): sequence of interest to evaluate
        reference_map (dict): family-to-reference sequences mapping
        hit_threshold (float): threshold used to count how many alignments are considered "strong"

    Returns:
        dict: numeric summary features plus the best matching family label.
    """
    #clean the incoming sequence as a safety measure
    seq = clean_sequence(seq)

    #return all missing/default values if the sequence or reference map is unusable.
    if seq is None or not reference_map:
        return {
            "local_alignment_best_score": np.nan,
            "local_alignment_mean_score": np.nan,
            "local_alignment_best_normalized": np.nan,
            "local_alignment_mean_normalized": np.nan,
            "local_alignment_score_std": np.nan,
            "local_alignment_hit_count": 0,
            "local_alignment_best_match_family": np.nan
        }
    
    #SMITH WATERMAN SCORE
    #Computing one numpy score array per family by aligning the query to that family's references
    family_scoring = {family: np.array([LOCAL_ALIGNER.score(seq, ref_seq) for ref_seq in refs], dtype=float)
                      for family, refs in reference_map.items() if refs}

    #keeping only the families that produced at least one alignment score total
    nonempty_scoring = {s: t for s, t in family_scoring.items() if t.size > 0}

    #again returning the default if no alignments were computed.
    if not nonempty_scoring:
        return {
            "local_alignment_best_score": np.nan,
            "local_alignment_mean_score": np.nan,
            "local_alignment_best_normalized": np.nan,
            "local_alignment_mean_normalized": np.nan,
            "local_alignment_score_std": np.nan,
            "local_alignment_hit_count": 0,
            "local_alignment_best_match_family": np.nan
        }
    
    #identify which family achieved the highest local alignment sore.
    best_family = max(nonempty_scoring, key = lambda fam: float(nonempty_scoring[fam].max()))
    
    #concatenate all of the family score arrays into one vector for summarizing stats
    all_scoring = np.concatenate(list(nonempty_scoring.values()))
    #storing the sequence length for normalization, using 1 to avoid 0 division
    seq_len = max(len(seq), 1)
    #extract the strongest local alignment score
    best_score = float(all_scoring.max())
    #compute the mean score across all family reference comparisons
    mean_score = float(all_scoring.mean())
   # print("score statistics:", np.min(all_scoring), np.mean(all_scoring), np.max(all_scoring))
    #print("percentiles", np.percentile(all_scoring, 90), np.percentile(all_scoring, 95))
    #returning the summary feature dictionary for this sequence
    return {
            "local_alignment_best_score": best_score,
            "local_alignment_mean_score": mean_score,
            "local_alignment_best_normalized": best_score / seq_len,
            "local_alignment_mean_normalized": mean_score / seq_len,
            "local_alignment_score_std": float(all_scoring.std()),
            "local_alignment_hit_count": int((all_scoring >= hit_threshold).sum()),
            "local_alignment_best_match_family": best_family
        }


def adding_local_align_features(df, ref_df = None, max_refs_per_family=3):
    """
    adding local alighment features to a dataframe

    Parameters:
        df (pd.Data): in this case, this is the step 2 dataframe
        max_refs_per_family (int): number of reference sequences per family

    Returns:
        pd.DataFrame: Original dataframe enriched with local alignment feature columns added
    """
    df = df.copy()
    #if a cleaned sequence exists, then use that
    if "sequence_clean" in df.columns:
        df["sequence_clean"] = df["sequence_clean"]
    #otherwise build a cleaned sequence column from the raw sequence column after cleaning
    elif "sequence" in df.columns:
        df["sequence_clean"] = df["sequence"].map(clean_sequence)
    #otherwise just stop if there is no useable sequence column present.
    else:
        print("no sequence column found for local alighment features")
        return df
    
    if ref_df is None:
        ref_df = df.copy()
    else:
        ref_df = ref_df.copy()
        #if a cleaned sequence exists, then use that
    if "sequence_clean" in ref_df.columns:
        ref_df["sequence_clean"] = ref_df["sequence_clean"]
    #otherwise build a cleaned sequence column from the raw sequence column after cleaning
    elif "sequence" in ref_df.columns:
        ref_df["sequence_clean"] = ref_df["sequence"].map(clean_sequence)
    #initializing family reference used for local alignment
    reference_map = family_reference_sequences(ref_df, max_refs_per_family=max_refs_per_family)
    #print the number of family reference groups that were built
    print("the number of local alightment reference families:", len(reference_map))

    #stop process is the reference is empty/not built
    if not reference_map:
        print("no local alightment references could be built")
        return df
    
    #extract the unique cleaned sequences so repeated rows dont do repetitive alignment work
    unique_seq = pd.Series(df["sequence_clean"].dropna().unique(), name="sequence_clean")
    
    
    #convering the unique sequence series to a dataframe for merging with features later on
    local_features_df = unique_seq.to_frame()

    #computing one local feature dictionary per unique sequence
    local_features_df["local_features"] = local_features_df["sequence_clean"].map(
        lambda s: computing_local_alignment_feats_seq(s, reference_map)
    )

    #expanding the dictionary column into ordinary dataframe columns
    expanding_features = pd.DataFrame(local_features_df["local_features"].tolist())
    #keep the sequence key and concat to the explanded local alignment columns
    local_features_df = pd.concat([local_features_df[["sequence_clean"]], expanding_features], axis=1)
    
    #merge thme local alignment features back into the step 2 dataframe by cleaned seq
    df = df.merge(local_features_df, on="sequence_clean", how="left")

    #creating a numeric indicator showing whether the best matching family agrees with the row label
    if "closest_kinase_family" in df.columns and "local_alignment_best_match_family" in df.columns:
        df["local_alignment_family_match"] =(
            df["local_alignment_best_match_family"].astype(str).str.strip()
            == df["closest_kinase_family"].astype(str).str.strip()
        ).astype(int)
    #returning the dataframe after alterations
    return df

#housekeeping safety function
def data_housekeeping_helper(df, feature_columns):
    """
    converting selected columns to a usable numeric matrix

    Parameters:
        df (pd.DataFrame): dataframe for input, in this case step 2 dataframe
        feature_columns (list[str]): columns intended for numeric converion/modeling

    Returns:
        df (pd.Dataframe): clean numeric matrix reading for scaling and clustering/plotting work
    """
    #copying only requested feature columns
    X_df = df[feature_columns].copy()
    #selected columns are given numeric values, non numeric will equal nan
    X_df = X_df.apply(pd.to_numeric, errors='coerce')
    #replacing positive or negative infinities with nan for filling
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    #crop any columns that are completely missing
    X_df = X_df.dropna(axis=1, how="all")
    #fill the remaining nan with the column median (may come back to improve this method though?)
    X_df = X_df.fillna(X_df.median(numeric_only=True))

    #return the cleaned numeric feature matric as dataframe
    return X_df

def main():
    #File Validations and Reading
    #============================================
    #Checking that input file actually exists
    if not os.path.exists(Checkpoint_2_step_3_input_path):
        print("NO Step 3 Input File Was Found!")
        return
    
    #Reading input file
    df = pd.read_csv(Checkpoint_2_step_3_input_path)
    print("Step 3 Input File Successfully Loaded!")
    #Making sure input file is not empty (there is data present)
    if df.empty:
        print("Step 3 Input File is Empty!")
        return
    #Checking that the label column is present for analysis
    if "closest_kinase_family" not in df.columns:
        raise ValueError("Missing Required Column! - closest_kinase_family")
    

    #print statements for sanity checks and data peeking/analysis
    print("\nTaking A Look At Input File from Step 2:")
    print(df.head()) #a few few look
    #looking at the number of unique families
    print("THe Number of families:", df["closest_kinase_family"].nunique())
    #class balance peek 
    print("Top 5 Family Counts:")
    print(df["closest_kinase_family"].value_counts().head())

    #Some File Housekeeping
    #===========================================
    #removing rows where the protein family label might be missing
    df = df.dropna(subset=["closest_kinase_family"]).copy()
    #converting the family label column to string type and removing any trailing or leading for consistency
    df["closest_kinase_family"] = df["closest_kinase_family"].astype(str).str.strip()
    #removes rows where the family label is empty after cleaning (handling blanks or spaces)
    df = df[df["closest_kinase_family"] != ""].copy()
    #checking is df is empty after housekeeping processes
    if df.empty:
        print("No good family labels present")
        return
    
    df = adding_local_align_features(df)
    #print shape after adding local alignment features
    print("Shape:", df.shape)
    #confirmation that the family column is existing
    print("File Has closest_kinase_family", "closest_kinase_family" in df.columns)
    #confirming protein labels are present
    print("File Has uniprot_id:", "uniprot_id" in df.columns)

    print("Step 3 Input File Passed Cleaning and Checks")

    #Data Preparations
    #=================================================
    #list of feature columns - alignment and embedding selection
    embedding_columns = [c for c in df.columns if c.startswith("emb_")]
    aa_columns = [c for c in df.columns if c.startswith("aa_")]
    phychem_columns = [
        #"seq_length", 
        "mean_hydrophobicity", 
        "mean_molecular_weight",
        "mean_polarity", 
        "net_charge",
        "mean_volume",
        "hydrophobic_ratio",
        "polar_ratio",
        "charged_ratio"
    ]
    similarity_columns = [
        "needleman_wunsch_score",
        "normalized_nw_score",
        "sequence_similarity",
        "mean_nw_score",
        "mean_normalized_nw_score",
        "mean_sequence_similarity",
        "family_reference_count"
    ]
    local_alighment_columns = ["local_alignment_best_score",
            "local_alignment_mean_score",
            "local_alignment_best_normalized",
            "local_alignment_mean_normalized",
            "local_alignment_score_std",
            "local_alignment_hit_count",
            #"local_alignment_family_match"
            ]

    assay_columns = ["ki", "kd", "ic50", "affinity", "log_affinity"] #not great for cold-start so considering removing!!!!
    feature_columns = embedding_columns + aa_columns + phychem_columns + similarity_columns + local_alighment_columns#+ assay_columns
    #checking out features and validity using some print statements
    print("\nPossible Feature Columns:")
    print(feature_columns)
    print("Any missing values in features?:", df[feature_columns].isna().sum().sum())
    #saving the altered datagrame for step 4
    df.to_csv(Checkpoint_2_step_3_output_path, index=False)
   

   ##VISUALIZATION SECTION FOR EDA##
   #========================================================================================================================================
    #Family Level Dendrogram Setup
    #==========================================================================
    family_df = (df.groupby("closest_kinase_family", as_index=False).median(numeric_only=True))
    #creating a feature matrix from the selected feature columns and cleaning using helper
    X_family = data_housekeeping_helper(family_df, feature_columns)
    print("\nFamily Dendrogam Feature Count:", X_family.shape[1])
    
    #standardizing the features before clustering
    scaler_fam = StandardScaler()
    X_family_scaled = scaler_fam.fit_transform(X_family)
    #computing the ward-linkage hierarchical clustering on scaled data
    z = linkage(X_family_scaled, method="ward")
    plt.figure(figsize=(30,20))
    labels = family_df["closest_kinase_family"].astype(str).tolist()
    
    #Plotting the Dendrogram
    dendrogram(z, labels=labels, orientation="right", leaf_font_size=25, color_threshold=10)
    plt.title("Hierarchical Clustering of Kinase Families", fontsize=50)
    plt.xlabel("Distance", fontsize=50)
    plt.ylabel("Kinase Family", fontsize=50)
    plt.tight_layout()
    plt.savefig(Checkpoint_2_step_3_family_dend_plot, dpi=400) #saving dendrogram to server directory
    plt.close()
    print("\nFamily Dendrogram PNG Saved to Directory!")
    
    #Attempt at Completing a Protein Level Subset Dendrogram
    #===========================================================================================
    if "uniprot_id" in df.columns:
        #grouping the rows into one median per protein family pair for visualization
        protein_df = (df.groupby(["uniprot_id", "closest_kinase_family"], as_index=False).median(numeric_only=True))
        print(f"\nThere are {len(protein_df)} unique proteins available")
        #breaking into a subset for plotting, keeping only up to 10 proteins per family - might need to decrease this
        protein_subset_df = protein_df.sort_values(by=["closest_kinase_family", "uniprot_id"]).groupby("closest_kinase_family", group_keys=False).head(10).reset_index(drop=True)

        #building the clean numeric feature matrix
        X_proteins = data_housekeeping_helper(protein_subset_df, feature_columns)
        #scaling the data for the standardization
        scaler_prot = StandardScaler()
        X_protein_scaled = scaler_prot.fit_transform(X_proteins)
        #computing the ward-linkage hierarchical clustering on the protein subset
        Z_protein = linkage(X_protein_scaled, method="ward")
        #attempt to build readable labels combining family name and unitprot_id
        labels = (protein_subset_df["closest_kinase_family"].astype(str) + " | " + protein_subset_df["uniprot_id"].astype(str)).tolist()
        
        #initializing figure and then building dendrogram
        plt.figure(figsize=(30,30))
        dendrogram(Z_protein, labels=labels, leaf_rotation=90, leaf_font_size=25)
        plt.title("Clustering of Protein Subset by Family", fontsize=40)
        plt.xlabel("Proteins", fontsize=40)
        plt.ylabel("Distance", fontsize=40)
        plt.tight_layout()
        plt.savefig(Checkpoint_2_step_3_protein_dend_plot, dpi=400) #saving dendrogram to server directory
        plt.close()
        print("\nProtein Subset Dendrogram PNG Saved to Directory!")


        ##ATTEMPTING UMAP REPRESENTATION##  #try biopython representations next
        #================================
        umap_model = umap.UMAP(random_state=42)
        X_umap = umap_model.fit_transform(X_protein_scaled)
        umap_df = pd.DataFrame({"UMAP1": X_umap[:,0], "UMAP2": X_umap[:,1], "family": protein_subset_df["closest_kinase_family"].values})
        plt.figure(figsize=(20,20))
        families = sorted(umap_df["family"].unique())
        for fam in families:
            subset = umap_df[umap_df["family"] == fam]
            plt.scatter(subset["UMAP1"], subset["UMAP2"], s=50, alpha=0.8, label=fam)
        plt.title("UMAP Projection of Protein Features by Kinase Family")
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
        plt.legend(bbox_to_anchor=(1.05,1), loc="upper right", fontsize=6)
        plt.tight_layout()
        plt.savefig(Checkpoint_2_step_3_umap, dpi=500)
        plt.close()
        print("UMAP Rows:", len(umap_df))
        print("UMAP Families:", umap_df["family"].nunique())
        print("UMAP Plot was Saved to Directory")

    #Yay! Celebrating that step 3 was able to complete
    print("Step 3 is Finished! Pursue Step 4!")

if __name__ == "__main__":
    main()