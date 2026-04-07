#Checkpoint-2 Step 3

#importing the necessary libraries
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram

#Setting up directory, input and output
Data_directory = "Checkpoint_2_data"
Checkpoint_2_step_3_input_path = os.path.join(Data_directory,"checkpoint_2_step_2_deepseq_embedding.csv") #input data  is output data from checkpoint 2 step 2
Checkpoint_2_step_3_output_path = os.path.join(Data_directory, "checkpoint_2_step_3_family_classification.csv") #output data
Checkpoint_2_step_3_family_plot = os.path.join(Data_directory, "checkpoint_2_family_tree_plot.png") #plot of family trees for visualization

def main():
    #File Validations and Reading
    #============================================
    #Checking that input file actually exists
    if not os.path.exists(Checkpoint_2_step_3_input_path):
        print("NO Step 3 Input File Was Found!")
        return
    
    #Reading input file
    df = pd.read_csv(Checkpoint_2_step_3_input_path)
    print("Data Loaded")
    #Making sure input file is not empty (there is data present)
    if df.empty:
        print("Step 3 Input File is Empty!")
        return
    #Checking that the label column is present for analysis
    if "closest_kinase_family" not in df.columns:
        raise ValueError("Missing Required Column! - closest_kinase_family")
    
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
    
    #Data Preparations
    #=================================================
    #list of non feature columns - these should not be used as model features
    non_feature_columns = ["target_name", "smiles", "sequence", "sequence_clean", "uniprot_id", "closest_kinase_family", "affinity", "log_affinity", "is_egfr"]
    #list of feature columns - alignment and embedding selection
    embedding_columns = [c for c in df.columns if c.startswith("emb_")]
    aa_columns = [c for c in df.columns if c.startswith("aa_")]

    phychem_columns = [
        "seq_length", 
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

    assay_columns = ["ki", "kd", "ic50", "affinity", "log_affinity"] #not great for cold-start so considering removing!!!!
    
    feature_columns = embedding_columns + aa_columns + phychem_columns + similarity_columns + assay_columns
    #creating a feature matrix from the selected feature columns
    X_df = df[feature_columns].copy()

    #converting all features to numeric
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    #dropping columns that are entirely NaN and are not useful
    X_df = X_df.dropna(axis=1, how="all")
    #replacing any infinite values with NaN
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    #filling any columns with NaN, median
    X_df = X_df.fillna(X_df.median(numeric_only=True))
    #printing the number of usable features
    print("Feature Count:", X_df.shape[1])

    #Grouping of duplicate protein rows if protein ids exist
    #=======================================================
    #columns that define the protein identity - metadata for grouping together                       
    group_columns = [c for c in ["target_name", "uniprot_id", "closest_kinase_family", "is_egfr"] if c in df.columns]
    #metadata columns that are being kept alongside averaged numeric features
    meta_columns = group_columns
    #combining metadata with cleaned numeric features
    no_repeat_df = pd.concat([df[meta_columns].reset_index(drop=True), X_df.reset_index(drop=True)], axis=1)
    #reducing duplicate proteins by averaging numeric features, grouped by group_columns
    end_df = (no_repeat_df.groupby(group_columns, dropna=False).mean(numeric_only=True).reset_index())
    print("After reducing duplicated proteins", end_df.shape)
    #saving output to csv for Step 4
    end_df.to_csv(Checkpoint_2_step_3_output_path, index=False)


    #Preparation for Dendrogram Visualization
    #===========================================
    #selecting label columns useful for plotting
    label_columns = [c for c in ["target_name", "uniprot_id", "closest_kinase_family"] if c in end_df.columns]
    
    #Creating labels for each leaf with family and protein id if id exists
    if "uniprot_id" in end_df.columns:
        labels = (end_df["closest_kinase_family"].astype(str) + " | " + end_df["uniprot_id"].astype(str)).tolist()
    else:
        #if not use family names
        labels = end_df["closest_kinase_family"].astype(str).tolist()
    
    #Generating the Dendrogram
    #===================================================================
    plt.figure(figsize=(10,10))
    #selecting numeric feature columns for clustering
    num_columns = [c for c in end_df.columns if c not in label_columns]
    num_columns = [c for c in num_columns if c != "is_egfr"]
    #converting feature table to numpy array
    X_plot = end_df[num_columns].values
    #Performing hierarchical clustering
    z = linkage(X_plot, method="ward")

    #Plotting the Dendrogram
    dendrogram(z, labels=labels, leaf_rotation=90, leaf_font_size=11)
    plt.title("Protein Family Tree")
    plt.xlabel("Proteins")
    plt.ylabel("Distance")
    plt.savefig(Checkpoint_2_step_3_family_plot, dpi=400, bbox_inches="tight") #saving dendrogram to server directory
    print("Dendrogram has been saved to directory as a png!")
    
    #Yay! Celebrating that step 3 was able to complete
    print("Step 3 is Finished! Pursue Step 4")

if __name__ == "__main__":
    main()