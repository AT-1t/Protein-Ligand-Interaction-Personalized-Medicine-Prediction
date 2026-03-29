# ===============
# Checkpoint-1 Step-3
# Ligand Features and Final Feature Table (chunked version)
# =================

import os
import warnings 
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs #represent loelcules, compares molecules
from rdkit.Chem import Descriptors #aculates molecular properties
from rdkit import RDLogger # this is stop warning spam in rdkit
from rdkit.Chem import rdFingerprintGenerator #generates finger prints 2D turned into binary
from tqdm import tqdm #progress bar on terminal

warnings.filterwarnings("ignore") #it will ignore basic warning messages and move along the program.
RDLogger.DisableLog('rdApp.*') #it will stop warnings coming in terminal

# ============
# Step-3 Data Output Previous data information
# ============

Data_directory = "New_checkpoint_1_data_here"
Checkpoint_1_step_2_input_path =os.path.join(Data_directory,"checkpoint_1_step_2_alignment_features_data.csv")
Checkpoint_1_step_3_table_path = os.path.join(Data_directory, "checkpoint_1_step_3_No_smiles_spam_feature_table_data.csv")

Fingerprint_bits = 128
Chunksize = 10000 # this is ram safe

# =================
# Smiles string clean-up
# ======================
#This is a new addtion to previous step-3 file to stop rdkit spam
def cleaned_smiles(smiles):
    if pd.isna(smiles):
        return None
    smile_string =str(smiles).strip().split()[0] #it is going to stop that trainilig [..] stuff.
    return smile_string

def valid_molecules_from_smiles(smiles):
    if smiles is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return mol
# ==================
# Ligand Finger Print
# ====================
def ligand_fingerprint(smiles):
    #mol for molecule
    mol = Chem.MolFromSmiles(smiles)
    numpy_arr = np.zeros(Fingerprint_bits, dtype=int) # dtype 

    if mol is not None:
        #it looks up 2 bonds away from each atom ,neighbor layers, to see local structural patterns.
        generator =rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=Fingerprint_bits)
        #molecular finger print 2d image to binary encode
        mol_finger_p = generator.GetFingerprint(mol)
        DataStructs.ConvertToNumpyArray(mol_finger_p, numpy_arr)
    return numpy_arr

# =================
# Ligand Description
# ==================
def ligand_des(smiles):
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return [np.nan, np.nan, np.nan, np.nan]
    #Provides Molecular Features
    molec_wt = Descriptors.MolWt(mol)
    #LogP --> Octanol water partition coeffecient to measure hyrophobic or lipohilic.
    #it is important for predicting memnrane permeability and drug absorption.
    logP =Descriptors.MolLogP(mol)
    hydrogen_donor = Descriptors.NumHDonors(mol)
    hydrogen_acceptor = Descriptors.NumHAcceptors(mol)

    return [molec_wt, logP, hydrogen_donor, hydrogen_acceptor]

# ===========
# Main
# ===========

def main():
    if not os.path.exists(Checkpoint_1_step_2_input_path):
        print("Step 2 file isn't found")
        return
    
    datafile = pd.read_csv(Checkpoint_1_step_2_input_path)
    print("Load check point -1 step2 data shape:", datafile.shape)

    if datafile.empty:
        print("Error step 2 input is empty")
        return
    
    #required columns so code fails safely before before the KeyError
    required_columns = [ 
        "smiles",
        "target_name",
        "log_affinity",
        "closest_kinase_family",
        "needleman_wunsch_score",
        "normalized_nw_score",
        "sequence_similarity",
        "mean_nw_score",
        "mean_normalized_nw_score",
        "mean_sequence_similarity",
        "family_reference_count"
    ]
    
    missing_col = [col for col in required_columns if col not in datafile.columns]
    if missing_col:
        print("Misiing columns:", missing_col)
        return

    datafile["smiles_clean"] = datafile["smiles"].apply(cleaned_smiles)
      #datafile["molecule"] =datafile["smiles_clean"].apply(valid_molecules_from_smiles)
    #this is to clean if there is any step3 file exist it delete it
    if os.path.exists(Checkpoint_1_step_3_table_path):
        os.remove(Checkpoint_1_step_3_table_path)
    
    output_rows_list = []
    chunk_count = 0
    
    #it loops through to data to create dictionary from each row
    #it computes molecular features ,fp, descp from smiles  
    #adds in the dataframe we created in step-2 
    for _, row in tqdm(datafile.iterrows(), total=len(datafile)):
        fp = ligand_fingerprint(row["smiles_clean"])
        descriptors = ligand_des(row["smiles_clean"])

        row_dict = {
            "target_name":row["target_name"],
            "uniprot_id": row["uniprot_id"] if "uniprot_id" in row.index else np.nan,
            "is_egfr": row ["is_egfr"] if "is_egfr" in row.index else False,
            "log_affinity": row["log_affinity"],

            "closest_kinase_family":row["closest_kinase_family"],
            "needleman_wunsch_score":row["needleman_wunsch_score"],
            "normalized_nw_score":row["normalized_nw_score"],
            "sequence_similarity":row["sequence_similarity"],
            
            "mean_nw_score": row["mean_nw_score"],
            "mean_normalized_nw_score": row["mean_normalized_nw_score"],
            "mean_sequence_similarity": row["mean_sequence_similarity"],
            "family_reference_count":row["family_reference_count"],

            "molec_wt":descriptors[0],
            "logP":descriptors[1],
            "hydrogen_donor":descriptors[2],
            "hydrogen_acceptor":descriptors[3]
            
        }

        for i in range(len(fp)):
            row_dict[f"fp_{i}"] = fp[i]
        output_rows_list.append(row_dict)

        ## chcunk saving for memeory usage
        if len(output_rows_list) >= Chunksize:
            chunk_data_df = pd.DataFrame(output_rows_list)
            chunk_data_df = chunk_data_df.dropna(subset=[
                "log_affinity",
                "normalized_nw_score",
                "sequence_similarity",
            ])
            
            if chunk_count == 0 : 
                chunk_data_df.to_csv(Checkpoint_1_step_3_table_path, index=False)
            else:
                chunk_data_df.to_csv(Checkpoint_1_step_3_table_path, mode ='a', header=False, index=False) #a = append do not overwrite the header
            
            output_rows_list = []
            chunk_count +=1


    ### saving in chuncks so it doesn't take so much ram and memeory
    #this part looks repeated but it does save the REMAINING rows that didn't reach the chunksize.
    #So no partial batch lost .
    if len(output_rows_list) > 0:
        chunk_data_df = pd.DataFrame(output_rows_list)
        chunk_data_df = chunk_data_df.dropna(subset = [
            "log_affinity",
            "normalized_nw_score",
            "sequence_similarity",
        ])
            
           
        if chunk_count == 0:
            chunk_data_df.to_csv(Checkpoint_1_step_3_table_path, index=False)
        else:
            chunk_data_df.to_csv(Checkpoint_1_step_3_table_path, mode='a', header=False, index=False)           

    print("Check point-1 step3 table saved in:", Checkpoint_1_step_3_table_path)

if __name__ == "__main__":
    main()




         
    