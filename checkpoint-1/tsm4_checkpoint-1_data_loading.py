# ==============================================
# Checkpoint-1
# Protein Ligand Binding Prediction( Regression)
#================================================

#The script is going to accomplish following tasks
# 1. Downloads the data from BindingDB
# 2. Cleans up the data 
# 3. Targets kinease proteins
# 4. Compute protein allignment feautures:
#  - Needleman-Wunsch (global alignment) as Markus suggested
#  - Smith-waterman (local allignment) as Markus suggested
# 6. Eexutes ligand features from SMILES
# 7. Build ML prediction
# 8. Trains on KNN regressor
# 9. Saves outputs so we do not need to re-runeverything 
# for Chckpoint-2 and CHeckpoint -3


# ==========
# Libraries
# ===========

import os #check files already exist
import re #cleans numeric strings 
import zipfile # unzip binding db data
import warnings #hides uncessary warnings we had this module-1 or 2
import requests #d wolaods files from web
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  #imports thre progress bar
from Bio.Align import PairwiseAligner
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from rdkit.Chem import rdFingerprintGenerator
warnings.filterwarnings("ignore")



# Step 1 Download Settings 
#This is going to create a checkpooint-1 directory to save the data files.
Data_file_dir ="checkpoint-1_data_loaded"
os.makedirs(Data_file_dir, exist_ok=True)

Bindingdb_tab_sep_val_url ="https://www.bindingdb.org/rwd/bind/downloads/BindingDB_All_202602_tsv.zip"
Tab_sep_val_zip_path = os.path.join(Data_file_dir, "BindingDB_All_202603_tab-sep-val.zip")


EGFR_Uniprot_id = "P00533"
Fingerpoint_bits =128
K_neighbors = 7
Chunk_size = 50000

#Downloading the files and extracting them

def file_download(url, outputfile_path):
    if os.path.exists(outputfile_path): #exists 
        print("File exists in:", outputfile_path)
        return
    print("Processing download:", url)
    r = requests.get(url, stream= True, timeout=120) #requests with s
    r.raise_for_status() # downloading status?

    with open (outputfile_path, "wb") as f: # write in binary mode
        for chunk in r.iter_content(1024*1024): #M chunkc  1024 by 1024 is std for avoding the downlaod the whole file into memory at once.
            if chunk:
                f.write(chunk) # writes each chuncks for file,skiiping empty chcunckc, so it saves the download piece by peice


def extract_tab_sep_val(zip_path, output_dir):
    with zipfile.ZipFile(zip_path, "r") as zf:
        tab_sep_val_files = [f for f in zf.namelist() if f.endswith(".tsv")]

        if len(tab_sep_val_files) == 0:
            raise ValueError("No tsv file inside the zip file.")
        
        name_tsv = tab_sep_val_files[0]
        output_file_path =os.path.join(output_dir, os.path.basename(name_tsv))

        if not os.path.exists(output_file_path):
            print("Extracting:", name_tsv)
            zf.extract(name_tsv, output_dir) # 
            
            old_path = os.path.join(output_dir, name_tsv)
            if old_path!= output_file_path:
                os.replace(old_path, output_file_path)
        return output_file_path
    

#Clean up the data

def parse_num(x):
    if pd.isna(x):
        return np.nan # is it saying if x panda datframe isn't there return nan
    x = str(x)
    x = re.sub(r"[<>=~]+", "", x) #regex it removes any  <, > ,  = , ~ from the string.
    try:
        return float(x)
    except:
        return np.nan
def select_columns(df):
    columns = df.columns

    def find_k(k):
        for c in columns:
            if k.lower() in c.lower():
                return c
        return None
        
    mapping = {
        "target_name": find_k("Target Name"),
        "smiles": find_k("SMILES"), 
        "ki": find_k("Ki"),
        "kd": find_k("Kd"),
        "ic50":find_k("IC50"),
        "uniprot_id": find_k("UniProt") or find_k("Primary ID") or find_k("Uniprot ID"),
        "sequence" :find_k("Sequence") or find_k("Chain Sequence") or find_k("Sequence 1")
    }
    #this is the debug uniprot id 
    
    print("Column mapping found:", mapping)

    if mapping["uniprot_id"] is None:
         print("Warning uniporot_id column isn't found")

    mapping = {v: k for k, v in mapping.items() if v }
    df = df[list(mapping.keys())].rename(columns=mapping)

    for c in ["ki", "kd", "ic50"]:
        if c in df:
            df[c] = df[c].apply(parse_num)
    return df
        
def affinity_val(r):
        if pd.notna(r.get("kd")):
            return r["kd"]
        if pd.notna(r.get("ki")):
            return r["ki"]
        if pd.notna(r.get("ic50")):
            return r["ic50"]
        return np.nan

def kinase_filteration(datafile):
    datafile = datafile[datafile["target_name"].str.contains("kinase|EGFR", case= False, na= False)]
    #this is to be safe in dropna
    required_columns = ["smiles", "sequence", "uniprot_id"]
    existing_columns = [ c for c in required_columns if c in datafile.columns]
    datafile = datafile.dropna(subset = existing_columns)

    datafile["affinity"] = datafile.apply(affinity_val, axis=1)
    datafile = datafile.dropna(subset=["affinity"])

    datafile["log_affinity"] = -np.log10(datafile["affinity"] *1e-9)
    return datafile


def chunk_loading(path):
        output_chunk = []
        for chunk in tqdm(pd.read_csv(path, sep="\t", chunksize=Chunk_size)):
            chunk = select_columns(chunk)
            chunk = kinase_filteration(chunk)
            if not chunk.empty:
                output_chunk.append(chunk)
        if len(output_chunk)==0:
             print("Warning no data filtering")
             return pd.DataFrame()
        
        return pd.concat(output_chunk, ignore_index=True)
       
        

# =========
# Features 
# ========

def ligand_fingerp(smiles):
        mol = Chem.MolFromSmiles(smiles)
        numpy_arr = np.zeros (Fingerpoint_bits, dtype = int)

        if mol:
            gen  = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=Fingerpoint_bits)
            mol_finger_p = gen.GetFingerprint(mol)
            DataStructs.ConvertToNumpyArray(mol_finger_p, numpy_arr)

        return numpy_arr

# =====
# MAIN 
# =====

def main():
        file_download(Bindingdb_tab_sep_val_url, Tab_sep_val_zip_path)
        tab_sep_val_path  = extract_tab_sep_val(Tab_sep_val_zip_path, Data_file_dir)
        
       
        datafile = chunk_loading(tab_sep_val_path)
        print("Final Shape of the chunks:", datafile.shape)

        if datafile.empty:
             print("error no data available for training")
             return

        # early model creation for demo
        X = []
        for smile in datafile["smiles"].head(1000):
             X.append(ligand_fingerp(smile))

        X = np.array(X)
        y = datafile ["log_affinity"].values[:1000] #use for 1000 to match X

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        chp_1_data_load_model = KNeighborsRegressor(n_neighbors=K_neighbors)
        chp_1_data_load_model.fit(X,y)

        print("Checkpoint-1 Data Loading Model training succesfully")

if __name__ == "__main__":
     main()
        









