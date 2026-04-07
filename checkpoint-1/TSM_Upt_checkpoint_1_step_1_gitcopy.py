# =========
# Checkpoint -1 Step-1
# Downloads Bindingdb kinase data
# Cleans the data, maps and filters it
# Goal : Keep kinase realated rows 
# it currently has EGFR rows for later use as test data for transfer learning.



import os
import re
import zipfile
import warnings
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm 

warnings.filterwarnings("ignore")

#URL to download the data
Data_directory = "New_checkpoint_1_data_here"
os.makedirs(Data_directory, exist_ok=True)

#The bindingdb url works even if it is open on safari.
#I checked it it downloads the zip file right away.
#if you guys want to check it up,Quit downlaod immediately
#it is a large file I don't recommend dowloading on your local desktop.

#This is going to save the zip file in the "New_checkpoint_1_data_here" folder in the server.
BINDINGDB_URL = "https://bindingdb.org/rwd/bind/downloads/BindingDB_All_202602_tsv.zip"
Zip_path = os.path.join(Data_directory, "BindingDB_All_202602_tab_sep_val.zip")
Chckp_1_step1_output_path =os.path.join(Data_directory, "checkpoint_1_step_1_kinase_filtered_data.csv")

#This is going to read the data 50,000 rows at a time
#Better than loading everything into memeory all at once,data is very large like millions of rows.
CHUNKSIZE = 50000
EGFR_UNIPROT_ID = "P00533"
#this will be useful for test in step4

# ========
# Downloading the file via URL and extracting kinase data
# ========

def download_file(url:str, outputfile_path:str) -> None:
    if os.path.exists(outputfile_path):
        print("File already created:", outputfile_path)
        return
    print("Processing download from Bindingdb:", url)
    #this line starts downloading the file in pieces so it doesn't crush.
    ##stream=True dowload in chunks wait 120 seconds for response before stopping with an error.
    response = requests.get(url, stream=True, timeout=120) 
    #this line gives an error if the download fails.
    response.raise_for_status()
    
    #wb --> write in binary code
    #Binding db is zip file is in binary format
    #then we extract and then do tab seperation and then pandas then the sequence will show up in the dataframe.
    #Bindingdb zip file(Binary format) -->extract to tsv(text-tab seprated values) -->pandas -->sequences
    with open(outputfile_path, "wb") as f:
        #this line reads the file in chunksof 1MB for 1024 by 1024 bites.
        for _ in map(lambda chunk: f.write(chunk) if chunk else None, response.iter_content(1024 * 1024)):
            pass          

        # may caused the memory crash 
        # list(map(lambda chunk: f.write(chunk) if chunk else None, response.iter_content(1024 * 1024)))

    #this is looping perpeously to read the lines in chunks it doesn't dump the data all at once.

def extract_tab_sep_val(zip_path: str, output_dir: str) -> str:
    with zipfile.ZipFile(zip_path, "r") as zf:
        tab_sep_val_files =[f for f in zf.namelist() if f.endswith(".tsv")]

        if not tab_sep_val_files:
            raise ValueError("There is no .tsv file found on the zip")
            
        tab_sep_val_name = tab_sep_val_files[0]
        tab_output_file_path = os.path.join(output_dir, os.path.basename(tab_sep_val_name))

        if not os.path.exists(tab_output_file_path):
            print("Extracting:", tab_sep_val_name)
            zf.extract(tab_sep_val_name, output_dir)

            previous_path = os.path.join(output_dir, tab_sep_val_name)
            if previous_path != tab_output_file_path:
                os.replace(previous_path, tab_output_file_path)
        return tab_output_file_path
    
# ==========
# Helper functions
# ===========

def parse_number(x):
    #Checkes if the values missing and returns nan if it is.
    if pd.isna(x):
        return np.nan
    #strip removes extra stpaces at thebegining and the end.
    x = str(x).strip()
    x = re.sub(r"[<>=~]+", "", x) 
    #regex saying remove the following symbols from the string
    #once the data checked and formatted (cleaned) is floats it.
    try:
        return float(x)
    except Exception:
        return np.nan #if this function fails it just returns  np.nan so it doesn't crush.

#This function going to find column names and keywords 
#By making it case in-senstive via .lower() 
def column_finder(columns, keywords_list):
    for keys in keywords_list:
        for col in columns:
            if keys.lower() in col.lower():
                return col
    return None

def columns_selections(df: pd.DataFrame) ->pd.DataFrame:
    columns = df.columns.tolist() #list is easier to loop through via col finder.

    #This is true style Uniprot columns style
    #previous code had to debug becuase of the uniprot styles weren't matched.
    #I also misspelled the uniprot sometimes 
    mapping_pos_proteins = {
        "target_name": [
            "Target Name",
            "Target",
            "Protein Name",
        ],
        "smiles": [
            "Ligand SMILES",
            "SMILES",
            "Canonical SMILES",
        ],
        "ki": [
            "Ki (nM)",
            "Ki",
        ],
        "kd" :[
            "Kd (nM)",
            "Kd",
        ],
        "ic50":[
            "IC50 (nM)",
            "IC50",
        ],
        "uniprot_id": [
            "UniProt (SwissProt) Primary ID of Target Chain 1",
            "Primary Uniprot Accession",
            "Uniprot ID",
            "Uniprot ID",
            "Primary ID",
            "SwissProt ID",
            "Swiss-Prot ID",
            "UniProt",
        ],
        "sequence": [
            "BindingDB Target Chain Sequence 1",
            "Target Chain Sequence 1",
            "Chain Sequence",
            "Sequence 1",
            "Amino Acid Sequence",
            "Target Sequence",
            "Sequence",
        ],
    }

    found_mapping_pos = dict(map(lambda item:(item[0],column_finder(columns,item[1])), mapping_pos_proteins.items()))
    print("Columns found in mapping:", found_mapping_pos)
    
    #v -> value orginal name id bingindb dataset k is cleaned name 
    #rev-flips the dictionary so we can rename 
    rev_mapping = {v: k for k, v in found_mapping_pos.items() if v is not None}
    missing_required_val = [ c for c in ["target_name", "smiles", "sequence"] if found_mapping_pos[c] is None]
    
    if missing_required_val:
        raise ValueError(f"Missing the rquired columns after the mapping: {missing_required_val}")
    
    df = df[list(rev_mapping.keys())].rename(columns=rev_mapping)
    
    list(map(lambda c: df.__setitem__(c, df[c].apply(parse_number)) if c in df.columns else None, ["ki", "kd", "ic50"]))

    if "uniprot_id" in df.columns:
        df["uniprot_id"] = df["uniprot_id"].astype(str).str.strip().str.upper()
    return df

#this is ging to check ki,kd ic50 values and return it if it exist.
#it picks whiever is availabe to represent binding affinity
def affinity_val(row):
    return next(map(lambda c: row[c], filter(lambda c: c in row.index and pd.notna(row[c]), ["kd", "ki", "ic50"])), np.nan)
  #np.nan workds better with pandas then just None.
#Ki:inhibition constant  lower ki stonger bining better ingibitor
#Kd: dissociation constant measure ligand bining lower kd tigheter bingind
#IC50: half maximal  inhibitory concetration, concentration needed to reduce activity by 50%
#Lower IC50 --> more potent drug, so we can use less of the drug to achieve the same effect.

#creates an affinity columns drops missing values and keeps only positive values
#then converts them into log scale
def add_affinity_col(df: pd.DataFrame) -> pd.DataFrame:
    df["affinity"] = df.apply(affinity_val, axis=1)
    df = df.dropna(subset=["affinity"])
    df = df[df["affinity"] > 0].copy()
    df["log_affinity"] = -np.log10(df["affinity"] * 1e-9) #pKi-pKd-PIC50 standard scaleable higher numbers stronger binding.
    return df
#this is going to help in step-4 with the trainin and test data seperation  
def egfr_rows_marked(df: pd.DataFrame) -> pd.DataFrame:
    if "uniprot_id" not in df.columns:
        df["is_egfr"] = False
        return df 
    #this is to make the test data easily seperable
    df["is_egfr"] = df["uniprot_id"].astype(str).str.contains(EGFR_UNIPROT_ID, na=False)
    return df

def kinase_and_egfr_rows_indata(df: pd.DataFrame) -> pd.DataFrame:
    """
    This part is going to keep rows that are kinase related by target_name,
    and also explicitly keep egfr rows by uniprot.
    """
    target_data = df["target_name"].astype(str).str.contains("kinase", case=False, na=False)
    if "uniprot_id" in df.columns:
        egfr_data = df["uniprot_id"].astype(str).str.contains(EGFR_UNIPROT_ID, na=False) #astype converts to string type str.contains pd dataframe
    else:
        egfr_data = pd.Series(False, index=df.index)
    df = df[target_data | egfr_data].copy()
    return df

def rows_clean_up(df: pd.DataFrame) -> pd.DataFrame:
    #checks the columns exist
    required_cols = [c for c in ["target_name", "smiles", "sequence"] if c in df.columns]
    df = df.dropna(subset=required_cols).copy() #removes Nan values

    list(map(lambda c: df.__setitem__(c, df[c].astype(str).str.strip()) if c in df.columns else None, ["target_name", "smiles", "sequence"])) #removes extra spaces
    df = df[df["target_name"] != ""] #removes empty strings from the target name column
    df = df[df["smiles"] != ""]
    df = df[df["sequence"] != ""]#removes empty strings from the sequence column
    return df
#it process one chunk at a time combining is at chunkloading 
def process_chunks(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = columns_selections(chunk)
    chunk = rows_clean_up(chunk)
    chunk= add_affinity_col(chunk)
    chunk= egfr_rows_marked(chunk)
    chunk= kinase_and_egfr_rows_indata(chunk)
    return chunk

def safely_process_chunks(chunk):
    try:
        processed = process_chunks(chunk)
        return processed if not processed.empty else None
    except Exception as e:
        print("Chunks skipped data due to error", e)
        return None
def chunkloading(path: str) -> pd.DataFrame:
    #chunks_output = []
    #CSV is better than ppickle now 
    #needleman wunsch and smith waterman needs the actual sequences 
    #I keep adding bew features and I have to be able to read the ouput to be able to debug.
    #tqdm progress bar to see the work in terminal
    chunks_output = list(filter(lambda x: x is not None, map(lambda chunk: safely_process_chunks(chunk), tqdm(pd.read_csv(path, sep="\t", chunksize=CHUNKSIZE)))))
    if not chunks_output:
            print("Warning: no data after filering.") #hope not
            return pd.DataFrame()
    return pd.concat(chunks_output, ignore_index=True) #concat joins multiple dataframe vertically(stacks in rows.)
    
# ====
# MAIN
# =====
def main():
    download_file(BINDINGDB_URL, Zip_path)
    tab_sep_path = extract_tab_sep_val(Zip_path, Data_directory)

    datafile = chunkloading(tab_sep_path)
    print("Final Shape:", datafile.shape)

    if datafile.empty:
        print("Error no available data after Step-1.")
        return
    if "is_egfr" in datafile.columns:
        print("EGFR rows found:", int (datafile["is_egfr"].sum()))
        print("Non EGFR rows found:", int((~datafile["is_egfr"]).sum()))
    if "target_name" in datafile.columns:
        print("Sample target names:")
        print(datafile["target_name"].head(10).tolist())
    if "uniprot_id" in datafile.columns:
        print("Sample Uniprot IDs:")
        print(datafile["uniprot_id"].dropna().head(10).tolist())
    datafile.to_csv(Chckp_1_step1_output_path , index=False)
    print("Step 1 output saved in",Chckp_1_step1_output_path)

if __name__ == "__main__":
    main()

        








