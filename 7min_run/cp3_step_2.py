""" 

This applies a regression ann model scalers, no softamx is neccessary.


"""

import os 
import re
import gc
import math
import pickle 
import warnings
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from cp1_step_4 import KinaseEgfrWeightedANN

warnings.filterwarnings("ignore")

#rng is better than random.randt
#becuase it uses fixed radnom seed for preprodicible Ann weight initializtion and drop filtet
#np.random.radn creates an uncontorilled generation


class LayerDense:
    def __init__(self, n_inputs, n_neurons, rng, l2_lambda=0.0):
        self.weights = (rng.randn(n_inputs, n_neurons) * np.sqrt(2.0/n_inputs)).astype(np.float32)
        self.biases = np.zeros((1,n_neurons), dtype =np.float32)
        self.l2_lambda = l2_lambda

 #class code direction application
 #part of the code is taken from ANNIII.ipynb 
 #we run into circularization of the data problem
 #i didn't know about it until last week's lecture that data canbe curcilzried so I had to add the following section to make
 #the fsuion. more and so due to time contraines I used the class code 

         
    def forward(self, inputs):
        self.output  = np.dot(inputs, self.weights) + self.biases
        self.inputs  = inputs

    def backward(self, dvalues):
        #gradients
        self.dweights = np.dot(self.inputs.T, dvalues) + (2.0 * self.l2_lambda * self.weights)
        self.dbiases  = np.sum(dvalues, axis = 0, keepdims = True)
        self.dinputs  = np.dot(dvalues, self.weights.T)
Layer_Dense = LayerDense # I will change this later for the naming issue in the previous file ann doesn't link.
class Activation_ReLU():
    
    def forward(self, inputs):
        self.output  = np.maximum(0, inputs)
        self.inputs  = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0#ReLU derivative
ReLU = Activation_ReLU # I will change this later for the naming issue in the previous file ann doesn't link
# ========
#https://github.com/Sentdex/nnfs_book/blob/main/Chapter_17/Ch17_Final.py
#I had got the binary_filtering from Sendex github repository he used is as binary_mask in his class layer_dense
# this is going to randomly turn off selected neurons during training 
#so it will prevent overfiiting  and force ann to learn more generalized deature patterns
#this is better than mememorxing the data.
#we leanred dropout in the class it was for stohastic neron masking 
#this is to create masking matrix via binary filtering
class Layer_Drop_out:
    def __init__(self, rate,rng):
        self.rate =rate
        self.rng = rng

    def forward(self, inputs, training=True):
        self.inputs = inputs 

        if training:
            self.binary_filtering = (self.rng.rand(*inputs.shape) > self.rate).astype(np.float32)
            self.output = inputs * self.binary_filtering /(1.0 - self.rate)

        else: 
            self.binary_filtering = None 
            self.output = inputs
    def backward(self, dvalues):
        if self.binary_filtering is None:
            self.dinputs = dvalues
        else:
            self.dinputs = dvalues * self.binary_filtering / (1.0 - self.rate)
Layer_Drop = Layer_Drop_out
# =======    
#class style code from ANNIII.ipynb adjusted to this project 

class Optimizer_SGD:
    #class version was initializing with a default learning rate of 0.01
    def __init__(self, learning_rate = 0.0005, decay = 0, momentum = 0.9):
        self.learning_rate         = learning_rate
        self.current_learning_rate = learning_rate
        self.decay                 = decay
        self.iterations            = 0
        self.momentum              = momentum
    
    def update_params(self, layer):
        
        #if we use momentum
        if self.momentum:
            
            #check if layer has attribute "momentum"
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights, dtype=np.float32) #dtype 
                layer.bias_momentums   = np.zeros_like(layer.biases, dtype=np.float32)
                
            #now the momentum parts
            #weights moved to oppsoite direction of grad to lower loss here.
            #previosu code had a dofferent section.
            weight_updates = self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            
            bias_updates = self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
            
        else:
            
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates   = -self.current_learning_rate * layer.dbiases
        
        layer.weights += weight_updates
        layer.biases  += bias_updates

class MSE_Loss:
    def forward(self, y_pred, y_true, sample_weights=None):
        y_true = y_true.reshape(-1,1).astype(np.float32)

        if sample_weights is None:
            return np.mean((y_pred -y_true) ** 2)
        
        sample_weights= sample_weights.reshape(-1,1).astype(np.float32)
        return np.sum(sample_weights * ((y_pred -y_true) **2)) / np.sum(sample_weights)

class SGradientD_Optimizer:
    def __init__ (self, learning_rate = 0.0005, momentum=0.9, min_learning_rate =0.00003):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.momentum = momentum
        self.min_learning_rate = min_learning_rate
    

class KinaseEgfrWeightedANN:

    def prediction(self, X):
        return self.forward(X, training=False).flatten().astype(np.float32)

    def forward(self, X, training=False):
        self.dense1.forward(X)
        self.relu1.forward(self.dense1.output)

        self.drop1.forward(
            self.relu1.output,
            training = training
        )

        self.dense2.forward(self.drop1.output)
        self.relu2.forward(self.dense2.output)

        self.drop2.forward(
            self.relu2.output,
            training=training
        )

        self.dense3.forward(self.drop2.output)
        self.relu3.forward(self.dense3.output)

        self.drop3.forward(
            self.relu3.output,
            training = training
        )

        self.out.forward(self.drop3.output)

        return self.out.output
EgfrregressorANN = KinaseEgfrWeightedANN # I will change this later naming issue in the previous file ann doesn't link 
    
  
        #predictions = self.forward(
       
           # training=False
       
        #return predictions.flatten().astype(np.float32)

# ====
# Paths
# =====

#I need to change the pkl named in the ann model 

BASE_DIRECTORY = "."
CP1_DIR = os.path.join(BASE_DIRECTORY, "New_checkpoint_1_data_here")
CP2_DIR = os.path.join(BASE_DIRECTORY, "Checkpoint_2_data")
CP3_DIR =os.path.join(BASE_DIRECTORY, "Checkpoint_3_data")
OUTPUT_DIRECTORY = os.path.join(BASE_DIRECTORY, "fused_data")

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

#they are in the base directory for the 7min run, it is in the server 7min_run folder.
CP3_PATIENT_PATH =os.path.join(BASE_DIRECTORY, "checkpoint_3_patient_mutation_data.csv")
CP1_STEP3_PATH = os.path.join(BASE_DIRECTORY, "checkpoint_1_step_3_No_smiles_spam_feature_table_data.csv")
CP2_PRED_PATH = os.path.join(BASE_DIRECTORY, "checkpoint_2_step_4_transfer_learning_family_predictions.csv")


#This is the new kinase update paths/ nedd to check the naming before run

MODEL_1_PATH = os.path.join(CP1_DIR, "ckp1_step4_model_1_egfr_only.pkl") #kinase_protein_egfr_weighted
MODEL_1_XSCALER_PATH = os.path.join(CP1_DIR, "ckp1_step4_model_1_egfr_only_scaler.pkl")
MODEL_1_YSCALER_PATH = os.path.join(CP1_DIR, "ckp1_step4_model_1_egfr_only_y_scaler.pkl")


OUTPUT_PATIENT_CLEAN = os.path.join(OUTPUT_DIRECTORY, " checkpoint3_patient_clean.parquet")
OUTPUT_CP1_EGFR = os.path.join(OUTPUT_DIRECTORY, "checkpoint1_kinase_weighted_feature_table.parquet")
OUTPUT_CP2_CLEAN =os.path.join(OUTPUT_DIRECTORY, "checkpoint2_predictions_clean.parquet")
OUTPUT_FUSION = os.path.join(OUTPUT_DIRECTORY, "checkpoint3_fusion_lightgbm_ready.parquet")



# =====
# Fusion
# ======
   
PATIENT_BIN_COLS = [
    "is_nonsynonymous",
    "has_exon19del",
    "has_L858R",
    "has_L861Q",
    "has_G719X",
    "has_exon20_alteration",
    "has_other_mutation",
    "is_compound_mutation",
    "is_egfr_hotspot",
]


PATIENT_NUM_COLS = [
    "mutation_count",
]

CP1_REQUIRED_COLS = [
    "target_name",
    "log_affinity",
    "closest_kinase_family",
    "needleman_wunsch_score",
    "normalized_nw_score",
    "mean_normalized_nw_score",
    "mean_sequence_similarity",
    "family_reference_count",
    "molec_wt",
    "logP",
    "hydrogen_donor",
    "hydrogen_acceptor",
]

CP2_POSSIBLE_KEY_COLS = ["uniprot_id", "target_name", "gene"]

FUSION_PAT_CHUNK_SIZE = 5


PATIENT_CAT_COLS = [
    "patient_id",
    "mutation",
    "EGFR_type",
    "cohort",
    "batch_domain",
    "gene",
    "egfr_hotspot_label",
    "mutation_class",
    "egfr_mutation_group",

]

FUSION_CAT_COLS =[
    "EGFR_type",
    "cohort",
    "batch_domain",
    "gene",
    "egfr_hotspot_label",
    "mutation_class",
    "egfr_mutation_group",
    "target_name",
    "uniprot_id",
    "closest_kinase_family",
    "true_family",
    "predicted_family",
    "ann_training_scope",
    "fusion_strategy"

]


#====
# Helper functions
# ====

#this part wwas neccsary for cleaning the patient data 
#I ran into errors

def clean_col_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        pd.Index(df.columns)
        .map(lambda x: str(x).strip())
        .map(lambda x: re.sub(r"\s+", "-", x))
    )
    return df


def remove_dup_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    return df.loc[:, ~df.columns.duplicated()]


def embeding_on_headers(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    header_rows = [
        idx
        for idx in df.index
        if sum(str(df.loc[idx, col]) == str(col) for col in df.columns) >= 2
    ]

    return df.drop(index=header_rows).copy()


def binary_num(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    numbered = s.map({
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
        "yes": 1,
        "no": 0,
        "y": 1,
        "n": 0,
    })

    numerical_data = pd.to_numeric(numbered.fillna(s), errors="coerce")
    return numerical_data.fillna(0).astype(np.int8)


def safe_numerical(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def merge_key_cleanup(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\s+", "", regex=True)
        .replace({"nan": np.nan, "None": np.nan, "": np.nan})
    )


def remove_list_organize(items):
    return list(dict.fromkeys(items))


def downline_numerical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    float_cols = df.select_dtypes(include=["float64"]).columns.tolist()
    int_cols = df.select_dtypes(include=["int64"]).columns.tolist()

    if float_cols:
        df[float_cols] = df[float_cols].apply(lambda s: pd.to_numeric(s, downcast="float"))

    if int_cols:
        df[int_cols] = df[int_cols].apply(lambda s: pd.to_numeric(s, downcast="integer"))

    return df


def pandas_cat(df: pd.DataFrame, category_cols) -> pd.DataFrame:
    df = df.copy()
    current_cols = [c for c in category_cols if c in df.columns]
    if current_cols:
        df[current_cols] = df[current_cols].apply(lambda s: s.astype("category"))
    return df


def patient_mutation_table(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_col_colnames(df)
    df = remove_dup_cols(df)
    df = embeding_on_headers(df)
    df = df.dropna(how="all").copy()

    if "patient_id" in df.columns:
        df["patient_id"] = merge_key_cleanup(df["patient_id"])

    if "mutation" in df.columns:
        df["mutation"] = df["mutation"].astype(str).str.strip()

    existing_binary = [c for c in PATIENT_BIN_COLS if c in df.columns]
    if existing_binary:
        df[existing_binary] = df[existing_binary].apply(binary_num)

    existing_numeric = [c for c in PATIENT_NUM_COLS if c in df.columns]
    if existing_numeric:
        df[existing_numeric] = df[existing_numeric].apply(safe_numerical).fillna(0)

    if "mutation" in df.columns and "mutation_count" not in df.columns:
        df["mutation_count"] = (
            df["mutation"]
            .fillna("")
            .map(lambda x: len([m for m in str(x).split() if m.strip() != ""]))
            .astype(np.int16)
        )

    return downline_numerical(df)

""" feature table table pred function applys the helper functions above to clean
fix, stablize and organize the data to make a mutaion based features for suing them later machine learning fusion model """

def prepare_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_col_colnames(df)
    df = remove_dup_cols(df)

    empty =list(filter(lambda c: c not in df.columns, CP1_REQUIRED_COLS))
    if empty:
        raise ValueError(f"checkpoint 1 step 3 misisng required columns:{empty}")
    
    #this is the changed part from previous egfr onluu model

    if "is_egfr" in df.columns:
        df["is_egfr"] = binary_num(df["is_egfr"])
    else:
        df["is_egfr"] = df["target_name"].astype(str).str.contains("EGFR|ERBB1", case= False, na= False).astype(np.int8) #mef
        #no space on the pipe don't forget

    if "uniprot_id" in df.columns:
        df["uniprot_id"] = merge_key_cleanup(df["uniprot_id"])

    if "target_name" in df.columns:
        df["uniprot_id"] = merge_key_cleanup(df["uniprot_id"])
    if "target_name" in df.columns:
        df["target_name"] = df["target_name"].astype(str).str.strip()
    
    sim_numerical_colums = list(filter(
        lambda c: c.startswith("fp_") or c in CP1_REQUIRED_COLS[1:],
        df.columns
    ))

    df[sim_numerical_colums] = df[sim_numerical_colums].apply(safe_numerical)

    model_num_baseline = [
        "normalized_nw_score",
        "sequence_similarity",
        "molec_wt",
        "logP",
        "hydrogen_donor",
        "hydrogen_acceptor",
    ]

    current_baseline = list(filter(lambda c : c in df.columns, model_num_baseline))
    df = df.dropna(subset=current_baseline).reset_index(drop=True)

    df["ligand_row_id"] = np.arange(len(df), dtype=np.int32)


    #new addtion for all inase data ann
#clears meta data so the fusion output is not interpated as experimental truth.
    df["ann_training_scope"] = "all_kinase_weighted_egfr_oversampled"
    df["is_proxy_affinity_score"] = np.int8(1) #val 1 stored as small integfer to save on memory
    df["fusion_strategy"] = "cartesian_pos_pat_generation"
    df["is_simulated_pair"] = np.int8(1) #it has to be mef, otheriwise server stops it

    df = downline_numerical(df)
    return df 
""" converts ann's y_scaler predition vals
back into ooriginal binidng afinity by suing y_scaler mean and standard devaition 

#reference 
"""
def y_scaler_inverse(raw_pred: np.ndarray, y_scaler_obj) -> np.ndarray:
    raw_pred = np.asarray(raw_pred). reshape(-1,1)

    if isinstance(y_scaler_obj, dict) and "y_mean" in y_scaler_obj and "y_std" in y_scaler_obj:
        return (raw_pred.reshape(-1) * float(y_scaler_obj["y_std"]) + float(y_scaler_obj["y_mean"]))
    
    #I should check it out with the scklaearn learn for this 

    if hasattr(y_scaler_obj, "inverse_transform"):
        return y_scaler_obj.inverse_transform(raw_pred).reshape(-1)
    
    return raw_pred.reshape(-1) *float(y_scaler_obj["std"])

def model_score(cp1_df: pd.DataFrame) -> pd.DataFrame:
    cp1_df = cp1_df.copy()

    feature_columns = [
        c
        for c in cp1_df.columns
        if c in [
            "normalized_nw_score",
            "sequence_similarity",
            "mean_nw_score",
            "mean_normalized_nw_score",
            "mean_sequence_similarity",
            "family_reference_count",
            "molec_wt",
            "logP",
            "hydrogen_donor",
            "hydrogen_acceptor",
            "needleman_wunsch_score",
            "closest_kinase_family",
        ]
        or c.startswith("fp_")
    ]
    feature_columns = remove_list_organize(feature_columns)

    if not (os.path.exists(MODEL_1_PATH) and os.path.exists(MODEL_1_XSCALER_PATH)):
        cp1_df["model_1_binding_pred"] = np.nan
        cp1_df["model_1_scoring_stats"] = "empty_model_or_scaler"
        return cp1_df

    X = cp1_df.loc[:, feature_columns].apply(safe_numerical).fillna(0).to_numpy(dtype=np.float32)

    try:
        with open(MODEL_1_PATH, "rb") as f:
            model = pickle.load(f)

        with open(MODEL_1_XSCALER_PATH, "rb") as f:
            ann_x_scaler = pickle.load(f)

        expected_num = getattr(ann_x_scaler, "n_features_in_", None)
        if expected_num is None:
            expected_num = getattr(ann_x_scaler, "num_features_in_", None)
        actual_num = X.shape[1]

        if expected_num is not None and actual_num != expected_num:
            print(
                f"Model 1 scoring skipped: X has {actual_num} features, "
                f"but scaler expects {expected_num} features."
            )
            cp1_df["model_1_binding_pred"] = np.nan
            cp1_df["model_1_scoring_stats"] = "mismatch_feat_count"
            return cp1_df

        X_scaled = ann_x_scaler.transform(X).astype(np.float32)
        X_scaled = np.nan_to_num(
            X_scaled,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)

        if hasattr(model, "prediction"):
            raw_pred = np.asarray(model.prediction(X_scaled)).reshape(-1, 1)
        elif hasattr(model, "predict"):
            raw_pred = np.asarray(model.predict(X_scaled)).reshape(-1, 1)
        elif isinstance(model, tuple):
            raise ValueError("The ANN pickle only contains parameter arrays, not the full model object")
        else:
            raise ValueError("Loaded model does not have prediction or predict")

        if os.path.exists(MODEL_1_YSCALER_PATH):
            with open(MODEL_1_YSCALER_PATH, "rb") as f:
                y_scaler = pickle.load(f)
            pred = y_scaler_inverse(raw_pred, y_scaler)
        else:
            pred = raw_pred.reshape(-1)

        cp1_df["model_1_binding_pred"] = pred.astype(np.float32)
        cp1_df["model_1_scoring_stats"] = "scored_with_all_kinase_weighted_egfr_ann"
#this added after getting model as nan 
#I had to add this exception code it was not linkng witht he ann model 
    except Exception as e:
        print("\nModel scoring error")
        print(type(e))
        print(e)
        import traceback
        traceback.print_exc()
        #print("Model 1 scoring skipped", e)
        cp1_df["model_1_binding_pred"] = np.nan
        cp1_df["model_1_scoring_stats"] = "scoring_failed"

    return downline_numerical(cp1_df)

def cp2_prediction_pred(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_col_colnames(df)
    df = remove_dup_cols(df)

    if "uniprot_id" in df.columns:
        df["uniprot_id"] = merge_key_cleanup(df["uniprot_id"])
    if "target_name" in df.columns:
        df["target_name"] = df["target_name"].astype(str).str.strip()
    if "gene" in df.columns:
        df["gene"] = df["gene"].astype(str).str.strip()

    possible_keys = [c for c in CP2_POSSIBLE_KEY_COLS if c in df.columns]
    if not possible_keys:
        raise ValueError("Could not find mergable key in CP2 predictions")

    key_merges = possible_keys[0]

    cols_to_keep_in_presence = [
        key_merges,
        "predicted_family",
        "prediction_confidence",
        "true_family",
        "closest_kinase_family",
        "is_egfr",
        "target_name",
        "uniprot_id",
        "gene",
    ]

    cols_to_keep = [c for c in cols_to_keep_in_presence if c in df.columns]
    cols_to_keep = remove_list_organize(cols_to_keep)

    possible_cols = [
        c for c in df.columns.tolist()
        if "prob" in c.lower() and c not in cols_to_keep
    ]

    final_columns = remove_list_organize(cols_to_keep + possible_cols)
    output_columns = df.loc[:, final_columns].copy()
    output_columns = remove_dup_cols(output_columns)

    if key_merges in output_columns.columns:
        output_columns = output_columns.dropna(subset=[key_merges]).reset_index(drop=True)
        output_columns = output_columns.drop_duplicates(subset=[key_merges]).copy()

    output_columns = pandas_cat(
        output_columns,
        ["target_name", "uniprot_id", "gene", "predicted_family", "true_family", "closest_kinase_family"]
    )

    return downline_numerical(output_columns)


# Reference: cross-join / cartesian product between pandas dataframes
# https://mkonrad.net/2016/04/16/cross-join-cartesian-product-between-pandas-dataframes.html

def cartesian_join(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    left = left.copy()
    right = right.copy()
    left["_tmpkey"] = 1
    right["_tmpkey"] = 1
    result = pd.merge(left, right, on="_tmpkey", how="inner").drop("_tmpkey", axis=1)
    return result


def last_feature_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = remove_dup_cols(df)

    object_columns = df.select_dtypes(include=["object"]).columns.tolist()
    patient_columns_num = [
        c for c in object_columns
        if c not in [
            "patient_id", "mutation_id", "EGFR_type", "cohort", "batch_domain",
            "gene", "egfr_hotspot_label", "mutation_class", "egfr_mutation_group",
            "target_name", "uniprot_id", "closest_kinase_family", "true_family",
            "predicted_family", "ann_training_scope", "fusion_strategy",
            "model_1_scoring_stats"
        ]
    ]

    if patient_columns_num:
        df[patient_columns_num] = df[patient_columns_num].apply(
            pd.to_numeric, errors="coerce"
        )

    category_columns = [
        c for c in FUSION_CAT_COLS + ["model_1_scoring_stats"]
        if c in df.columns
    ]
    if category_columns:
        df[category_columns] = df[category_columns].apply(lambda s: s.astype("category"))

    specific_linkages = [
        ("model_1_binding_pred", "is_egfr_hotspot", "proxy_model1_x_hotspot"),
        ("model_1_binding_pred", "mutation_count", "proxy_model1_x_mutcount"),
        ("model_1_binding_pred", "EGFR_RNA", "proxy_model1_x_egfr_rna"),
        ("model_1_binding_pred", "EGFR_PROTEIN", "proxy_model1_x_egfr_protein"),
    ]

    for i, j, output_col in specific_linkages:
        if i in df.columns and j in df.columns:
            df[output_col] = (
                pd.to_numeric(df[i], errors="coerce").fillna(0).astype(np.float32)
                * pd.to_numeric(df[j], errors="coerce").fillna(0).astype(np.float32)
            )

    df["is_simulated_pair"] = np.int8(1)
    df["is_proxy_affinity_score"] = np.int8(1)

    df = downline_numerical(df)
    return remove_dup_cols(df).reset_index(drop=True)


def creating_fusion_in_chunks(cp3: pd.DataFrame, cp1_fused: pd.DataFrame, output_path: str, chunk_size: int) -> None:
    total_patients = len(cp3)
    total_ligands = len(cp1_fused)
    total_rows = total_patients * total_ligands
    total_chunks = (total_patients + chunk_size - 1) // chunk_size

    print("creating the patient x ligand fusion table in chunks")
    print("Patient rows:", total_patients)
    print("Ligand_rows:", total_ligands)
    print("Expected fused rows:", total_rows)
    print("Patient chunk size:", chunk_size)
    print("total fusion chunks:", total_chunks)

    writer = None

    try:
        for chunk_idx, start in enumerate(range(0, total_patients, chunk_size), start=1):
            finish = min(start + chunk_size, total_patients)
            print(f"Building fusion chunk {chunk_idx}/{total_chunks} for patient rows {start}:{finish}")

            cp3_chunk = cp3.iloc[start:finish].copy()
            cp3_chunk = pandas_cat(cp3_chunk, PATIENT_CAT_COLS)

            fusion_chunk = cartesian_join(cp3_chunk, cp1_fused)
            fusion_chunk["fusion_row_id"] = np.arange(
                start * total_ligands,
                start * total_ligands + len(fusion_chunk),
                dtype=np.int64,
            )
            fusion_chunk = last_feature_cleanup(fusion_chunk)

            table = pa.Table.from_pandas(fusion_chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
            writer.write_table(table)

            print(f"Saved fusion chunk {chunk_idx}/{total_chunks} with shape: {fusion_chunk.shape}")
            del cp3_chunk
            del fusion_chunk
            del table
            gc.collect()

    finally:
        if writer is not None:
            writer.close()


def main():
    print("Loading and cleaning Checkpoint 3 patient data")

    cp3 = pd.read_csv(CP3_PATIENT_PATH)
    cp3 = patient_mutation_table(cp3)
    cp3.to_parquet(OUTPUT_PATIENT_CLEAN, index=False)
    print("Saved cleaned patient table:", OUTPUT_PATIENT_CLEAN)
    print("Patient clean shape:", cp3.shape)

    print("Loading and preparing checkpoint 1 step3 kinase feature table")
    cp1 = pd.read_csv(CP1_STEP3_PATH)
    cp1 = prepare_feature_table(cp1)
    cp1 = model_score(cp1)

    cp1["ann_training_scope"] = "all_kinase_weighted_egfr_oversampled"
    cp1["is_proxy_affinity_score"] = np.int8(1)
    cp1["fusion_strategy"] = "cartesian_candidate_generation"
    cp1["is_simulated_pair"] = np.int8(1)

    cp1 = remove_dup_cols(cp1)
    cp1 = pandas_cat(
        cp1,
        [
            "target_name",
            "uniprot_id",
            "closest_kinase_family",
            "ann_training_scope",
            "fusion_strategy",
            "model_1_scoring_stats",
        ],
    )
    cp1 = downline_numerical(cp1)
    cp1.to_parquet(OUTPUT_CP1_EGFR, index=False)
    print("Saved CP1 kinase weighted feature table:", OUTPUT_CP1_EGFR)
    print("CP1 kinase weighted shape:", cp1.shape)

    print("Loading checkpoint 2 step 4 predictions")
    cp2 = pd.read_csv(CP2_PRED_PATH)
    cp2_cleaned = cp2_prediction_pred(cp2)
    cp2_cleaned = remove_dup_cols(cp2_cleaned)
    cp2_cleaned.to_parquet(OUTPUT_CP2_CLEAN, index=False)
    print("Saved cleaned CP2 prediction:", OUTPUT_CP2_CLEAN)
    print("CP2 clean shape:", cp2_cleaned.shape)

    cp2_keys = [c for c in CP2_POSSIBLE_KEY_COLS if c in cp2_cleaned.columns]
    cp1_keys = [c for c in CP2_POSSIBLE_KEY_COLS if c in cp1.columns]
    keys_shared_btw = [c for c in cp1_keys if c in cp2_keys]

    if keys_shared_btw:
        key_merges = keys_shared_btw[0]
        print("Merging CP1 and CP2 on:", key_merges)
        cp1_fused = cp1.merge(cp2_cleaned, on=key_merges, how="left")
    else:
        print("No direct CP1 and CP2 merge key found.")
        cp2_first = cp2_cleaned.head(1).copy()
        cp1_fused = cartesian_join(cp1, cp2_first)

    cp1_fused = remove_dup_cols(cp1_fused)
    cp1_fused["ann_training_scope"] = "all_kinase_weighted_egfr_oversampled"
    cp1_fused["is_proxy_affinity_score"] = np.int8(1)
    cp1_fused["fusion_strategy"] = "cartesian_candidate_generation"
    cp1_fused["is_simulated_pair"] = np.int8(1)

    cp1_fused = pandas_cat(
        cp1_fused,
        [
            "target_name",
            "uniprot_id",
            "closest_kinase_family",
            "true_family",
            "predicted_family",
            "ann_training_scope",
            "fusion_strategy",
            "model_1_scoring_stats",
        ],
    )
    cp1_fused = downline_numerical(cp1_fused)

    print("finally, CP1 fused shape:", cp1_fused.shape)
    print("Writing final fusion parquet to:", OUTPUT_FUSION)

    creating_fusion_in_chunks(
        cp3=cp3,
        cp1_fused=cp1_fused,
        output_path=OUTPUT_FUSION,
        chunk_size=FUSION_PAT_CHUNK_SIZE,
    )

    print("Saved fusion dataset:", OUTPUT_FUSION)
    print("Done.")


if __name__ == "__main__":
    main()

