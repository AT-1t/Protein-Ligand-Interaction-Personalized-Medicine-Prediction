#Checkpoint-2 Step 4
"""
Checkpoint 2 Step 4
Protein Family Classification using Machine Learning
This script trains and evaluates a classification model using features from
sequence embeddings, physicochemical properties, and alignment scores.

This Includes:
    - Protein Holdout Testing
    - feature importance analysis
    - visualization of model performance
"""
#Importing necessary libraries
from checkpoint_2_step_3 import adding_local_align_features, clean_sequence
import os
import pandas as pd
import warnings 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
warnings.filterwarnings("ignore")

#Setting Up The Directories for Input and Output
Data_directory = "Checkpoint_2_data"
Checkpoint_2_step_4_input_path = os.path.join(Data_directory,"checkpoint_2_step_3_family_classification.csv")
Checkpoint_2_step_4_output_path = os.path.join(Data_directory, "checkpoint_2_step_4_transfer_learning_family_predictions.csv")
Checkpoint_2_metrics_path = os.path.join(Data_directory, "checkpoint_2_step_4_family_classification_metrics.csv")
Checkpoint_2_confusion_path = os.path.join(Data_directory, "checkpoint_2_step_4_family_classification_confusion_matrix.csv")
Checkpoint_2_importance_path = os.path.join(Data_directory, "checkpoint_2_step_4_feature_importance.csv")
Checkpoint_2_importance_png = os.path.join(Data_directory, "checkpoint_2_step_4_feature_importance.png")
Checkpoint_2_confusion_png = os.path.join(Data_directory, "checkpoint_2_step_4_confusion_matrix.png")
Checkpoint_2_metrics_png = os.path.join(Data_directory, "checkpoint_2_step_4_metrics.png")
Checkpoint_2_correctness_png = os.path.join(Data_directory, "checkpoint_2_step_4_cvsi.png")
Checkpoint_2_confidence_pnd = os.path.join(Data_directory, "checkpoint_2_step_4_conhist.png")
Checkpoint_2_pkl = os.path.join(Data_directory, "checkpoint_2_step_4_transfer_learning_family_predictions.pkl")
Checkpoint_2_classification = os.path.join(Data_directory, "checkpoint_2_step_4_classification_report_proteins.csv")
Checkpoint_2_map = os.path.join(Data_directory, "checkpoint_2_step_4_classification_map.png")

def saving_confusion_mattrix(y_true, y_pred, class_names, normalize=True):
    """
    creates and then saves a confusion matrix to a .png

    Parameters:
        cm_df (pd.DataFrame): confusion matrix with rows as true labels and columns as prediction labels
    
    Returns:
        None, instead saves .png file to directory
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)


    #initializing confusion matrix with heatmap figure
    plt.figure(figsize=(10,10))
    sns.heatmap(cm_df, cmap="Blues", annot=True, linewidths=0.5)
    plt.title("Step 4 Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(Checkpoint_2_confusion_png, dpi=400)
    plt.close()

def save_feature_importance(importance_df, top_n=15):
    """
    creates and saves a plot showing feature importance

    Parameters:
        importance_df (pd.DataFrame): DataFrame with Columns: ["feature", "importance"]
        top_n (int): number of top features to display. based on trial and error, 5-20 seem to be good range for this dataset
    
    Returns:
        None, but saves the .png file to the specified directory
    """
    #keeping only top_n features for plotting
    top_df = importance_df.head(top_n).copy()
    #initializing, creating, and saving the  plot
    plt.figure(figsize=(10,10))
    plt.barh(top_df["feature"][::-1], top_df["importance"][::-1]) #horizontal bar chart
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Feature Importance")
    plt.tight_layout()
    plt.savefig(Checkpoint_2_importance_png, dpi=400) #saving to png in directory
    plt.close()

def save_metrics_plot(metrics_df):
    """
    saves and plots the metric data for model performance

    Parameters:
        metrics_df (pd.DataFrame): DataFrame containing metric names and values
    
    Returns:
        None, but saves barplot as .png to directory specified
    """
    #keep only the predictive score metrics
    plot_df = metrics_df[metrics_df["metric"].isin(["accuracy", "f1", "f1_weighted", "f1_micro"])].copy()
    #initializing and creating barplot showing metrics
    plt.figure(figsize=(10,10))
    plt.bar(plot_df["metric"], plot_df["value"])
    plt.ylim(0,1)
    plt.ylabel("Score")
    plt.title("Step 4 Model Performance")
    plt.tight_layout()
    plt.savefig(Checkpoint_2_metrics_png, dpi=400) #saving to png in server directory
    plt.close()

def save_correct_incorrect_plot(results_df):
    """
    saving plot of predictions correct vs incorrect

    Parameters:
        results_df (pd.DataFrame): DataFrame containing a "prediction_correct" boolean columns
    
    Results:
        None, but saves par plot as .png in directory specified.
    """
    #counting how many predictions are correct vs incorrect
    counts = results_df["prediction_correct"].value_counts()
    #initializing barplot with breakdown of correct vs incorrect assignment
    plt.figure(figsize=(10,10))
    plt.bar(["Incorrect", "Correct"], [counts.get(False, 0), counts.get(True, 0)])
    plt.ylabel("Count")
    plt.title("Prediction Accuracy Breakdown")
    plt.tight_layout()
    plt.savefig(Checkpoint_2_correctness_png, dpi=400) #saving to a png in the server directory
    plt.close()

def save_confidence_histogram(results_df):
    """
    saving and plotting the prediction confidence distribution

    Parameters:
        results_df (pd.DataFrame): DataFrame containing a "prediction_confidence" columns
    
    Results:
        None, but saves histogram to specified directory as .png
    """
    #only continue with prediction_confidence even exists
    if "prediction_confidence" not in results_df.columns:
        return
    #initializing histogram to model the confidence
    plt.figure(figsize=(10,10))
    plt.hist(results_df["prediction_confidence"], bins=20)
    plt.xlabel("Prediction Confidence")
    plt.ylabel("Counts")
    plt.title("Prediction Confidence Distribution")
    plt.tight_layout()
    plt.savefig(Checkpoint_2_confidence_pnd, dpi=400) #save to png
    plt.close()

def loading_too_much_data(path, chunksize=50000):

    cols=pd.read_csv(path, nrows=0).columns.tolist()
    embedding_columns = [c for c in cols if c.startswith("emb_")]
    aa_columns = [c for c in cols if c.startswith("aa_")]

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
    
    meta_columns = ["target_name", 
                    "smiles", 
                    "uniprot_id", 
                    "sequence", 
                    "sequence_clean", 
                    "closest_kinase_family", 
                    "is_egfr"]
    #assay_columns = ["ki", "kd", "ic50", "affinity", "log_affinity"]
    #assay columns left out to try and improve model accuracy
    base_feature_cols = [c for c in (embedding_columns + aa_columns + phychem_columns + similarity_columns) if c in cols]
    usecols = [c for c in meta_columns + base_feature_cols if c in cols]

    chunks = []
    rows_read = 0
    rows_keep = 0
    reading = pd.read_csv(path, usecols=usecols, chunksize=chunksize, engine="python", on_bad_lines="skip")
    for i, chunk in enumerate(reading, start=1):
        if "closest_kinase_family" in chunk.columns:
            chunk = chunk.dropna(subset=["closest_kinase_family"]).copy()
            chunk["closest_kinase_family"] = chunk["closest_kinase_family"].astype(str).str.strip()
            chunk = chunk[chunk["closest_kinase_family"] != ""].copy()

        for col in base_feature_cols:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce', downcast='float')
        
        if "closest_kinase_family" in chunk.columns:
            chunk["closest_kinase_family"] = chunk["closest_kinase_family"].astype("string")
        if "uniprot_id" in chunk.columns:
            chunk["uniprot_id"] = chunk["uniprot_id"].astype("string")
        
        rows_keep += len(chunk)
        chunks.append(chunk)

       # if i % 10 ==0:
           # print(f"chunks read: {i}")

    if not chunks:
        return pd.DataFrame(columns=usecols), base_feature_cols
    
    print("concatenating filtered chunks")
    df = pd.concat(chunks, ignore_index=True)

    for col in base_feature_cols:
        if col in df.columns:
            df[col] = df[col].astype("float32")
    
    return df, base_feature_cols


def prepare_feature_matrix(train_df, test_df, base_feature_cols,):


    local_alighment_columns = ["local_alignment_best_score",
            "local_alignment_mean_score",
            "local_alignment_best_normalized",
            "local_alignment_mean_normalized",
            "local_alignment_score_std",
            "local_alignment_hit_count",
            #"local_alignment_family_match"
           ]
    
    feature_columns = [c for c in (base_feature_cols +local_alighment_columns)
                       if c in train_df.columns and c in test_df.columns]
    if not feature_columns:
        raise ValueError("no usable feature columns present")
    
    X_train = train_df[feature_columns].apply(pd.to_numeric, errors="coerce")
    X_test = test_df[feature_columns].apply(pd.to_numeric, errors="coerce")
    X_train= X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    train_median = X_train.median(numeric_only=True)
    X_train = X_train.fillna(train_median)
    X_test = X_test.fillna(train_median)

    valid_cols = [c for c in feature_columns if not X_train[c].isna().any() and not X_test[c].isna().any()]
    if not valid_cols:
        raise ValueError("all feature columns are invalid")
    
    X_train = X_train[valid_cols].astype("float32")
    X_test = X_test[valid_cols].astype("float32")

    return X_train, X_test, valid_cols

def looking_at_protein_level(results_df):
    df = results_df.dropna(subset=["uniprot_id"]).copy()

    true_df = df.groupby("uniprot_id")["true_family"].agg(lambda x: x.mode()[0])

    pred_df = df.groupby("uniprot_id")["predicted_family"].agg(lambda x: x.mode()[0])
        
    
    protein_df = pd.DataFrame({"true_family": true_df, "predicted_family": pred_df}).reset_index()
    
    protein_df["prediction_correct"] = (protein_df["true_family"] == protein_df["predicted_family"])
   
   
    return protein_df

def group_holdout(df, label_col="closest_kinase_family", group_col="uniprot_id", test_fraction=0.2, random_state=42):
    rng = np.random.default_rng(random_state)
    protein_df = df[[group_col, label_col]].drop_duplicates().copy()
    train_groups = []
    test_groups = []
    for family, fam_df in protein_df.groupby(label_col):
        proteins = fam_df[group_col].astype(str).dropna().unique()
        if len(proteins) < 2:
            continue
        proteins = proteins.copy()
        rng.shuffle(proteins)
        n_test = max(1, int(round(len(proteins)*test_fraction)))
        n_test = min(n_test, len(proteins)-1)
        test_groups.extend(proteins[:n_test].tolist()) 
        train_groups.extend(proteins[n_test:].tolist())

    train_mask = df[group_col].astype(str).isin(set(train_groups))
    test_mask = df[group_col].astype(str).isin(set(test_groups))
    return df.loc[train_mask].copy(), df.loc[test_mask].copy()


def main():
    """
    Main pipline for protein family classification, checkpoint 2 step 4

    Current Workflow as of 4/8/2026:
    1. loads processed data from step 3
    2. cleans and prepares feature matrices
    3. filters for valid protein families
    4. splits data into training with protein holdout
    5. encodes labels for classification using LabelEncoder
    6. trains a XGB Boost Model- also exploring XGBoost
    7. Evaluates performance using accuracy and f1 scores
    8. saves predictions, metrics, confusion matrix, and feature importance as .csv and .png
    9. generates visualizations for understanding and analysis
    """
    #Checking that input file exists
    if not os.path.exists(Checkpoint_2_step_4_input_path):
        print("No Step 3 File Found - Run Step 3!!")
        return
    
    #Loading Data from Step 3
    df, base_feature_cols = loading_too_much_data(Checkpoint_2_step_4_input_path, chunksize=1000)
    print("Loaded Step 4 Input File - Shape:", df.shape)
    #Dataframe empty check
    if df.empty:
        print("The Input File is EMPTY!")
        return
    #column check for closest kinase family target column
    if "closest_kinase_family" not in df.columns:
        raise ValueError("Step 3 output is missing closest_kinase_column!")
    
    #Keeping Families with Enough Representation - Consider Removing, using as test for accuracy improvements
    #=============================================================================
    df = df.dropna(subset=["uniprot_id", "closest_kinase_family"]).copy()
    df["closest_kinase_family"] = df["closest_kinase_family"].astype(str).str.strip()
    df["uniprot_id"] = df["uniprot_id"].astype(str).str.strip()
    df = df[(df["closest_kinase_family"] != "") & (df["uniprot_id"] != "")].copy()
    
    min_samples = 20
    protein_counting = df[["closest_kinase_family", "uniprot_id"]].drop_duplicates()["closest_kinase_family"].value_counts()
    valid_families = protein_counting[protein_counting >= min_samples].index
    df = df[df["closest_kinase_family"].isin(valid_families)]
    #there must be two families overall
    if df["closest_kinase_family"].nunique() < 2:
        print("Please have at least 2 families for classification!!")
        return
   
    meta_keeping = ["uniprot_id", "closest_kinase_family", "target_name", "sequence", "sequence_clean", "is_egfr"]
    numeric_cols = [c for c in df.columns if c not in meta_keeping + ["smiles"]]
    numeric_cols = [c for c in numeric_cols if pd.api.types.is_numeric_dtype(df[c])]

    protein_labels = (df.groupby("uniprot_id")["closest_kinase_family"].agg(lambda x: x.mode().iloc[0]).reset_index())
    protein_numeric = (df.groupby("uniprot_id", as_index=False)[numeric_cols].median())
    
    metas = {}
    for col in ["target_name", "sequence", "sequence_clean", "is_egfr"]:
        if col in df.columns:
            metas[col] = "first"
    
    protein_meta_df = (df.groupby(["uniprot_id"], as_index=False).agg(metas))

    df = protein_numeric.merge(protein_labels, on="uniprot_id", how="left")
    df = df.merge(protein_meta_df, on="uniprot_id", how="left")

    base_feature_cols = [c for c in df.columns if c not in ["uniprot_id", "closest_kinase_family", "target_name", "sequence", "sequence_clean", "is_egfr", "smiles"] 
                         and pd.api.types.is_numeric_dtype(df[c])]
    train_df, test_df = group_holdout(df, test_fraction=0.2, random_state=42)

    protein_overlapping = set(train_df["uniprot_id"]).intersection(set(test_df["uniprot_id"]))
    print("Protein Overlap for Training and Testing:", len(protein_overlapping))

    train_df = train_df.copy()
    
    train_df = adding_local_align_features(train_df, ref_df=train_df, max_refs_per_family=5)
    test_df = adding_local_align_features(test_df, ref_df=train_df, max_refs_per_family=5)
    
    #extracting the train and test labels
    y_train = train_df["closest_kinase_family"].copy()
    y_test = test_df["closest_kinase_family"].copy()

    #computing the majority-class baseline 
    maj_base = y_test.value_counts(normalize=True).max()
    #printing the family distribution for analysis/confirmation
    print("majority-class baseline accuracy:",maj_base)
    print("Train Family Counts:", y_train.value_counts())
    print("Test Family Counts:", y_test.value_counts())
    print("Number of Train Classes:", y_train.nunique())
    print("Number of Test Classes:", y_test.nunique())

    #encoding the training family labels as integers
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    #keepings only test rows whose classes were seen during training
    known_classes = y_test.isin(label_encoder.classes_)
    test_df = test_df.loc[known_classes].copy()
    y_test = y_test.loc[known_classes].copy()
    #Building the train and test feature matrices
    X_train, X_test, feature_columns = prepare_feature_matrix(train_df, test_df, base_feature_cols)

    #encode the filtered test labels
    y_test_encoded = label_encoder.transform(y_test)
                     
    #Setting Up and Training Random Forest Classifier
    #============================================================================================
    weighting_sample = compute_sample_weight("balanced", y_train)
    xgb_model = XGBClassifier(objective="multi:softprob", 
                             n_estimators=300,
                            max_depth=5, 
                             learning_rate=0.03, 
                          #   subsample=0.7, 
                           #  colsample_bytree=0.7,
                            min_child_weight=3,
                            gamma=0.5,
                             reg_alpha=2.0, 
                            reg_lambda=5.0, 
                             random_state=42, 
                             n_jobs=-1, 
                             eval_metric="mlogloss", 
                             tree_method="hist")

    #fitting the model on training data
    xgb_model.fit(X_train, y_train_encoded, sample_weight=weighting_sample)
    selector = SelectFromModel(xgb_model, threshold="mean", prefit=True)
    selected = selector.get_support()
    selected_feature_cols = [f for f, keep in zip(feature_columns, selected) if keep]

    X_train = selector.transform(X_train)
    X_test = selector.transform(X_test)
    print("original features:", len(feature_columns))
    print("selected features:", X_train.shape[1])

    xgb_model_c = XGBClassifier(objective="multi:softprob", 
                             n_estimators=200,
                            max_depth=5, 
                             learning_rate=0.03, 
                          #   subsample=0.7, 
                           #  colsample_bytree=0.7,
                            min_child_weight=3,
                            gamma=0.5,
                             reg_alpha=2.0, 
                            reg_lambda=5.0, 
                             random_state=42, 
                             n_jobs=-1, 
                             eval_metric="mlogloss", 
                             tree_method="hist")

    #fitting the model on training data
    xgb_model_c.fit(X_train, y_train_encoded, sample_weight=weighting_sample)
    #measuring the training accuracy to detect overfitting
    train_pred = xgb_model_c.predict(X_train).astype(int)
    train_accuracy = accuracy_score(y_train_encoded, train_pred)
    print("Training Accuracy:", train_accuracy)

    #Prediction ofr test labels
    y_pred = xgb_model_c.predict(X_test)
    current_labels = np.unique(y_test_encoded)
    #Metrics - Accuracy, F1 Macro and F1 Weighted
    #====================================================================
    accuracy = accuracy_score(y_test_encoded, y_pred)
    f1 = f1_score(y_test_encoded, y_pred, labels=current_labels, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test_encoded, y_pred, labels=current_labels, average="weighted", zero_division=0)
    f1_micro = f1_score(y_test_encoded, y_pred, labels=current_labels, average="micro", zero_division=0)

    #Saving the Predictions
    #===========================================================================
    results = test_df.copy()
    results["true_family"] = y_test.values
    results["predicted_family"] = label_encoder.inverse_transform(y_pred)
    results["prediction_correct"] = (results["true_family"] == results["predicted_family"])
    pred_prob = xgb_model_c.predict_proba(X_test)
    results["prediction_confidence"] = pred_prob.max(axis=1)
    prot_results = looking_at_protein_level(results)
    prot_accuracy = accuracy_score(prot_results["true_family"], prot_results["predicted_family"])
    prot_f1 = f1_score(prot_results["true_family"], prot_results["predicted_family"], average="macro", zero_division=0)
    prot_f1_weighted = f1_score(prot_results["true_family"], prot_results["predicted_family"], average="weighted", zero_division=0)
    prot_f1_micro = f1_score(prot_results["true_family"], prot_results["predicted_family"], average="micro", zero_division=0)






    results.to_csv(Checkpoint_2_step_4_output_path, index=False)
    results.to_pickle(Checkpoint_2_pkl)
    print("Step 4 Predictions Have Been Saved!")



    #Saving Metrics Summary
    #==================================================================================
    metrics_df = pd.DataFrame({"metric": ["accuracy", "f1", "f1_weighted", "f1_micro", "n_train", "n_test", "n_features"], 
                               "value": [accuracy, f1, f1_weighted, f1_micro, len(X_train), len(X_test), len(feature_columns)]})
    metrics_df.to_csv(Checkpoint_2_metrics_path, index=False)
    print(metrics_df)
    print("Step 4 Metrics Have Been Saved!")


    #Saving Confusion Matrix
    #=======================================================================
    cm_labels = sorted(set(y_test_encoded) | set(y_pred))
    cm = confusion_matrix(y_test_encoded, y_pred, labels=cm_labels)
    class_names = label_encoder.inverse_transform(cm_labels)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(Checkpoint_2_confusion_path, index=True)


    #Saving Feature Importance
    #====================================================================
    importance_df = pd.DataFrame({"feature": selected_feature_cols, 
                                  "importance": xgb_model_c.feature_importances_}).sort_values("importance", ascending=False)
    importance_df.to_csv(Checkpoint_2_importance_path, index=False)

    #Saving PNGS of Everything Possible
    #===================================
    saving_confusion_mattrix(y_test_encoded, y_pred, class_names)
    save_feature_importance(importance_df, top_n=30)
    save_metrics_plot(metrics_df)
    save_correct_incorrect_plot(results)
    save_confidence_histogram(results)

    #Printing Summary of Model
    #===================================================
    print("Step4 Accuracy:", accuracy)
    print("Step4 F1 MACRO:", f1)
    print("Step4 F1 WEIGHTED:", f1_weighted)
    print("Step4 F1 MICRO:", f1_micro)
    print("unique test proteins:", len(prot_results))
    print("Step4 Protein Accuracy:", prot_accuracy)
    print("Step4 Protein F1 MACRO:", prot_f1)
    print("Step4  Protein F1 WEIGHTED:", prot_f1_weighted)
    print("Step4 Protein F1 MICRO:", prot_f1_micro)
    print("\nClassification Report:")
    print(classification_report(y_test_encoded, y_pred,
                                labels=np.unique(y_test_encoded),
                                target_names=label_encoder.inverse_transform(np.unique(y_test_encoded)), 
                                zero_division=0))
    print("\nProtein-Level Classification Report:")   
    print(classification_report(prot_results["true_family"], prot_results["predicted_family"], 
                                zero_division=0))
    report = classification_report(prot_results["true_family"], prot_results["predicted_family"], 
                                zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(Checkpoint_2_classification)
    print("total rows", len(df))
    print("unique proteins:", df["uniprot_id"].nunique())

    print("Step 4 Has Completed!")
    print("=====================================================================================")


if __name__ == "__main__":
    main()
        



    
