#Checkpoint-2 Step 4

#Importing necessary libraries
import os
import pandas as pd
import warnings 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
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

def find_egfr(df):
    """
    Function for finding egfr

    """
    #creating false (initially) boolean masking
    cond1 = pd.Series(False, index=df.index)
    cond2 = pd.Series(False, index=df.index)
    cond3 = pd.Series(False, index=df.index)
    #if is_egfr exists, identify the rows marked as true / 1 / yes
    if "is_egfr" in df.columns:
        cond1 = df["is_egfr"].astype(str).str.lower().isin(["true", "1", "yes"])
    #if uniprot_id exists, indentigy the egfr by uniprot accession p00533
    if "uniprot_id" in df.columns:
        cond2 = df["uniprot_id"].astype(str).str.upper().str.contains("P00533", na=False)
    #if target_name exists identify egfr or erbb1 by the name
    if "target_name" in df.columns:
        cond3 = df["target_name"].astype(str).str.contains("EGFR|ERBB1", case=False, na=False)
    #mark a row as egfr if any of the three checks is truthful
    return cond1 | cond2 | cond3


def saving_confusion_mattrix(cm_df):
    """
    creates and then saves a confusion matrix to a .png
    """
    #initializing confusion matrix with heatmap figure
    plt.figure(figsize=(10,10))
    plt.imshow(cm_df.values, aspect="auto")
    plt.colorbar()

    #labeling axes 
    plt.xticks(range(len(cm_df.columns)), cm_df.columns, rotation=90)
    plt.yticks(range(len(cm_df.index)), cm_df.index)

    for i in range(cm_df.shape[0]):
        for j in range(cm_df.shape[1]):
            plt.text(j, i, str(cm_df.iloc[i,j]), ha="center", va="center")
    plt.xlabel("Predicted Family")
    plt.ylabel("True Family")
    plt.title("Step 4 Confusion Matrix")
    plt.savefig(Checkpoint_2_confusion_png, dpi=400)
    plt.close()

def save_feature_importance(importance_df, top_n=15):
    """
    creates and saves a plot showing feature importance
    """
    #keeping only top_n features for plotting
    top_df = importance_df.head(top_n).copy()
    #initializing, creating, and saving the  plot
    plt.figure(figsize=(10,10))
    plt.barh(top_df["feature"][::-1], top_df["importance"][::-1]) #horizontal bar chart
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Feature Importance")
    plt.savefig(Checkpoint_2_importance_png, dpi=400) #saving to png in directory
    plt.close()

def save_metrics_plot(metrics_df):
    """
    saves and plots the metric data for model performance
    """
    #keep only the predictive score metrics
    plot_df = metrics_df[metrics_df["metric"].isin(["accuracy", "f1", "f1_weighted"])].copy()
    #initializing and creating barplot showing metrics
    plt.figure(figsize=(10,10))
    plt.bar(plot_df["metric"], plot_df["value"])
    plt.ylim(0,1)
    plt.ylabel("Score")
    plt.title("Step 4 Model Performance")
    plt.savefig(Checkpoint_2_metrics_png, dpi=400) #saving to png in server directory
    plt.close()

def save_correct_incorrect_plot(results_df):
    """
    saving plot of predictions correct vs incorrect
    """
    #counting how many predictions are correct vs incorrect
    counts = results_df["prediction_correct"].value_counts()
    #initializing barplot with breakdown of correct vs incorrect assignment
    plt.figure(figsize=(10,10))
    plt.bar(["Incorrect", "Correct"], [counts.get(False, 0), counts.get(True, 0)])
    plt.ylabel("Count")
    plt.title("Prediction Accuracy Breakdown")
    plt.savefig(Checkpoint_2_correctness_png, dpi=400) #saving to a png in the server directory
    plt.close()

def save_confidence_histogram(results_df):
    """
    saving and plotting the prediction confidence distribution
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
    plt.savefig(Checkpoint_2_confidence_pnd, dpi=400) #save to png
    plt.close()



def main():
    #Checking that input file exists
    if not os.path.exists(Checkpoint_2_step_4_input_path):
        print("No Step 3 File Found - Run Step 3!!")
        return
    
    #Loading Data from Step 3
    df = pd.read_csv(Checkpoint_2_step_4_input_path)
    print("Loaded Step 4 Input File - Shape:", df.shape)
    #Dataframe empty check
    if df.empty:
        print("The Input File is EMPTY!")
        return
    #column check for closest kinase family target column
    if "closest_kinase_family" not in df.columns:
        raise ValueError("Step 3 output is missing closest_kinase_column!")
    
    
    #Building A Numeric Feature Table
    #============================================================================
    #non feature columns for separating metadata from features
    non_feature_columns = ["target_name", "smiles", "uniprot_id", "sequence", "sequence_clean", "closest_kinase_family", "is_egfr"]
    #feature columns
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

    assay_columns = ["ki", "kd", "ic50", "affinity", "log_affinity"]
    #similarity and assay columns left out to try and improve model accuracy
    feature_cols = embedding_columns + aa_columns + phychem_columns + similarity_columns

    #creation of feature only matrix
    X_df = df[feature_cols].copy()
    #safety checks for X_df
    #converting to numeric
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    #replacing ant infinities with NaN
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    #filling any missing values with the median of the column
    X_df = X_df.fillna(X_df.median(numeric_only=True))
    #saving list of final features after cleaning
    feature_columns = X_df.columns.tolist()
    #printing the number of usable features after cleaning
    print("Useable Feature Count", len(feature_columns))
    

    #Keeping Families with Enough Representation
    #=============================================================================
    #there must be two families overall
    if df["closest_kinase_family"].nunique() < 2:
        print("Please have at least 2 families for classification!!")
        return
    #counting how many examples each family has
    family_count = df["closest_kinase_family"].value_counts()
    #keeping only families with at least two samples
    valid_family = family_count[family_count >=5].index.tolist()
    df = df[df["closest_kinase_family"].isin(valid_family)].copy()
    X_df = X_df.loc[df.index].copy()
    #checking again that at least two families are remaining
    if df["closest_kinase_family"].nunique() < 2:
        print("Please have at least 2 families for classification!!")
        return
    

    #Setting Up Model Data - Train/Test Splitting
    #===================================================================================
    #marking egfr rows using find_egfr function
    egfr_look = find_egfr(df)
    #training on all non-egfr proteins
    train_df = df.loc[~egfr_look].copy()
    #testing on egfr proteins only
    test_df = df.loc[egfr_look].copy()
    #checking if either is empty to verify
    if train_df.empty:
        print("there are no on-EGFR proteins available for training")
        return
    if test_df.empty:
        print("there are no egfr proteins for cold start testing")
        return

    #Building the train and test feature matrices
    X_train = X_df.loc[train_df.index].copy()
    X_test = X_df.loc[test_df.index].copy()

    #extracting the train and test labels
    y_train = train_df["closest_kinase_family"].copy()
    y_test = test_df["closest_kinase_family"].copy()

    #computing the majority-class baseline on the egfr only test set
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
    known_test_mask = y_test.isin(label_encoder.classes_)
    test_df = test_df.loc[known_test_mask].copy()
    X_test = X_test.loc[test_df.index].copy()
    y_test = y_test.loc[test_df.index].copy()
    #testing if empty just in case
    if test_df.empty:
        print("All EGFR test labels are unseen in training")
        return
    
    #encode the filtered test labels
    y_test_encoded = label_encoder.transform(y_test)
                     
    #Setting Up and Training Random Forest Classifier
    #============================================================================================
    
    rf_model = RandomForestClassifier(
        n_estimators=400, 
        max_depth=6,  
        n_jobs=-1, 
        random_state=42, 
        class_weight="balanced")
    
    
    #xgb_model = XGBClassifier(objective="multi:softprob", 
     #                        n_estimators=300, num_class=num_classes,
     #                        max_depth=3, 
     #                        learning_rate=0.03, 
     #                        subsample=0.7, 
      #                       colsample_bytree=0.7,
       #                      min_child_weight=5,
       #                      reg_alpha=1.0, 
       #                      reg_lambda=3.0, 
       #                      random_state=42, 
       #                      n_jobs=-1, 
      
      
      #                       eval_metric="mlogloss")

    #fitting the model on training data
    rf_model.fit(X_train, y_train_encoded)

    #measuring the training accuracy to detect overfitting
    train_pred = rf_model.predict(X_train).astype(int)
    train_accuracy = accuracy_score(y_train_encoded, train_pred)
    print("Training Accuracy:", train_accuracy)

    #Prediction ofr test labels
    y_pred = rf_model.predict(X_test)

    #Metrics - Accuracy, F1 Macro and F1 Weighted
    #====================================================================
    accuracy = accuracy_score(y_test_encoded, y_pred)
    f1 = f1_score(y_test_encoded, y_pred, average="macro")
    f1_weighted = f1_score(y_test_encoded, y_pred, average="weighted")


    #Saving the Predictions
    #===========================================================================
    results = test_df.copy()
    results["true_family"] = y_test.values
    results["predicted_family"] = label_encoder.inverse_transform(y_pred)
    results["prediction_correct"] = (results["true_family"] == results["predicted_family"])
    pred_prob = rf_model.predict_proba(X_test)
    results["prediction_confidence"] = pred_prob.max(axis=1)
    results.to_csv(Checkpoint_2_step_4_output_path, index=False)
    print("Step 4 Predictions Have Been Saved!")



    #Saving Metrics Summary
    #==================================================================================
    metrics_df = pd.DataFrame({"metric": ["accuracy", "f1", "f1_weighted", "n_train", "n_test", "n_features"], 
                               "value": [accuracy, f1, f1_weighted, len(X_train), len(X_test), len(feature_columns)]})
    metrics_df.to_csv(Checkpoint_2_metrics_path, index=False)
    print(metrics_df)
    print("Step 4 Metrics Have Been Saved!")


    #Saving Confusion Matrix
    #=======================================================================
    cm_labels = sorted(set(y_test_encoded) | set(y_pred))
    cm = confusion_matrix(y_test_encoded, y_pred, labels=cm_labels)
    class_names = label_encoder.inverse_transform(cm_labels)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(Checkpoint_2_confusion_path, index=False)


    #Saving Feature Importance
    #====================================================================
    importance_df = pd.DataFrame({"feature": feature_columns, "importance": rf_model.feature_importances_}).sort_values("importance", ascending=False)
    importance_df.to_csv(Checkpoint_2_importance_path, index=False)

    #Saving PNGS of Everything Possible
    #===================================
    saving_confusion_mattrix(cm_df)
    save_feature_importance(importance_df, top_n=100)
    save_metrics_plot(metrics_df)
    save_correct_incorrect_plot(results)
    save_confidence_histogram(results)

    #Printing Summary of Model
    #===================================================
    print("Step4 Accuracy:", accuracy)
    print("Step4 F1 MACRO:", f1)
    print("Step4 F1 WEIGHTED:", f1_weighted)
    print("\nClassification Report:")
    #print(classification_report(y_test, y_pred, target_names=class_names))

    print("total rows", len(df))
    print("unique proteins:", df["uniprot_id"].nunique())

    print("Step 4 Has Completed!")

    print("=====================================================================================")
    print("=====================================================================================")
    print("=====================================================================================")
    
    print("Beginning of Protein-Hold Out Experiment for Cold-Start")

    #Loading Data from Step 3
    df = pd.read_csv(Checkpoint_2_step_4_input_path)
    print("Loaded Step 4 Input File - Shape:", df.shape)
    #Dataframe empty check
    if df.empty:
        print("The Input File is EMPTY!")
        return
    #column check for closest kinase family target column
    if "closest_kinase_family" not in df.columns:
        raise ValueError("Step 3 output is missing closest_kinase_column!")
    
    df = df.dropna(subset=["closest_kinase_family", "uniprot_id"]).copy()
    #Building A Numeric Feature Table
    #============================================================================
    #non feature columns for separating metadata from features
    non_feature_columns = ["target_name", "smiles", "uniprot_id", "sequence", "sequence_clean", "closest_kinase_family", "is_egfr"]
    #feature columns
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

    assay_columns = ["ki", "kd", "ic50", "affinity", "log_affinity"]
    #similarity and assay columns left out to try and improve model accuracy
    feature_cols = embedding_columns + aa_columns + phychem_columns + similarity_columns

    #creation of feature only matrix
    X_df = df[feature_cols].copy()
    #safety checks for X_df
    #converting to numeric
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    #replacing ant infinities with NaN
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    #filling any missing values with the median of the column
    X_df = X_df.fillna(X_df.median(numeric_only=True))
    #saving list of final features after cleaning
    feature_columns = X_df.columns.tolist()
    #printing the number of usable features after cleaning
    print("Useable Feature Count", len(feature_columns))
    

    #Keeping Families with Enough Representation
    #=============================================================================
    #there must be two families overall
    if df["closest_kinase_family"].nunique() < 2:
        print("Please have at least 2 families for classification!!")
        return
    #counting how many examples each family has
    family_count = df["closest_kinase_family"].value_counts()
    #keeping only families with at least two samples
    valid_family = family_count[family_count >=2].index.tolist()
    df = df[df["closest_kinase_family"].isin(valid_family)].copy()
    X_df = X_df.loc[df.index].copy()
    #checking again that at least two families are remaining
    if df["closest_kinase_family"].nunique() < 2:
        print("Please have at least 2 families for classification!!")
        return

    #Group-Based Splitting so Same Protein Does Not Appear in Train and Test
    #================================================================================
    groups = df["uniprot_id"].astype(str)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, df["closest_kinase_family"], groups=groups))
   
    #Setting Up Testing and Training Data using New Split
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    y_train = train_df["closest_kinase_family"].copy()
    y_test = test_df["closest_kinase_family"].copy()

    X_train = X_df.loc[train_df.index].copy()
    X_test = X_df.loc[test_df.index].copy()

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
   
    known_mask_testing = y_test.isin(label_encoder.classes_)
    test_df = test_df.loc[known_mask_testing].copy()
    
    X_test = X_test.loc[test_df.index]
    y_test = y_test.loc[test_df.index].copy()
    
    y_test_encoded = label_encoder.transform(y_test)
    maj_base = y_test.value_counts(normalize=True).max()
    print("majority-class baseline accuracy:",maj_base)
    print("Train Family Counts:", y_train.value_counts())
    print("Test Family Counts:", y_test.value_counts())
    print("Number of Train Classes:", y_train.nunique())
    print("Number of Test Classes:", y_test.nunique())
    training_proteins = set(train_df["uniprot_id"].astype(str))
    testing_proteins = set(test_df["uniprot_id"].astype(str))
    overlap = training_proteins & testing_proteins
    print("Number of Training Proteins:", len(training_proteins))
    print("Number of Testng Proteins:", len(testing_proteins))
    print("Protein Overlap Between Train and Test:", len(overlap))
   # num_classes = len(label_encoder.classes_)
    
    #Setting Up and Training Random Forest Classifier
    #============================================================================================
    rf_model = RandomForestClassifier(
        n_estimators=400, 
        max_depth=6,  
        n_jobs=-1, 
        random_state=42, 
        class_weight="balanced")


    rf_model.fit(X_train, y_train_encoded)
    train_pred = rf_model.predict(X_train).astype(int)
    train_accuracy = accuracy_score(y_train_encoded, train_pred)
    print("Training Accuracy:", train_accuracy)

    #Predictions
    y_pred = rf_model.predict(X_test).astype(int)

    #Metrics - Accuracy, F1 Macro and F1 Weighted
    #====================================================================
    accuracy = accuracy_score(y_test_encoded, y_pred)
    f1 = f1_score(y_test_encoded, y_pred, average="macro")
    f1_weighted = f1_score(y_test_encoded, y_pred, average="weighted")
    print("Protein-HoldOut Test Accuracy:", accuracy)
    predicted_labels = label_encoder.inverse_transform(y_pred)
    print("\nPredicted Family Counts")
    print(pd.Series(predicted_labels).value_counts())

if __name__ == "__main__":
    main()

        



    
