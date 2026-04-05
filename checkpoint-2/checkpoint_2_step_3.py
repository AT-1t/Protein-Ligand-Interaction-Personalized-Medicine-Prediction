#Checkpoont-2 Step 3
import os
import pandas as pd
import warnings 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
warnings.filterwarnings("ignore")
import numpy as np

#Setting Up The Directories
Data_directory = "Checkpoint_2_data"
Checkpoint_2_step_3_input_path = os.path.join(Data_directory,"checkpoint_2_step_2_deepseq_embedding.csv")
Checkpoint_2_step_3_output_path = os.path.join(Data_directory, "checkpoint_2_step_3_family_classification.csv")

#Function for Building a Numeric Feature Table
def building_nfeature_table(df):

    non_feature_columns = ["target_name", "smiles", "uniprot_id", "sequence", "sequence_clean", "closest_kinase_family", "affinity", "log_affinity"]
    if "is_egfr" in df.columns:
        non_feature_columns.append("is_egfr")
    feature_cols = [c for c in df.columns if c not in non_feature_columns]
    X_df = df[feature_cols].copy()
    #to numeric
    for c in X_df.columns:
        X_df[c] = pd.to_numeric(X_df[c], errors="coerce")
    all_nan_columns = X_df.columns[X_df.isna().all()].tolist()
    #taking care of nan columns and inf
    if all_nan_columns:
        print("Dropping all NaN columns:", all_nan_columns)
        X_df = X_df.drop(columns=all_nan_columns)
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    #filling missing with median(??)
    X_df = X_df.fillna(X_df.median(numeric_only=True))
    #dropping any other icky columns though
    icky_columns = X_df.columns[X_df.isna().any()].tolist()
    if icky_columns:
        print("Dropping Columns", icky_columns)
        X_df = X_df.drop(columns=icky_columns)
    feature_cols = X_df.columns.tolist()
    return X_df, feature_cols

def main():

    if not os.path.exists(Checkpoint_2_step_3_input_path):
        print("No Step 2 Embeddings File Found - Run Step 2!!")
        return
    #Loading Embedding Data - with column check
    dataframe = pd.read_csv(Checkpoint_2_step_3_input_path)
    print("Loaded Embeddings:", dataframe.shape)
    #Dataframe empty check
    if dataframe.empty:
        print("The Input File is EMPTY!")
        return
    #column check for closest kinase family
    if "closest_kinase_family" not in dataframe.columns:
        raise ValueError("Step 2 output is missing closest_kinase_column!")
    #checking for missing required columns
    target_columns = ["closest_kinase_family"]
    cols = [c for c in target_columns if c not in dataframe.columns]
    if cols:
        print("You are MISSING REQUIRED COLUMNS!")
        return
    
    #removing rows with missing family labels
    dataframe = dataframe.dropna(subset=["closest_kinase_family"]).copy()
    dataframe["closest_kinase_family"] = dataframe["closest_kinase_family"].astype(str).str.strip()

    #removing any blank labels
    dataframe = dataframe[dataframe["closest_kinase_family"] != ""].copy()
    if dataframe.empty:
        print("no valid family labels!")
        return
    
    #Building a Numeric Feature Table
    X_df, feature_columns = building_nfeature_table(dataframe)
    print("Useable Feature Count", len(feature_columns))

    dataframe = dataframe.loc[X_df.index].copy()
    df_model = dataframe.copy()
    df_model[feature_columns] = X_df.copy()

    if df_model["closest_kinase_family"].nunique() < 2:
        print("Please have at least 2 families for classification!!")
        return
    
    #removing rare cases??? possibly implement this but see how model behaves now
    if df_model["closest_kinase_family"].nunique() < 2:
        print("Please have at least 2 families for classification!!")
        return

    
    #Setting up Data for Model
    y = df_model["closest_kinase_family"].copy()
    X = df_model[feature_columns].copy()

    print("Label is Features?", "closest_kinase_family" in X.columns)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)



    X_train, X_test, y_train, y_test, train_idx, test_idx  = train_test_split(X, y_encoded, df_model.index, test_size=0.2, random_state=42, stratify=y_encoded)
                                                         

    rf_model = RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1, random_state=42, class_weight="balanced")
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")


    print("Step3 Accuracy:", accuracy)
    print("Step3 F1 MACRO:", f1)
    print("Step3 F1 WEIGHTED:", f1_weighted)


    #predictions table

    results = df_model.loc[test_idx].copy()
    results["true_family"] = label_encoder.inverse_transform(y_test)
    results["predicted_family"] = label_encoder.inverse_transform(y_pred)
    results["prediction_correct"] = (results["true_family"] == results["predicted_family"])

    results.to_csv(Checkpoint_2_step_3_output_path, index=False)
    print("Step 3 Predictions Have Been Saved!")

    metrics_df = pd.DataFrame({"metric": ["accuracy", "f1", "f1_weighted", "n_train", "n_test", "n_features"], 
                               "value": [accuracy, f1, f1_weighted, len(X_train), len(X_test), len(feature_columns)]})

    print(metrics_df)

    #confusion matrix
    class_names = label_encoder.classes_
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)

    print("Step 3 has Completed - Pursue Step 4!")
    
if __name__ == "__main__":
    main()