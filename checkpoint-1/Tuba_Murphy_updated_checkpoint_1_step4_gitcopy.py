# ========
# CheckPoint -1 Step -4
# KNN regression
# Training on non-efgr
# Test on EGFR
# Numberic cleaned up
# Pickle

import os
import warnings
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

Data_directory = "New_checkpoint_1_data_here"
Checkpoint_1_step_3_table_path = os.path.join(Data_directory, "cleaned_step3.csv")
Chck1_step4_model_output_path = os.path.join(Data_directory,"checkpoint_1_step4_knn_model_dat_.pkl" )
Scaler_output_path = os.path.join(Data_directory, "checkpoint_1_scaler_data.pkl")
Prediction_output_path = os.path.join(Data_directory,"checkpoint_1_egfr_predictions_data.csv")
Feature_importance_path = os.path.join(Data_directory, "checkpoint_1_feature_importance_data.csv")


K_values = [3,5,7,9,11]

def is_egfr_row(df):
    condition_1 = df["uniprot_id"].astype(str).str.contains("P00533", na=False)
    condition_2 = df["target_name"].astype(str).str.contains("EGFR|ERBB1", case=False, na=False)
    return condition_1 | condition_2

def numeric_feature_table(df):
    non_feature_cols = ["target_name", "uniprot_id", "is_egfr", "log_affinity"]
    candidate_cols = [ c for c in df.columns if c not in non_feature_cols]
    
    X_df = df[candidate_cols].copy()
    
    #convert candidate feature columns to numeric
    for c in X_df.columns:
        X_df[c] = pd.to_numeric(X_df[c], errors="coerce")

    #drops columns that are entrley nan

    all_nan_cols = X_df.columns[X_df.isna().all()].tolist()
    if all_nan_cols:
        print("Dropping the all Nan feature columns:", len(all_nan_cols))
        print(all_nan_cols[:20], "..." if len(all_nan_cols) >20 else "")
        X_df = X_df.drop(columns = all_nan_cols)

    #replace inf with Nan
    X_df = X_df.replace([np.inf, -np.inf], np.nan)

    #Keeps rows with a non missing target 
    valid_target = pd.to_numeric(df["log_affinity"], errors="coerce").notna()
    cleaned_df = df.loc[valid_target].copy()
    cleaned_df["log_affinity"] = pd.to_numeric(cleaned_df["log_affinity"], errors="coerce")
    X_df =X_df.loc[valid_target].copy()


    #impute reamiang missing feature values with columns and medians 
    medians = X_df.median(numeric_only=True)
    X_df = X_df.fillna(medians)

    #remove columns still cointaing Nan after median fill
    remaining_bad_cols = X_df.columns[X_df.isna().any()].tolist()
    if remaining_bad_cols:
        print("Nan columns dropped after median fill:", len (remaining_bad_cols))
        print(remaining_bad_cols[:20], "..." if len(remaining_bad_cols) > 20 else "")
        X_df = X_df.drop(columns=remaining_bad_cols)

    #Last alignment
    cleaned_df = cleaned_df.loc[X_df.index].copy()
    feature_cols =X_df.columns.tolist()

    print("Final usable feature count:", len(feature_cols))
    return cleaned_df, X_df, feature_cols

def main():
    if not os.path.exists(Checkpoint_1_step_3_table_path): #low_memory=False)
        print("Srep 3 table isn't found")
        return 
    
    datafile = pd.read_csv(Checkpoint_1_step_3_table_path , low_memory=False)
    print("Load step3 table shape:", datafile.shape)

    if datafile.empty:
        print("Error step-3 input is empty")
        return
    
    needed = ["uniprot_id", "target_name", "log_affinity"]
    missing = [c for c in needed if c not in datafile.columns]
    if missing:
        print("Missing required columns:", missing)
        return
    
    datafile, X_df, feature_cols = numeric_feature_table(datafile)
    print("Shape of the numeric after cleanup:", datafile.shape)

    train_filter = ~is_egfr_row(datafile) # it only takes non egfr for training
    test_filter  = is_egfr_row(datafile)

    train_df = datafile.loc[train_filter].copy()
    test_df = datafile.loc[test_filter].copy()

    X_train_df =X_df.loc[train_filter].copy()
    X_test_df = X_df.loc[test_filter].copy()

    print("Training on non-EGFR:", train_df.shape[0])
    print("Testing rows EGFR data:", test_df.shape[0])

    if train_df.empty:
        print("Error train is data is empty")
        return
    if test_df.empty:
        print("Error EGFR test dara is empty")
        return

    X = X_train_df.values 
    y = train_df["log_affinity"].values
    
    X_test = X_test_df.values
    y_test = test_df["log_affinity"].values

    X_train, X_val,y_train, y_val = train_test_split(
        X,y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    parameter_grid = { "n_neighbors" : K_values}
    grid = GridSearchCV(
        KNeighborsRegressor(),
        parameter_grid,
        scoring="r2",
        cv=3
    )

    grid.fit(X_train, y_train)
    great_model =grid.best_estimator_

    print("Best k:", grid.best_params_)
    y_val_pred = great_model.predict(X_val)
    val_r2 = r2_score(y_val, y_val_pred)
    print("Validation R^2:", val_r2)

    X_full = scaler.fit_transform(X)
    great_model.fit(X_full, y)

    X_test_last = scaler.transform(X_test)
    y_pred = great_model.predict(X_test_last)


    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    np_mse = np.sqrt(mse)

    print("Checkpoint-1 EFGR Test R^2 Score:", r2)
    print("Checkpoint-1 EGFR Tests RMSE:", np_mse)

    test_df["pred_log_affinity"] = y_pred
    test_df["prediction_error"] = test_df["pred_log_affinity"] - test_df["log_affinity"]
    test_df.to_csv(Prediction_output_path, index= False)

    result = permutation_importance(
        great_model,
        X_test_last,
        y_test,
        n_repeats=5,
        random_state=42
    )

    importance_score_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": result.importances_mean
    }).sort_values(by="importance", ascending=False)

    importance_score_df.to_csv(Feature_importance_path, index=False) 

    with open(Chck1_step4_model_output_path, "wb") as f:
        pickle.dump(great_model,f)

    with open(Scaler_output_path, "wb") as f:
        pickle.dump(scaler,f)
    
    print("Yay! Prediction saved in:", Prediction_output_path)
    print("Yay!Feature importance saved in:", Feature_importance_path)
    print("Yay! Model saved in:", Chck1_step4_model_output_path)
    print("yay!scaler saved in:", Scaler_output_path)


if __name__ == "__main__":
    main()
   
    









              


    




