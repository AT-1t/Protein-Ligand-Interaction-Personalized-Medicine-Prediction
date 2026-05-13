import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb

# =============================================================================
# Load data
# =============================================================================

file_path = "checkpoint3_fusion_lightgbm_sample_1M.parquet"
output_dir = "checkpoint_3_ann_outputs"
os.makedirs(output_dir, exist_ok=True)

print("Loading from:", file_path)
df = pd.read_parquet(file_path)

target_col = "model_1_binding_pred" 


print("Loaded shape:", df.shape)
print("Target column exists:", target_col in df.columns)
print("Target nulls:", df[target_col].isna().sum())

# =============================================================================
# Separate y and X
# =============================================================================
y = df[target_col].copy()

drop_cols = [
    target_col,
    "patient_id",
    "ligand_row_id",
    "fusion_row_id",
    "predicted_family",
    "prediction_confidence",
    "prediction_correct",
    "true_family",
]
drop_cols = [c for c in drop_cols if c in df.columns]
X = df.drop(columns=drop_cols).copy()


#dropping target_name to see if r2 improves from 0.78
if "target_name_x" in X.columns:
	X = X.drop(columns= ["target_name_x"])

print("X columns preview:", X.columns[:10].tolist())
print("X shape:", X.shape)
print("y shape:", y.shape)

# =============================================================================
# Patient-level split
# =============================================================================
print("\n--- Patient-level train/val/test split ---")

unique_patients = df["patient_id"].dropna().unique()
print("Total unique patients:", len(unique_patients))

train_patients, temp_patients = train_test_split(
    unique_patients,
    test_size=0.30,
    random_state=42
)

val_patients, test_patients = train_test_split(
    temp_patients,
    test_size=0.50,
    random_state=42
)

train_mask = df["patient_id"].isin(train_patients)
val_mask = df["patient_id"].isin(val_patients)
test_mask = df["patient_id"].isin(test_patients)

X_train = X.loc[train_mask].copy()
X_val = X.loc[val_mask].copy()
X_test = X.loc[test_mask].copy()

y_train = y.loc[train_mask].copy()
y_val = y.loc[val_mask].copy()
y_test = y.loc[test_mask].copy()

print("Row counts:")
print("  Train:", len(X_train))
print("  Val:  ", len(X_val))
print("  Test: ", len(X_test))

print("Patient counts:")
print("  Train:", len(train_patients))
print("  Val:  ", len(val_patients))
print("  Test: ", len(test_patients))
### Added 
print("Target means:")
print("  Train:", y_train.mean())
print("  Val:  ", y_val.mean())
print("  Test: ", y_test.mean())

print("Leakage checks:")
print("  Train/Val overlap:", len(set(train_patients) & set(val_patients)))
print("  Train/Test overlap:", len(set(train_patients) & set(test_patients)))
print("  Val/Test overlap:", len(set(val_patients) & set(test_patients)))
### Added above
# =============================================================================
# LightGBM categorical handling
# =============================================================================
categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
print("\nCategorical columns:", categorical_cols)

for col in categorical_cols:
    X_train[col] = X_train[col].astype("category")
    X_val[col] = X_val[col].astype("category")
    X_test[col] = X_test[col].astype("category")
##### Addedd
# =============================================================================
# First-pass LightGBM regressor
# =============================================================================
print("\n--- Training first-pass LightGBM regressor ---")
####
model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
#running time for model 
import time
start_time = time.time()


model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",
    categorical_feature=categorical_cols
)

end_time = time.time()
training_time_seconds = end_time - start_time
training_time_minutes = training_time_seconds / 60

print(f"\nTraining time: {training_time_seconds:.2f} seconds ({training_time_minutes:.2f} minutes)")
# =============================================================================
# Predictions
# =============================================================================
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# =============================================================================
# Metrics
# =============================================================================
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
##adede
val_mae = mean_absolute_error(y_val, y_val_pred)

val_r2 = r2_score(y_val, y_val_pred)
###
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
####Added
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\nValidation Metric (no target name)s:")
print("  RMSE:", val_rmse)
print("  MAE: ", val_mae)
print("  R²:  ", val_r2)

print("\nTest Metrics:")
print("  RMSE:", test_rmse)
print("  MAE: ", test_mae)
print("  R²:  ", test_r2)

# =============================================================================
# Save predictions /the test results now have ligand id and and Log P
# =============================================================================
###changed 
output_columns =["patient_id", "ligand_row_id", "logP"] #just to make sure previous data has all the coulmns
output_columns = [c for c in output_columns if c in df.columns] #so it doesn't give an error 
test_results = df.loc[test_mask, output_columns].copy()
###
test_results["y_test"] = y_test.values
test_results["y_pred"] = y_test_pred
test_results["residual"] = test_results["y_test"] - test_results["y_pred"]

test_results_path = os.path.join(output_dir, "checkpoint3_lightgbm_test_predictions.csv")
test_results.to_csv(test_results_path, index=False)
###Addition for ranked ligand file to get the easiest attachments
ranked_ligandpath = os.path.join(output_dir, "checkpoint3_ligand_ranked_predictions.csv")
test_results.sort_values("y_pred", ascending =False).to_csv(ranked_ligandpath, index=False)
print("saved ranked ligand predictions to", ranked_ligandpath)
######
print("\nSaved test predictions to:", test_results_path)


# =============================================================================
# Save metrics
# =============================================================================
metrics_df = pd.DataFrame({
    "split": ["val", "test"],
    "rmse": [val_rmse, test_rmse],
    "mae": [val_mae, test_mae],
    "r2": [val_r2, test_r2]
})

metrics_path = os.path.join(output_dir, "checkpoint3_lightgbm_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)

print("Saved metrics to:", metrics_path)

# =============================================================================
# Feature importances
# =============================================================================
importance_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

importance_path = os.path.join(output_dir, "checkpoint3_lightgbm_feature_importance.csv")
importance_df.to_csv(importance_path, index=False)

print("Saved feature importances to:", importance_path)
print("\nTop 20 features:")
print(importance_df.head(20))

# =============================================================================
# Presentation PLots
# =============================================================================

import os
plt.figure(figsize=(8, 5))
#test v train
plt.hist(y_train, bins=50, alpha=0.5, label="Train")
plt.hist(y_val, bins=50, alpha=0.5, label="Validation")
plt.hist(y_test, bins=50, alpha=0.5, label="Test")

plt.title("Target Distribution: Train vs Validation vs Test")
plt.xlabel("Binding Affinity (binding_affinity_y)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()

dist_plot_path = os.path.join(output_dir, "target_distribution_train_val_test.png")
plt.savefig(dist_plot_path, dpi=300, bbox_inches="tight")
plt.close()

print("Saved target distribution plot to:", dist_plot_path)

#pred v true
plt.figure(figsize=(6, 6))
plt.scatter(y_val, y_val_pred, alpha=0.3)

min_val = min(y_val.min(), y_val_pred.min())
max_val = max(y_val.max(), y_val_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

plt.title("Validation: True vs Predicted Binding Affinity")
plt.xlabel("True binding_affinity_y")
plt.ylabel("Predicted binding_affinity_y")
plt.tight_layout()

val_scatter_path = os.path.join(output_dir, "val_true_vs_pred.png")
plt.savefig(val_scatter_path, dpi=300, bbox_inches="tight")
plt.close()

print("Saved validation scatter plot to:", val_scatter_path)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_test_pred, alpha=0.3)

min_test = min(y_test.min(), y_test_pred.min())
max_test = max(y_test.max(), y_test_pred.max())
plt.plot([min_test, max_test], [min_test, max_test], linestyle="--")

plt.title("Test: True vs Predicted Binding Affinity")
plt.xlabel("True binding_affinity_y")
plt.ylabel("Predicted binding_affinity_y")
plt.tight_layout()

test_scatter_path = os.path.join(output_dir, "test_true_vs_pred.png")
plt.savefig(test_scatter_path, dpi=300, bbox_inches="tight")
plt.close()

print("Saved test scatter plot to:", test_scatter_path)


#residuals to remove cluttered points
residuals = y_test - y_test_pred

plt.figure(figsize=(6,5))
plt.scatter(y_test_pred, residuals, alpha=0.3)
plt.axhline(0, linestyle='--')

plt.title("Residuals vs Predicted (Test)")
plt.xlabel("Predicted binding_affinity_y")
plt.ylabel("Residual (True - Predicted)")
plt.tight_layout()

plt.savefig("residual_plot.png", dpi=300)
plt.close()

plt.figure(figsize=(6,5))
plt.hist(residuals, bins=50)

plt.title("Error Distribution (Test)")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig("error_distribution.png", dpi=300)
plt.close()

plt.figure(figsize=(6,6))
plt.hexbin(y_test, y_test_pred, gridsize=50)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--')

plt.title("Test: True vs Predicted (Density)")
plt.xlabel("True")
plt.ylabel("Predicted")
plt.tight_layout()

plt.savefig("hexbin_plot.png", dpi=300)
plt.close()
