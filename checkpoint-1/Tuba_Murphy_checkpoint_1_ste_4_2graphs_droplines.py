# =========
# Checkpoint -1 Step-4
# ANN Model with cold start model
# Non egfr for training data, egfr for test
# this file has updated epocs, batch and drop the noise lines

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") #generates plot without displaying, neccessery for the server usage.
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from  sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
import pickle

warnings.filterwarnings("ignore")

Data_directory = "New_checkpoint_1_data_here"
Input_path =os.path.join(Data_directory, "cleaned_step3.csv")

Model_path = os.path.join(Data_directory, "ckp_1_step4_model.pkl")
Scale_path = os.path.join(Data_directory, "ckp_1_step4_scaler.pkl")
Log_path = os.path.join(Data_directory, "ckp_1_step4_training.log.txt")

# ==========
# Parameters 
# ==========
Hidden_units_1 = 128
Hidden_units_2 = 64

Learning_rate = 0.001
Epochs = 200
Batch_size = 256

L2_lambda = 5e-4
Dropout_rate = 0.1

Random_seed =42

# === helpers
def is_egfr_row(df):
    return (
        df["uniprot_id"].astype(str).str.contains("P00533", na=False) |
        df["target_name"].astype(str).str.contains("EGFR|ERBB1", case=False, na=False)
    )
def relu(z):
    return np.maximum(0,z)

def relu_derivative(z):
    return (z>0).astype(float)

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def count_bad_values(arr):
    arr = np.asarray(arr, dtype=np.float32)
    return np.isnan(arr).sum(), np.isinf(arr).sum()

##This is the drop line it will clear the noise
def dropout_forward(H, rate):
    mask = (np.random.rand(*H.shape) > rate).astype(float)
    return H *mask / (1-rate), mask

#=== DATA

def num_feature_table(df):
    X= df.drop(columns=["target_name", "uniprot_id", "log_affinity", "egfr", "is_egfr"], errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    return df, X, X.columns.tolist()

## Paramter weights/biases

def in_parameters(input_dim):
    np.random.seed(Random_seed)
    
    W1 = (np.random.randn(input_dim, Hidden_units_1)* np.sqrt(2/input_dim)).astype(np.float32)
    B1 = np.zeros((1, Hidden_units_1), dtype=np.float32)

    W2 = (np.random.randn(Hidden_units_1, Hidden_units_2)* np.sqrt(2/Hidden_units_1)).astype(np.float32)
    B2 = np.zeros((1, Hidden_units_2), dtype=np.float32)

    W3 = (np.random.randn(Hidden_units_2, 1)* np.sqrt(2/Hidden_units_2)).astype(np.float32)
    B3 = np.zeros((1,1), dtype=np.float32)

    return W1, B1, W2, B2, W3,B3

# forward step
def forward (X, W1, B1, W2, B2, W3, B3, training = True):
    
    Z1 = X @ W1 + B1
    H1 = relu(Z1)

    if training:
        H1, mask1 = dropout_forward(H1, Dropout_rate)
    else:
        mask1 = None
    Z2 = H1 @ W2 +B2
    H2 = relu(Z2)

    if training:
        H2, mask2 = dropout_forward(H2, Dropout_rate)
    else:
        mask2 = None
    
    y_hat = H2 @W3 +B3

    return y_hat, Z1 , H1, Z2, H2, mask1, mask2


# Backward 

def backward(X, y, y_hat, Z1, H1, Z2, H2, mask1, mask2, W1, W2,W3):

    n =X.shape[0]
    y = y.reshape(-1,1)

    dY = (2/n) * (y_hat -y)
    
    dW3 = H2.T @dY + 2 * L2_lambda *W3
    dB3 = np.sum(dY, axis =0, keepdims =True)

    dH2 = dY @ W3.T
    if mask2 is not None:
        dH2 *= mask2/(1-Dropout_rate)
    
    dZ2 = dH2 * relu_derivative(Z2)
    dW2 = H1.T @dZ2 + 2 * L2_lambda * W2
    dB2 = np.sum(dZ2, axis =0, keepdims=True)

    dH1 = dZ2 @ W2.T
    if mask1 is not None:
        dH1 *= mask1 / (1 -Dropout_rate)
    
    dZ1 =dH1 * relu_derivative(Z1)
    dW1 = X.T @ dZ1 + 2 * L2_lambda *W1
    dB1 = np.sum(dZ1, axis=0, keepdims=True)

    return dW1, dB1, dW2, dB2, dW3, dB3

# Training

def train(X_train, y_train, X_val, y_val):
    W1,B1,W2,B2,W3,B3 = in_parameters(X_train.shape[1])

    best_r2_score = -999
    best_parameters = None
    log_lines = []

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype = np.float32)
    y_val = np.asarray(y_val, dtype = np.float32)

    #nans
    x_tr_nan, xtr_inf = count_bad_values(X_train)
    y_tr_nan, y_tr_inf = count_bad_values (y_train)
    xval_nan, xval_inf = count_bad_values(X_val)
    yval_nan, yval_inf = count_bad_values(y_val)

    print(f"Debug x_train nan={x_tr_nan} inf ={xtr_inf}", flush=True)
    print(f"Debug y_train nan={y_tr_nan} inf ={y_tr_inf}", flush=True)
    print(f"Debug x_val nan={xval_nan} inf ={xval_inf}", flush=True)
    print(f"Debug y_val nan={yval_nan} inf ={yval_inf}", flush=True)
    
    if not np.isfinite(X_train).all():
        raise ValueError("X train contains Nan or inf")
    if not np.isfinite(y_train).all():
        raise ValueError(" y train contains Nan or inf")
    if not np.isfinite(X_val).all():
        raise ValueError("X val contains Nan or Inf")
    if not np.isfinite(y_val).all():
        raise ValueError(" y_val contains Nan or inf")

    for epoch in range(Epochs):

        idx = np.random.permutation(len(X_train))
        X_train = X_train[idx]
        y_train = y_train[idx]

        for i in range(0, len(X_train), Batch_size):
        
            Xb = X_train[i:i+Batch_size]
            yb = y_train[i:i+Batch_size]

            y_hat, Z1, H1, Z2, H2, m1, m2 = forward(Xb, W1, B1, W2, B2, W3,B3)
            
            if not np.isfinite(y_hat).all():
                raise ValueError(f"Training batch produced Nan/inf at epoch {epoch}, batch start {i}")


            gr = backward(Xb, yb, y_hat, Z1, H1, Z2, H2, m1, m2, W1, W2, W3)

            dW1, dB1, dW2, dB2, dW3, dB3 = gr

            W1 -= Learning_rate * dW1
            B1 -= Learning_rate * dB1
            W2 -= Learning_rate *dW2
            B2 -= Learning_rate * dB2
            W3 -= Learning_rate * dW3
            B3 -= Learning_rate * dB3

            if not np.isfinite(W1).all() or not np.isfinite(W2).all() or not np.isfinite(W3).all():
                raise ValueError(f"Model weights became Nan/inf at epoch {epoch}, batch start {i}")

        pred_val = forward (X_val, W1, B1, W2, B2, W3, B3, training=False)[0].flatten()
        pred_val = np.asarray(pred_val, dtype=np.float32)
        y_val_loc = np.asarray(y_val, dtype=np.float32)

        pred_nan, pred_if = count_bad_values(pred_val)
        yv_nan, yv_inf = count_bad_values(y_val_loc)

        if pred_nan > 0 or pred_if > 0:
            #line = f"epoch {epoch} | skipped pre_val contains NaN={pred_nan} inf={pred_if}"
            line = f"epoch {epoch} | val R2: {r2_val:.4f} | Val Loss:{value_loss:.4f}"
            log_lines.append(line)
            print(line, flush=True)
            continue
        
        if yv_nan > 0 or yv_inf > 0 :
            raise ValueError(f"y_val contains Nan={yv_nan} inf={yv_inf}")
        value_loss = mse_loss(y_val_loc, pred_val)
        r2_val = r2_score(y_val_loc, pred_val)

        #line = f"epoch {epoch} | val r2: {r2_val:.4f} | val loss {value_loss:.4f}"
        line = f"epoch {epoch} | val R2: {r2_val:.4f} | Val Loss:{value_loss:.4f}"
        log_lines.append(line)

        if epoch %20 == 0:
            print(line, flush=True)
        if r2_val > best_r2_score:
            best_r2_score = r2_val
            best_parameters = (W1.copy(), B1.copy(),W2.copy(), B2.copy(), W3.copy(), B3.copy())
    if best_parameters is None:
        raise ValueError("No valid model was found during trianing, epochs may have produced nan predictions.")
    return best_parameters, best_r2_score,log_lines

# ---
# Main

def main():

    os.makedirs(Data_directory, exist_ok=True)

    df = pd.read_csv(Input_path, low_memory=True)

    if "is_egfr" not in df.columns:
        df["is_egfr"] = is_egfr_row(df)
    
    df, X_df, cols = num_feature_table(df)
    train_df = df[df["is_egfr"] == False].copy()
    test_df = df[df["is_egfr"] == True].copy()
    #train_df = df[df["egfr"].astype(str).str.lower()=="false"].copy()
    #test_df = df[df["egfr"].astype(str).str.lower() =="true"].copy()

    if len (train_df) == 0 :
        raise ValueError("No traininf rows found after egfr split")
    
    if len(test_df) == 0 :
        raise ValueError("Nontest rows found after egfr split")
    
    #features
    X_train_full = X_df.loc[train_df.index].to_numpy(dtype =np.float32)
    X_test = X_df.loc[test_df.index].to_numpy(dtype=np.float32)

    #targets
    y_train_full = train_df["log_affinity"].to_numpy(dtype=np.float32)
    y_test = test_df["log_affinity"].to_numpy(dtype=np.float32)

    y_train_full = np.nan_to_num(y_train_full, nan =np.nanmedian(y_train_full), posinf=np.nanmedian(y_train_full), neginf=np.median(y_train_full))
    y_test = np.nan_to_num(y_test, nan = np.nanmedian(y_test), posinf=np.nanmedian(y_test), neginf=np.nanmedian(y_test))

    scaler = StandardScaler()
    X_train_full = scaler.fit_transform(X_train_full).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    X_train_full = np.nan_to_num(X_train_full, nan =0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0).astype(np.float32)

    #split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size= 0.2,
        random_state=Random_seed

    )
    #train only non egfr
    best_parameters, best_r2_score, log_lines =train(X_train, y_train,X_val, y_val)

    W1, B1, W2, B2, W3, B3 = best_parameters

    test_pred = forward(X_test, W1, B1, W2, B2, W3,B3, training=False)[0].flatten()
    test_pred = np.asarray(test_pred, dtype=np.float32)

    if not np.isfinite(test_pred).all():
        raise ValueError("Test predcition contains na or inf")
    
    test_r2 = r2_score(y_test, test_pred)
    test_mse = mse_loss(y_test, test_pred)

    plt.figure(figsize=(6,6))
    plt.scatter(y_test, test_pred, alpha=0.4)
    plt.xlabel("True log_affinity")
    plt.ylabel("Predicted log_affinity")
    plt.title("EGFR test set: true vs predicted")

    min_val = min(y_test.min(), test_pred.min())
    max_val = max(y_test.max(), test_pred.max())
    plt.plot([min_val, max_val], [min_val,max_val], linestyle= "--")

    plt.tight_layout()
    plt.savefig(os.path.join(Data_directory, "egfr_true_vs_pred.png"))
    plt.close()


   
    val_r2s = [float(line.split("Val R2: ")[1].split(" | ")[0]) for line in log_lines if "Val R2:" in line]
    
    plt.figure(figsize=(7,5))
    plt.plot(range(len(val_r2s)), val_r2s)
    plt.xlabel("Epoch")
    plt.ylabel("Validation r2 over epochs (non egfr)")
    plt.tight_layout()
    plt.savefig(os.path.join(Data_directory, "r2_value_curve.png"))
    plt.close()

    with open(Model_path, "wb") as f:
        pickle.dump(best_parameters, f)
    
    with open(Scale_path, "wb") as f:
        pickle.dump(scaler, f)
    
    with open(Log_path, "w") as f:
        for line in log_lines:
            f.write(line + "\n")

    print("Non egfr train rows:", len(X_train))
    print("Non egfr val rows:", len(X_val))
    print("EGFR test rows:", len(X_test))
    print(f"Best Val R2: {best_r2_score:.4f}")
    print(f"egfr test r2: {test_r2:.4f}")
    print(f"egfr test mse: {test_mse:.4f}")


if __name__ == "__main__":
    main()









