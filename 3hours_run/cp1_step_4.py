"""
All kinase data.Weights added and EGFR oversampled."
3 hidden layers --> 
early stopping --> fto reduce overfitting and save time and gpu.
debugging --> nans, infinte vals and missing vals got checked and fixed.
I had a lot simpler code, my project friends requested add all kinase dataset to preveent
circularization in fusion step for checkpoint 3 step2.
I learned a lot of new codes some of them weren't included during class hours 
so I reference some github repos and other online blogs with similar code.

Model 1's  saved x and y scaler outputa are used during 
chekpoint3 step2  fusion process so the trained ANN can generate 
bining affinity predictions before 
training on the lightGBM proxy model.


"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pickle 

warnings.filterwarnings("ignore")

DIRECTORY ="New_checkpoint_1_data_here"
INPUT_PATH = os.path.join(DIRECTORY, "checkpoint_1_step_3_No_smiles_spam_feature_table_data.csv") #cleaned_step3.csv"

Model_1_Egfr_w_path = os.path.join(DIRECTORY,"ckp1_step4_model_1_egfr_weights.pkl")
Scale_ckp1_step4_egfr_only_path = os.path.join(DIRECTORY, "ckp1_step4_model_1_egfr_weights_scaler.pkl")
Y_scale_path = os.path.join(DIRECTORY, "ckp1_step4_model_1_egfr_weights_y_scaler.pkl")
log_path = os.path.join(DIRECTORY, "ckp1_step4_model_1_egfr_weights_log.txt")

#I ran and stop this ann over 100 times with different paramters.
#this one seesm to work best for all kinase data.

Hidden_unit_1 = 224
Hidden_unit_2 = 112
Hidden_unit_3 = 56

Learning_rate = 0.0003
Min_learning_rate = 0.00003
Epochs = 1800
Batch_size = 256

L2_lambda = 2.5e-4
Dropout_rate = 0.16
Grad_clip_value = 5.0

#It will stop training after 140 epoch if there no improvement
#leanring rate will wait 50 epoch wihout improvement that lower learning rate, it prints to view the ouput.
#learning rate decay is going to loweer traing rate by 30% if model deosn't improve.
#min delta  is the min improvement for early stopping.
#Reference:
#https://keras.io/api/callbacks/early_stopping/
Early_stop_pat = 140
Learning_rate_pat = 50
Learning_rate_decay = 0.7
Min_delta = 3e-4

Random_seed = 42

train_data_frac = 0.80
val_frac_in_train = 0.20

#this is the balance egfr overall kinase data.
Egfr_class_weight = 2.5
Non_egfr_class_weight = 1.0
Egfr_oversample_multiplier = 2


def is_egfr(df):
    return(
        df["uniprot_id"].astype(str).str.contains("P00533", na=False) |
        df["target_name"].astype(str).str.contains("EGFR|ERBB1", case=False, na= False)
    )
#or ERBB1 is another name for EGFR
def loss_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def loss_rmse(y_true, y_pred):
    return np.sqrt(loss_mse(y_true, y_pred))

#it checks nan, inf and missing vals.
#so model doesn't crash.
#https://numpy.org/devdocs/reference/generated/numpy.isinf.html
#it helped debugging increasing the R2 scores for all kinase data somehow.
def values_bad_count(input_arr):
    input_arr = np.asarray(input_arr, dtype=np.float32)
    return np.isnan(input_arr).sum(), np.isinf(input_arr).sum()

def feature_tab_num(df):
    X = df.drop(
        columns = ["target_name", "uniprot_id", "log_affinity", "egfr", "is_egfr"],
        errors ="ignore"
    )
    X = X.apply(pd.to_numeric, errors ="coerce")
    X= X.replace([np.inf, -np.inf], np.nan)
    X= X.fillna(X.median())
    return df, X, X.columns.tolist()
"""
It arrages data into ranges instead of using raw values.
Bining also prevents erros, if data has to few unique values or duplicated ranges.
Reference fot his code:
#https://github.com/search?/q=pd.cut+labels+False+duplicates+drop&type=code
This a newly learned code.
"""
def make_bins(input_series, n_bins=10):
    input_series = pd.Series(input_series).astype(np.float32)
    different_count =input_series.nunique()
    bin_count = int(max(2, min(n_bins, different_count)))
    if bin_count < 2:
        return None
    try:
        return pd.qcut(input_series, q=bin_count, labels = False, duplicates ="drop")
    except Exception:
        return None
    
""" 
our data was imbalanced comprate  non EGFR rows, EGFR rows were minority.
and our patient sample has egfr spefic mutations, so we want model to egfr speicifc biding more.
Adding sample weights made model to pay attention to EGFR rowsmore.

#Reference code:
#https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_sample_weight.html
#https://github.com/HealthRex/CDSS/blob/a24edec8ac6dbb0bf7984d6ef82f59cc2f3805e4/scripts/Monitoring_Induced_Feedback-main/monitoring.py#L25
#he uses sampling weights and adherence weights, a lot more than this model uses.

"""

def make_sample_weights(df):
    return np.where(
        df["is_egfr"].astype(bool).to_numpy(),
        Egfr_class_weight,
        Non_egfr_class_weight
    ).astype(np.float32)


#rng is better than random.randt
#becuase it uses fixed radnom seed for preprodicible Ann weight initializtion and drop filtet
#np.random.radn creates an uncontorolled generation.
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, rng, l2_lambda=0.0):
        self.weights = (rng.randn(n_inputs, n_neurons) * np.sqrt(2.0/n_inputs)).astype(np.float32)
        self.biases = np.zeros((1, n_neurons), dtype=np.float32)
        self.l2_lambda = l2_lambda
        self.learning_rate_pat = Learning_rate_pat #need to chec this later
        #self.weight_std = np.sqrt(2.0/n_inputs)

    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs @ self.weights + self.biases
    
    def backward(self, dvalues):
        self.dweights = self.inputs.T @ dvalues + (2.0 * self.l2_lambda *self.weights)
        self.dbiases = np.sum(dvalues, axis =0, keepdims=True)
        self.dinputs = dvalues @ self.weights.T

class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
#Binary filter is going to droput radnomly tured off nerons
#it will prevent overfitting
#I can use this later for G protein couples
class Layer_Drop:
    def __init__(self, rate, rng):
        self.rate = rate
        self.rng = rng
    
    def forward(self, inputs, training=True):
        #new = np.sum(self.binary_filter)
        #addition =self.binary_filter.size
        #return new, addition
        self.inputs = inputs
        if training:
            self.binary_filter = (self.rng.rand(*inputs.shape) > self.rate).astype(np.float32)
            #drop_rate = np.mean(self.binary_filter)
            self.output = inputs * self.binary_filter/(1.0 - self.rate)

        else: 
            self.binary_filter = None
            self.output = inputs

    def backward(self, dvalues):
        if self.binary_filter is None:
            self.dinputs = dvalues
        else:
            self.dinputs = dvalues * self.binary_filter / (1.0 - self.rate)

class MSE_Loss:
    def forward(self, y_pred, y_true, sample_weights=None):
        y_true = y_true.reshape(-1,1).astype(np.float32)

        if sample_weights is None:
            return np.mean((y_pred -y_true) ** 2)
        
        sample_weights= sample_weights.reshape(-1,1).astype(np.float32)
        return np.sum(sample_weights * ((y_pred -y_true) **2)) / np.sum(sample_weights)
    
    def backward(self, y_pred, y_true, sample_weights=None):
        y_true = y_true.reshape(-1,1).astype(np.float32)

        if sample_weights is None:
            n = y_true.shape[0]
            self.dinputs = (2.0/ n) * (y_pred - y_true)
        else: 
            sample_weights = sample_weights.reshape(-1,1).astype(np.float32)
            weight_sum = np.sum(sample_weights)
            if weight_sum == 0:
                weight_sum = 1.0
            self.dinputs = (2.0 / weight_sum) * sample_weights *(y_pred -y_true)
#I printed learning rate reduction in output to follow models performance.
class SGradientD_Optimizer:
    def __init__ (self, learning_rate = 0.0005, momentum=0.9, min_learning_rate =0.00003):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.momentum = momentum
        self.min_learning_rate = min_learning_rate

    def param_update(self, layer):
        if not hasattr(layer, "weight_momentums"):
            layer.weight_momentums = np.zeros_like(layer.weights, dtype=np.float32)
            layer.bias_momentums = np.zeros_like(layer.biases, dtype=np.float32)

        layer.weight_momentums = (
            self.momentum * layer.weight_momentums -
            self.current_learning_rate * layer.dweights
        )   

        layer.bias_momentums =(
            self.momentum * layer.bias_momentums -
            self.current_learning_rate * layer.dbiases
    
        )

        layer.weights += layer.weight_momentums
        layer.biases += layer.bias_momentums

    def lower_learning_rate(self, factor =0.7):
        self.current_learning_rate = max(
            self.min_learning_rate,
            self.current_learning_rate *factor
        ) 
#Learning_rate_pat = Learning_rate_pat
#I had to change the class name to kinase after addding all the dataset
#Learning rate decay is crucial for this data 
#it wwill slow the training if impprovements become small.
#so we get more stable training and better results.
class KinaseEgfrWeightedANN:
    def __init__(
            self,
            input_dim,
            hidden_units=(192, 96,48),
            learning_rate =0.0003,
            dropout_rate=0.18,
            l2_lambda =3e-4,
            random_seed=42,
            min_learning_rate =0.00003,
            gradient_clip_value = 5.0,
            Early_stop_pat = 120,
            Learning_rate_pat= 45,
            Learning_rate_decay = 0.7,
            Min_delta= 3e-4,
            
    ): 

            np.random.seed(random_seed)
            self.rng = np.random.RandomState(random_seed)

            h1, h2, h3 = hidden_units

            self.dense1 = Layer_Dense(input_dim, h1, self.rng, l2_lambda=l2_lambda)
            self.relu1 =  ReLU()
            self.drop1 = Layer_Drop(dropout_rate, self.rng)


            self.dense2 = Layer_Dense(h1, h2, self.rng, l2_lambda=l2_lambda)
            self.relu2 =  ReLU()
            self.drop2 = Layer_Drop(dropout_rate, self.rng)


            self.dense3 = Layer_Dense(h2, h3, self.rng, l2_lambda=l2_lambda)
            self.relu3 =  ReLU()
            self.drop3 = Layer_Drop(dropout_rate, self.rng)

            self.out = Layer_Dense(h3, 1, self.rng, l2_lambda=l2_lambda)

            self.loss_function= MSE_Loss()
            self.optimizer = SGradientD_Optimizer(
                learning_rate=learning_rate,
                momentum= 0.9,
                min_learning_rate= min_learning_rate
            )

            self.gradient_clip_value = gradient_clip_value
            self.early_stop_patience = Early_stop_pat
            self.learning_rate_pat = Learning_rate_pat
            self.learning_rate_decay = Learning_rate_decay
            self.min_delta = Min_delta
    
            #it starts r2 very low so model replace it with newly trained score right away.
            #https;//docs.python.org/3/library/functions/htm
            self.best_r2_score = -999.0
            self.best_val_rmse = np.inf
            self.best_parameters = None
            self.log_lines = []
            self.history = {
                "epoch" : [],
                "train_r2":[],
                "train_rmse":[],
                "val_r2":[],
                "val_rmse":[]
            }
    
    def get_parameters(self):
        return(
            self.dense1.weights.copy(), self.dense1.biases.copy(),
            self.dense2.weights.copy(), self.dense2.biases.copy(),
            self.dense3.weights.copy(), self.dense3.biases.copy(),
            self.out.weights.copy(), self.out.biases.copy()
        )


    def set_parameters(self, params):
        (
         
         self.dense1.weights, self.dense1.biases,
         self.dense2.weights, self.dense2.biases,
         self.dense3.weights, self.dense3.biases,
         self.out.weights, self.out.biases
        ) = map(lambda x: x.copy(), params)

    def forward(self, X, training=True):
        self.dense1.forward(X)
        self.relu1.forward(self.dense1.output)
        self.drop1.forward(self.relu1.output, training =training)

        self.dense2.forward(self.drop1.output)
        self.relu2.forward(self.dense2.output)
        self.drop2.forward(self.relu2.output, training =training)

        self.dense3.forward(self.drop2.output)
        self.relu3.forward(self.dense3.output)
        self.drop3.forward(self.relu3.output, training =training)

        self.out.forward(self.drop3.output)
        return self.out.output
    
    def backward(self, y_pred, y_true, sample_weights=None):
        self.loss_function.backward(y_pred, y_true, sample_weights=sample_weights)
        self.out.backward(self.loss_function.dinputs)

        self.drop3.backward(self.out.dinputs)
        self.relu3.backward(self.drop3.dinputs)
        self.dense3.backward(self.relu3.dinputs)


        self.drop2.backward(self.dense3.dinputs)
        self.relu2.backward(self.drop2.dinputs)
        self.dense2.backward(self.relu2.dinputs)


        self.drop1.backward(self.dense2.dinputs)
        self.relu1.backward(self.drop1.dinputs)
        self.dense1.backward(self.relu1.dinputs)
    #this keeps weight updates in stable range, so ann learneasily and avoids nan or any other unstable vals.
    def clip_gradients(self):
        list(
            map(
                lambda layer:(
                    setattr(
                        layer,
                        "dweights",
                        np.clip(layer.dweights, -self.gradient_clip_value, self.gradient_clip_value)
                    ),

                    setattr(
                        layer,
                        "dbiases",
                        np.clip(layer.dbiases, -self.gradient_clip_value, self.gradient_clip_value)    
                    )

                ),
                [self.dense1, self.dense2, self.dense3, self.out]
            )
        )
    #this going to cut off the large gradience for training stabilty.
    #it uodates with weeights ,biases and the optimzer.
    
    def update_grad_params(self):
        self.clip_gradients()
        list(
            map(
                lambda layer: self.optimizer.param_update(layer),
                [self.dense1, self.dense2, self.dense3, self.out]
            )
        )
    #had a prediction function in class ANN III code I think.
    def prediction(self, X):

        return self.forward(X, training= False).flatten().astype(np.float32)
    
    def train(self, X_train, y_train, X_val, y_val,sample_weights_train=None, epochs=1800, batch_size=256):
        if sample_weights_train is None:
            sample_weights_train = np.ones(len(y_train), dtype=np.float32)
        else:
            sample_weights_train = np.asarray(sample_weights_train, dtype=np.float32)

        
        
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        X_val = np.asarray(X_val, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=np.float32)

        #i had to add the count bad vals,all kinase data doesn't learn easily.r2scores were low.
        #need to check this later for ion channels data as well.
        x_train_nan, x_train_inf = values_bad_count(X_train)
        y_train_nan, y_train_inf = values_bad_count(y_train)
        x_val_nan, x_val_inf =values_bad_count(X_val)
        y_val_nan, y_val_inf = values_bad_count(y_val)


        print(f"Debug x_train nan:{x_train_nan} inf:{x_train_inf}" ,flush=True)
        print(f"Debug y_train nan:{y_train_nan} inf:{y_train_inf}" ,flush=True)
        print(f"Debug x_val nan:{x_val_nan} inf:{x_val_inf}" ,flush=True)
        print(f"Debug y_val nan:{y_val_nan} inf:{y_val_inf}" ,flush=True)

        ###### Debugging this part is really important
        ### i had mismatch values wihout it.
        #

        if not np.isfinite(X_train).all():
            raise ValueError("X train contains nan or inifinte")
        
        if not np.isfinite(y_train).all():
            raise ValueError("Y train contains nan or inifinte")
        
        if not np.isfinite(X_val).all():
            raise ValueError("X val contains nan or inifinte")
        
        if not np.isfinite(y_val).all():
            raise ValueError("Y validation contains nan or inifinte")

        #if not np.isfinite(sample_weights_train).all():
        batching = lambda X, y, w: list(
            map(
                lambda start: (X[start:start +batch_size],
                                y[start:start +batch_size],
                                w[start:start + batch_size],
                                start
                ),
                range(0, len(X), batch_size)
            ) 

        )
        
        #w_b  is going to help egfr get influenced more on training
        def batching_dist(batch, epoch):
            X_b, y_b,w_b, start = batch
            #X_b,y_b,w_b = map(lambda x: np.asarray(x, dytype =np.float32))

            y_hat = self.forward(X_b, training=True)

            if not np.isfinite(y_hat).all():
                raise ValueError(f"Training batch executed nan and infinite at epoch {epoch}, batch start{start}")
            
            self.backward(y_hat, y_b, sample_weights=w_b)
            self.update_grad_params()

            weights_wrong= list(
                map(
                    lambda w: not np.isfinite(w).all(),
                    [self.dense1.weights, self.dense2.weights, self.dense3.weights, self.out.weights]

                )
            )
            ######## newly learned code 
            #weights wring checks numerical imbalance.
            if any(weights_wrong):
                raise ValueError(f"Not, good.Model weights turned nan and inifnite at epoch{epoch}, batch start{start}")
            
            return 0
        #early stopping helps to prevent uncnessasry cpu/gpu usage and overfitting.
        early_stop_waiting = [0]
        waiting_lr = [0]

        def compute_epoch(epoch):
            idx= self.rng.permutation(len(X_train))
            X_epoch =X_train[idx]
            y_epoch =y_train[idx]
            w_epoch =sample_weights_train[idx]

            list(
                map(
                    lambda batch: batching_dist(batch, epoch),
                    batching(X_epoch, y_epoch, w_epoch)
                )
            )

            pred_train = self.prediction(X_train)
            pred_val = self.prediction(X_val)
            #pred_test =self.prediction(X_test) not here

            y_train_local = np.asarray(y_train, dtype=np.float32)
            y_val_local= np.asarray(y_val, dtype=np.float32)

            pred_train_nan, pred_train_inf = values_bad_count(pred_train)
            pred_val_nan, pred_val_inf = values_bad_count(pred_val)
            y_train_nan, y_train_inf = values_bad_count(y_train_local)
            y_val_nan, y_val_inf= values_bad_count(y_val_local)

            if pred_train_nan > 0 or pred_train_inf > 0:
                line = (
                    f" epoch {epoch}  train r2: skipped  train rmse: skipped"
                    f" pred_train contains nan :{pred_train_nan} inf:{pred_train_inf}"
                )

                self.log_lines += [line]
                print(line, flush=True)
                return 0
            
            if pred_val_nan > 0 or pred_val_inf > 0:
                line = (
                    f" epoch {epoch}  val r2: skipped  val rmse:skipped"
                    f" pred_val contains nan = {pred_val_nan} inf:{pred_val_inf}"
                )
                self.log_lines+= [line]
                print(line, flush =True)
                return 0
            
            if y_train_nan > 0 or y_train_inf >0:
                raise ValueError(f"y_train contains nan:{y_train_nan}, infinity :{y_train_inf}")
            
            if y_val_nan > 0 or y_val_inf >0:
                raise ValueError(f"y_val contains nan:{y_val_nan}, infinity :{y_val_inf}")
            
            train_mse = loss_mse(y_train_local, pred_train)
            train_rmse = np.sqrt(train_mse)
            train_r2 = r2_score (y_train_local, pred_train)

            val_rmse = loss_mse(y_val_local, pred_val)
            val_rmse = np.sqrt(val_rmse)
            val_r2 =r2_score(y_val_local, pred_val)

            #history is good for tracking models progress during training dataset curshed multiple time
            #it monitor the progress of the model during training going over each epochs
            #it had to debug the code a lot this part helped to see the improvent
            # we learned somthing similar to this in pytorch or tensorflow 
            #but their calculations were diferent 
            #https://github.com/skorch-dev/skorch/issues/245
            #they use the history for bathcing similar but  not epochs.
            self.history["epoch"] += [epoch+1]
            self.history["train_r2"] += [float(train_r2)]
            self.history["train_rmse"] += [float(train_rmse)]
            self.history["val_r2"] += [float(val_r2)]
            self.history["val_rmse"] += [float(val_rmse)]

            line = (
            f"epoch {epoch +1} "
            f"train r2 {train_r2:.4f}  train rmse:{train_rmse:.5f}   "
            f"val r2: {val_r2:.4f}  val rmse {val_rmse:.4f} "
            f"lr: {self.optimizer.current_learning_rate:.4f}"
            )

            self.log_lines += [line]
            
            if (epoch +1) % 10 == 0:
                print(line, flush=True)

            improved_rmse = val_rmse <(self.best_val_rmse - self.min_delta)

            if improved_rmse:
                self.best_val_rmse = val_rmse
                self.best_r2_score =val_r2
                self.best_parameters = self.get_parameters()
                early_stop_waiting[0] =0
                waiting_lr[0] = 0
            else:
                early_stop_waiting[0] += 1
                waiting_lr[0] += 1

            if waiting_lr[0] >= self.learning_rate_pat:
                previous_lr = self.optimizer.current_learning_rate
                #current_step = self.history(epoch)
                self.optimizer.lower_learning_rate(self.learning_rate_decay) #frpgt this before
                waiting_lr[0] = 0

                if self.optimizer.current_learning_rate < previous_lr:
                    lr_line = (
                        f" epoch {epoch +1} learning rate reduced to"
                        f" {self.optimizer.current_learning_rate:.4f}"
                    )

                    self.log_lines += [lr_line]
                    print(lr_line, flush=True)

            if early_stop_waiting[0] >= self.early_stop_patience:
                stop_line = (
                    f"early stopping at epcpch {epoch +1} "
                    f"best val rmse: {self.best_val_rmse:.4f} "
                    f"best val r2:{self.best_r2_score:.4f}"
                )

                self.log_lines +=[stop_line]
                print(stop_line, flush=True)
                return 1
                
            return 0
            
        epoch = 0
        stop_training = 0

        while epoch < epochs and stop_training == 0:
            stop_training = compute_epoch(epoch)
            epoch +=1

        if self.best_parameters is None:
            raise ValueError("No  valid model was found ruing training :/")
          #self.set_parameters(self.best_parameters)  
        return self.best_parameters, self.best_r2_score, self.log_lines
        #return 0 #this is to contine runng epoch 

def main():
    os.makedirs(DIRECTORY, exist_ok=True)

    df = pd.read_csv(INPUT_PATH, low_memory=True)
    if "is_egfr" not in df.columns:
        df["is_egfr"] = is_egfr(df)
    
    #df, X_df, cols = feature_tab_num(df)
    
    def feature_tab_num_no_fill(df):
        X = df.drop(
            columns=["target_name", "uniprot_id", "log_affinity", "egfr", "is_egfr"],
            errors ="ignore" 
        )

        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)
        return df, X, X.columns.tolist()
    
    
    df, X_df_raw, cols = feature_tab_num_no_fill(df)

    all_kinase_df = df.copy()
    egfr_df = all_kinase_df[all_kinase_df["is_egfr"] == True].copy()
    non_egfr_df = all_kinase_df[all_kinase_df["is_egfr"] == False].copy()

    if len(all_kinase_df) == 0:
        raise ValueError("No kinase rows found")
    if len(egfr_df) == 0:
        raise ValueError("No egfr rows found for weighted training")

    
    stratify_full = make_bins(all_kinase_df["log_affinity"], n_bins=10)
    
    full_train_df, test_df = train_test_split(
        all_kinase_df,
        train_size= train_data_frac,
        random_state=Random_seed,
        stratify=stratify_full
    )
    stratify_train = make_bins(full_train_df["log_affinity"], n_bins=10)
    
    train_base_df, validation_df = train_test_split(
        full_train_df,
        test_size=val_frac_in_train,
        random_state=Random_seed,
        stratify = stratify_train
    )

    egfr_train_rows = train_base_df[train_base_df["is_egfr"] == True].copy()


    if Egfr_oversample_multiplier > 1 and len (egfr_train_rows) > 0:
        extra_egfr_rows = pd.concat(
            [egfr_train_rows.copy() for _ in range(Egfr_oversample_multiplier -1)],
            axis = 0
        )
        train_df = pd.concat([train_base_df.copy(), extra_egfr_rows], axis=0)
    else:
        train_df = train_base_df.copy()
    train_df = train_df.sample(frac=1.0, random_state=Random_seed).copy()
    #train_df = train_df.copy()
   # validation_df = egfr_val_df.copy() #####
    #test_df = egfr_test_df.copy()


    #if len(train_df) == 0:
        #raise ValueError("No training rows found after EGFR split")
    #if len(validation_df) ==0:
        #raise ValueError("No valifation rows foiund after EGFRsplit")
    #if len(test_df) ==0:
       # raise ValueError("No test rows found after egfr split, hope not!!")
    
    sample_weights_train = make_sample_weights(train_df)

    
    #X_train_df = X_df_raw.loc[train_df.index].to_numpy(dtype=np.float32)
    #X_val = X_df.loc[validation_df.index].to_numpy(dtype=np.float32)
    #X_test = X_df.loc[test_df.index].to_numpy(dtype=np.float32)
    
    #data leakage preventaion this part wasn't in the previous one 
    X_train_df = X_df_raw.loc[train_df.index].copy()
    X_val_df = X_df_raw.loc[validation_df.index].copy()
    X_test_df = X_df_raw.loc[test_df.index].copy()

    train_medians = X_train_df.median()

    X_train_df = X_train_df.fillna(train_medians).fillna(0)
    X_val_df =X_val_df.fillna(train_medians).fillna(0)
    X_test_df= X_test_df.fillna(train_medians).fillna(0)


    y_train = train_df["log_affinity"].to_numpy(dtype=np.float32)
    y_val = validation_df["log_affinity"].to_numpy(dtype=np.float32)
    y_test = test_df["log_affinity"].to_numpy(dtype=np.float32)

    y_mean = y_train.mean()
    y_std = y_train.std()

    y_train = (y_train -y_mean)/y_std
    y_val = (y_val -y_mean)/y_std
    y_test = (y_test - y_mean) /y_std

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_val = scaler.transform(X_val_df)
    X_test = scaler.transform(X_test_df)


    
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf= 0.0).astype(np.float32)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf= 0.0).astype(np.float32)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf= 0.0).astype(np.float32)
    

    Model_1_Egfr_w = KinaseEgfrWeightedANN(
            input_dim =X_train.shape[1],
            hidden_units=(Hidden_unit_1, Hidden_unit_2, Hidden_unit_3),
            learning_rate = Learning_rate,
            dropout_rate= Dropout_rate,
            l2_lambda =L2_lambda,
            random_seed=Random_seed,
            min_learning_rate =Min_learning_rate,
            gradient_clip_value = Grad_clip_value,
            Early_stop_pat = Early_stop_pat,
            Learning_rate_pat= Learning_rate_pat,
            Learning_rate_decay = Learning_rate_decay,
            Min_delta= Min_delta
    )

    best_paramters, best_r2_score, log_lines= Model_1_Egfr_w.train(
        X_train= X_train,
        y_train =y_train,
        X_val =X_val,
        y_val =y_val,
        sample_weights_train=sample_weights_train,
        epochs =Epochs,
        batch_size = Batch_size

    )

    
    train_pred = Model_1_Egfr_w.prediction(X_train)
    val_pred = Model_1_Egfr_w.prediction(X_val)
    test_pred = Model_1_Egfr_w.prediction(X_test)
    
    if not np.isfinite(test_pred).all():
        raise ValueError("Test prediction contains nan or infinite")
    
    y_train_actual =(y_train * y_std) + y_mean
    y_val_actual = (y_val * y_std) + y_mean
    y_test_actual = (y_test * y_std) +y_mean

    train_pred_actual = (train_pred * y_std) +y_mean
    val_pred_actual = (val_pred * y_std) + y_mean
    test_pred_actual =(test_pred * y_std) +y_mean

    train_r2_final = float(r2_score(y_train_actual, train_pred_actual))
    val_r2_final = float(r2_score(y_val_actual, val_pred_actual))
    test_r2 = float(r2_score(y_test_actual, test_pred_actual))

    train_rmse_final =loss_rmse(y_train_actual, train_pred_actual)
    val_rmse_final = loss_rmse(y_val_actual, val_pred_actual)
    test_rmse =loss_rmse(y_test_actual, test_pred_actual)
    test_mse = loss_mse(y_test_actual, test_pred_actual)

    epochs_plot = Model_1_Egfr_w.history["epoch"]
    train_r2_hist = list(map(float, Model_1_Egfr_w.history["train_r2"]))
    val_r2_hist = list(map(float, Model_1_Egfr_w.history["val_r2"]))
    train_rmse_hist = list(map(float, Model_1_Egfr_w.history["train_rmse"]))
    val_rmse_hist = list(map(float, Model_1_Egfr_w.history["val_rmse"]))

    fig, axes = plt.subplots(2,2, figsize=(16,12))

    ax1 = axes[0,0]
    ax1.scatter(y_test_actual, test_pred_actual, alpha=0.4)
    min_val = min (y_test_actual.min(), test_pred_actual.min())
    max_val = max(y_test_actual.max(), test_pred_actual.max())
    ax1.plot([min_val, max_val], [min_val,max_val], linestyle = "--")

    ax2 = axes[0,1]
    ax2.plot(epochs_plot, train_rmse_hist, label ="Train RMSE")
    ax2.plot(epochs_plot, val_rmse_hist, label="Validation RMSE")
    ax2.axhline(y=test_rmse, linestyle="--", label=f"Test RMSE :{test_rmse:.3f}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("RMSE")
    ax2.set_title("RMSE per Epoch")
    ax2.legend()

    #R\u00b2 is for r square 
    ax3 = axes[1,0]
    ax3.plot(epochs_plot, train_r2_hist, label ="Train R\u00B2")
    ax3.plot(epochs_plot, val_r2_hist, label="Validation R\u00B2")
    ax3.axhline(y=test_r2, linestyle="--", label=f"Test R\u00B2 :{test_r2:.3f}")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("R\u00B2")
    ax3.set_title("R\u00b2 per Epoch")
    ax3.legend()

    ax4 = axes[1,1]
    ax4.bar(
        ["Train", "Validation", "Test"],
        [train_rmse_final, val_rmse_final, test_rmse]
    )

    ax4.set_xlabel("Dataset")
    ax4.set_ylabel("RMSE")
    ax4.set_title("Final RMSE Comparsion")

    plt.tight_layout()
    plt.savefig(os.path.join(DIRECTORY, "Checkpoint_1_kinase_egfr_weightedann_model1_graphs.png"))
    plt.close()

    with open(Model_1_Egfr_w_path, "wb") as f:
        pickle.dump(Model_1_Egfr_w, f)

    Best_paramter_path = os.path.join(
        DIRECTORY,
        "cp1_step4_all_kinase_egfr_weighted_best_param.pkl"
    )

    with open(Best_paramter_path, "wb") as f:
        pickle.dump(best_paramters, f)
            
    with open(Scale_ckp1_step4_egfr_only_path, "wb") as f:
        pickle.dump(scaler, f)
            
    with open(Y_scale_path, "wb") as f:
        pickle.dump({"y_mean": y_mean, "y_std": y_std}, f)
            
    with open(log_path, "w") as f:
        list(map(lambda line: f.write(line +"\n"), log_lines))

    print("Model type: All kinase EGFR weighted  ANN MODEL (Model-1)")
    print("Data change all kinase protein inclused instead of using EGFR only dataset ")
    print("Egfr class weight:", Egfr_class_weight)
    print("Non egfr class weight:", Non_egfr_class_weight)
    print("Egfr oversampling multiplier", Egfr_oversample_multiplier)
    print("Training: loss gradient includes egfr weighted rows and oversampled egfr rows")
    
    print("Original full dataset rows:", len(df))
    print("All kinase protein rows used before split:", len(all_kinase_df))
    print("Orignal egfr rows", len(egfr_df))
    print("Original non egfr rows:", len(non_egfr_df))

    print("Best train rows before egfr oversampling:", len(train_base_df))
    print("Train rows total after Egfr oversampling:", len(X_train))
    print("Validations rows total:", len(X_val))
    print("Test rows total:", len(X_test))


    print("Egfr rows used in base before oversampling:", int(train_base_df["is_egfr"].sum()))
    print("Egfr rows in train after oversamoling:", int(train_df["is_egfr"].sum()))

    print("EGFR rows used in validation:", int(validation_df["is_egfr"].sum()))
    print("EGFR rows used un test:",  int(test_df["is_egfr"].sum()))

    print("Egfr rows in base train before oversampling:", int((train_base_df["is_egfr"] == True).sum()))
    print("Egfr rows in train after oversamling", int((train_df["is_egfr"] == True).sum()))
    
    print("Non Egfr rows in train after oversamling", int((train_df["is_egfr"] == False).sum()))
    print("Non egfr rows in data validation:", int((validation_df["is_egfr"] == False).sum()))
    print("Non Egfr rows in test:", int((test_df["is_egfr"] == False).sum()))



    print(f"Best validation r2 score: {best_r2_score:.4f}")
    print(f"Final train r2 score: {train_r2_final:.4f}")
    print(f"Final validaton r2 score: {val_r2_final:.4f}")
    print(f"Final test r2 score: {test_r2:.4f}")
    print(f"Final train rmse: {train_rmse_final:.4f}")
    print(f"Final validation rmse:{val_rmse_final:.4f}")
    print(f"Final test mse:{test_mse:.4f}")
    print(f"Done!Final test rmse:{test_rmse:4f}")
    print("This took so lon time to debug and rewrite it was to messy.")

if __name__ == "__main__":
    main()
              


                         



    
