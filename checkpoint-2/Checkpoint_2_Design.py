# ============================
# checkpoint-2 Step 1-4 Roadmap
# =============================
# Protein family classification
# Seq feature extraction
# Directoriy set up as in checkpoint -1 updated for checkpoint -2
import os 
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
#you will need add more libraries for each steps
#### checkpoint 2 step 1 
Data_directory = "Checkpoint_2_data"
os.makedirs(Data_directory, exist_ok=True)

Checkpoint_2_step_1_input_path = os.path.join("New_checkpoint_1_data_here", "checkpoint_1_step_2_alignment_features_data.csv")
Checkpoint_2_step_1_output_path = os.path.join(Data_directory, "checkpoint_2_step_1_sequence_feature_data.csv")

def main():
    if not os.path.exists(Checkpoint_2_step_1_input_path):
        print("Check point 2 step 1 input file is not found")
        return 
df = pd.read_csv(Checkpoint_2_step_1_input_path, low_memory=False)
print("Input loaded, its shape:", df.shape)

Required_cols = [ "sequence", "target_name"]
missing_cols = [c for c in Required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Misisng reuired columns:", {missing_cols})
print("Checkpoint-2 set up ready")
print("Sampple target names:")
print(df["target_name"].dropna().head(10).tolist())

print("Sample seqs:")
print(df["sequence"].dropna().head(3).tolist())
#After you erite your code you will need to save it to the output file 
#delete hastags in the followinf two lines
#df.to_csv(Checkpoint_2_step_1_output_path, index=False)
#print("Checkpoin-1 step2 output saved:",Checkpoint_2_step_1_output_path)


if __name__ == "__main__":
    main()
#![Screenshot 2026-03-29 at 1.12.01 PM.png](<attachment:Screenshot 2026-03-29 at 1.12.01 PM.png>)
### checkpoint 2 step 2

#Checkpoint-2 step2
Data_directory = "Checkpoint_2_data"
Checkpoint_2_step_2_input_path = os.path.join(Data_directory,"checkpoint_2_step_1_sequence_feature_data.csv")
Checkpoint_2_step_2_output_path = os.path.join(Data_directory, "checkpoint_2_step_2_deepseq_embedding.csv")


### checkpoint 2 step3 
Data_directory = "Checkpoint_2_data"
Checkpoint_2_step_3_input_path = os.path.join(Data_directory,"checkpoint_2_step_2_deepseq_embedding.csv")
Checkpoint_2_step_3_output_path = os.path.join(Data_directory, "checkpoint_2_step_3_family_classification.csv")

### checkpoint 2 step4
Data_directory = "Checkpoint_2_data"
Checkpoint_2_step_4_input_path = os.path.join(Data_directory,"checkpoint_2_step_3_family_classification.csv")
Checkpoint_2_step_4_output_path = os.path.join(Data_directory, "checkpoint_2_step_4_transfer_learning_family_predictions.csv")