# Predicting Protein Ligand Ineractions, Family, and Environmental Influence Using Multi Model Machine Learning 

This project focuses on predicting ineractions between kinase family proteins by analyzing their sequence similarity, kinase ligand ineractions, and how these ineractions change under different environmental conditions.Kinase family proteins play a crucial role in cell signalling, regulating  growth and their dysregulation is heavily linked to several types of cancer.Unfortunately,current public datasets often lack integrated approaches that account for tissue specific conditions.As a result, they do not fully capture whether a kinase would bind to a ligand in a specific bilogical context.Our approach constructs on these datasets by integrating protein ligand, and enviromental information into a single model to better reflect real world conditions and support more personalized medicine applications.
Currently, the project emphasizes kinase proteins and introduces a cold start problem where the model must predict binding for protein it has never seen before.To test this, we exclude EGFR from training and use it only for evalution.
Protein features include sequence similairty using Needleman-Wunsch and Smith Waterman motif detection methods.Ligand features are generated from SMILES strings and include properties like hydrophobicity and binding affintiy.These features are combined into a multi modal machine learning model.The projects is divided into three checkpoints:binding predictions, protein family classifcation, and enviromental efects.In the final phase we include factors like pH and tissue specific conditions to better understand how context affects binding.The model ouputs similarity scores,binding affinity,and critical contributing environmental features.
We are going to use datasets such as BindingDB,UniProt,Protein Databank, and ChemBL across different checkpoints.Our aim is to improve prediction accuracy for unseen kinase proteins by combining biological, chemical and environmental information into a unified framework to create more personalized therapeutics.
# Checkpoint-1
Chuckpoint 1 creates the full pipeline for predicting protein ligand binding affinity.It starts by loading and filtering the dataset to only include kinase proteins.Then, protein and ligand features are generated separately.Finally, these features are combined into a machine learning model and evaluated using a cold start setup where EGFR is not inlcuded during training.

## Checkpoint -1 Step -1 
Checkpoint-1 step1 downloads the full BindingDB dataset, which contains protein ligand ineractions data for many types of proteins and then filters it to keep only kinase related entries.Data gets loaded in smaller chunks so it doesn't overload ram.It cleans the data, selects important columns,convets binding values into usable format, and filters the dataset to keep only kinase related rows while marking EGFR for later testing.Finally, it combines all the prepared data and saves a clean kinase specific datset that will be used i the next steps for the project.

    
    •nohup python -u Tuba_Murphy_Upt_checkpoint_1_step_1.py > cp1_step_1_for_run.log 2>&1 & 
    OR
    •python Tuba_Murphy_Upt_checkpoint_1_step_1.py

    Data Directory:New_checkpoint_1_data_here
    File saved:
    checkpoint_1_step_1_kinase_filtered_data.csv

### Terminal General Sanity Check
![Terminal sanity check](images/sanity_check_ckp1_stp1.png)

## Checkpoint -1 Step -2 
Checkpoint-1 Step-2 takes the cleaned kinase data from step-1 <checkpoint_1_step_1_kinase_filtered_data.csv> and compares each protein sequence to reference kinase family sequences using Needleman Wunsh allignment to evaluate similairty. It calculates features like allignment score,normalized similarity, and closest kinase family for each protein.The output is a new dataset with these sequence based features added which will be used for machine learning in Checkpoint-1 step3 and checkpoint-2 step1.

    •nohup python -u  Tuba_Murphy_chckp1_step_2_gitcopy.py > cp1_step_2_for_run.log 2>&1 &
    OR
    python Tuba_Murphy_chckp1_step_2_gitcopy.py
    Data Directory:New_checkpoint_1_data_here
    File saved:()
    checkpoint_1_step_2_alignment_features_data.csv
    checkpoint_1_kinase_reference_sequence_data.csv


### Terminal General Sanity Check
![Terminal sanity check](images/sanity_check_ckp1_st2_1.png)


![Terminal sanity check](images/sanity_check_ckp1_stp2.png)




## Checkpoint -1 Step -3
Checkpoint-1 step-3 takes dataset from step2 <checkpoint_1_step_2_alignment_features_data.csv> and converts SMILES strings into numerical features like molecular fingerprints and chemical properties ie.molecular weight, hydrophobicity.This crucial beacuse machine learning models cannot use raw SMILES strings, so we transform them into numbers that provides ligand structure and behavior.
These ligand features are then combined with protein similarity features to create a final dataset that allows the model to learn and predict protein ligand binding affinity.

    •nohup python -uTuba_Murphy_chckp1_step_3_mef-2_smiles-spam-off-troubleshhoot.py > cp1_step_2_for_run.log 2>&1 &
    OR
    •python Tuba_Murphy_chckp1_step_3_mef-2_smiles-spam-off-troubleshhoot.py
    Data Directory:New_checkpoint_1_data_here
    File saved:
    checkpoint_1_step_3_No_smiles_spam_feature_table_data.csv

### Terminal> General Sanity Check
![Terminal sanity check](images/sanity_check_ckp1_stp3_1.png)

![Terminal sanity check](images/sanity_check_ckp1_stp3_2.png)


Data structure is correct, features complete,values are realistic,missining values minimal(1%) pipeline seems like working perfectly.




## Checkpoint -1 Step -4
This code takes the final dataset from step3 <checkpoint_1_step_3_No_smiles_spam_feature_table_data.csv>
cleans and prepares all features, and trains a KNN regression model to predict binding affinity.It trains the model only on non-EGFR proteins and then tests it on EGFR to evaluate how well it generalizes to unseen data.The output includes predictions for EGFR, feature importance scores, and saved files for the trained model and scaler.

    •nohup python -u  Tuba_Murphy_updated_checkpoint_1_step4.py > cp1_step_2_for_run.log 2>&1 &
    OR
    •python Tuba_Murphy_updated_checkpoint_1_step4.py
    Data Directory:New_checkpoint_1_data_here
    File saved:
    checkpoint_1_step4_knn_model_dat_.pkl
    checkpoint_1_scaler_data.pkl
    checkpoint_1_egfr_predictions_data.csv
    checkpoint_1_feature_importance_data.csv



# Checkpoint-2 

## Steps

# Checkpoint -3 
### Steps