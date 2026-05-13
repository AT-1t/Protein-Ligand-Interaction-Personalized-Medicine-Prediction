## Server Usage Information

### HPC Account Information and env activation

    •Open your terminal
    •Copy paste this --->     ssh tubmu@hpcctl.ocf.berkeley.edu
    •Password: —-------->      group8chem277B@ucb
    •Activate environment ->  source chem277b-env2/bin/activate
    •Go to project folder      cd 3hours_run



# Environment necessary to run this pipeline
We already have all these libraries installed in the chem277b-env2 environment on the server.The libraries listed below are only required for usage outside the server environment.

        conda create -n group7-env python=3.11
        condat activate group7-env 
        
        pip install numpy pandas matplotlib scikit-learn scipy seaborn tqdm biopython umap-learn 
        xgboost pyarrow requests

        conda install -c conda-forge rdkit




pip install numpy pandas matplotlib scikit-learn scipy seaborn tqdm biopython umap-learn xgboost pyarrow requests

conda install -c conda-forge rdkit

## Running the Full Pipeline
This is stepwise transfer learning pipeline, so it will run through three checkpoints.We laready have the folder on the server with all the files, and we have already done multiple trials.

### To run checkpoint 1

</> Bash 
        
        make Checkpoint1

### To run checkpoint 2

</> Bash 
        
        make Checkpoint2

### To run checkpoint 2

</> Bash 
    
    make Checkpoint3






