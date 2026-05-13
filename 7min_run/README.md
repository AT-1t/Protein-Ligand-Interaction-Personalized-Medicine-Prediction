## Server Usage Information

### HPC Account Information and env activation

    •Open your terminal
    •Copy paste this --->     ssh tubmu@hpcctl.ocf.berkeley.edu
    •Password: —-------->      group8chem277B@ucb
    •Activate environment ->  source chem277b-env2/bin/activate
    •Go to project folder      cd 7min_run

This must be run on the server becuase it requires large datafiles generated in previous steps.It creates a fsuion dataset with nearly 48 million rows, while the LightGBM Model reads only a 1M ros sample of the dataset due to memory limitations.

This will run step of checkpoint3 step2, which combines the Kinase EGFR Weighted ANN Model1 and XGBoost Family Classifaction Model2 datasets with patient mutation data.It then executes the LightGBM Model 3, which builds on ANN binding affinity predictions .There is also an pdf file in the directory containing the top ten highest binding affinity small molecules predicted to bind to specific patients who carry those mutations.


#### To run checkpoint3 (LightGBM +Excel output)

</> Bash 

        make ProxyModel

T

##### To visualize project outputs
All visualization results are already included in the folder as PNG files.It will generate and open the outputs for all three models.

### Visualization for Mac users
</> Bash 

    make visualize-mac 

### Visualization for Linux 

    make visualize-linux

### Visualization for Windows 

    make visualize-windows







