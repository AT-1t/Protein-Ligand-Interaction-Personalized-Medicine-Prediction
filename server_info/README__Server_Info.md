## Server Usage Information

### HPC Account Information and env activation

    •Open your terminal
    •Copy paste this --->      ssh tubmu@hpcctl.ocf.berkeley.edu
    •Password: —-------->      group8chem277B@ucb
    •Activate environment ->  source chem277b-env2/bin/activate
    •Go to project folder      cd new_chem_277b
#### Project data has all the checkpoint-1 steps 1,2,3 and 4.

    •Go to output csv files --> cd New_checkpoint_1_data_here
#### I saved all four steps outputs here. 

#### How to save this current folder in your local computer
    exit (you need to exit the server)
    scp tubmu@hpcctl.ocf.berkeley.edu:/home/t/tu/tubmu/new_chem_277b/filename.py ~/Desktop/your/folder/name

#### How to save your file from your local computer to the server
    
    
     scp /full/path/to/yourlocal/file.py tubmu@hpcctl.ocf.berkeley.edu:/home/t/tu/tubmu/new_chem_277b/


#### How to run file even the computer shuts down 
    nohup python -u filename.py > step_name_for_run.log 2>&1 &

#### How to check if the file is progressing
    tail -f step_name_for_run.log (-f gives live updates)
    Ctrl + C to escape, programs still runs on the background

#### How to find the running process

    ps aux | grep filename.py ()  #it shows all the runing progress

#### How to check how long it is going to run 

    ps -p <PID> -o etime,cmd  #grep code above provides the Process ID.

### How to stop running 
    kill <PID>







