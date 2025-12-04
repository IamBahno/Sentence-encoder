#!/bin/bash
#PBS -N mnr_last_layer_attention_8h_8q
#PBS -l select=1:ncpus=2:ngpus=1:gpu_mem=46gb:mem=32gb:scratch_local=10gb
#PBS -l walltime=6:00:00
# TODO add the label redo, and add data.yaml and toher things
    
 # append a line to a file "jobs_info.txt" containing the ID of the job, the hostname of the node it is run on, and the path to a scratch directory
# this information helps to find a scratch directory in case the job fails, and you need to remove the scratch directory manually 
echo "$PBS_JOBID is running on node `hostname -f` "
    
module add mambaforge
mamba activate /storage/brno2/home/xbahou00/zpja_env

cd /storage/brno2/home/xbahou00/Sentence-encoder

python main.py --config config_8h_8q.yaml

python test.py --config config_8h_8q.yaml