########################################################################################
# 25_mar_26_hpc_job.sh - Script for executing a single `scdo` job on the HPC.          #
#                                                                                      #
# Author: Ben Winchester                                                               #
# Copyright: Ben Winchester, 2026                                                      #
# Date created: 25/03/2026                                                             #
#                                                                                      #
# For more information, please email:                                                  #
#   benedict.winchester@gmail.com                                                      #
########################################################################################
#PBS -lwalltime=72:00:00
#PBS -lselect=1:ncpus=1:mem=11800Mb

echo -e "HPC array script executed for an individual run"

# Load the anaconda environment
module load anaconda3/personal
source activate py312
conda activate py312

cd $PBS_O_WORKDIR

echo -e "Running PVTModel HPC python script."

# Change to the submission directory
CURRENT_DIR=$(pwd)
cd $PBS_O_WORKDIR

python -m src.collector_optimisation -l imperial_chemical_engineering -i 64 -n 8128 \
    -wf weather_data_25_1000.csv -bc autotherm.yaml -bmfs autotherm.yaml \
    -d 25/03/26 -t 16:00:00 -hpc 1

echo "Completed. Exiting."
