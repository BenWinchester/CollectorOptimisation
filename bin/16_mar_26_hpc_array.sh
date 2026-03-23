########################################################################################
# 16_mar_26_hpc_array.sh - Script for executing `scdo` as an array job on the HPC.     #
#                                                                                      #
# Author: Ben Winchester                                                               #
# Copyright: Ben Winchester, 2026                                                      #
# Date created: 16/03/2026                                                             #
#                                                                                      #
# For more information, please email:                                                  #
#   benedict.winchester@gmail.com                                                      #
########################################################################################
#PBS -J 1-7
#PBS -lwalltime=72:00:00
#PBS -lselect=1:ncpus=1:mem=11800Mb

echo -e "HPC array script executed for 7 runs"

# Load the anaconda environment
module load anaconda3/personal
source activate py312

cd $PBS_O_WORKDIR

echo -e "Running PVTModel HPC python script."
if ./bin/16_mar_26_run_in_array_job.sh ; then
    echo -e "PVTModel successfully run."
else
    echo -e "FAILED. See logs for details."
fi
