########################################################################################
# 31_jul_24_hpc_array.sh - Script for executing PVTModel as an array job on the HPC.   #
#                                                                                      #
# Author: Ben Winchester                                                               #
# Copyright: Ben Winchester, 2024                                                      #
# Date created: 22/02/2024                                                             #
#                                                                                      #
# For more information, please email:                                                  #
#   benedict.winchester@gmail.com                                                      #
########################################################################################
#PBS -J 1-272
#PBS -lwalltime=72:00:00
#PBS -lselect=1:ncpus=1:mem=11800Mb

echo -e "HPC array script executed for 2552 runs"

# Load the anaconda environment
module load anaconda3/personal
source activate clover

cd $PBS_O_WORKDIR

echo -e "Running PVTModel HPC python script."
if ./bin/31_jul_24_run_in_array_job.sh ; then
    echo -e "PVTModel successfully run."
else
    echo -e "FAILED. See logs for details."
fi
