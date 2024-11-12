########################################################################################
# 31_jul_24_hpc_array.sh - Script for executing PVTModel as an array job on the HPC.   #
#                                                                                      #
# Author: Ben Winchester                                                               #
# Copyright: Ben Winchester, 2024                                                      #
# Date created: 05/09/2024                                                             #
#                                                                                      #
# For more information, please email:                                                  #
#   benedict.winchester@gmail.com                                                      #
########################################################################################
#PBS -J 1-2930
#PBS -lwalltime=08:00:00
#PBS -lselect=1:ncpus=1:mem=100Gb

echo -e "HPC array script executed for 2552 runs"

# Load the anaconda environment
module load matlab/R2018a

cd $PBS_O_WORKDIR

echo -e "Running PVTModel HPC python script."
if ./06_sep_24_run_in_array_job.sh ; then
    echo -e "SSPV-T model successfully run."
else
    echo -e "FAILED. See logs for details."
fi
