########################################################################################
# 24_mar_26_run_in_array_job.sh - Runs the `scdo` model in an HPC array job.           #
#                                                                                      #
# Author(s): Ben Winchester                                                            #
# Copyright: Ben Winchester, 2026                                                      #
# Date created: 16/03/2026                                                             #
# License: Open source                                                                 #
# Most recent update: 16/03/2026                                                       #
#                                                                                      #
# For more information, please email:                                                  #
#     benedict.winchester@gmail.com                                                    #
########################################################################################

# Depending on the environmental variable, run the appropriate HPC job.
module load anaconda3/personal
source activate py312
conda activate py312

# Change to the submission directory
CURRENT_DIR=$(pwd)
cd $PBS_O_WORKDIR

python -m src.collector_optimisation -l imperial_chemical_engineering -i 64 -n 8128 \
    -wf weather_data_25_1000.csv -bc autotherm.yaml -bmfs autotherm.yaml \
    -d 24/03/26 -t 10:00:00 -hpc $PBS_ARRAY_INDEX

echo "Completed. Exiting."
