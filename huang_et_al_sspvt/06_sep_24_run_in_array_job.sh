########################################################################################
# 31_jul_24_run_in_array_job.sh - Runs the PV-T model in an HPC array job.             #
#                                                                                      #
# Author(s): Ben Winchester                                                            #
# Copyright: Ben Winchester, 2024                                                      #
# Date created: 05/09/2024                                                             #
# License: Open source                                                                 #
# Most recent update: 05/09/2024                                                       #
#                                                                                      #
# For more information, please email:                                                  #
#     benedict.winchester@gmail.com                                                    #
########################################################################################

# Depending on the environmental variable, run the appropriate HPC job.
module load matlab/R2018a

# Change to the submission directory
OUTPUT_DIR="$PBS_O_WORKDIR/sspvt_sensitivity_output"
CURRENT_DIR=$(pwd)
cd $PBS_O_WORKDIR

matlab -nodesktop -nosplash -nojvm -r "index=$PBS_ARRAY_INDEX;hpc_sspvt_performance"
