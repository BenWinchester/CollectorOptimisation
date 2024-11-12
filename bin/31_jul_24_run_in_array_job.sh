########################################################################################
# 31_jul_24_run_in_array_job.sh - Runs the PV-T model in an HPC array job.             #
#                                                                                      #
# Author(s): Ben Winchester                                                            #
# Copyright: Ben Winchester, 2024                                                      #
# Date created: 22/02/2024                                                             #
# License: Open source                                                                 #
# Most recent update: 22/02/2024                                                       #
#                                                                                      #
# For more information, please email:                                                  #
#     benedict.winchester@gmail.com                                                    #
########################################################################################

# Depending on the environmental variable, run the appropriate HPC job.
module load anaconda3/personal
source activate clover

# Change to the submission directory
OUTPUT_DIR="$PBS_O_WORKDIR/output_files"
CURRENT_DIR=$(pwd)
cd $PBS_O_WORKDIR

# Set the steady-state data file name
STEADY_STATE_DATA_FILE="$PBS_O_WORKDIR/system_data/steady_state_data/sensitivity_31_jul_2024.yaml"
echo "Steady-state data file: $STEADY_STATE_DATA_FILE"

# Determine the panel to use based on the array job index
PANEL_FILE=$(head 31_jul_24_panels.txt -n $PBS_ARRAY_INDEX | tail -n 1)
OUTPUT_NAME=$(echo $PANEL_FILE | awk -F/ '{print $NF}' | awk -F ".yaml" '{print $1}')
OUTPUT_FILE="$PBS_O_WORKDIR/output_files/31_jul_24/$OUTPUT_NAME"
echo "Panel file: $PANEL_FILE"

python3.7 -u -m src.pvt_model --skip-analysis \
    --output $OUTPUT_FILE \
    --steady-state-data-file $STEADY_STATE_DATA_FILE \
    --decoupled --steady-state --initial-month 7 --location system_data/london_ilaria/ \
    --portion-covered 1 \
    --pvt-data-file $PANEL_FILE \
    --x-resolution 31 --y-resolution 11 --average-irradiance --skip-2d-output \
    --layers g pv a p f --disable-logging
