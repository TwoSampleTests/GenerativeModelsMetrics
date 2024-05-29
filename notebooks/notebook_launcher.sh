#!/bin/bash

# Check if the notebook name is provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <notebook-name.ipynb>"
    exit 1
fi

NOTEBOOK_PATH=$(realpath "$1")
NOTEBOOK_NAME=$(basename "$NOTEBOOK_PATH")
SCRIPT_NAME="${NOTEBOOK_NAME%.*}.py"
BASE_NAME="${NOTEBOOK_NAME%.*}"
OUT_FILE="${BASE_NAME}.out"
ERR_FILE="${BASE_NAME}.err"
JOB_NAME="submit_${BASE_NAME}.sh"
SCRIPT_PATH=$(dirname "$NOTEBOOK_PATH")

# Source the python environment
source /leonardo_scratch/fast/INF24_pmlhep_1/rtorre00/envs/tf2_custom/bin/activate

# Convert the notebook to a Python script
jupyter nbconvert --to script "$NOTEBOOK_PATH"

# Create the SLURM job script
echo "#!/bin/bash
#SBATCH --account=INF24_pmlhep_1
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=$BASE_NAME
#SBATCH --output=$OUT_FILE
#SBATCH --error=$ERR_FILE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=00:01:00

source /leonardo_scratch/fast/INF24_pmlhep_1/rtorre00/envs/tf2_custom/bin/activate
srun python $SCRIPT_NAME
" > $JOB_NAME

# Submit the job to SLURM
sbatch $JOB_NAME

# Launch with the following command:
# source /leonardo_scratch/fast/INF24_pmlhep_1/rtorre00/envs/tf2_custom/bin/activate
# chmod +x notebook_launcher.sh
# ./notebook_launcher.sh Analysis_100D_unimodal_leo20.ipynb