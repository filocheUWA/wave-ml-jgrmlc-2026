#!/bin/bash -login

#SBATCH --account=pawsey0106-gpu
#SBATCH --partition=gpu
#SBATCH --job-name=specx_tune
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=23:50:00

module load pytorch/2.7.1-rocm6.3.3

# Virtual env location
export MYENV=env_DL
export VENV_PATH=$MYSOFTWARE/manual/software/pythonEnvironments/pytorchContainer-environments/${MYENV}

#----
# Additional Settings needed when using Pytorch module:
export TMPDIR="/tmp/${USER}-${SLURM_JOB_ID}"
mkdir -p $TMPDIR

# Hide ~/.local packages everywhere in this job:
export PYTHONNOUSERSITE=1

# ==============================================================================
# ROCm / MIOpen STABILIZATION FLAGS (Fix for "Unknown layout" / NaN errors)
# ==============================================================================
# 1. Disable unstable implicit GEMM convolutions
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0

# 2. Force MIOpen to evaluate kernels safely rather than guessing
export MIOPEN_FIND_MODE=3

# 3. Isolate the MIOpen cache database to prevent parallel task corruption
export MIOPEN_USER_DB_PATH="/tmp/miopen_db_$SLURM_JOB_ID"
# ==============================================================================

echo "monitor GPU process started with PID $PID"

echo "Rank $SLURM_PROCID --> $(taskset -p $$) "
rocm-smi --showhw
echo "Rank $SLURM_PROCID --> HIP Devices: $HIP_VISIBLE_DEVICES"
echo "Rank $SLURM_PROCID --> ROCR Devices: $ROCR_VISIBLE_DEVICES"

# --- CONFIGURATION ---
# You can override these here if needed
export OUTPUT_ROOT="/scratch/pawsey0106/afiloche/output_tuning"

# Run the Tuning Script
# We use -u for unbuffered output to see logs in real-time
# Change the srun line to this:
srun bash -c "export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0; export MIOPEN_FIND_MODE=3; export MIOPEN_USER_DB_PATH=/tmp/miopen_db_$SLURM_JOB_ID; source $VENV_PATH/bin/activate && python3 -u tune.py"

#----
#Done
echo -e "\n\n#------------------------#"
echo "Done"