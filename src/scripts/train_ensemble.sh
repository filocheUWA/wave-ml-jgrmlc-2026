#!/bin/bash -login
 
#SBATCH --account=pawsey0106-gpu
#SBATCH --partition=gpu
#SBATCH --job-name=ens_train
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00

module load pytorch/2.7.1-rocm6.3.3
 
# Virtual env location
export MYENV=env_DL
export VENV_PATH=$MYSOFTWARE/manual/software/pythonEnvironments/pytorchContainer-environments/${MYENV}
 
#----
#Additional Settings needed when using Pytorch module:
export TMPDIR="/tmp/${USER}-${SLURM_JOB_ID}"
mkdir -p $TMPDIR
 
# Hide ~/.local packages everywhere in this job:
export PYTHONNOUSERSITE=1
 
echo "monitor GPU process started with PID $PID"
 
echo "Rank $SLURM_PROCID --> $(taskset -p $$) "
rocm-smi --showhw
echo "Rank $SLURM_PROCID --> HIP Devices: $HIP_VISIBLE_DEVICES"
echo "Rank $SLURM_PROCID --> ROCR Devices: $ROCR_VISIBLE_DEVICES"
 
srun bash -c "source $VENV_PATH/bin/activate && python3 -u train_ensemble.py"
 
#----
#Done
echo -e "\n\n#------------------------#"
echo "Done"