#!/bin/bash
#SBATCH --job-name=umap_atlas
#SBATCH --account=jespinoza@kgi.edu_1537
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/project/jespinoza_1537/umap_atlas_%j.log

source ~/.bashrc
module load conda/25.11.0
conda activate /project/jespinoza_1537/envs/metagenome

set -eo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Job $SLURM_JOB_ID on $(hostname) with $SLURM_CPUS_PER_TASK CPUs"
python /project/jespinoza_1537/umap_atlas.py
echo "Finished"
