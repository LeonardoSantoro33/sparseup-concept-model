#!/bin/bash
# =====================================================================
#  GENERIC GPU job. Usage (from the repository directory):
#     sbatch -J job_name jobs/gpu.sbatch  script.py  --script-arguments
#  (the script is looked up in src/, so the name alone is enough)
#  Example:
#     sbatch -J baseline jobs/gpu.sbatch 04_baseline_sparseup.py --query-sets dev
#  Output in logs/job_name_<id>.out (stdout) and logs/job_name_<id>.err (errors).
#  The logs directory must exist:  mkdir -p logs
#  Data go to $SPARSEUP_WORK_DIR (default /datafast/$USER/sparseup_concept, see src/config.py).
# =====================================================================
#SBATCH --job-name=sparseup
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/miniforge3/etc/profile.d/conda.sh
conda activate lce
cd "$SLURM_SUBMIT_DIR"
set -e

echo "Node: $(hostname) | Job: $SLURM_JOB_ID | Start: $(date)"
echo "SPARSEUP_WORK_DIR=${SPARSEUP_WORK_DIR:-(src/config.py default)}"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || true

# Script name: bare name (04_baseline_sparseup.py) or path (src/04_baseline_sparseup.py) both work.
SCRIPT="$1"; shift
[ -f "$SCRIPT" ] || SCRIPT="src/$SCRIPT"
[ -f "$SCRIPT" ] || SCRIPT="${SCRIPT}.py"
[ -f "$SCRIPT" ] || { echo "Script not found: $1 (looked in ./ and src/)"; exit 1; }

python -u "$SCRIPT" "$@"

echo "End: $(date)"
