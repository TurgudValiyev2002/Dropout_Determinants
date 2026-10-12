#!/bin/bash
#SBATCH --job-name=dropout-llm-final
#SBATCH --partition=IFIgpuL40S
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40S:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --array=0-5%1
#SBATCH --chdir=/home/turgud.valiyev/dropout-llm-study
#SBATCH --output=/home/turgud.valiyev/dropout-llm-study/logs/final_llm_%A_%a.out
#SBATCH --error=/home/turgud.valiyev/dropout-llm-study/logs/final_llm_%A_%a.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
exec /home/turgud.valiyev/rdcc-distillation/.venv/bin/python -B -u \
  /home/turgud.valiyev/dropout-llm-study/scripts/final_llm_experiment.py run \
  --experiment /home/turgud.valiyev/dropout-llm-study/outputs/final_experiment_v2 \
  --model /home/turgud.valiyev/rdcc-distillation/models/nemotron_3_5_lightning_bf16 \
  --chunk "${SLURM_ARRAY_TASK_ID}"
