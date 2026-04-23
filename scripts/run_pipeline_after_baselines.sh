#!/bin/bash
# Run bicycle training + final evaluation after GOKU/DVBF/V2P finish

set -e
cd /e/Desktop/PIWM/piwm
source "$(/c/ProgramData/anaconda3/condabin/conda.bat shell.bash hook)"
conda activate piwm

echo "Starting bicycle dynamics training..."
python train/train_piwm_bicycle.py

echo "Running full comparison..."
python eval/compare_baselines_rel.py
