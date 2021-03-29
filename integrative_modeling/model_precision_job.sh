#! /bin/bash

# add conda environment
eval "$(conda shell.bash hook)"
conda activate impenv

imp_sampcon exhaust -n smc56 -p .. \
-m cpu_omp -c 20 \
-a \  # -s - ct 50 \
-sa sample_A_scores.txt -sb sample_B_scores.txt \
-ra sample_A_models.rmf3 -rb sample_B_models.rmf3 \
-d density_ranges.txt




