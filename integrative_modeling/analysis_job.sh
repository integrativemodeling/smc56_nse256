#$ -S /bin/bash
#$ -cwd
#
#$ -r n
#$ -j y
#$ -R y
#
#$ -l mem_free=4G
#$ -l h_rt=200:00:00
#$ -l scratch=200G
#
#$ -N c_smc56
#$ -o /wynton/scratch/tsanyal
#$ -e /wynton/scratch/tsanyal

hostname
date

# add conda environment
eval "$(conda shell.bash hook)"
conda activate impenv

$IMPENV python analyse_score_distributions.py -o analys -np 20

hostname
date

qstat -j $JOB_ID
