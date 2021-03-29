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
#$ -N e_smc56
#$ -o /wynton/scratch/tsanyal
#$ -e /wynton/scratch/tsanyal

hostname
date

# add conda environment
eval "$(conda shell.bash hook)"
conda activate impenv

mkdir -p structural_clustering
$IMPENV python get_good_scoring_models.py -a analys -np 20 -o ./structural_clustering

hostname
date

qstat -j $JOB_ID
