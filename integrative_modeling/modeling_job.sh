#$ -S /bin/bash
#$ -cwd
#
#$ -r n
#$ -j y
#$ -R y
#
#$ -l mem_free=4G
#$ -l h_rt=160:00:00
#$ -l scratch=200G
#
#$ -pe smp 8
#$ -t 1-100
#$ -N smc56
#$ -o /wynton/scratch/tsanyal
#$ -e /wynton/scratch/tsanyal

hostname
date

# add conda environment
eval "$(conda shell.bash hook)"
conda activate impenv

# run directory for this independent run
i=$(expr $SGE_TASK_ID)
RUNDIR=./run_$i

# run sampling for this run
if [ ! -d $RUNDIR ]; then
    mkdir   $RUNDIR 
    cd $RUNDIR    
    $IMPENV mpirun -np $NSLOTS python ../modeling.py -d ../data
    cd ..
fi

hostname
date

qstat -j $JOB_ID
