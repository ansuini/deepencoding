#!/bin/bash
#PBS -N gpu_feats
#PBS -l walltime=12:00:00
#PBS -l nodes=1:ppn=20
#PBS -T flush_cache
#PBS -q gpu

JOBNAME=gpu_feats
WD='/scratch/ansuini/repositories/machine_learning/deepencoding/features'
cd $WD
echo 'Working directory is' $WD


OUTFILE='times_gpu_vectorized'
rm -f $JOBNAME*
export PATH="/home/ansuini/shared/programs/x86_64/anaconda2/bin:$PATH"
module load cudatoolkit/6.5

echo "Version complete" >> $OUTFILE
THEANO_FLAGS=device=gpu,floatX=float32 python extract_gpu_vectorized.py >> $OUTFILE
