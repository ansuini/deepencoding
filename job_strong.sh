#!/bin/bash
#PBS -N parallel_feats
#PBS -l walltime=12:00:00
#PBS -l nodes=1:ppn=20
#PBS -T flush_cache
#PBS -q regular

NAME=parallel_feats
WD='/scratch/ansuini/repositories/machine_learning/deepencoding'
cd $WD
echo 'Working directory is' $WD
rm -f $WD/parallel_feats*

for j in 1 2 4 8 16 20
do
python features_extraction_joblib.py --numprocs $j >> out
done
