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

module load openmpi

for j in 16
do
mpirun -np $j python features_extraction_mpi.py >> out
done