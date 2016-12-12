#!/bin/bash
#PBS -N mpi_feats
#PBS -l walltime=12:00:00
#PBS -l nodes=1:ppn=20
#PBS -T flush_cache
#PBS -q regular

NAME=mpi_feats
WD='/scratch/ansuini/repositories/machine_learning/deepencoding/features'
cd $WD
echo 'Working directory is' $WD


OUTFILE='times_mpi_vectorized'
rm -f mpi_feats*

export PATH="/home/ansuini/shared/programs/x86_64/anaconda2/bin:$PATH"
module purge
module load openmpi/1.8.3/gnu/4.9.2

for j in 1 2 4 8 16 20
do
mpirun -np $j python extract_mpi_balanced_vectorized.py >> $OUTFILE
done
