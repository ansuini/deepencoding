#!/bin/bash
#PBS -N splin
#PBS -l walltime=12:00:00
#PBS -l nodes=1:ppn=20
#PBS -T flush_cache
#PBS -q regular

JOBNAME=splin
WD='/scratch/ansuini/repositories/machine_learning/deepencoding/regression'
cd $WD
echo 'Working directory is' $WD


OUTFILE='times_regression_block4_20'
rm -f $JOBNAME*
export PATH="/home/ansuini/shared/programs/x86_64/anaconda2/bin:$PATH"


python sparse_linear.py >> $OUTFILE
