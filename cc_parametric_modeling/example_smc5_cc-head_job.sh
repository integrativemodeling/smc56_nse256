#!/bin/bash

# input dirs
DATADIR=../data

# output dir
OUTDIR=./smc5/cc-head
mkdir -p $OUTDIR

# run
SCRIPT=generate_cc.py
mpirun -np 8 python -W ignore $SCRIPT -p smc5 -m1 208 266 -m2 885 946 -f $DATADIR/holocomplex.fasta.txt -xl $DATADIR/xl/xl_all.csv -o $OUTDIR
