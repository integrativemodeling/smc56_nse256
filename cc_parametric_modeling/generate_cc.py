"""
Generate a library of antiparallel coiled-coil dimers using the 
ISAMBARD package (https://github.com/isambard-uob/isambard). This alternate
coiled-coils are energy minimized using a molecular-mechanics forcefield
and thus have low steric strain.

Needs mpi4py.

Author:
Tanmoy Sanyal
Sali lab, UCSF
Email: tsanyal@salilab.org
"""

import os
import argparse
import numpy as np
import pandas as pd
import json
import itertools
import contextlib
from tqdm import trange

import matplotlib.pyplot as plt

from Bio import SeqIO
from Bio.PDB import PDBParser, PDBIO

import cc_utils as utils

from mpi4py import MPI
comm = MPI.COMM_WORLD
me = comm.Get_rank()
nproc = comm.Get_size()
ROOT = 0

# crosslink cutoffs
XL_cutoffs = {"DSSO": 30.0, "CDI": 20.0}

# genetic algorithm specific
POP_SIZE = 100
NGEN = 50
NCORES_GA = 10
NMODELS = 50


def _estimate_register(seq1, seq2, outdir=os.getcwd()):
    """
    Helper function to estimate the starting register of an antiparallel
    coiled-coil dimer such that the overall register is as close as possible
    to an ideal heptad.
    
    Args: seq1 (string): Fasta sequence of 1st monomer of coiled-coil dimer.
    
    seq2 (string): Fasta string of 2nd monomer of coiled-coil dimer.
    
    outdir (str, optional): Output directory. Defaults to current dir.
    """
    startres1 = utils.hp_model_tab[seq1[0]]
    startres2 = utils.hp_model_tab[seq2[0]]
    allowed_registers_1 = utils.REGISTERS[startres1]
    allowed_registers_2 = utils.REGISTERS[startres2]

    register_scores_1 = np.array([utils.get_register_score(seq1, r)[1]
                                  for r in allowed_registers_1])
    register_scores_2 = np.array([utils.get_register_score(seq2, r)[1]
                                  for r in allowed_registers_2])

    registers1 = [allowed_registers_1[i] for i in
                  range(len(allowed_registers_1))
                  if register_scores_1[i] == min(register_scores_1)]

    registers2 = [allowed_registers_2[i] for i in
                  range(len(allowed_registers_2))
                  if register_scores_2[i] == min(register_scores_2)]

    # write log
    s = """
Best register/(s) determined for monomer-1: %s
Best register/(s) determined for monomer-2: %s
    """ % ((", ".join([i for i in registers1])),
           (", ".join([i for i in registers2])))
    print(s)

    # make the register plot
    fig = plt.figure(figsize=(10, 4))
    bar_width = 0.6

    ax1 = fig.add_subplot(1, 2, 1)
    x1 = np.arange(len(allowed_registers_1))
    y1 = 100 * (1 - register_scores_1)
    ax1.bar(x1, y1, width=bar_width, color="deepskyblue")
    ax1.set_xticks(x1)
    ax1.set_xticklabels([i for i in allowed_registers_1])
    ax1.set_ylim([0, 100])
    ax1.set_xlabel("register of starting residue", fontsize=15)
    ax1.set_ylabel("ideal heptad propensity (%)", fontsize=15)
    ax1.set_title("CC-N: %s (%d, %d)" % (protein_name, b1, e1))

    ax2 = fig.add_subplot(1, 2, 2)
    x2 = np.arange(len(allowed_registers_2))
    y2 = 100 * (1 - register_scores_2)
    ax2.bar(x2, y2, width=bar_width, color="salmon")
    ax2.set_xticks(x2)
    ax2.set_xticklabels([i for i in allowed_registers_2])
    ax2.set_ylim([0, 100])
    ax2.set_xlabel("register of starting residue", fontsize=15)
    ax2.set_yticklabels([])
    ax2.set_title("CC-C: %s (%d, %d)" % (protein_name, b2, e2))

    plt.subplots_adjust(wspace=0)
    figname = os.path.join(outdir, "registers.png")
    fig.savefig(figname, bbox_inches="tight")
    return registers1, registers2


def _save_model(model, outdir=os.getcwd(), prefix="cc"):
    io = PDBIO()
    io.set_structure(model)
    io.save(os.path.join(outdir, prefix + ".pdb"))


def _save_params(nmodels, outdir=os.getcwd()):
    paramlist = []
    for i in range(nmodels):
        param_fn = os.path.join(outdir, "_params_%d.json" % i)
        if not os.path.isfile(param_fn):
            print("Warning: optimized parameters for model %d not found" % i)
            continue
        with open(param_fn, "r") as of:
            params = json.load(of)
        params["model"] = i
        paramlist.append(params)
        os.remove(param_fn)

    df = pd.DataFrame(paramlist)
    df.to_csv(os.path.join(outdir, "params.csv"),
              index=False,
              float_format="%.2f")


def _save_scores(nmodels, outdir=os.getcwd()):
    # save the scores
    scorelist = []
    for i in range(nmodels):
        score_fn = os.path.join(outdir, "_scores_%d.json" % i)
        if not os.path.isfile(score_fn):
            print("Warning: scores for optimized model %d not found" % i)
            continue
        with open(score_fn, "r") as of:
            scores = json.load(of)
        scores["model"] = i
        scorelist.append(scores)
        os.remove(score_fn)

    df = pd.DataFrame(scorelist)
    df.to_csv(os.path.join(outdir, "scores.csv"),
              index=False,
              float_format="%.2f")


# -------------
# MAIN SCRIPT
# -------------
# user arguments
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument("-p", "--protein_name", help="Protein name (smc5 or smc6)")
parser.add_argument("-m1", "--monomer1", type=int, nargs="+",
                    help="start and end residues of monomer1")
parser.add_argument("-m2", "--monomer2", type=int, nargs="+",
                    help="start and end residues of monomer2")
parser.add_argument("-f", "--fasta_fn",
                    help="Fasta file containing sequences for smc5 and smc6")
parser.add_argument("-xl", "--xl_fn", help="Crosslink dataset file.")
parser.add_argument("-o", "--outdir", default=os.getcwd(),
                    help="Output directory where all files are written.")

# parse arguments
args = parser.parse_args()
protein_name = args.protein_name
b1, e1 = args.monomer1
b2, e2 = args.monomer2
fasta_fn = os.path.abspath(args.fasta_fn)
xl_fn = os.path.abspath(args.xl_fn)
outdir = os.path.abspath(args.outdir)

# make output dir and start logfile
if me == ROOT:
    os.makedirs(outdir, exist_ok=True)

# get sequences
fasta = {r.id: r.seq._data for r in SeqIO.parse(fasta_fn, format="fasta")}
seq1 = fasta[protein_name][(b1 - 1): e1]
seq2 = fasta[protein_name][(b2 - 1): e2]

# get the cc_def tuple
cc_def = (b1, e1, seq1), (b2, e2, seq2)
s = """
PROTEIN: %s
N-terminal coiled coil (monomer-1): (%d, %d, %s)
C-terminal coiled coil (monomer-2): (%d, %d, %s)
""" % (protein_name, *cc_def[0], *cc_def[1])
if me == ROOT:
    print(s)

# process crosslinks to get intra crosslinks with cutoffs
df = pd.read_csv(xl_fn)
xls = []
for i in range(len(df)):
    xl = tuple(df.iloc[i])[0:5]
    p1, r1, p2, r2, linker = xl
    if p1 == p2 == protein_name:
        xls.append(xl)

# ------------------
# ESTIMATE REGISTER
# ------------------
if me == ROOT:
    registers = _estimate_register(seq1, seq2, outdir=outdir)
else:
    registers = None
registers = comm.bcast(registers, root=ROOT)

# --------------------------------------
# OPTIMIZE FOR EACH REGISTER COMBINATION
# ----------------------------------------
regpairs = list(itertools.product(*registers))
nmodels = len(regpairs) * NMODELS

# find my chunk of models to process
chunksize = int(nmodels / nproc)
my_start = me * chunksize
my_stop = (me + 1) * chunksize
if me == nproc - 1:
    my_stop = nmodels

if me == ROOT:
    s = """
Register pairs to examine: %s
Each pair re-optimized %d times
Starting optimization...
""" % ((", ".join([str(i) for i in regpairs])),
        NMODELS)
    print(s)

regpairs *= NMODELS
if me == ROOT and nproc > 1:
    print("\nGenerating %d models per processor" % (my_stop-my_start))
    iterable = trange(my_start, my_stop)
else:
    iterable = range(my_start, my_stop)

for i in iterable:
    # run ISAMBARD's GA optimization
    if nproc > 1:
        with contextlib.redirect_stdout(None):
            model, param_dict = utils.optimize_cc_budeff(cc_def=cc_def,
                                                         registers=regpairs[i],
                                                         pop_size=POP_SIZE,
                                                         ngen=NGEN,
                                                         ncores=NCORES_GA)
    else:
        model, param_dict = utils.optimize_cc_budeff(cc_def=cc_def,
                                                     registers=regpairs[i],
                                                     pop_size=POP_SIZE,
                                                     ngen=NGEN,
                                                     ncores=NCORES_GA)

    # score the model
    n_cov, n_sat, frac_sat = utils.get_xl_sat(model, xls)
    l1, l2, l3, l4 = utils.get_cc_len(model)

    # save the optimized model
    _save_model(model=model, outdir=outdir, prefix="cc_%d" % i)

    # save model params
    tmp_param_fn = os.path.join(outdir, "_params_%d.json" % i)
    with open(tmp_param_fn, "w") as of:
        json.dump(param_dict, of)

    # save scores
    score_dict = {"n_XL_cov"   : n_cov,
                  "n_XL_sat"   : n_sat,
                  "frac_XL_sat": frac_sat,
                  "b1-e1"      : l1,
                  "b2-e2"      : l2,
                  "b1-b2"      : l3,
                  "e1-e2"      : l4}

    tmp_score_fn = os.path.join(outdir, "_scores_%d.json" % i)
    with open(tmp_score_fn, "w") as of:
        json.dump(score_dict, of)

# wait till all procs have finished
comm.barrier()

# release excess procs
if me == ROOT:
    s = """
%d models optimized and scored
""" % nmodels
    print(s)
else:
    exit()

# write optimized params to file
_save_params(nmodels, outdir=outdir)

# write scores to file
_save_scores(nmodels, outdir=outdir)
