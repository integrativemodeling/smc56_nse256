"""
This script analyses score distributions from the structural sampling
This assumes that the multiple runs of the system have been kept in directories
called run_1/output, run_2/output, etc.

Author:
Tanmoy Sanyal,
Sali lab, UCSF
Email: tsanyal@salilab.org
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
from analysis_trajectories import AnalysisTrajectories


# take user input
parser = argparse.ArgumentParser(description=__doc__,
                        formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("-i", "--indir", default=".",
                    help="input directory containing independent runs")
parser.add_argument("-o", "--outdir", default="analys",
                    help="name of output directory")
parser.add_argument("-np", "--nproc", type=int, default=1,
                    help="number of processors for parallel run")
parser.add_argument("-r", "--rerun", action="store_true",
                    help="True for re-runs with only HDBSCAN clustering skipping slow stat-file-reading")

args = parser.parse_args()
topdir = os.path.abspath(args.indir)
outdir = os.path.abspath(args.outdir)
nproc = args.nproc
rerun = args.rerun

os.makedirs(outdir, exist_ok=True)

# get the run dirs for independent runs
dir_head = "run_"
run_dirs = glob.glob(os.path.join(topdir, dir_head+"*", "output"))

XLs_cutoffs = {"intra_DSSO": 30.0, "inter_DSSO": 30.0, "CDI": 20.0}

# init analyzer
AT = AnalysisTrajectories(out_dirs=run_dirs, dir_name=dir_head,
                          analysis_dir=outdir, nproc=nproc)
if not rerun:
    # add restraints
    AT.set_analyze_Connectivity_restraint()
    AT.set_analyze_Excluded_volume_restraint()
    AT.set_analyze_XLs_restraint(XLs_cutoffs=XLs_cutoffs,
                                Multiple_XLs_restraints=True)

    # read stat files (this is slow)
    AT.read_stat_files()
    AT.write_models_info()
    AT.get_psi_stats()
else:
    # directly read stored model info
    AT.read_models_info(XLs_cutoffs)

# do HDBSCAN clustering
AT.hdbscan_clustering(["EV_sum", "XLs_intra_DSSO", "XLs_inter_DSSO", "XLs_CDI"])

# summarize XL satisfaction info
AT.summarize_XLs_info()
