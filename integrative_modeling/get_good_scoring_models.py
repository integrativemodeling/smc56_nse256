""" 
This script extracts good scoring models after all scores have been analyzed
and frame numbers (i.e. monte carlo iterations) for good scoring models have
been noted. If the total number of good scoring models > NMAX, then they are 
further subsampled and NMAX models are retained. Usually NMAX=30,000.

This assumes that the multiple runs of the system have been kept in directories
called run_1/output, run_2/output, etc.

## Author:
Tanmoy Sanyal, Sali lab, UCSF, tsanyal@salilab.org
"""

import os
import glob
import pandas as pd
import numpy as np
import argparse

from analysis_trajectories import AnalysisTrajectories

# take user input
parser = argparse.ArgumentParser(description=__doc__,
                        formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument("-a", "--analysis_dir", default="analys", 
                    help="Directory containing the results of score distribution analysis.")

parser.add_argument("-t", "--traj_dir", default=".", 
                    help="Directory containing the independent monte carlo runs")

parser.add_argument("-o", "--outdir", default="structural_clustering", help="Output directory.")

parser.add_argument("-np", "--nproc", type=int, 
                    help="Number of processors for parallel model extraction")

args = parser.parse_args()
analysis_dir = os.path.abspath(args.analysis_dir)
traj_dir = os.path.abspath(args.traj_dir)
outdir = analysis_dir if args.outdir is None else os.path.abspath(args.outdir)
nproc = args.nproc

dir_head = "run_"
run_dirs = glob.glob(os.path.join(traj_dir, dir_head+"*", "output"))

# initialize the analyzer
AT = AnalysisTrajectories(out_dirs=run_dirs, dir_name=dir_head,
                          analysis_dir=analysis_dir, nproc=nproc)

# get the top cluster (nonzero cluster id with largest population)
summary_fn = os.path.join(analysis_dir, "summary_hdbscan_clustering.dat")
df = pd.read_csv(summary_fn)
clust_indices_ = df["cluster"].values
nmodels_ = df["N_models"].values

clust_indices = []
nmodels = []
for ii, i in enumerate(clust_indices_):
    if i >= 0:
        clust_indices.append(i)
        nmodels.append(nmodels_[ii])    
top_clust_idx = clust_indices[np.argmax(nmodels)]

# get the names of good scoring model files
fmt = os.path.join(analysis_dir, "selected_models_%s_cluster%d")
gsms_A = fmt % ("A", top_clust_idx) + "_detailed.csv"
gsms_B = fmt % ("B", top_clust_idx) + "_detailed.csv"
if not os.path.isfile(gsms_A) and os.path.isfile(gsms_B):
    raise IOError("Good scoring model info for top cluster %d not found" % top_clust_idx)
msg = "\nExtracting good scoring models from top cluster %d..." % top_clust_idx

gsms_subsampled_A = fmt % ("A", top_clust_idx) + "_detailed_random.csv"
gsms_subsampled_B = fmt % ("B", top_clust_idx) + "_detailed_random.csv"
if os.path.isfile(gsms_subsampled_A) and os.path.isfile(gsms_subsampled_B):
    gsms_A = gsms_subsampled_A
    gsms_B = gsms_subsampled_B
    msg = "\nExtracting (subsampled) good scoring models from top cluster %d..." % top_clust_idx    

# extract good scoring models from the top cluster into a single RMF file
print(msg)
HA = AT.get_models_to_extract(gsms_A)
HB = AT.get_models_to_extract(gsms_B)

rmf_fn_A = "sample_A_models.rmf3"
scores_prefix_A = "sample_A_scores"

rmf_fn_B = "sample_B_models.rmf3"
scores_prefix_B = "sample_B_scores"

AT.do_extract_models_single_rmf(HA, rmf_fn_A, traj_dir + "/", outdir, 
                                scores_prefix=scores_prefix_A)

AT.do_extract_models_single_rmf(HB, rmf_fn_B, traj_dir + "/", outdir,
                                scores_prefix=scores_prefix_B)

