"""
This script calculates crosslink (XL) satisfaction for all good scoring models
(generated from sampling) of the Smc5/6-Nse2/5/6 complex, belonging
to the requested cluster. Usually this is the most populated cluster, with
cluster index 0.

XL satisfaction is calculated for the entire dataset as well as separately for
DSSO and CDI. A crosslink is considered satisfied if any model in the cluster
satisfies it, i.e. has a CA-CA distance less than the cutofff length for the XL
(which is 30 A for DSSO, and 20 A for CDI).

This script needs the XL dataset as a CSV file with the following columns:
protein1, residue1, protein2, residue2, linker where linker is either "DSSO" or
"CDI".

Outputs are a list of crosslinks with a 1 or 0 indicating XL satisfaction or
violation, a plain txt file contatining the average satisfaction values, and
plots of the CA-CA distance distribution for DSSO and CDI crosslinked residues. 

Author: 
Tanmoy Sanyal
Sali lab, UCSF
Email: tsanyal@salilab.org
"""

import os
import argparse

import pandas as pd
import numpy as np
from collections import OrderedDict
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

import IMP
import RMF
import IMP.core
import IMP.atom
import IMP.rmf

pd.options.display.float_format = "{:,2.2f}".format

XLS_cutoffs = {"DSSO": 30.0, "CDI": 20.0}
MOL_RES = {"smc5": (1, 1093), "smc6": (1, 1114), "nse2": (1, 267), 
           "nse5": (1, 556), "nse6": (1, 464)}


def _get_xl_sat(ps, xl):
    """
    Get XL satisfaction for a given crosslink.
    
    Args: ps (dict): Dict mapping a tuple of the form (protein, residue) 
    to a IMP hierarchy particle.

    xl (tuple): Tuple characterizing a single crosslink, of the form 
    (protein1, residue1, protein2, residue2, linker)

    Returns: (tuple): Distance between crosslinked residues, and a boolean to
    indicate if that distance is less than XL cutoff.
    """
    p1, r1, p2, r2, linker = xl
    particle_1 = ps[(p1, r1)]
    coord_1 = IMP.core.XYZ(particle_1).get_coordinates()
    
    particle_2 = ps[(p2, r2)]
    coord_2 = IMP.core.XYZ(particle_2).get_coordinates()
    
    dist = np.linalg.norm(coord_1 - coord_2)
    sat = dist <= XLS_cutoffs[linker]
    return dist, sat


#### MAIN ####
# user input
parser = argparse.ArgumentParser(description=__doc__,
                        formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument("-r", "--rmf_fn", 
                    help="RMF file containing good scoring sampled models.")

parser.add_argument("-c", "--cluster", type=int,
                    default=0, help="Index of cluster of to process.")

parser.add_argument("-m", "--cluster_members_fn",
help="File containing indices of models belonging to the desired cluster.")

parser.add_argument("-xl", "--xl_fn", help="Crosslink dataset.")

parser.add_argument("-o", "--outdir", default=os.getcwd(), 
                    help="Output directory.")

# parse args
args = parser.parse_args()
cluster = args.cluster
cluster_members_fn = os.path.join(args.cluster_members_fn)
rmf_fn = os.path.abspath(args.rmf_fn)
xl_fn = os.path.abspath(args.xl_fn)
outdir = os.path.abspath(args.outdir)

# make output dir
os.makedirs(outdir, exist_ok=True)

# get indices of models belonging to requested cluster
model_indices = [int(i) for i in np.loadtxt(cluster_members_fn)]

# read XL data
df = pd.read_csv(xl_fn)
xls = [tuple(df.iloc[i]) for i in range(len(df))]

# read system topology from the rmf file
model = IMP.Model()
of = RMF.open_rmf_file_read_only(rmf_fn)
hier = IMP.rmf.create_hierarchies(of, model)[0]
nmodels_all = of.get_number_of_frames()
    
particles = {}
for molname, resrange in MOL_RES.items():
    s = IMP.atom.Selection(hier, molecule=molname)
    if not s.get_selected_particles(): continue
    for i in range(resrange[0], resrange[1]+1):
        this_s = IMP.atom.Selection(hier, molecule=molname, residue_index=i)
        p = this_s.get_selected_particles()[0]
        particles[(molname, i)] = p

# calculate XL satisfaction
xls_dist = {"DSSO": [], "CDI": []}
xls_sat = OrderedDict()
print("\nCalculating XL satisfaction for %d models in cluster %d" % \
     (len(model_indices), cluster))

for ii, i in enumerate(tqdm(model_indices)):
    IMP.rmf.load_frame(of, i)
    for xl in xls:
        p1, r1, p2, r2, linker = xl
        if not (p1 in MOL_RES and p2 in MOL_RES): continue
        dist, sat = _get_xl_sat(ps=particles, xl=xl)
        xls_dist[linker].append(dist)
        if (xl not in xls_sat) or (xl in xls_sat and xls_sat[xl] == 0):
            xls_sat[xl] = int(sat)
        
# save to file
xls_out = [k[:-1] + (v, ) for k, v in xls_sat.items()]
columns = ["protein1", "residue1", "protein2", "residue2", "sat"]
df_xls_out = pd.DataFrame(xls_out, columns=columns)
out_fn = os.path.join(outdir, "XLs_ensemble_satisfaction.csv")
df_xls_out.to_csv(out_fn, index=False)

# calculate overall statistics
DSSO_sat = np.mean([v for k, v in xls_sat.items() if k[-1] == "DSSO"])
CDI_sat = np.mean([v for k, v in xls_sat.items() if k[-1] == "CDI"])
all_sat = np.mean(list(xls_sat.values()))
out_fn = os.path.join(outdir, "XLs_ensemble_satisfaction_avg.txt")
with open(os.path.join(outdir, out_fn), "w") as of:
    of.write("Satisfaction of DSSO crosslinks : %2.2f%%\n" % (100.0 * DSSO_sat))
    of.write("Satisfaction of CDI crosslinks  : %2.2f%%\n" % (100.0 * CDI_sat))
    of.write("Satisfaction of all crosslinks  : %2.2f%%\n" % (100.0 * all_sat))

# plot the distance distributions
fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(121)
ax1.hist(xls_dist["DSSO"], density=False)
ax1.axvline(XLS_cutoffs["DSSO"], ls="--", color="black", lw="1.5")
ax1.set_xlabel("CA-CA distance " + r"$(\AA)$", fontsize=20)
ax1.set_ylabel("distribution", fontsize=20)
ax1.set_title("DSSO", fontsize=20)

ax2 = fig.add_subplot(122)
ax2.hist(xls_dist["CDI"], density=False)
ax2.axvline(XLS_cutoffs["CDI"], ls="--", color="black", lw="1.5")
ax2.set_xlabel("CA-CA distance " + r"$(\AA)$", fontsize=20)
ax2.set_ylabel("distribution", fontsize=20)
ax2.set_title("CDI", fontsize=20)

fig.tight_layout()
out_fn = os.path.join(outdir, "XLs_distance_distributions.svg")
fig.savefig(out_fn, bbox_inches="tight", dpi=100)
