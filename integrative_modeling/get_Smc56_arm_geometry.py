"""
This script calculates the length of the arm, and the angle formed at the elbow
region in the Smc5/6-Nse2/5/6 complex. These quantities are averaged over
all good scoring models (generated from sampling), belonging to the requested cluster. 
Usually this is the most populated cluster, with cluster index 0.

Based on the structurally covered regions of the centroid model:
a) The arm extremities are defined as (CA coordinates):
from  Smc5:523 to Smc6:886

b) The vectors for the two parts of the arm that form the elbow are 
(in CA coordinates):
for Smc5: v1 from Smc5:388 to Smc5:811,  v2 from Smc5:399 to Smc5:459
for Smc6: v1 from Smc6:430 to Smc6:881,  v2 from Smc6:437 to Smc6:501   

The angle calculated between these two vectors for Smc5 and Smc6 are averaged to get an 
approximate measure of the elbow angle. 

Output is a plain txt file contatining the average arm length and elbow angle values. 

Author:
Tanmoy Sanyal
Sali lab, UCSF
Email: tsanyal@salilab.org
"""

import os
import argparse

import pandas as pd
import numpy as np
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

import IMP
import RMF
import IMP.core
import IMP.atom
import IMP.rmf

pd.options.display.float_format = "{:,2.2f}".format

MOL_RES = {"smc5": (1, 1093), "smc6": (1, 1114), "nse2": (1, 267), 
           "nse5": (1, 556), "nse6": (1, 464)}

# ARM_LEN_VEC = [("smc6", 521), ("smc5", 890)]
# SMC5_ELBOW_VECS = [(388, 811), (399, 459)]
# SMC6_ELBOW_VECS = [(430, 881), (437, 501)]

ARM_LEN_VEC = [("smc6", 527), ("smc5", 886)]
SMC5_ELBOW_VECS = [(388, 880), (399, 459)]
SMC6_ELBOW_VECS = [(430, 881), (437, 501)]


def _get_arm_length(ps):
    """
    Get length of the Smc5/6 arm.
    
    Args: ps (dict): Dict mapping a tuple of the form (protein, residue) 
    to a IMP hierarchy particle.

    Returns: (float): Distance between ends of the arm.
    """
    p1, r1 = ARM_LEN_VEC[0]
    particle_1 = ps[(p1, r1)]
    coord_1 = IMP.core.XYZ(particle_1).get_coordinates()
    
    p2, r2 = ARM_LEN_VEC[1]
    particle_2 = ps[(p2, r2)]
    coord_2 = IMP.core.XYZ(particle_2).get_coordinates()
    
    return np.linalg.norm(coord_1 - coord_2)


def _get_elbow_angle(ps):
    """
    Get avg. angle at the elbow.
    
    Args: ps (dict): Dict mapping a tuple of the form (protein, residue) 
    to a IMP hierarchy particle.

    Returns: (float): Avg. angle at the elbow.
    """
    # Smc5 elbow angle
    r0 = IMP.core.XYZ(ps[("smc5", SMC5_ELBOW_VECS[0][0])]).get_coordinates()
    r1 = IMP.core.XYZ(ps[("smc5", SMC5_ELBOW_VECS[0][1])]).get_coordinates()
    smc5_vec1 = (r0-r1) / np.linalg.norm(r0-r1)
    
    r0 = IMP.core.XYZ(ps[("smc5", SMC5_ELBOW_VECS[1][0])]).get_coordinates()
    r1 = IMP.core.XYZ(ps[("smc5", SMC5_ELBOW_VECS[1][1])]).get_coordinates()
    smc5_vec2 = (r0-r1) / np.linalg.norm(r0-r1)
    
    smc5_elbow_angle = np.arccos(np.dot(smc5_vec1, smc5_vec2)) * (180. / np.pi)
    
    # Smc6 elbow angle
    r0 = IMP.core.XYZ(ps[("smc6", SMC6_ELBOW_VECS[0][0])]).get_coordinates()
    r1 = IMP.core.XYZ(ps[("smc6", SMC6_ELBOW_VECS[0][1])]).get_coordinates()
    smc6_vec1 = (r0-r1) / np.linalg.norm(r0-r1)
    
    r0 = IMP.core.XYZ(ps[("smc6", SMC6_ELBOW_VECS[1][0])]).get_coordinates()
    r1 = IMP.core.XYZ(ps[("smc6", SMC6_ELBOW_VECS[1][1])]).get_coordinates()
    smc6_vec2 = (r0-r1) / np.linalg.norm(r0-r1)
    
    smc6_elbow_angle = np.arccos(np.dot(smc6_vec1, smc6_vec2)) * (180. / np.pi)

    return smc5_elbow_angle, \
           smc6_elbow_angle, \
           0.5 * (smc5_elbow_angle + smc6_elbow_angle)
    

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

parser.add_argument("-o", "--outdir", default=os.getcwd(), 
                    help="Output directory.")

# parse args
args = parser.parse_args()
cluster = args.cluster
cluster_members_fn = os.path.join(args.cluster_members_fn)
rmf_fn = os.path.abspath(args.rmf_fn)
outdir = os.path.abspath(args.outdir)

# make output dir
os.makedirs(outdir, exist_ok=True)

# get indices of models belonging to requested cluster
model_indices = [int(i) for i in np.loadtxt(cluster_members_fn)]

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

# calculate arm length and angle
arm_lengths = []
elbow_angles = []

print("\nCalculating Smc5/6 arm geometry for %d models in cluster %d" % \
     (len(model_indices), cluster))

for ii, i in enumerate(tqdm(model_indices)):
    IMP.rmf.load_frame(of, i)
    arm_lengths.append(_get_arm_length(particles))
    elbow_angles.append(_get_elbow_angle(particles))
        
# save to file
avg_arm_length = 0.1 * np.mean(arm_lengths)  # angstrom --> nm
std_arm_length = 0.1 * np.std(arm_lengths, ddof=1) # angstrom --> nm

avg_smc5_elbow_angle = np.mean([a[0] for a in elbow_angles])
std_smc5_elbow_angle = np.std([a[0] for a in elbow_angles], ddof=1)

avg_smc6_elbow_angle = np.mean([a[1] for a in elbow_angles])
std_smc6_elbow_angle = np.std([a[1] for a in elbow_angles], ddof=1)

avg_elbow_angle = np.mean([a[2] for a in elbow_angles])
std_elbow_angle = np.std([a[2] for a in elbow_angles], ddof=1)

out_fn = os.path.join(outdir, "Smc56_arm_geometry.txt")
with open(os.path.join(outdir, out_fn), "w") as of:
    of.write("Arm length       :  %3.2f +/- %3.2f nm\n" % \
             (avg_arm_length, std_arm_length))
        
    of.write("Smc5 elbow angle :  %3.2f +/- %3.2f degrees\n" % \
            (avg_smc5_elbow_angle, std_smc5_elbow_angle))

    of.write("Smc6 elbow angle :  %3.2f +/- %3.2f degrees\n" % \
            (avg_smc6_elbow_angle, std_smc6_elbow_angle))

    of.write("Elbow angle      :  %3.2f +/- %3.2f degrees\n" % \
            (avg_elbow_angle, std_elbow_angle))
