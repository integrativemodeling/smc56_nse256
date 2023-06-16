"""
This script builds the system representation, adds the spatial restraints
and performs the conformational sampling for the integrative modeling of the 
yeast Smc5/6-Nse2/5/6 complex, with DSSO and CDI crosslinks.

Input:
A data directory containing:
a) fasta file with all chain sequences called smc56_nse256.fasta.txt
b) subdirectory pdb: contains pdb files (3D co-ordinates) for different parts of the complex.
c) subdirectory xl: contains the crosslink dataset divided into intra- and inter-subunit
partitions for both DSSO, and all CDI crosslinks.

For the analysis scripts to work well, please keep output of independent runs in directories
called run_1, run_2, ...etc.

Author:
Tanmoy Sanyal,
Sali lab, UCSF
Email: tsanyal@salilab.org
"""

import os
import argparse
import pandas as pd

import IMP
import IMP.pmi
import IMP.pmi.tools
import IMP.pmi.macros

import IMP.pmi.topology

import IMP.pmi.restraints
import IMP.pmi.restraints.stereochemistry
import IMP.pmi.io.crosslink
import IMP.pmi.restraints.crosslinking

import RMF
import IMP.rmf
import ihm.cross_linkers

# coarse grained resolution for different restraints
XL_BEAD_RES = 1  # for crosslink restraint
EV_BEAD_RES = 10 # for excluded volume restraint

# rigid body movement parameters
MAX_RB_TRANS = 1.00
MAX_RB_ROT = 0.05
MAX_SRB_TRANS = 1.00
MAX_SRB_ROT = 0.05
MAX_BEAD_TRANS = 2.00  # flexible region / bead movement

# crosslink parameters
XL_CUT = {"DSSO": 30.0, "CDI": 20.0}
XL_SLOPE = 0.02

# randomize initial config parameters
SHUFFLE_ITER = 100
INIT_MAX_TRANS = 30
SHUFFLE_BEADS_ITER = 500

# replica exchange parameters
MC_TEMP = 1.0
MIN_REX_TEMP = 1.0
MAX_REX_TEMP = 2.5

# sampling iterations
MC_FRAMES_1 = 20000
MC_FRAMES_2 = 80000
MC_STEPS_PER_FRAME_1 = 25
MC_STEPS_PER_FRAME_2 = 50

# read and parse user input
parser = argparse.ArgumentParser(description=__doc__,
                        formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument("-d", "--datadir", default="data",
                    help="data directory")

parser.add_argument("-t", "--test", action="store_true",
                    help="true to test with very limited sampling")

args = parser.parse_args()
datadir = os.path.abspath(args.datadir)
is_test = args.test

# test mode sampling iterations
if is_test:
    SHUFFLE_BEADS_ITER = 10
    MC_FRAMES_1 = 20
    MC_FRAMES_2 = 20
    MC_STEPS_PER_FRAME_1 = 50
    MC_STEPS_PER_FRAME_2 = 50

# input filenames
topo_fn = os.path.join(datadir, "topology.txt")
xl_intra_DSSO_fn = os.path.join(datadir, "xl", "xl_intra_DSSO.csv")
xl_inter_DSSO_fn = os.path.join(datadir, "xl", "xl_inter_DSSO.csv")
xl_CDI_fn = os.path.join(datadir, "xl", "xl_CDI.csv")

# -------------------------
# REPRESENTATION
# -------------------------
m = IMP.Model()
bs = IMP.pmi.macros.BuildSystem(m)
t = IMP.pmi.topology.TopologyReader(topology_file=topo_fn,
                                    pdb_dir=os.path.join(datadir, "pdb"),
                                    fasta_dir=os.path.join(datadir))
bs.add_state(t)

root_hier, dof = bs.execute_macro(max_rb_trans=MAX_RB_TRANS,
                                  max_rb_rot=MAX_RB_ROT,
                                  max_bead_trans=MAX_BEAD_TRANS,
                                  max_srb_trans=MAX_SRB_TRANS,
                                  max_srb_rot=MAX_SRB_ROT)

# -------------------------
# RESTRAINTS
# -------------------------
output_objects = []

# connectivity
mols = root_hier.get_children()[0].get_children()
for mol in mols:
    cr = IMP.pmi.restraints.stereochemistry.ConnectivityRestraint(mol)
    cr.add_to_model()
    output_objects.append(cr)

# excluded volume
evr = IMP.pmi.restraints.stereochemistry.ExcludedVolumeSphere(
    included_objects=root_hier,
    resolution=EV_BEAD_RES)
output_objects.append(evr)

# crosslink restraint

# set up a cross link database keyword converter
xldbkc = IMP.pmi.io.crosslink.CrossLinkDataBaseKeywordsConverter()
xldbkc.set_protein1_key("protein1")
xldbkc.set_residue1_key("residue1")
xldbkc.set_protein2_key("protein2")
xldbkc.set_residue2_key("residue2")

# intra-DSSO
xldb_intra_DSSO = IMP.pmi.io.crosslink.CrossLinkDataBase()
xldb_intra_DSSO.create_set_from_file(xl_intra_DSSO_fn, converter=xldbkc)
xlr_intra_DSSO = IMP.pmi.restraints.crosslinking.\
    CrossLinkingMassSpectrometryRestraint(
    root_hier=root_hier,
    database=xldb_intra_DSSO,
    length=XL_CUT["DSSO"],
    resolution=XL_BEAD_RES,
    slope=XL_SLOPE,
    label="intra_DSSO",
    filelabel="intra_DSSO",
    linker=ihm.cross_linkers.dsso)
output_objects.append(xlr_intra_DSSO)
xlr_intra_DSSO.set_psi_is_sampled(True)
dof.get_nuisances_from_restraint(xlr_intra_DSSO)

# inter-DSSO
xldb_inter_DSSO = IMP.pmi.io.crosslink.CrossLinkDataBase()
xldb_inter_DSSO.create_set_from_file(xl_inter_DSSO_fn, converter=xldbkc)
xlr_inter_DSSO = IMP.pmi.restraints.crosslinking.\
    CrossLinkingMassSpectrometryRestraint(
    root_hier=root_hier,
    database=xldb_inter_DSSO,
    length=XL_CUT["DSSO"],
    resolution=XL_BEAD_RES,
    slope=XL_SLOPE,
    label="inter_DSSO",
    filelabel="inter_DSSO",
    linker=ihm.cross_linkers.dsso)
output_objects.append(xlr_inter_DSSO)
xlr_inter_DSSO.set_psi_is_sampled(True)
dof.get_nuisances_from_restraint(xlr_inter_DSSO)

# intra-CDI
# define an ihm CDI cross_linker object
# chemical information src: https://en.wikipedia.org/wiki/Carbonyldiimidazole
cdi_ihm_obj = ihm.ChemDescriptor(auth_name="CDI",
  chemical_name="1,1'-carbonyldiimidazole",
  smiles="O=C(N1CNCC1)N2CCNC2",
  inchi="1S/C7H6N4O/c12-7(10-3-1-8-5-10)11-4-2-9-6-11/h1-6H",
  inchi_key="PFKFTWBEEFSNDU-UHFFFAOYSA-N")

# intra-CDI
xldb_CDI = IMP.pmi.io.crosslink.CrossLinkDataBase()
xldb_CDI.create_set_from_file(xl_CDI_fn, converter=xldbkc)
xlr_CDI = IMP.pmi.restraints.crosslinking.\
    CrossLinkingMassSpectrometryRestraint(
    root_hier=root_hier,
    database=xldb_CDI,
    length=XL_CUT["CDI"],
    resolution=XL_BEAD_RES,
    slope=XL_SLOPE,
    label="CDI",
    filelabel="CDI", linker=cdi_ihm_obj)
output_objects.append(xlr_CDI)
xlr_CDI.set_psi_is_sampled(True)
dof.get_nuisances_from_restraint(xlr_CDI)

# -------------------------
# SAMPLING
# -------------------------
# shuffle all particles to randomize the system
IMP.pmi.tools.shuffle_configuration(root_hier,
                                    max_translation=INIT_MAX_TRANS,
                                    niterations=SHUFFLE_ITER)

# optimize flexible beads to relax large connectivity restraint scores
# note-model object m contains only connectivity restraints at this point
dof.optimize_flexible_beads(nsteps=SHUFFLE_BEADS_ITER)

# add all other restraints to the model
evr.add_to_model()
xlr_intra_DSSO.add_to_model()
xlr_inter_DSSO.add_to_model()
xlr_CDI.add_to_model()

# run replica exchange Monte-Carlo for the first time
print("\n\nWARM-UP RUNS\n\n")
rex1 = IMP.pmi.macros.ReplicaExchange(m, root_hier,
                            monte_carlo_sample_objects=dof.get_movers(),
                            global_output_directory="./output_warmup",
                            output_objects=output_objects,
                            write_initial_rmf=True,

                            monte_carlo_steps=MC_STEPS_PER_FRAME_1,
                            number_of_frames=MC_FRAMES_1,
                            number_of_best_scoring_models=0,

                            monte_carlo_temperature=MC_TEMP,
                            replica_exchange_minimum_temperature=MIN_REX_TEMP,
                            replica_exchange_maximum_temperature=MAX_REX_TEMP)
rex1.execute_macro()

# run replica exchange Monte-Carlo again
print("\n\nPRODUCTION RUNS\n\n")
rex2 = IMP.pmi.macros.ReplicaExchange(m, root_hier,
                           monte_carlo_sample_objects=dof.get_movers(),
                           global_output_directory="./output",
                           output_objects=output_objects,
                           write_initial_rmf=True,
                           
                           monte_carlo_steps=MC_STEPS_PER_FRAME_2,
                           number_of_frames=MC_FRAMES_2,
                           number_of_best_scoring_models=5,
                           
                           monte_carlo_temperature=MC_TEMP,
                           replica_exchange_minimum_temperature=MIN_REX_TEMP,
                           replica_exchange_maximum_temperature=MAX_REX_TEMP,

                           replica_exchange_object=rex1.replica_exchange_object)
rex2.execute_macro()
