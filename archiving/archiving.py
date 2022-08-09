"""
This script builds the system representation, adds the details of the spatial restraints and structural sampling, and prepares the mmcif file of the integrative model of the yeast Smc5/6-Nse2/5/6 complex for deposition into PDB-Dev.

Author:
Tanmoy Sanyal,
Sali lab, UCSF
Email: tsanyal@salilab.org
"""

import os
import copy
import numpy as np
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

import ihm
import ihm.cross_linkers
import ihm.dumper
import ihm.location 
import ihm.model
import IMP.pmi.mmcif

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

# input filenames
datadir = "../data"
topo_fn = os.path.join(datadir, "topology.txt")
xl_intra_DSSO_fn = os.path.join(datadir, "xl", "xl_intra_DSSO.csv")
xl_inter_DSSO_fn = os.path.join(datadir, "xl", "xl_inter_DSSO.csv")
xl_CDI_fn = os.path.join(datadir, "xl", "xl_CDI.csv")

# output file names (produced after structural clustering)
output_centroid_rmf_file = "../integrative_modeling/structural_clustering/sp_50/cluster.0/cluster_center_model.rmf3"

output_trajectory_rmf_file = "../integrative_modeling/structural_clustering/good_scoring_models.rmf3"

output_cluster_0_indices_file = "../integrative_modeling/structural_clustering/sp_50/cluster.0.all.txt"

density_ranges_file = "../integrative_modeling/density_ranges.txt"

density_dir = "../integrative_modeling/structural_clustering/sp_50/cluster.0"


# -------------------------
# REPRESENTATION
# -------------------------
m = IMP.Model()
bs = IMP.pmi.macros.BuildSystem(m)

# start recording the modeling protocol from here
po = IMP.pmi.mmcif.ProtocolOutput()
bs.system.add_protocol_output(po)
po.system.title = "Integrative analysis reveals unique structural and functional features of the Smc5/6 complex"

# add publication
po.system.citations.append(ihm.Citation.from_pubmed_id(33941673))

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
    filelabel="CDI",
    linker=cdi_ihm_obj)
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

# run a dummy replica exchange
print("\n\nWARM-UP RUNS\n\n")
rex1 = IMP.pmi.macros.ReplicaExchange0(m, root_hier,
                            monte_carlo_sample_objects=dof.get_movers(),
                            global_output_directory="./output_warmup",
                            output_objects=output_objects,
                            write_initial_rmf=True,

                            monte_carlo_steps=MC_STEPS_PER_FRAME_1,
                            number_of_frames=MC_FRAMES_1,
                            number_of_best_scoring_models=0,

                            monte_carlo_temperature=MC_TEMP,
                            replica_exchange_minimum_temperature=MIN_REX_TEMP,
                            replica_exchange_maximum_temperature=MAX_REX_TEMP,
                            
                            test_mode=True)
rex1.execute_macro()

# run replica exchange Monte-Carlo again
print("\n\nPRODUCTION RUNS\n\n")
rex2 = IMP.pmi.macros.ReplicaExchange0(m, root_hier,
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

                           replica_exchange_object=rex1.replica_exchange_object,
                           
                           test_mode=True)
rex2.execute_macro()


# -----------------------------
# CREATING THE FINAL MMCIF FILE
# ----------------------------- 
# collect all information from the modeling protocol
po.finalize()
s = po.system

# ============ 1. SEQUENCES ==============
# add uniprot references for subunits
Uniprot_accesion_codes = {"smc5.0": "Q08204",
                          "smc6.0": "Q12749",
                          "nse2.0": "P38632",
                          "nse5.0": "Q03718"}
for p, c in Uniprot_accesion_codes.items():
    ref = ihm.reference.UniProtSequence.from_accession(c)
    po.asym_units[p].entity.references.append(ref)


# ============ 2. CROSSLINKS =============
# note: XLs are the only restraints recorded by the protocol-output
# since excluded volume and connectivity are physics-based and assumed
# to be satisfied by default by PDB-Dev

# first link to the PRIDE database where the spectra have been deposited
l = ihm.location.PRIDELocation("PXD023164")
pride_db = ihm.dataset.MassSpecDataset(location=l)
for r in s.restraints:
    r.dataset.add_primary(pride_db)

xl_dataset_group = ihm.dataset.DatasetGroup(
                                    elements=[r.dataset for r in s.restraints],
                                    name="All Smc5/6-Nse2/5/6 crosslinks",
                                    application="Percentage satisfaction")


# ============ 3. SAMPLING ===============
# add the total number of mcmc iteration steps (warm-up + production)
last_step = s.orphan_protocols[-1].steps[-1]
last_step.num_models_end = 100000


# ============ 4. ANALYSIS ===============
protocol = po.system.orphan_protocols[-1]
a = ihm.analysis.Analysis()
protocol.analyses.append(a)
structural_clustering_step = ihm.analysis.ClusterStep(feature="RMSD", 
                                       num_models_begin=10000000,
                                       num_models_end=29975)
a.steps.append(structural_clustering_step)


# ============ 5. CENTROID MODEL FROM TOP CLUSTER =============
# make a separate group for the (only) structural cluster and add to last state
mg = ihm.model.ModelGroup(name="Cluster 0")
po.system.state_groups[-1][-1].append(mg)

# now describe this cluster
e = ihm.model.Ensemble(model_group=mg,
                       num_models=29975,
                       post_process=a.steps[-1],
                       name="Cluster 0",
                       clustering_method="Other",
                       details="Density based threshold-clustering",
                       clustering_feature="RMSD",
                       precision=38.97)  # Angstroms
po.system.ensembles.append(e)

# add the model coordinates of the centroid model of this cluster
rh = RMF.open_rmf_file_read_only(output_centroid_rmf_file)
IMP.rmf.link_hierarchies(rh, [root_hier])
IMP.rmf.load_frame(rh, 0)
del rh
model = po.add_model(e.model_group)


# ============ 7. ZENODO HARD-LINK ==============
repo = ihm.location.Repository(doi="10.5281/zenodo.4685414", root="../",
       top_directory="smc56_nse256",
       url="https://zenodo.org/record/4685415")


# ============ 8. LOCALIZATION DENSITIES =============
with open(density_ranges_file, "r") as of:
    density_ranges_str = of.read()
density_ranges = eval(density_ranges_str.split("=")[-1].strip())
for subunit in ["smc5", "smc6", "nse2", "nse5", "nse6"]:
    density_mrc_file = os.path.join(density_dir, "LPD_%s.mrc" % subunit)
    loc = ihm.location.OutputFileLocation(density_mrc_file, repo=repo)
    density = ihm.model.LocalizationDensity(
        file=loc, asym_unit=po.asym_units[subunit + ".0"])
    e.densities.append(density)


# update everything and write mmcif to file
po.system.update_locations_in_repositories([repo])
po.finalize()
with open("smc56_nse256.cif", "w") as of:
    ihm.dumper.write(of, [po.system])

# delete un-necessary files
os.system("rm *.xl.db")
