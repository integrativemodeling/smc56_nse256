"""
This module contains utilities for building coiled-coil members of the 
Smc5/6 arm region using ISAMBARD and clustering them using crosslink satisfaction data.

In principle, one can also use a more targeted coiled-coil builder based
on ISAMBARD (and designed by the same group) called CCBuilder. But directly
using ISAMBARD gives more fine-grained control on the design aspects. ISAMBARD
can be downloaded from: https://github.com/isambard-uob/isambard

AUTHOR
Tanmoy Sanyal
Sali lab, UCSF
Email: tsanyal@salilab.org
"""

import os
import sys
import io
import numpy as np
import fnmatch
import pandas as pd
import json

from Bio import SeqUtils
from Bio.PDB import PDBParser, PDBIO, Select, Structure, Model, Chain, Residue
from Bio.PDB import Superimposer

import isambard
from isambard.specifications import CoiledCoil
from isambard.optimisation.evo_optimizers import Parameter, GA
import budeff
import ampal

# XL cutoffs
xl_cutoffs = {"DSSO": 30.0, "CDI": 20.0}

# coiled coil registers
REGISTERS = {"H": "ad", "P": "bcefg"}

# parameter ranges for ideal coiled-coil geometry
# Source: "Probing designability via a generalized model of helical
# bundle geometry", Grigoryan, DeGrado, JMB, 2011
RADIUS_RANGE = (5.75, 2.5)
PITCH_RANGE = (150, 50)

# above paper says that z=+2.5 is the avg. for antiparallel ccs
# I'm still gonna use avg z = 0
ZSHIFT_RANGE = (0, 5.0)

# Source: "CCBuilder: an interactive web-based tool for building, designing
# and assessing coiled-coil protein assemblies, Wood, Bruning, et. al,
# Bioinformatics, 2014
INTERFACE_ANGLE_RANGE = (20, 10)

#TODO: figure out typical mean and sd values for superhelical rotation
#SUPER_HELICAL_ROTATION_RANGE = ()

# random number seed for reproducibility
RNG_SEED = 85431
np.random.seed(RNG_SEED)

# reduced H/P alphabet for sequences
POLAR_AA = "AGTSNQDEHRKP"
HYDROPHOBIC_AA = "CMFILVWY"

def hp_model_tab(x):
    if x in POLAR_AA: return "P"
    elif x in HYDROPHOBIC_AA: return "H"


def CoiledCoilDimer(*params):
    """
    ADAPTED FROM CCBuilder2.0:
    https://github.com/woolfson-group/ccbuilder2/blob/master/web/ccbmk2/model_building.py
    
    Builds a model of an anti-parallel coiled coil dimer using the input geometrical parameters.

    Args: params (tuple): parameters of first and second monomer concatenated
    with each other. Parameters of each monomer are:
    (# of residues, radius, pitch, z-shift, superhelical rotation, register)

    Returns: (isambard.specification.CoiledCoil): ISAMBARD assembly object
    specifying a coiled coil
    """

    REGISTER_ADJUST = {
        'a': 0,
        'b': 102.8,
        'c': 205.6,
        'd': 308.4,
        'e': 51.4,
        'f': 154.2,
        'g': 257
    }

    # extract parameters
    aa1, r1, p1, zs1, ia1, shr1, reg1, \
    aa2, r2, p2, zs2, ia2, shr2, reg2 = params

    coiled_coil = CoiledCoil(2, auto_build=False)
    coiled_coil.aas = [aa1, aa2]
    coiled_coil.major_radii = [r1, r2]
    coiled_coil.major_pitches = [p1, p2]
    coiled_coil.raw_phi = [ia1, ia2]
    coiled_coil.phi_c_alphas = [ia1 + REGISTER_ADJUST[reg1],
                                ia2 + REGISTER_ADJUST[reg2]]
    coiled_coil.z_shifts = [zs1, zs2]
    lshr_adjust_1 = (zs1 / p1) * 360
    lshr_adjust_2 = (zs2 / p2) * 360
    old_offsets = coiled_coil.rotational_offsets
    coiled_coil.rotational_offsets = [old_offsets[0] + shr1 - lshr_adjust_1,
                                      old_offsets[1] + shr2 - lshr_adjust_2]

    coiled_coil.orientations = [1, -1]
    coiled_coil.build()
    return coiled_coil


def build_cc_dimer(args):
    """
    Wrapper on the build function of the CoiledCoilDimer object. Called by
    geometry optimizers.

    Args: args (tuple): contains (ISAMBARD assembly object,list of sequences,
    geometrical parameters in the same order as required by CoiledCoilDimer)

    Returns: (isambard.specification.CoiledCoil object): parameterized ISAMBARD
    assembly object specifying a coiled coil.
    """

    specification, sequences, params = args
    model = specification(*params)
    return model


def get_budeff_score(ampal_obj):
    """
    Returns the total energy of coiled coil using the BUDE force-field
    (http://www.bris.ac.uk/biochemistry/research/bude)

    Args: ampal_obj: coiled-coil dimer object using the AMPAL data-structure
    that is native to ISAMBARD.

    Returns: (float): total energy of the assembly using the BUDE forcefield.
    """

    return budeff.get_internal_energy(ampal_obj).total_energy


def optimize_cc_budeff(cc_def, registers, pop_size=10, ngen=10, ncores=1):
    """
    Genetic-algorithm based geometry optimizer for coiled-coil dimers, using
    the BUDE force-field (http://www.bris.ac.uk/biochemistry/research/bude)
    as a physics-based scoring function

    Args: cc_def (tuple): (cc1, cc2) where cc_i is a tuple
    containing (start residue, end residue, sequence) of i^th monomer in the
    coiled coil assembly

    registers (tuple): registers of starting residue of each monomer

    pop_size (int): population size in each generation of the GA

    param ngen (int): number of generations to run

    ncores (int): number of cores to use for parallel GA

    Returns: (tuple): best model (biopython model object) and best params (dict)
    """

    # set specification
    specification = CoiledCoilDimer

    # get sequences
    (b1, e1, seq1), (b2, e2, seq2) = cc_def
    sequences = [seq1, seq2]

    parameters = [
        Parameter.static("aa1", len(seq1)),
        Parameter.dynamic("r1", *RADIUS_RANGE),
        Parameter.dynamic("p1", *PITCH_RANGE),
        Parameter.static("zs1", 0),
        Parameter.dynamic("ia1", *INTERFACE_ANGLE_RANGE),
        Parameter.static("shr1", 0),
        Parameter.static("reg1", registers[0]),

        Parameter.static("aa2", len(seq2)),
        Parameter.dynamic("r2", *RADIUS_RANGE),
        Parameter.dynamic("p2", *PITCH_RANGE),
        Parameter.dynamic("zs2", *ZSHIFT_RANGE),
        Parameter.dynamic("ia2", *INTERFACE_ANGLE_RANGE),
        Parameter.static("shr2", 0),
        Parameter.static("reg2", registers[1])
    ]

    # set the build function
    build_fn = build_cc_dimer

    # set up the eval function
    eval_fn = get_budeff_score

    # set up the optimizer
    opt_ga = GA(specification=specification,
                sequences=sequences,
                parameters=parameters,
                build_fn=build_fn,
                eval_fn=eval_fn)

    # run optimization
    opt_ga.run_opt(pop_size=pop_size, generations=ngen, cores=ncores)

    # get best parameters
    best_ind = opt_ga.halloffame[0]
    best_params = opt_ga.parse_individual(best_ind)
    aa1, r1, p1, zs1, ia1, shr1, reg1, \
    aa2, r2, p2, zs2, ia2, shr2, reg2 = best_params

    best_model = ampal2biopython(opt_ga.best_model, cc_def)
    best_params_dict = {"r1"  : float(r1),
                        "p1"  : float(p1),
                        "zs1" : float(zs1),
                        "ia1" : float(ia1),
                        "shr1": float(shr1),
                        "reg1": reg1,

                        "r2"  : float(r2),
                        "p2"  : float(p2),
                        "zs2" : float(zs2),
                        "ia2" : float(ia2),
                        "shr2": float(shr2),
                        "reg2": reg2}

    return best_model, best_params_dict


def ampal2biopython(ampal_obj, cc_def):
    """
    Convert a protein from the AMPAL to the Biopython data structure

    Args: ampal_obj: coiled-coil dimer object using the AMPAL data-structure
    that is native to ISAMBARD.

    cc_def (tuple): (cc1, cc2) where cc_i is a tuple
    containing (start residue, end residue, sequence) of i^th monomer in the
    coiled coil assembly

    Returns: (Bio.PDB.Model.Model): biopython model object
    """

    pdbstr = io.StringIO(ampal_obj.pdb)
    pdb_obj = PDBParser().get_structure("x", pdbstr)[0]

    # renumber and rename the residues
    (b1, e1, seq1_), (b2, e2, seq2_) = cc_def
    seq1 = [SeqUtils.seq3(i).upper() for i in seq1_]
    seq2 = [SeqUtils.seq3(i).upper() for i in seq2_]
    offset1 = b1 - 1
    offset2 = b2 - 1
    residues1 = list(pdb_obj["A"].get_residues())
    residues2 = list(pdb_obj["B"].get_residues())

    model = Model.Model(0)
    chain1 = Chain.Chain("A")
    chain2 = Chain.Chain("B")

    for i, r in enumerate(residues1):
        new_id = (r.id[0], r.id[1] + offset1, r.id[2])
        new_r = Residue.Residue(id=new_id, resname=seq1[i], segid=r.segid)
        [new_r.add(a) for a in r.get_atoms()]
        chain1.add(new_r)

    for i, r in enumerate(residues2):
        new_id = (r.id[0], r.id[1] + offset2, r.id[2])
        new_r = Residue.Residue(id=new_id, resname=seq2[i], segid=r.segid)
        [new_r.add(a) for a in r.get_atoms()]
        chain2.add(new_r)

    model.add(chain1)
    model.add(chain2)

    return model


def get_xl_sat(model, xls):
    """
    Calculate intra-coiled-coil xl satisfaction

    Args: model (Bio.PDB.Model.Model): biopython model object

    xls (list): tuples each containing
    (protein1, residue1, protein2, residue2, linker). All crosslinks must
    be intra to the protein of interest.

    Returns: (float): total fractional intra crosslink violation on this model
    """

    c_alpha = {r.id[1]: r["CA"] for r in model.get_residues()}

    n_sat = 0
    n_cov = 0
    for xl in xls:
        p1, r1, p2, r2, linker = xl
        if not (r1 in c_alpha and r2 in c_alpha):
            continue
        n_cov += 1
        if c_alpha[r1] - c_alpha[r2] <= xl_cutoffs[linker]:
            n_sat += 1

    if n_cov == 0:
        score = 0.0
    else:
        score = float(n_sat) / n_cov
    return n_cov, n_sat, score


def get_cc_len(model):
    """
    Calculate cc end-to-end lengths.

    Args: model (Bio.PDB.Model.Model): biopython model object

    Returns: (float): max possible end-to-end length of this assembly
    """

    residues_1 = {r.id[1]: r["CA"] for r in model["A"].get_residues()}
    residues_2 = {r.id[1]: r["CA"] for r in model["B"].get_residues()}

    b1 = residues_1[min(residues_1)]
    e1 = residues_1[max(residues_1)]
    b2 = residues_2[min(residues_2)]
    e2 = residues_2[max(residues_2)]

    cc_len = (float(b1 - e1),
              float(b2 - e2),
              float(b1 - b2),
              float(e1 - e2))
    return cc_len


def _get_hamming_distance(s1, s2):
    """
    Helper function to get the hamming distance between two equal length
    strings

    Args: s1 (str): string-1

    s2 (str): string-2 (must be of equal length as string-1)

    Returns: (int): un-normalized hamming distance between s1 and s2
    """
    return sum(x != y for x, y in zip(s1, s2))


def get_register_score(sequence, register):
    """
    Compares the 2-alphabet (HP) reduced version of a sequence to an ideal
    heptad repeat that starts with the given register and accounts for any
    offset that results because of not starting from register "a"

    Args: sequence (str): FASTA sequence of a CC monomer

    register (char): register of the starting residue (one of
    "a", "b", "c", "d", "e", "f", "g", "h")

    Returns: (float): hamming distance between reduced sequence and ideal
    """
    ideal_heptad = "HPPHPPP"   # ih
    ideal_register = "abcdefg"
    heptad_len = 7 #  duh!

    seq = "".join([hp_model_tab[i] for i in sequence])

    # start
    start = ideal_register.index(register)
    if start == 0:
        offset = 0
        target_start = []
        template_start = []
    else:
        offset = heptad_len - start
        target_start = seq[0: offset]
        template_start = ideal_heptad[ideal_register.index(register):]

    # middle
    n_heptad = int(len(seq[offset:]) / heptad_len)
    targets_mid = []
    templates_mid = []
    if n_heptad > 0:
        for i in range(n_heptad):
            startres = offset + i * heptad_len
            stopres = offset + (i+1) * heptad_len
            targets_mid = seq[startres:stopres]

        templates_mid = [ideal_heptad]*n_heptad

    # end
    if n_heptad > 0:
        end_len = len(seq[offset:]) % n_heptad
    else:
        end_len = len(seq[offset:])
    if end_len == 0:
        target_end = []
        template_end = []
    else:
        target_end = seq[-end_len:]
        template_end = ideal_heptad[0:end_len]

    d = []
    d_total = 0.0

    if offset > 0:
        d_start = _get_hamming_distance(target_start, template_start)
        d_total += d_start
        d.extend([d_start/float(offset)] * offset)

    for (target, template) in zip(targets_mid, templates_mid):
        d_mid = _get_hamming_distance(target, template)
        d_total += d_mid
        d.extend([d_mid/float(n_heptad)] * n_heptad)

    if end_len > 0:
        d_end = _get_hamming_distance(target_end, template_end)
        d_total += d_end
        d.extend([d_end / float(end_len)]*end_len)

    return d, d_total / float(len(seq))


def filter_models(score_fn, len_bounds=None, xl_bounds=None):
    """
    Filter models based on satisfaction of XL and avg. end-to-end lengths
    within given upper and lower bounds.

    Args: score_fn (str): file containing dataframe of scores of optimized
    models returned by ISAMBARD.
    
    len_bounds (tuple): upper and lower bounds for avg. cc length.

    xl_bounds (tuple): upper and lower bounds for xl satisfaction.

    Returns: (list): indices of filtered models
    """

    df_score = pd.read_csv(score_fn)

    filtered_indices = []
    for i in range(len(df_score)):
        this_df = df_score.iloc[i]
        xl_viol = 1.0 - this_df["frac_XL_sat"]
        
        test_xl, test_len = True, True
        
        if xl_bounds is not None:
            test_xl = xl_bounds[0] <= xl_viol <= xl_bounds[1]
        if len_bounds is not None:
            test_len = len_bounds[0] <= avg_len <= len_bounds[1]
        
        if test_xl and test_len:
            filtered_indices.append(int(this_df["model"]))

    return filtered_indices


def cluster(model_indices, model_dir, param_fn, score_fn,
            rmsd_cutoff=2.0, outdir=os.getcwd()):
    """
    RMSD based clustering of models using Daura's algorithm
    ("Folding-unfolding thermodynamics of a beta-heptapeptide
    from equilibrium simulations", Daura, Gunsteren, Mark, Proteins, 1999)

    Args: models_indices (list): indices of (filtered) models

    model_dir (str): directory containing model pdbs

    param_fn (str): file containing dataframe of optimized parameters

    score_fn (str): file containing dataframe of scores of optimized
    models

    rmsd_cutoff (float): models within this (post-alignment) rmsd are
    considered part of the same cluster

    outdir (str): output directory

    Returns: (dict): {k: f} k = cluster index,
    f = fraction of models belonging to this cluster. Also saves the members
    in each cluster to outdir as <model_num>.pdb. center.pdb is the center
    """

    # helper function to calculate rmsd
    def _calc_rmsd(model1, model2):
        atoms1 = [r[atom_name] for r in model1.get_residues()
                  for atom_name in ["N", "CA", "C", "O"]]

        atoms2 = [r[atom_name] for r in model2.get_residues()
                  for atom_name in ["N", "CA", "C", "O"]]

        # calculate RMSD
        aln = Superimposer()
        aln.set_atoms(atoms1, atoms2)
        return aln.rms

    # parse pdbs into biopython model objects
    models = []
    for ii, i in enumerate(model_indices):
        pdb_fn = os.path.join(model_dir, "cc_%d.pdb" % i)
        this_model = PDBParser(QUIET=True).get_structure("x", pdb_fn)[0]
        models.append(this_model)

    # calculate rmsd distance matrix
    nmodels = len(models)
    distmat = np.zeros([nmodels, nmodels])
    for i in range(nmodels-1):
        for j in range(i+1, nmodels):
            rmsd = _calc_rmsd(models[i], models[j])
            distmat[i, j] = rmsd
            distmat[j, i] = rmsd

    # ------------------
    # DAURA'S ALGORITHM
    # ------------------

    # populate neighbors of a given model
    neigh = []
    for i in range(nmodels):
        neigh.append([i])  # model is always a neighbor of itself

    for idx in np.argwhere(distmat <= rmsd_cutoff):
        i = idx[0]
        j = idx[1]
        if i > j:
            neigh[i].append(j)
            neigh[j].append(i)

    # set all models as un-clustered (i.e. not visited)
    pool = list(range(nmodels))
    visited = [False for _ in range(nmodels)]

    # init arrays for accumulating stats
    clust_members = []
    clust_centers = []

    # main loop: get the most populated cluster and iterate
    while len(pool) > 0:
        # get cluster with max population
        max_neigh = 0
        curr_center = -1
        # note: doesn't account for multiple clusters with same population
        for p in pool:
            if len(neigh[p]) > max_neigh:
                max_neigh = len(neigh[p])
                curr_center = p

        # form a new cluster with current center and its neighbors
        clust_centers.append(curr_center)
        clust_members.append([n for n in neigh[curr_center]])

        # update neighbors: remove from pool and add to visited
        for n in neigh[curr_center]:
            pool.remove(n)
            visited[n] = True

        # update neighbors:
        # remove neighbors of current center from their parents
        for n in neigh[curr_center]:
            for nn in neigh[n]:
                if not visited[nn]:
                    neigh[nn].remove(n)

    # write clustering results
    clust_count = 0
    clust_frac = {}
    io = PDBIO()

    for (center, members) in zip(clust_centers, clust_members):
        if center == -1:
            continue

        # prevent overcounting of center in members
        if center in members:
            members.remove(center)

        # make cluster dir
        clustdir = os.path.join(outdir, "cluster_%d" % clust_count)
        os.makedirs(clustdir, exist_ok=True)
        clust_models = [models[center]] + \
                       [models[i] for i in members]

        # write center and member models to pdb files
        for i, model in enumerate(clust_models):
            io.set_structure(model)
            if i == 0:
                pdb_fn = "center.pdb"
            else:
                pdb_fn = str(i-1) + ".pdb"
            model_pdb_fn = os.path.join(clustdir, pdb_fn)
            io.save(model_pdb_fn)

        clust_frac[clust_count] = float(len(clust_models)) / nmodels

        center_idx = model_indices[center]
        neigh_indices = [model_indices[x] for x in members]

        # parameters of the center and parameter variability
        df_params = pd.read_csv(param_fn)
        param_keys = list(df_params.keys())
        param_keys.remove("model")
        param_outdict = {}
        df_center = df_params.query("model == %d" % center_idx)
        for k in param_keys:
            if k in ["reg1", "reg2"]:
                param_outdict[k] = df_center[k].values[0]
            else:
                param_outdict[k] = [df_center[k].values[0], 0.0]

        for i in neigh_indices:
            this_df = df_params.query("model == %d" % i)
            for k in param_keys:
                if k in ["reg1", "reg2"]:
                    continue
                x = (this_df[k].values[0] - df_center[k].values[0])
                param_outdict[k][1] += x * x

        for k in param_keys:
            if k in ["reg1", "reg2"]:
                continue
            param_outdict[k][1] = np.sqrt(param_outdict[k][1]) / \
                                  len(neigh_indices)

        clust_param_fn = os.path.join(clustdir, "params.csv")
        pd.DataFrame(param_outdict).to_csv(clust_param_fn, index=False)

        # score of the center
        df_scores = pd.read_csv(score_fn)
        score_keys = list(df_scores.keys())
        score_keys.remove("model")
        scoredict =[{k: 0.0 for k in score_keys}]
        df_center = df_scores.query("model == %d" % center_idx)
        for k in score_keys:
            scoredict[0][k] = df_center[k].values[0]

        clust_score_fn = os.path.join(clustdir, "score_center.csv")
        pd.DataFrame(scoredict).to_csv(clust_score_fn, index=False)

        clust_count += 1

    return clust_frac

