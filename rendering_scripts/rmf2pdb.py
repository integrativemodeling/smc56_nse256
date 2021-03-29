"""
This script converts structurally available parts of an integrative modeling
generated RMF file to a PDB file and orients these parts to align with the
RMF file. This RMF is usually the centroid model of the most populated cluster
obtained by clustering all good scoring models.

It needs as input the PMI topology file and access to all the pdb files
of rigid bodies that were used in the integrative modeling protocol.

Author
Tanmoy Sanyal
Sali lab, UCSF
Email: tsanyal@salilab.org
"""

import os
import numpy as np
import string
from collections import OrderedDict
import argparse

from Bio.PDB import PDBParser, PDBIO
from Bio.PDB import Residue, Chain, Model
from Bio.SVDSuperimposer import SVDSuperimposer

import RMF
import IMP
import IMP.rmf
import IMP.atom
import IMP.core
import IMP.pmi.topology


def _align_component_with_rmf(component, model, rmf_hier):
    pdb_resrange = component.residue_range
    pdb_residues = list(range(pdb_resrange[0], pdb_resrange[1]+1))
    
    pdb_coords, pdb_atoms = [], []
    rmf_coords = []
    for r in model[component.chain].get_residues():
        if (r.id[0] != " ") or (r.id[1] not in pdb_residues): continue
        pdb_coords.append(r["CA"].coord)
        pdb_atoms.extend([a for a in r.get_atoms()])
        
        rmf_sel = IMP.atom.Selection(rmf_hier, molecule=component.molname,
                            resolution=1,
                            residue_index=r.id[1]+component.pdb_offset)
        rmf_particle = IMP.core.XYZ(rmf_sel.get_selected_particles()[0])
        rmf_coords.append(rmf_particle.get_coordinates())
        
    rmf_coords = np.array(rmf_coords, dtype=np.float32)
    pdb_coords = np.array(pdb_coords, dtype=np.float32)
    
    aln = SVDSuperimposer()
    aln.set(rmf_coords, pdb_coords)
    aln.run()
    rotmat, vec = aln.get_rotran()
    [a.transform(rotmat, vec) for a in pdb_atoms]


def _make_structured_res_from_component(component, model, target_chain):
    pdb_resrange = component.residue_range
    pdb_residues = list(range(pdb_resrange[0], pdb_resrange[1]+1))
    residues = []
    for r in model[component.chain].get_residues():
        if (r.id[0] != " ") or (r.id[1] not in pdb_residues): continue
        new_resid = (r.id[0], r.id[1]+component.pdb_offset, r.id[2])
        new_r = Residue.Residue(id=new_resid, resname=r.resname,
                                segid=r.segid)
        [new_r.add(a) for a in r.get_atoms()]
        residues.append(new_r)
    return residues


def _assemble_structured_model_from_topology(topology, model_dict):
    mol2chain = OrderedDict()
    chains = OrderedDict()
    
    for component in topology.get_components():
        # ignore unstructured residues
        if component.pdb_file == "BEADS": continue
        
        # get target chain
        mol = component.molname
        if mol not in mol2chain:
            count = len(chains)
            c = string.ascii_uppercase[count]
            chains[c] = []
            mol2chain[mol] = c
        target_chain = mol2chain[mol]
        
        # get target model
        pdb_prefix = os.path.basename(component.pdb_file).split(".pdb")[0]
        target_model = model_dict[pdb_prefix]
        
        # build residues for this part of the target molecule
        residues = _make_structured_res_from_component(component, 
                                                       target_model,
                                                       target_chain)
        chains[target_chain].extend(residues)
    
    model = Model.Model(0)
    for c, residues in chains.items():
        chain = Chain.Chain(c)
        [chain.add(r) for r in residues]
        model.add(chain)
    
    return model, mol2chain
    
    
#### MAIN ####
parser = argparse.ArgumentParser(description=__doc__,
                        formatter_class=argparse.RawDescriptionHelpFormatter)
    
parser.add_argument("-r", "--rmf_fn", help="RMF file.")

parser.add_argument("-t", "--topology_fn", help="PMI style topology file.")
    
parser.add_argument("-p", "--pdb_dir", 
                    help="Directory containing all PDB files.")

parser.add_argument("-o", "--out_pdb_fn", default=None,
                    help="Output pdb file name.")

args = parser.parse_args()
rmf_fn = os.path.abspath(args.rmf_fn)
topology_fn = os.path.abspath(args.topology_fn)
pdb_dir = os.path.abspath(args.pdb_dir)
if args.out_pdb_fn is not None:
    out_pdb_fn = os.path.abspath(args.out_pdb_fn)
else:
    prefix = os.path.basename(rmf_fn).split(".rmf3")[0]
    out_pdb_fn = prefix + ".pdb"

# get topology
topology = IMP.pmi.topology.TopologyReader(topology_fn, pdb_dir=pdb_dir)

# get all biopython models
model_dict = OrderedDict()
for component in topology.get_components():
    if component.pdb_file == "BEADS": continue
    pdb_fn = os.path.abspath(component.pdb_file)
    pdb_prefix = os.path.basename(component.pdb_file).split(".pdb")[0]
    if pdb_prefix not in model_dict:
        model = PDBParser(QUIET=True).get_structure("x", pdb_fn)[0]
        model_dict[pdb_prefix] = model

# read rmf file
rh = RMF.open_rmf_file_read_only(rmf_fn)
rmf_model = IMP.Model()
rmf_hier = IMP.rmf.create_hierarchies(rh, rmf_model)[0]
IMP.rmf.load_frame(rh, 0)

# align all components
for component in topology.get_components():
    if component.pdb_file == "BEADS": continue
    pdb_prefix = os.path.basename(component.pdb_file).split(".pdb")[0]
    model = model_dict[pdb_prefix]
    _align_component_with_rmf(component, model, rmf_hier)

# assemble the final pdb file
model, mol2chain = _assemble_structured_model_from_topology(
                    topology, model_dict)

io = PDBIO()
io.set_structure(model)
io.save(out_pdb_fn)

print("Chain map:")
for k, v in mol2chain.items():
    print("mol: %s --> chain: %s" % (k, v))

