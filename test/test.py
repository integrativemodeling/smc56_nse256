import ihm.reader
import sys
import os
import glob
import shutil
import subprocess
import unittest
import RMF

TOPDIR = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..'))

MODELLER_DIR = os.path.join(TOPDIR, "non_cc_comparative_modeling",
                            "smc56_head_MODELLER")
N_COMPARATIVE_MODELS = 20

INTEGRATIVE_MODELING_DIR = os.path.join(TOPDIR, "ntegrative_modeling")
DATADIR = os.path.join(TOPDIR, "data")
N_WARMUP_MODELS = 20
N_PRODUCTION_MODELS = 20


class Tests(unittest.TestCase):
    def test_smc56_head_model(self):
        """
        Test if MODELLER runs succesfully on the Smc5/6 head-dimer model
        and produces 20 comparative models
        """
        currdir = os.getcwd()
        
        # run the modeling script
        os.chdir(MODELLER_DIR)
        p = subprocess.check_call(["python", "modeling.py"])
        
        # check that 20 models have been produced
        model_pdb_files = glob.glob(os.path.join(MODELLER_DIR, "smc56_head.B*.pdb"))
        assert len(model_pdb_files) == N_COMPARATIVE_MODELS

        # delete the files
        del_files = glob.glob(os.path.join(MODELLER_DIR, "smc56_head.*"))
        [os.remove(f) for f in del_files]
        
        # switch to current dir
        os.chdir(currdir)

    def test_mmcif(self):
        """Test generation of mmCIF output"""
        mmcif = 'smc56_nse256.cif'
        os.chdir(os.path.join(TOPDIR, 'archiving'))
        if os.path.exists(mmcif):
            os.unlink(mmcif)
        # Potentially override methods that need network access
        env = os.environ.copy()
        env['PYTHONPATH'] = \
            os.path.join(TOPDIR, 'test', 'mock') + ':' \
            + env.get('PYTHONPATH', '')
        subprocess.check_call(["python", 'archiving.py'], env=env)
        self._check_mmcif(mmcif)

    def _check_mmcif(self, fname):
        with open(fname) as fh:
            s, = ihm.reader.read(fh)
        self.assertEqual(len(s.citations), 0)
        self.assertEqual(len(s.software), 4)
        self.assertEqual(len(s.orphan_starting_models), 27)
        # Should be 1 state
        self.assertEqual(len(s.state_groups), 1)
        state1, = s.state_groups[0]
        # Should be 1 model
        self.assertEqual(sum(len(x) for x in state1), 1)
        # Check # of spheres and atoms in one model
        m = state1[0][0]
        self.assertEqual(len(m._spheres), 2805)
        self.assertEqual(len(m._atoms), 0)
        # Should be 1 ensemble
        self.assertEqual([e.num_models for e in s.ensembles], [29975])
        # Three restraints - crosslinks
        xl1, xl2, xl3 = s.restraints
        self.assertEqual(len(xl1.experimental_cross_links), 159)
        self.assertEqual(len(xl1.cross_links), 159)
        self.assertEqual(len(xl2.experimental_cross_links), 109)
        self.assertEqual(len(xl2.cross_links), 109)
        self.assertEqual(len(xl3.experimental_cross_links), 67)
        self.assertEqual(len(xl3.cross_links), 67)

    def test_integrative_model(self):
        """
        Test if IMP structural sampling on the Smc5/6 pentamer is successful
        and produces warmup and output rmf files with 20 models in each.
        """
        currdir = os.getcwd()
        
        # run the modeling script from a tmp dir
        outdir = os.path.abspath("tmp_run")
        os.makedirs(outdir, exist_ok=True)
        os.chdir(outdir)
        
        script = os.path.join(INTEGRATIVE_MODELING_DIR, "modeling.py")
        p = subprocess.check_call(["python", script, "-d", DATADIR, "-t"])
        
        rmf_fn_1 = os.path.join(outdir, "output_warmup", "rmfs", "0.rmf3")
        assert os.path.isfile(rmf_fn_1)
        rh1 = RMF.open_rmf_file_read_only(rmf_fn_1)
        assert rh1.get_number_of_frames() == N_WARMUP_MODELS
        
        rmf_fn_2 = os.path.join(outdir, "output", "rmfs", "0.rmf3")
        assert os.path.isfile(rmf_fn_2)
        rh2 = RMF.open_rmf_file_read_only(rmf_fn_2)
        assert rh2.get_number_of_frames() == N_PRODUCTION_MODELS
        
        # switch to current dir
        os.chdir(currdir)
        
        # delete the tmp output dir
        shutil.rmtree(outdir, ignore_errors=True)
        

if __name__ == "__main__":
    unittest.main()
