import os
import glob
import shutil
import subprocess
import unittest
import RMF

MODELLER_DIR = os.path.abspath("../non_cc_comparative_modeling/smc56_head_MODELLER")
N_COMPARATIVE_MODELS = 20

INTEGRATIVE_MODELING_DIR = os.path.abspath("../integrative_modeling")
DATADIR = os.path.abspath("../data")
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
    