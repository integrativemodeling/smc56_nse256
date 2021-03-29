from modeller import *
from modeller.automodel import *

class MyModel(automodel):
    def special_patches(self, aln):
        self.rename_segments(segment_ids=["A", "B", "C", "D"],
                             renumber_residues=[42, 949, 80, 988])

env = environ()
env.io.hetatm = False
a = MyModel(env, alnfile="smc56_6qpwAC_6zz6AB.ali", 
            knowns=("6qpw_AC", "6zz6_AB"), 
            sequence="smc56_head",
            assess_methods=(assess.DOPE, assess.GA341, assess.normalized_dope))
a.starting_model=1
a.ending_model=20
a.make()
