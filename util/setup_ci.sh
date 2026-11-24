#!/bin/bash -e

# Set up an environment to run tests under Travis CI (see toplevel .travis.yml)
# or GitHub Actions (see toplevel .github/workflows/build.yml)

IMP_CONDA="imp"

eval "$(conda shell.bash hook)"
conda activate test
conda install --yes -q matplotlib pandas ${IMP_CONDA}

# Don't try to run with MPI even if mpich is available
echo "import pathlib, IMP.mpi; m=pathlib.Path(IMP.mpi.__file__).parent; m.rename(m.with_stem('mpi.bak'))" | python3
