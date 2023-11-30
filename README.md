# ptflow
Nonlinear modeling of small-scale matter field using perturbation theory

## running on NERSC
```
# Example to run in default mode at NERSC on a Perlmutter login node

% module use /global/cfs/cdirs/mp107/exgal/env/xgsmenv/20231013-0.0.0/modulefiles/
% module load xgsmenv
% source /global/cfs/cdirs/mp107/exgal/env/xgsmenv/20231013-0.0.0/conda/bin/activate

% export XLA_PYTHON_CLIENT_ALLOCATOR=platform
% export XLA_PYTHON_CLIENT_PREALLOCATE=false
% export PTFLOW_COMPARISON_DATA=/global/cfs/cdirs/mp107/exgal/data/IllustrisTNG/TNG300-1

% cd $SCRATCH
% rm -rf ptflow_test.bak
% if [ -d ptflow_test ] ; then mv ptflow_test ptflow_test.bak ; fi
% git clone git@github.com:marcelo-alvarez/ptflow.git ptflow_test
% cd ptflow_test
% python runptflow.py
```
