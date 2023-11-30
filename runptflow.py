import sys
import jax
import jax.numpy as jnp
import numpy as np

import ptflow   as ptf
import analysis as pfa
import sampling as pfs

args   = ptf.parsecommandline()

config = ptf.configfromargs(args)
params = ptf.paramsfromargs(args)

config, params = ptf.setupflowprofile(args,config,params)

# fiducial model from command line / default
aux, [loss, rholpt, rhopfl, mask] = ptf.flowloss(params,config)
pfa.analyze(params,config,rholpt,rhopfl,mask)

# find optimal parameters in sample parameter space
pfs.optfromsample(args,params,config)
