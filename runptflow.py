import ptflow   as ptf
import analysis as pfa
import sampling as pfs

# create configuration and load fiducial model parameters
config, params = ptf.initialize()

# find optimal parameters in sample parameter space
pfs.optfromsample(config,params)
