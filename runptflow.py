import ptflow   as ptf
import analysis as pfa
import sampling as pfs

config, params = ptf.initialize()

# fiducial model from command line / default
loss, [rhopfl, mask] = ptf.flowloss(config,params)
pfa.analyze(config,params,rhopfl,mask)

# find optimal parameters in sample parameter space
pfs.optfromsample(config,params)
