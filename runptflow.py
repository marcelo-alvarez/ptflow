import ptflow       as ptf
import analysis     as pfa
import optimization as pfo

# create configuration and load fiducial model parameters
config, params = ptf.initialize()

if config.gradopt:
    # find optimal parameters by gradient descent
    pfo.optfromgrad(config,params)

if config.sampl:
    # run for a sample of model parameters
    pfo.sampleparams(config,params)
