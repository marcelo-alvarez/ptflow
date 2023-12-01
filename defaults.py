# model configuration
N     = 625
N0    = 625
nx0   = 200
x2yz  = 1
nm    = 20
logM1 = 10.
logM2 = 15.
zoom  = 0.666
kmax0 = 6
fltr  = "matter"
ctype = "int8"
mtype = "int8"
excl  = False
mask  = True
soft  = False
test  = False
sampl = True
sqrtN = 5
ctdwn = True
ftdwn = False
flowl = True
ploss = True

# model parameters
d0    =  200 
gamma =  1.1
beta  = -2.0
alpha = 0.0
logM0 = 13
sigma = 1.0 # lptsigma in params

# sampling parameters
sprms = "d0,gamma,sigma"

# sampling bounds
samplbnds = {}
samplbnds['d0']       = {'lower':    50, 'upper':  500, 'logscale' :  True}
samplbnds['gamma']    = {'lower':  1.01, 'upper':  1.3, 'logscale' :  True}
samplbnds['beta']     = {'lower':  -2.9, 'upper':-0.05, 'logscale' : False}
samplbnds['alpha']    = {'lower':  -0.1, 'upper':  0.1, 'logscale' : False}
samplbnds['logM0']    = {'lower':    11, 'upper':   15, 'logscale' : False}
samplbnds['lptsigma'] = {'lower':   0.3, 'upper':  2.0, 'logscale' : False}
