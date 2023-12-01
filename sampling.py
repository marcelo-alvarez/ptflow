import numpy as np
import jax.numpy as jnp
import ptflow   as ptf
import analysis as pfa
import pathlib
import os.path
import copy
from time import time

def reportsparams(config,params,sparamnames,loss,i,nval):
    print(f" {i:>3}/{nval:<3} loss: {loss:<7.3f} ",end="")
    for sparam in sparamnames:
        print(f"{sparam}: {params[sparam]:<7.3f} ",end="")
    for param in config.samplbnds:
        if param not in sparamnames:
            print(f"{param}: {params[param]:<7.3f} ",end="")    

def optfromsample(config,inparams):

    datadir = "./data/"
    pathlib.Path(datadir).mkdir(parents=True, exist_ok=True)     
    lossfile  = datadir + 'loss.npz'
    params = copy.deepcopy(inparams)

    print()

    from scipy.stats import qmc

    sparamnames = config.samplprms
    nvar = len(sparamnames)

    l_bounds = []
    u_bounds = []
    fids = []
    for sparam in sparamnames:

        smplbnds = config.samplbnds[sparam]

        l_bound = np.log10(smplbnds['lower']) if smplbnds['logscale'] else smplbnds['lower']
        u_bound = np.log10(smplbnds['upper']) if smplbnds['logscale'] else smplbnds['upper']
        fid     = np.log10(params[sparam])    if smplbnds['logscale'] else params[sparam]

        l_bounds.append(l_bound)
        u_bounds.append(u_bound)
        fids.append(fid)

    if config.sampl:
        sampler = qmc.LatinHypercube(d=nvar, strength=2)
        vparams = sampler.random(n=config.sqrtN**2)
        vparams = qmc.scale(vparams, l_bounds, u_bounds)
    else:
        vparams = np.zeros((0,nvar))

    # insert central point by hand
    vparams = np.asarray(vparams)
    vparams = np.insert(vparams,[0],[fids],axis=0)
    nval=np.shape(vparams)[0]

    for i in range(nval):
        for j in range(nvar):
            sparam = sparamnames[j]
            smplbnds = config.samplbnds[sparam]
            if smplbnds['logscale']: vparams[i,j] = 10**vparams[i,j]

    sparamvalsl = []
    print("sampling parameter space")
    lossl = []
    for i in range(nval):
        t0 = time()
        for j in range(nvar):
            sparam = sparamnames[j]
            params[sparam] = vparams[i,j]
        if i>0: config.verbose = False 

        if any(param in sparamnames for param in ("d0","beta","gamma")):
            params = ptf.setupflowprofile(config,params)
        closs, [rhopfl, mask] = ptf.flowloss(config,params)

        # if first iteration then analyze fiducial model
        if i==0: pfa.analyze(config,params,rhopfl,mask)
        sparamvalsl.append(vparams[i,:])
        lossl.append(closs)

        sparamvals = np.asarray(sparamvalsl)
        loss = np.asarray(lossl)

        dm          = loss.argsort()
        loss        = loss[dm]
        for j in range(nvar):
            sparamvals[:,j] = sparamvals[:,j][dm]

        if config.sampl:
            np.savez(lossfile,loss=loss,sparamvals=sparamvals,sparamnames=sparamnames)

        i+=1
        reportsparams(config,params,sparamnames,closs,i,nval)
        print(f"minloss: {loss[0]:<7.3f} dt: {time()-t0:<6.2} ")
    print()

    if os.path.isfile(lossfile):
        data = np.load(lossfile)

        closs   = data['loss'][0]
        sparamvals  = data['sparamvals'][0]
        sparamnames = data['sparamnames']

        nvar = len(sparamnames)
        for i in range(nvar):
            params[sparamnames[i]] =  sparamvals[i]
 
        print(f"running for optimal sampled parameters")
        reportsparams(config,params,sparamnames,closs,1,1)
        print()

        if any(param in sparamnames for param in ("d0","beta","gamma")): params = ptf.setupflowprofile(config,params)
        config.verbose = True
        loss, [rhopfl,mask] = ptf.flowloss(config,params)
        pfa.analyze(config,params,rhopfl,mask,opt=True)
        print()
