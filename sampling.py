import numpy as np
import jax.numpy as jnp
import ptflow   as ptf
import analysis as pfa
import pathlib
import os.path
import copy

def optfromsample(config,inparams):

    datadir = "./data/"
    pathlib.Path(datadir).mkdir(parents=True, exist_ok=True)     
    lossfile  = datadir + 'loss.npz'
    params = copy.deepcopy(inparams)

    print()
    if config.sampl:

        from scipy.stats import qmc

        sparams = config.samplprms
        nvar = len(sparams)

        l_bounds = []
        u_bounds = []
        fids = []
        for sparam in sparams:
    
            smplbnds = config.samplbnds[sparam]
    
            l_bound = np.log10(smplbnds['lower']) if smplbnds['logscale'] else smplbnds['lower']
            u_bound = np.log10(smplbnds['upper']) if smplbnds['logscale'] else smplbnds['upper']
            fid     = np.log10(params[sparam])    if smplbnds['logscale'] else params[sparam]

            l_bounds.append(l_bound)
            u_bounds.append(u_bound)
            fids.append(fid)

        sampler = qmc.LatinHypercube(d=nvar, strength=2)
        vparams = sampler.random(n=config.sqrtN**2)
        vparams = qmc.scale(vparams, l_bounds, u_bounds)

        # insert central point by hand
        vparams = np.asarray(vparams)
        vparams = np.insert(vparams,[0],[fids],axis=0)
        nval=np.shape(vparams)[0]

        for i in range(nval):
            for j in range(nvar):
                sparam = sparams[j]
                smplbnds = config.samplbnds[sparam]
                if smplbnds['logscale']: vparams[i,j] = 10**vparams[i,j]

        sampledparamsl = []
        print("sampling parameter space")
        lossl = []
        for i in range(nval):
            for j in range(nvar):
                sparam = sparams[j]
                params[sparam] = vparams[i,j]
            config.verbose = False

            if any(param in sparams for param in ("d0","beta","gamma")):
                params = ptf.setupflowprofile(config,params)
            closs, [rhopfl, mask] = ptf.flowloss(config,params)
            if i==1:
                # analyze fiducial model (i=1) from command line / default
                pfa.analyze(config,params,rhopfl,mask)
            sampledparamsl.append(vparams[i,:])
            lossl.append(closs)
    
            sampledparams = np.asarray(sampledparamsl)
            loss = np.asarray(lossl)

            dm          = loss.argsort()
            loss        = loss[dm]
            for j in range(nvar):
                sampledparams[:,j] = sampledparams[:,j][dm]

            np.savez(lossfile,loss=loss,sampledparams=sampledparams)
            i+=1
            print(f" {i:>3}/{nval:<3} curloss: {closs:<7.3f} minloss: {loss[0]:<7.3f} ",end="")
            for j in range(nvar):
                sparam = sparams[j]
                print(f"{sparams[j]}: {params[sparams[j]]:<7.3f} ",end="")
            print()
        print()

    if os.path.isfile(lossfile):
        data = np.load(lossfile)

        loss          = data['loss']
        sampledparams = data['sampledparams']

        print(f"running for optimal sampled parameters with loss: {loss[0]:<7.3f} ",end="")
        for j in range(nvar):
            sparam = sparams[j]
            params[sparam] = sampledparams[0,j]
            print(f"{sparams[j]}: {params[sparam]:<7.3f} ",end="")
        print()
        config.verbose = True
        if any(param in sparams for param in ("d0","beta","gamma")):
            params = ptf.setupflowprofile(config,params)
        loss, [rhopfl,mask] = ptf.flowloss(config,params)
        pfa.analyze(config,params,rhopfl,mask,opt=True)
        print()
