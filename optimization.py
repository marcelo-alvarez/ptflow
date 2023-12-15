import numpy as np
import jax.numpy as jnp
import ptflow   as ptf
import analysis as pfa
import pathlib
import os.path
import copy
from time import time

def reportsparams(config,params,sparamnames,loss,i,nval,grad=False):
    print(f" {i:>3}/{nval:<3} loss: {loss:<5.2f} ",end="")
    for sparam in sparamnames:
        print(f"{sparam}: {params[sparam]:<5.2f} ",end="")
    for param in config.pbounds:
        if param not in sparamnames:
            pname=param
            if grad: pname=f"dloss/d{param}"
            print(f"{pname}: {params[param]:<5.2f} ",end="")

def reportgrads(params,pnames,grads,loss,step,i):
    print(f" {i:>3} loss: {loss:<.5e} ",end="")
    for sparam in pnames:
        print(f"{sparam}: {params[sparam]:<.5e} ",end="")
    for sparam in pnames:
        print(f"grad-{sparam}: { grads[sparam]:<.5e} ",end="")
    print(f"step: {step}")

def optfromgrad(config,params):
    config.verbose = False

    gamma = 1e-3
    maxfail = 5
    gradmin = 1e-8

    losses=np.empty(0,dtype=np.float32)
    grads=np.empty(0,dtype=np.float32)
    gparams=np.empty(0,dtype=np.float32)
    oldparams={}
    oldgrad={}
    minloss=1e10
    dloss = 1e10
    loss = 1e10
    i = 0
    ifail=0
    while i<30 or ifail < maxfail:
        cparams,fparams,pparams,params = ptf.parseallparams(params)

        oldloss=loss
        # loss and grad update
        lossgrad,loss,rhopfl,xf,yf,zf,mask = ptf.flowgrad(config,params,fparams)
        if i==0:
            datastring = pfa.savedata(config,params,rhopfl,xf,yf,zf,mask)
            pfa.analyze(datastring)

        minloss = min(minloss,loss)
        dloss = loss-oldloss

        if loss == minloss:
            optparams = copy.deepcopy(params)
        if (dloss > 0 or loss > minloss) and i>0:
            for pname in config.goptprms:
                params[pname]  = oldparams[pname]
                lossgrad[pname] = oldgrad[pname]
            gamma /= 2
            ifail += 1
        else:
            gamma *= 2
            losses=np.append(losses,loss)
            grads=np.append(grads,lossgrad[config.goptprms[0]])
            gparams=np.append(gparams,params[config.goptprms[0]])

        if loss > minloss and abs(dloss)/(loss-minloss) < 1e-3:
            print(f"convergence too slow with dloss={dloss} and loss-minloss={loss-minloss}. stopping...")
            break

        minstep = 1e10
        for pname in config.goptprms:
            oldparams[pname]  = params[pname]
            oldgrad[pname] = lossgrad[pname]
            if abs(lossgrad[pname]) < gradmin: continue
            step = loss/lossgrad[pname]**2
            if step < minstep:
                minstep = step
                diffparam = pname

        updatelist = config.goptprms
        #updatelist = [diffparam]

        reportgrads(fparams,updatelist,lossgrad,loss,gamma,i)

        for pname in updatelist:

            l_bound = config.pbounds[pname]['lower']
            u_bound = config.pbounds[pname]['upper']

            oldval = params[pname]
            newval = oldval - gamma * minstep * lossgrad[pname]

            if l_bound - newval > 0:
                newval = 0.5 * (l_bound+oldval)
                #raise Exception(f"below lower bound for {pname}: {oldval} --> {newval} < {l_bound}")
            elif newval - u_bound > 0:
                newval = 0.5 * (u_bound+oldval)
                #raise Exception(f"above upper bound for {pname}: {oldval} --> {newval} > {u_bound}")

            params[pname] = newval
        i+=1

        if len(losses) > 0:
            losses=np.array(losses)
            gparams=np.array(gparams)
            grads=np.array(grads)
            np.savez("glosses.npz",losses=losses,gparams=gparams,grads=grads)

    print(f"running and analyzing model with optimal parameters")
    cparams,fparams,pparams,params = ptf.parseallparams(optparams)
    lossgrad,loss,rhopfl,xf,yf,zf,mask = ptf.flowgrad(config,params,fparams)
    datastring = pfa.savedata(config,optparams,rhopfl,xf,yf,zf,mask,opt=True)
    reportgrads(fparams,config.goptprms,lossgrad,loss,gamma,0)
    pfa.analyze(datastring)

    return

def sampleparams(config,inparams):

    import defaults as pfd

    pparamnames = pfd.allparams['pparams'].keys()
    datadir = "./data/"
    pathlib.Path(datadir).mkdir(parents=True, exist_ok=True)     
    lossfile  = datadir + 'loss.npz'
    params = copy.deepcopy(inparams)

    from scipy.stats import qmc

    sparamnames = config.soptprms
    nvar = len(sparamnames)

    l_bounds = []
    u_bounds = []
    fids = []
    for sparam in sparamnames:

        smplbounds = config.pbounds[sparam]

        smplbounds['lower'] *= inparams[sparam]
        smplbounds['upper'] *= inparams[sparam]

        l_bound = np.log10(smplbounds['lower']) if smplbounds['logscale'] else smplbounds['lower']
        u_bound = np.log10(smplbounds['upper']) if smplbounds['logscale'] else smplbounds['upper']
        fid     = np.log10(params[sparam])    if smplbounds['logscale'] else params[sparam]

        l_bounds.append(l_bound)
        u_bounds.append(u_bound)
        fids.append(fid)

    if config.smplopt:
        sampler = qmc.LatinHypercube(d=nvar, strength=2)
        vparams = sampler.random(n=config.sqrtN**2)
        vparams = qmc.scale(vparams, l_bounds, u_bounds)
    else:
        vparams = np.zeros((0,nvar))

    # sort samples by first variable value
    dm = vparams[:,0].argsort()
    vparams = vparams[dm,:]

    # insert fiducial point by hand
    vparams = np.asarray(vparams)
    vparams = np.insert(vparams,[0],[fids],axis=0)
    nval=np.shape(vparams)[0]

    for i in range(nval):
        for j in range(nvar):
            sparam = sparamnames[j]
            smplbounds = config.pbounds[sparam]
            if smplbounds['logscale']: vparams[i,j] = 10**vparams[i,j]

    sparamvalsl = []
    if config.smplopt:
        print("optimizing over sampled parameter space")
    else: 
        print("running single model")
    lossl = []
    for i in range(nval):
        t0 = time()
        for j in range(nvar):
            sparam = sparamnames[j]
            params[sparam] = vparams[i,j]
        if i>0: config.verbose = False 

        #if any(param in sparamnames for param in pparamnames):
        modparams = set(list(sparamnames)).intersection(pparamnames)
        if len(modparams) > 0:
            params = ptf.setupflowprofile(config,params)
        closs,rhopfl,xf,yf,zf,mask = ptf.fullflow(config,params)

        # if first iteration then analyze fiducial model
        if i==0:
            datastring = pfa.savedata(config,params,rhopfl,xf,yf,zf,mask)
            pfa.analyze(datastring)


        sparamvalsl.append(vparams[i,:])
        lossl.append(closs)

        sparamvals = np.asarray(sparamvalsl)
        loss = np.asarray(lossl)

        dm          = loss.argsort()
        loss        = loss[dm]

        if closs == loss[0]:
            datastring = pfa.savedata(config,params,rhopfl,xf,yf,zf,mask,opt=True)
            pfa.analyze(datastring)

        for j in range(nvar):
            sparamvals[:,j] = sparamvals[:,j][dm]

        if config.smplopt:
            np.savez(lossfile,loss=loss,sparamvals=sparamvals,sparamnames=sparamnames)

        i+=1
        reportsparams(config,params,sparamnames,closs,i,nval)
        print(f"minloss: {loss[0]:<6.2f} dt: {time()-t0:<6.2} ")
    print()

    if config.smplopt:

        closs      = loss[0]
        sparamvals = sparamvals[0]

        nvar = len(sparamnames)
        for i in range(nvar):
            params[sparamnames[i]] =  sparamvals[i]
 
        print(f"running for optimal sampled parameters")
        reportsparams(config,params,sparamnames,closs,1,1)
        print()

        modparams = set(list(sparamnames)).intersection(pparamnames)
        if len(modparams) > 0:
            params = ptf.setupflowprofile(config,params)

        config.verbose = True
        loss,rhopfl,xf,yf,zf,mask = ptf.fullflow(config,params)

        datastring = pfa.savedata(config,params,rhopfl,xf,yf,zf,mask,opt=True)
        pfa.analyze(datastring)

        print()
