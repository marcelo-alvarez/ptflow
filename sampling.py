import numpy as np
import jax.numpy as jnp
import ptflow   as ptf
import analysis as pfa
import pathlib

def optfromsample(args,params,config):

    datadir = "./data/"
    pathlib.Path(datadir).mkdir(parents=True, exist_ok=True)     
    lossfile  = datadir + 'loss.npz'

    print()
    if args.sampl:

        from scipy.stats import qmc

        sampler = qmc.LatinHypercube(d=3, strength=2)
        vparams = sampler.random(n=args.sqrtN**2)

        #             alpha  log[M0] log[lptsigma]
        l_bounds = [  -0.5,     12,        -0.2]
        u_bounds = [   0.5,     14,         0.2]
        vparams = qmc.scale(vparams, l_bounds, u_bounds)

        # insert central point by hand
        vparams = np.asarray(vparams)
        vparams = np.insert(vparams,[0],[[0.0,13.0,0.0]],axis=0)

        sampledparamsl = []
        nval=np.shape(vparams)[0]
        print("sampling parameter space")

        for i in range(nval):
            alpha    =      vparams[i,0]
            M0       = 10.**vparams[i,1]
            lptsigma = 10.**vparams[i,2]

            params['alpha']    = alpha
            params['M0']       = M0
            params['lptsigma'] = lptsigma

            config.verbose = False
            arg, aux = ptf.flowloss(params,config)
            closs,rholpt,rhopfl,mask = aux
            sampledparamsl.append([alpha,M0,lptsigma,closs])

            sampledparams = np.asarray(sampledparamsl)
            loss        = sampledparams[:,3]
            dm          = loss.argsort()
            loss        = loss[dm]

            alpha    = sampledparams[:,0][dm]
            M0       = sampledparams[:,1][dm]
            lptsigma = sampledparams[:,2][dm]
            loss     = sampledparams[:,3][dm]

            np.savez(lossfile,
                    alpha    = alpha,
                    M0       = M0,
                    lptsigma = lptsigma,
                    loss     = loss
                    )
            i+=1
            print(f" {i:>3}/{nval:<3} curloss: {closs:>11.3f} alpha: {params['alpha']:>5.2f} logM0: {np.log10(params['M0']):>5.2f} lptsigma: {params['lptsigma']:>4.2f} minloss: {loss[0]:>11.3f}")
        print()

    data = np.load(lossfile)

    alpha    = data['alpha']
    M0       = data['M0']
    lptsigma = data['lptsigma']

    params['alpha']    = alpha[0]
    params['M0']       = M0[0]
    params['lptsigma'] = lptsigma[0]

    print(f"running for optimal sampled parameters alpha: {alpha[0]:>5.2f} logM0: {np.log10(M0[0]):>5.2f} lptsigma: {lptsigma[0]:>4.2f}")
    aux, [loss, rholpt, rhopfl, mask] = ptf.flowloss(params,config)

    pfa.analyze(params,config,rholpt,rhopfl,mask,opt=True)
    print()
