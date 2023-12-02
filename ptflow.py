import numpy as np
import scipy
import jax
import jax.numpy as jnp
from time import time
from jax import vmap
import gc

from mutil import crosspower, powerspectrum, tophat, heavileft, heaviright

def parsecommandline():
    import argparse
    import defaults as pfd

    parsbool = argparse.BooleanOptionalAction
    parser   = argparse.ArgumentParser(description='Commandline interface to ptflow')

    for params in pfd.allparams:
        for param in pfd.allparams[params]:
            pdval = pfd.allparams[params][param]['val']
            ptype = pfd.allparams[params][param]['type']
            pdesc = pfd.allparams[params][param]['desc']
            if ptype == 'bool':
                parser.add_argument('--'+param, default=pdval, help=f'{pdesc} [{pdval}]', action=parsbool)
            else:
                parser.add_argument('--'+param, default=pdval, help=f'{pdesc} [{pdval}]', type=ptype)

    return vars(parser.parse_args())

def parseallparams(allparams):

    import defaults as pfd

    cparams = {}
    params  = {}
    for param in allparams:
        if param in pfd.allparams['cparams']:
            cparams[param] = allparams[param]
        else:
            params[param] = allparams[param]

    return cparams, params

def setupflowprofile(config,params):

    import funcs

    # flow profiles using duffy nfw 
    params['xL'] = np.zeros(config.nm,dtype=jnp.float32)
    flowparams = {}
    na = 10000
    qa = np.logspace(-3,1,na)
    qa = np.insert(qa,0,0.0)
    fa = np.zeros((config.nm,len(qa)))
    for i in range(config.nm):
        z = 0.0
        M200m = config.fmass[i] * config.h # mass scale in Msun/h
        M200c = M200m * np.sqrt(config.omegam)
        flowparams['cnfw']  = funcs.duffycnfw(M200c,z)
        flowparams['beta']  = params['beta']
        flowparams['d0']    = params['d0']
        flowparams['gamma'] = params['gamma']
        flowparams['inner'] = params['inner']
        flowparams['outer'] = params['outer']
        params['xL'][i], flowfunc = funcs.flowgen(flowparams)
        fa[i,:] = flowfunc(qa)
    params['na'] = na
    params['qa'] = jnp.asarray(qa)
    params['fa'] = jnp.asarray(fa)
    params['xL'] = jnp.asarray(params['xL'])
    return params

def getloss(config,xfl,yfl,zfl):

    # set up xcorr grids
    zoom = config.zoom
    kmax = config.kmax

    dx = (config.sbox[0][1]-config.sbox[0][0])*zoom
    dy = (config.sbox[1][1]-config.sbox[1][0])*zoom
    dz = (config.sbox[2][1]-config.sbox[2][0])*zoom

    xc = 0.5 * (config.sbox[0][0]+config.sbox[0][1])
    yc = 0.5 * (config.sbox[1][0]+config.sbox[1][1])
    zc = 0.5 * (config.sbox[2][0]+config.sbox[2][1])

    x0 = xc - dx / 2; x1 = xc + dy / 2
    y0 = yc - dy / 2; y1 = yc + dy / 2
    z0 = zc - dz / 2; z1 = zc + dz / 2

    extent=[z0,z1,y0,y1]

    i0 = int(x0/config.dsub) - config.sboxrange[0][0] ; i1 = int(x1/config.dsub) - config.sboxrange[0][0]
    j0 = int(y0/config.dsub) - config.sboxrange[1][0] ; j1 = int(y1/config.dsub) - config.sboxrange[1][0]
    k0 = int(z0/config.dsub) - config.sboxrange[2][0] ; k1 = int(z1/config.dsub) - config.sboxrange[2][0]

    si = slice(i0,i1)
    sj = slice(j0,j1)
    sk = slice(k0,k1)

    rhopfl = config.binpoints(xfl,yfl,zfl)
    rhodmg = config.rhodmg

    rhopfl2 = rhopfl[si,sj,sk].mean(axis=0)
    rhodmg2 = rhodmg[si,sj,sk].mean(axis=0) * (config.N/2500)**3

    # get (cross) power spectra
    k, cl_dmg = powerspectrum(np.asarray(rhodmg2))
    k, cl_hfl = powerspectrum(np.asarray(rhopfl2))

    k, cl_dh  = crosspower(np.asarray(rhodmg2),np.asarray(rhopfl2))

    r_dh = cl_dh / np.sqrt(cl_dmg*cl_hfl)

    # convert k from pixel units [0:npixel] to wavenumbers h/Mpc
    k = k * 2*np.pi / dy

    loss = (1.-r_dh)**2 / k**1.5*20
    if config.ploss: 
        loss *= abs(np.log(cl_dmg)-np.log(cl_hfl)) / cl_dmg / 1e3
    else:
        loss *= 1e2
    loss *= heavileft(k,cen=kmax,soft=config.soft)

    loss = loss.cumsum()[-1]
    return loss, rhopfl

def collkernele(x,soft=True):
    return x**2*heavileft(x,1,soft=soft)

def collkernelm(x,soft=True):
    return heavileft(x,1,soft=soft)

def cexclusion(config,cfield,deltasmooth,R,deltac):
    epsilon = 1e-3

    iter = 0
    while iter < 5:
        iter += 1
        nlfield = (deltasmooth-deltac)**25
        counts    = config.convolve(cfield,R,norm=False)
        nlfield=jnp.array(cfield*nlfield).astype(jnp.float32)
        con = config.convolve(nlfield,R,norm=False) / (counts + epsilon)
        cfield *= jnp.heaviside(nlfield-con,1.0)

    return cfield

def getcfield(config, params, i, cfields, mask):

    Rpix = config.cRLag[i] / config.dsub

    if config.filter == "matter":
        collkernel = collkernelm # "matter" =~ int delta(r)r^2dr
    else:
        collkernel = collkernele # "energy" =~ int delta(r)r^4dr
    deltasmooth = config.convolve(config.deltai, Rpix, wfunc=collkernel)
    deltasigma  = jnp.sqrt(deltasmooth.var())

    # deltac_scale = 1 + (jnp.log10(config.cmass[i]) - params['logM0']) * params['alpha'] 
    # deltasigma *= deltac_scale
    delc0 = params['delc0']
    delca = params['delca']
    if config.filter == "matter":
        # halo-collapse-inspired thresholds using fits from Musso & Sheth 2021
        #deltac =  1.56 +  0.63 * deltasigma # "matter" Musso & Sheth 2021
        deltac = delc0 + delca * deltasigma 
    elif config.filter == "energy":
        deltac = 1.78 + 0.81 * deltasigma # "energy" Musso & Sheth 2021

    cfield  = jnp.array(heaviright(deltasmooth,cen=deltac,scale=1e-5*deltac),dtype=config.cftype)

    if config.masking:
        cfield *= mask

    if config.exclusion:
        cfield = cexclusion(config,cfield,deltasmooth,Rpix,deltac)

    cmask = config.convolve(cfield, Rpix, norm=False)
    mask *= jnp.array(heavileft(cmask,cen=0.1,soft=config.soft),dtype=config.masktype)

    cfields += jnp.array((tophat(cfield,cen=1,soft=config.soft)*(i+1)),dtype=config.cftype)

    return cfields, mask, deltasigma.item()
#getcfield = jax.jit(getcfield,static_argnums=[0,])

def particleflow(config,params,i,cfield,sigmas,xf,yf,zf):

    RLagpix = config.fRLag[i] / config.dsub
    r0pix   = config.fRLag[i] / config.dsub / params['xL'][i] # config.xL[i]
    smooth1 = params['smooth1']
    smooth2 = params['smooth2']
    fsigma = (sigmas[i]-sigmas[0]) / (sigmas[-1]-sigmas[0])
    smooth = smooth1 * (1-fsigma) + smooth2 * fsigma
    Rsmooth = config.fRLag[i] * smooth

    # count number of convergence points within RLagpix pixels of each point and set flowing --> 1 when count > 1/2
    count  = config.convolve(cfield, RLagpix, norm=False) 
    count *= jnp.heaviside(count,0.0) 
    count += 1e-10 # TBD jax-friendly don't divide by zero
    flowing = heaviright(count, 0.5, soft=config.soft)

    # displacements at smoothing scale
    sxc,syc,szc = config.getslpt(Rsmooth=Rsmooth)

    # convergence displacement field is mean of all overlapping convergence centers, otherwise zero
    sxc = flowing * config.convolve(sxc*cfield, RLagpix, norm=False) / count 
    syc = flowing * config.convolve(syc*cfield, RLagpix, norm=False) / count 
    szc = flowing * config.convolve(szc*cfield, RLagpix, norm=False) / count 

    # final positions of convergence centers
    xs,ys,zs = config.advect([sxc,syc,szc])
    del sxc ; gc.collect()
    del syc ; gc.collect()
    del szc ; gc.collect()

    # flow field infalling towards cfield with profile in flowfunc
    qa = params['qa']
    fa = params['fa'][i]
    flowf = lambda q: jnp.interp(q,qa,fa)
    flow = config.convolve(cfield, r0pix, flowf, norm = False) * flowing
    flow /= count

    # particle field flows towards cfield from [x0,y0,z0]
    if config.flowlpt:
        # flow from LPT positions
        x0 = config.xl
        y0 = config.yl
        z0 = config.zl
    else:
        # flow from positions at previous step
        x0 = xf
        y0 = yf
        z0 = zf

    # nonlinear displacement correction [dx,dy,dz] from convergence flow where flowing > 0
    dx = flow * (config.convolve(xs*cfield, RLagpix, norm=False) / count - x0)
    xc = x0 + dx                         ; del dx ; gc.collect()
    xf = xc * flowing + xf * (1-flowing) ; del xc ; gc.collect()

    dy = flow * (config.convolve(ys*cfield, RLagpix, norm=False) / count - y0)
    yc = y0 + dy                         ; del dy ; gc.collect()
    yf = yc * flowing + yf * (1-flowing) ; del yc ; gc.collect()

    dz = flow * (config.convolve(zs*cfield, RLagpix, norm=False) / count - z0)
    zc = z0 + dz
    zf = zc * flowing + zf * (1-flowing) ; del zc ; gc.collect()
 
    return xf,yf,zf
particleflow = jax.jit(particleflow,static_argnums=[0,])

def scaleflow(config,params,i,cfield,sigmas,xf,yf,zf):

    # remove cfield within specified distance of boundary to avoid wrapping artefacts
    # TBD do this in jax-friendly way inside of jitted particleflow function
    nbx0 = config.nbx[i,0]
    nby0 = config.nby[i,0]
    nbz0 = config.nbz[i,0]
    nbx1 = config.nbx[i,1]
    nby1 = config.nby[i,1]
    nbz1 = config.nbz[i,1]
    x = config.xyz[0] ; y = config.xyz[1] ; z = config.xyz[2]
    cfield *= jnp.heaviside(x-nbx0,1)*jnp.heaviside(y-nby0,1)*jnp.heaviside(z-nbz0,1)
    cfield *= jnp.heaviside(nbx1-x,1)*jnp.heaviside(nby1-y,1)*jnp.heaviside(nbz1-z,1)

    # cfield = cfield.at[:nbx0 ,:,:].set(0)
    # cfield = cfield.at[-nbx0:,:,:].set(0)
    # cfield = cfield.at[:,:nby0 ,:].set(0)
    # cfield = cfield.at[:,-nby0:,:].set(0)
    # cfield = cfield.at[:,:,: nbz0].set(0)
    # cfield = cfield.at[:,:,-nbz0:].set(0)

    xf,yf,zf = particleflow(config,params,i,cfield,sigmas,xf,yf,zf)

    return xf,yf,zf
scaleflow = jax.jit(scaleflow,static_argnums=[0,])

def cfieldstep(config,params,i,cfields,mask):
    t0 = time()
    cfields, mask, sigma = getcfield(config,params,i,cfields,mask)
    cfield = tophat(cfields,cen=i+1,width=(i+1)/2,soft=config.soft)
    nc = int(cfield.sum())
    if config.verbose:
        print(f"threshold: {i+1:>4}/{config.nm:<4} nc={nc:<8} logM={np.log10(config.cmass[i]):<5.2f} "+
              f"dt={time()-t0:<6.3f} RLag={config.cRLag[i]:<6.3f}",end='\r')
    return cfields, mask, sigma

def flowstep(config,params,i,cfields,sigmas,xf,yf,zf):
    t0 = time()
    ci = i+1 if config.ctdwn == config.ftdwn else config.nm-i
    cfield = tophat(cfields,cen=ci,width=ci/2,soft=config.soft)
    nc = int(cfield.sum())
    if int(cfield.sum())==0: return xf,yf,zf

    xf,yf,zf = scaleflow(config,params,i,cfield,sigmas,xf,yf,zf)
    if config.verbose:
        print(f" dynamics: {i+1:>4}/{config.nm:<4} nh={nc:<8} logM={np.log10(config.fmass[i]):<5.2f} "+
              f"dt={time()-t0:<6.3f} RLag={config.fRLag[i]:<6.3f}",end='\r')
    return xf,yf,zf

def cfieldall(config,params,cfields,mask,sigmas):

    # iterate over cfield scales
    for i in range(config.nm):
        cfields, mask, sigma = cfieldstep(config,params,i,cfields,mask)
        sigmas.append(sigma)
    if config.verbose: print()

    return cfields, mask, sigmas

def flowall(config,params,cfields,sigmas,xf,yf,zf):

    # iterate over flow scales
    for i in range(config.nm):
        xf,yf,zf = flowstep(config,params,i,cfields,sigmas,xf,yf,zf)
    if config.verbose: print()

    return xf,yf,zf

def cfieldflowall(config,params,cfields,mask,xf,yf,zf):
    # iterate over cfield scales
    for i in range(config.nm):
        cfields,mask = cfieldstep(config,params,i,cfields,mask)
        xf,yf,zf = flowstep(config,params,i,cfields,xf,yf,zf)
    if config.verbose: print()

    return xf,yf,zf,mask

def fullflow(config,params):

    # cfield = 1 --> convergence point
    #   mask = 0 --> already a convergence point from previous iterations
    mask    = jnp.array(jnp.ones( config.sboxdims),dtype=config.masktype)    
    cfields = jnp.array(jnp.zeros(config.sboxdims),dtype=config.cftype)
    sigmas  = []
    # [xf,yf,zf] = nonlinear positions initially set to unsmoothed LPT positions
    xf = config.xl.copy()
    yf = config.yl.copy()
    zf = config.zl.copy()

    # config.ctdwn --> cfield field starting from smoothed on largest scales first [default]
    # config.ftdwn -->   flow field starting from smoothed on largest scales first [default]

    # if ordering between cfield and flow is different, run cfield and flow separately
    if config.ctdwn != config.ftdwn:
        cfields,mask,sigmas = cfieldall(config,params,cfields,mask,sigmas)
        sigmas = jnp.asarray(sigmas)
        xf,yf,zf = flowall(config,params,cfields,sigmas,xf,yf,zf)
    else:
        xf,yf,zf,mask = cfieldflowall(config,params,cfields,mask,xf,yf,zf)

    return xf,yf,zf,mask
#fullflow = jax.jit(fullflow,static_argnums=[0,])

def flowloss(config,params):

    xf,yf,zf,mask = fullflow(config,params)

    loss, rhopfl = getloss(config,xf,yf,zf)

    return loss, [rhopfl,mask]

def initialize():
    import config as ptc

    allparams = parsecommandline()

    cparams, params = parseallparams(allparams)

    config = ptc.PTflowConfig(**cparams)

    params = setupflowprofile(config,params)

    return config, params