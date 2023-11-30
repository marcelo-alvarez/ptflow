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

    parser.add_argument('--N',     default=pfd.N,     help=f'input xyz-dim [{pfd.N}]',   type=int)
    parser.add_argument('--N0',    default=pfd.N0,    help=f'input xyz-dim [{pfd.N0}]',  type=int)
    parser.add_argument('--nx0',   default=pfd.nx0,   help=f'xdim truth [{pfd.nx0}]',    type=int)
    parser.add_argument('--x2yz',  default=pfd.x2yz,  help=f'ydim/xdim [{pfd.x2yz}]',    type=int)
    parser.add_argument('--nm',    default=pfd.nm,    help=f'N scales [{pfd.nm}]',       type=int)
    parser.add_argument('--sqrtN', default=pfd.sqrtN, help=f'N^2 samples [{pfd.sqrtN}]', type=int)
    parser.add_argument('--logM1', default=pfd.logM1, help=f'logM1 [{pfd.logM1}]',       type=float)
    parser.add_argument('--logM2', default=pfd.logM2, help=f'logM2 [{pfd.logM2}]',       type=float)
    parser.add_argument('--zoom',  default=pfd.zoom,  help=f'train zoom [{pfd.zoom}]',   type=float)
    parser.add_argument('--kmax',  default=pfd.kmax,  help=f'train kmax [{pfd.kmax}]',   type=float)
    parser.add_argument('--d0',    default=pfd.d0,    help=f'deltavir [{pfd.d0}]',       type=float)
    parser.add_argument('--gamma', default=pfd.gamma, help=f'M0 / Mh[{pfd.gamma}]',      type=float)
    parser.add_argument('--beta',  default=pfd.beta,  help=f'exterior plaw[{pfd.beta}]', type=float)
    parser.add_argument('--alpha', default=pfd.alpha, help=f'deltac tilt [{pfd.alpha}]', type=float)
    parser.add_argument('--M0',    default=pfd.M0,    help=f'tilt pivot [{pfd.M0}]',     type=float)
    parser.add_argument('--sigma', default=pfd.sigma, help=f'Rs / RLag [{pfd.sigma}]',   type=float)
    parser.add_argument('--fltr',  default=pfd.fltr,  help=f'filter [{pfd.fltr}]',  type=str)
    parser.add_argument('--ctype', default=pfd.ctype, help=f'cfield type [{pfd.ctype}]', type=str)
    parser.add_argument('--mtype', default=pfd.mtype, help=f'mask type [{pfd.mtype}]',   type=str)
    parser.add_argument('--mask',  default=pfd.mask,  help=f'do masking [{pfd.mask}]',   action=parsbool)
    parser.add_argument('--excl',  default=pfd.excl,  help=f'do exclusion [{pfd.excl}]', action=parsbool)
    parser.add_argument('--soft',  default=pfd.soft,  help=f'thresholding [{pfd.soft}]', action=parsbool)
    parser.add_argument('--test',  default=pfd.test,  help=f'test [{pfd.test}]',         action=parsbool)
    parser.add_argument('--sampl', default=pfd.sampl, help=f'sample [{pfd.sampl}]',      action=parsbool)
    parser.add_argument('--tdown', default=pfd.tdown, help=f'topdown [{pfd.tdown}]',     action=parsbool)
    parser.add_argument('--flowl', default=pfd.tdown, help=f'inflow LPT [{pfd.flowl}]',  action=parsbool)

    args = parser.parse_args()

    return args

def configfromargs(args):
    import config
 
    config = config.PTflowConfig(
        report = True,
        N     = args.N,
        N0    = args.N0,
        nx    = args.nx0  * (args.N//args.N0),
        ny    = args.x2yz * args.nx0  * (args.N//args.N0),
        nz    = args.x2yz * args.nx0  * (args.N//args.N0),
        nm    = args.nm,
        logM1 = args.logM1,
        logM2 = args.logM2,
        zoom  = args.zoom,
        kmax  = args.kmax,
        fltr  = args.fltr,
        ctype = args.ctype,
        mtype = args.mtype,
        mask  = args.mask,
        excl  = args.excl,
        soft  = args.soft
    )
    config.topdown = args.tdown
    config.flowlpt = args.flowl

    return config

def paramsfromargs(args):
    params = {
        'd0'       : args.d0,
        'gamma'    : args.gamma,
        'beta'     : args.beta,
        'alpha'    : args.alpha,
        'M0'       : args.M0,
        'lptsigma' : args.sigma
    }
    return params

def setupflowprofile(args,config,params):

    import funcs

    # flow profiles using duffy nfw 
    config.xL       = np.zeros(config.nm,dtype=np.float32)
    config.flowfunc = [None] * config.nm
    flowparams = {}
    na = 10000
    qa = np.logspace(-3,1,na)
    qa = np.insert(qa,0,0.0)
    fa = np.zeros((config.nm,len(qa)))
    xL = np.zeros(config.nm)
    for i in range(config.nm):
        z = 0.0
        M200m = config.masses[i] * config.h # mass scale in Msun/h
        M200c = M200m * np.sqrt(config.omegam)
        flowparams['cnfw']  = funcs.duffycnfw(M200c,z)
        flowparams['beta']  = params['beta']
        flowparams['d0']    = params['d0']
        flowparams['gamma'] = params['gamma']
        config.xL[i], config.flowfunc[i] = funcs.flowgen(flowparams)
        fa[i,:] = config.flowfunc[i](qa)
        xL[i]   = config.xL[i]
    params['na'] = na
    params['qa'] = jnp.asarray(qa)
    params['fa'] = jnp.asarray(fa)
    return config, params

def getloss(config,xlpt,ylpt,zlpt,xfl,yfl,zfl):

    rholpt0 = config.binpoints(xlpt,ylpt,zlpt)
    rhopfl0 = config.binpoints(xfl,yfl,zfl)

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

    rhodmg = config.rhodmg

    rhopfl = rhopfl0[si,sj,sk].mean(axis=0)
    rhodmg = rhodmg[ si,sj,sk].mean(axis=0) * (config.N/2500)**3

    # get (cross) power spectra
    k, cl_dmg = powerspectrum(np.asarray(rhodmg))
    k, cl_hfl = powerspectrum(np.asarray(rhopfl))

    k, cl_dh  = crosspower(np.asarray(rhodmg),np.asarray(rhopfl))

    r_dh = cl_dh / np.sqrt(cl_dmg*cl_hfl)

    # convert k from pixel units [0:npixel] to wavenumbers h/Mpc
    k = k * 2*np.pi / dy

    ploss = abs(np.log(cl_dmg)-np.log(cl_hfl)) / cl_dmg * (1.-r_dh**2) / k 
    ploss *= heavileft(k,cen=kmax,soft=config.soft)

    loss = ploss.cumsum()[-1] / 1e3

    # ny = j1 - j0

    # rhodmg = config.rhodmg

    # rhopfl = rhopfl0[si,sj,sk].mean(axis=0)
    # rhodmg = rhodmg[ si,sj,sk].mean(axis=0) * (config.N/2500)**3

    # # convert k from pixel units [0:npixel] to wavenumbers h/Mpc
    # dk  = 2 * np.pi / dy
    # kx, ky = jnp.meshgrid(jnp.arange(ny)-ny//2,jnp.arange(ny)-ny//2)
    # k2 = jnp.sqrt(kx**2+ky**2) * dk

    # # use 2d (cross) power spectra
    # cl2_dmg = crosspower2(rhodmg,rhodmg)
    # cl2_pfl = crosspower2(rhopfl,rhopfl)
    # cl2_dh  = crosspower2(rhodmg,rhopfl)
    # r2_dh   = cl2_dh / jnp.sqrt(cl2_dmg*cl2_pfl)

    # # apply k cutoff
    # r2_dh *= heavileft(k2,cen=kmax,soft=config.soft)

    # k1 = jnp.linspace(0.5,ny//2-0.5) * dk
    # loss = 0.0
    # for k in k1:

    #     filter = tophat(k2,cen=k,scale=1e-3*dk,soft=True)

    #     r_dh   = ( r2_dh  * filter).mean()
    #     cl_dmg = (cl2_dmg * filter).mean()
    #     cl_pfl = (cl2_pfl * filter).mean()

    #     dr  = 1.-r_dh**2
    #     dloss  = abs(jnp.log(cl_dmg)-jnp.log(cl_pfl)) / cl_dmg * dr

    #     loss  += dloss / k
    
    return loss, rholpt0, rhopfl0
#getloss = jax.jit(getloss,static_argnums=0)

def collkerneles(x):
    return x**2*heavileft(x,1,soft=True)

def collkernelms(x):
    return heavileft(x,1,soft=True)

def collkernele(x):
    return x**2*heavileft(x,1,soft=True)

def collkernelm(x):
    return heavileft(x,1,soft=True)

def meff(m,mfac_m=1.0,alpha_m=0):
    return mfac_m * m * (m/1e13)**-alpha_m

def cexclusion(config,cfield,field,R,deltac):
    epsilon = 1e-3

    iter = 0
    while iter < 5:
        iter += 1
        nlfield = (field-deltac)**25
        counts    = config.convolve(cfield,R,norm=False)
        nlfield=jnp.array(cfield*nlfield).astype(jnp.float32)
        con = config.convolve(nlfield,R,norm=False) / (counts + epsilon)
        cfield *= jnp.heaviside(nlfield-con,1.0)

    return cfield

def getcfield(params, config, i, M, cfieldall, mask):

    masking   = config.masking
    exclusion = config.exclusion
    filter    = config.filter

    R=(3*M/4./np.pi/config.rho)**(1./3.)*config.h
    Rpix=R/config.dsub

    if filter == "matter":
        collkernel = collkernelm
        if config.soft: collkernel = collkernelms
    else:
        collkernel = collkernele
        if config.soft: collkernel = collkerneles
    collfield = config.convolve(config.deltai, Rpix, wfunc=collkernel)
    collsigma  = jnp.sqrt(collfield.var())

     # halo-collapse-inspired thresholds using fits from Musso & Sheth 2021
    if filter == "matter":
        deltac = 1.56 + 0.63 * collsigma # "matter" =~ r^2dr
    elif filter == "energy":
        deltac = 1.78 + 0.81 * collsigma # "energy" =~ r^4dr
    deltac *= (M / params['M0'])**params['alpha']

    cfield  = jnp.array(heaviright(collfield,cen=deltac,scale=1e-5*deltac),dtype=config.cftype)

    if masking:
        cfield *= mask

    if exclusion:
        cfield = cexclusion(config,cfield,collfield,Rpix,deltac)

    cmask = config.convolve(cfield, Rpix, norm=False)
    mask *= jnp.array(heavileft(cmask,cen=0.1,soft=config.soft),dtype=config.masktype)

    cfieldall += jnp.array((tophat(cfield,cen=1,soft=config.soft)*(i+1)),dtype=config.cftype)

    return cfieldall, cfield, mask
getcfield = jax.jit(getcfield,static_argnums=1)

def particleflow(config,params,i,cfield,RLagpix,Rsmooth,r0pix,xl,yl,zl,xf,yf,zf):

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
        x0 = xl
        y0 = yl
        z0 = zl
    else:
        # flow from positions at previous step
        x0 = xf
        y0 = yf
        z0 = zf

    dx = flow * (config.convolve(xs*cfield, RLagpix, norm=False) / count - x0)
    xc = x0 + dx                         ; del dx ; gc.collect()
    xf = xc * flowing + xf * (1-flowing) ; del xc ; gc.collect()

    dy = flow * (config.convolve(ys*cfield, RLagpix, norm=False) / count - y0)
    yh = y0 + dy                         ; del dy ; gc.collect()
    yf = yh * flowing + yf * (1-flowing) ; del yh ; gc.collect()

    dz = flow * (config.convolve(zs*cfield, RLagpix, norm=False) / count - z0)
    zh = z0 + dz                         ; del dz ; gc.collect()
    zf = zh * flowing + zf * (1-flowing) ; del zh ; gc.collect()
 
    return xf,yf,zf
particleflow = jax.jit(particleflow,static_argnums=[0,])

def scaleflow(params,config,i,M,cfield,xl,yl,zl,xf,yf,zf):

    d0        = params['d0']
    lptsigma  = params['lptsigma']

    r0      = (3*M/4./jnp.pi/config.rho/d0)**(1./3.)*config.h
    Rsmooth = (3*M/4./jnp.pi/config.rho         )**(1./3.)*config.h * lptsigma
    RLag    = r0 * config.xL[i]

    RLagpix = RLag/config.dsub
    r0pix   = r0/config.dsub

    # remove cfield within specified distance of boundary to avoid wrapping artefacts
    ntophat = int(RLagpix)+1
    nbx = min(ntophat,config.sboxdims[0]//2-1)
    nby = min(ntophat,config.sboxdims[1]//2-1)
    nbz = min(ntophat,config.sboxdims[2]//2-1)
    cfield = cfield.at[:nbx ,:,:].set(0)
    cfield = cfield.at[-nbx:,:,:].set(0)
    cfield = cfield.at[:,:nby ,:].set(0)
    cfield = cfield.at[:,-nby:,:].set(0)
    cfield = cfield.at[:,:,: nbz].set(0)
    cfield = cfield.at[:,:,-nbz:].set(0)

    xf,yf,zf = particleflow(config,params,i,cfield,RLagpix,Rsmooth,r0pix,xl,yl,zl,xf,yf,zf)

    return xf,yf,zf,Rsmooth,RLag

def flow(params,config):

    # mass scales
    nm     = config.nm
    masses = config.masses

    mask      = jnp.array(jnp.ones( config.sboxdims),dtype=config.masktype)    
    cfieldall = jnp.array(jnp.zeros(config.sboxdims),dtype=config.cftype)

    if config.topdown:
        for i in range(nm):
            t0 = time()
            RLag = (3*masses[i]/4./jnp.pi/config.rho)**(1./3.) * config.h # cMpc/h
            cfieldall, cfield, mask = getcfield(params,config,i,masses[i],cfieldall,mask)
            nc = int(cfield.sum())
            if config.verbose:
                print(f"threshold: {i+1:>4}/{nm:<4} nc={nc:<8} logM={np.log10(masses[i]):<5.2f} dt={time()-t0:<6.3f} RLag={RLag:<6.3f}",end='\r')
        if config.verbose:
            print()

    sx,sy,sz = config.getslpt()
    xl,yl,zl = config.advect([sx,sy,sz])
    del sx ; gc.collect()
    del sy ; gc.collect()
    del sz ; gc.collect()
    xf=xl+0.;yf=yl+0.;zf=zl+0.

    for i in reversed(range(nm)):
        t0 = time()
        if config.topdown:
            cfield = tophat(cfieldall,cen=i+1,width=(i+1)/2,soft=config.soft)
        else:
            cfieldall, cfield, mask = getcfield(params,config,i,masses[i],cfieldall,mask)
        nc = int(cfield.sum())
        if nc>0:
            xf, yf, zf, Rsmooth, RLag = scaleflow(params,config,i,masses[i],cfield,xl,yl,zl,xf,yf,zf)
            if config.verbose:
                print(f" dynamics: {i+1:>4}/{nm:<4} nh={nc:<8} logM={np.log10(masses[i]):<5.2f} dt={time()-t0:<6.3f} RLag={RLag:<6.3f}",end='\r')
    if config.verbose:
        print()
    return xl,yl,zl,xf,yf,zf,mask

def flowloss(params,config):

    xlpt,ylpt,zlpt,xfl,yfl,zfl,mask = flow(params,config)

    loss, rholpt, rhopfl = getloss(config,xlpt,ylpt,zlpt,xfl,yfl,zfl)
    return loss, [loss,rholpt,rhopfl,mask]
