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

    # modeling configuration
    parser.add_argument('--N',     default=pfd.N,     help=f'     xyz-dim [{pfd.N}]    ', type=int)
    parser.add_argument('--N0',    default=pfd.N0,    help=f' fid xyz-dim [{pfd.N0}]   ', type=int)
    parser.add_argument('--nx0',   default=pfd.nx0,   help=f'  xdim truth [{pfd.nx0}]  ', type=int)
    parser.add_argument('--x2yz',  default=pfd.x2yz,  help=f' ydim / xdim [{pfd.x2yz}] ', type=int)
    parser.add_argument('--nm',    default=pfd.nm,    help=f' # of scales [{pfd.nm}]   ', type=int)
    parser.add_argument('--sqrtN', default=pfd.sqrtN, help=f' N^2 samples [{pfd.sqrtN}]', type=int)
    parser.add_argument('--logM1', default=pfd.logM1, help=f'       logM1 [{pfd.logM1}]', type=float)
    parser.add_argument('--logM2', default=pfd.logM2, help=f'       logM2 [{pfd.logM2}]', type=float)
    parser.add_argument('--zoom',  default=pfd.zoom,  help=f'  train zoom [{pfd.zoom}] ', type=float)
    parser.add_argument('--kmax0', default=pfd.kmax0, help=f'  train kmax [{pfd.kmax0}]', type=float)
    parser.add_argument('--fltr',  default=pfd.fltr,  help=f'      filter [{pfd.fltr}] ', type=str)
    parser.add_argument('--ctype', default=pfd.ctype, help=f' cfield type [{pfd.ctype}]', type=str)
    parser.add_argument('--mtype', default=pfd.mtype, help=f'   mask type [{pfd.mtype}]', type=str)
    parser.add_argument('--mask',  default=pfd.mask,  help=f'  do masking [{pfd.mask}] ', action=parsbool)
    parser.add_argument('--excl',  default=pfd.excl,  help=f'do exclusion [{pfd.excl}] ', action=parsbool)
    parser.add_argument('--soft',  default=pfd.soft,  help=f'thresholding [{pfd.soft}] ', action=parsbool)
    parser.add_argument('--test',  default=pfd.test,  help=f'        test [{pfd.test}] ', action=parsbool)
    parser.add_argument('--sampl', default=pfd.sampl, help=f'      sample [{pfd.sampl}]', action=parsbool)
    parser.add_argument('--ctdwn', default=pfd.ctdwn, help=f'cfield order [{pfd.ctdwn}]', action=parsbool)
    parser.add_argument('--ftdwn', default=pfd.ftdwn, help=f'  flow order [{pfd.ftdwn}]', action=parsbool)
    parser.add_argument('--flowl', default=pfd.flowl, help=f'  inflow LPT [{pfd.flowl}]', action=parsbool)

    # fiducial parameters
    parser.add_argument('--d0',    default=pfd.d0,    help=f'    deltavir [{pfd.d0}]   ', type=float)
    parser.add_argument('--gamma', default=pfd.gamma, help=f'     M0 / Mh [{pfd.gamma}]', type=float)
    parser.add_argument('--beta',  default=pfd.beta,  help=f'    ext plaw [{pfd.beta}] ', type=float)
    parser.add_argument('--alpha', default=pfd.alpha, help=f' deltac tilt [{pfd.alpha}]', type=float)
    parser.add_argument('--logM0', default=pfd.logM0, help=f'  tilt pivot [{pfd.logM0}] ', type=float)
    parser.add_argument('--sigma', default=pfd.sigma, help=f'   Rs / RLag [{pfd.sigma}]', type=float)
    
    return parser.parse_args()

def configfromargs(args):
    import config
 
    config = config.PTflowConfig(
        report = True,
        N     = args.N,
        N0    = args.N0,
        nx    = args.nx0  * (args.N // args.N0),
        ny    = args.x2yz * args.nx0  * (args.N // args.N0),
        nz    = args.x2yz * args.nx0  * (args.N // args.N0),
        nm    = args.nm,
        logM1 = args.logM1,
        logM2 = args.logM2,
        zoom  = args.zoom,
        kmax  = args.kmax0 * args.N / args.N0,
        fltr  = args.fltr,
        ctype = args.ctype,
        mtype = args.mtype,
        mask  = args.mask,
        excl  = args.excl,
        soft  = args.soft,
        ctdwn = args.ctdwn,
        ftdwn = args.ftdwn,
        flowl = args.flowl,
        sampl = args.sampl,
        sqrtN = args.sqrtN
    )

    return config

def paramsfromargs(args):
    params = {
        'd0'       : args.d0,
        'gamma'    : args.gamma,
        'beta'     : args.beta,
        'alpha'    : args.alpha,
        'logM0'    : args.logM0,
        'lptsigma' : args.sigma
    }
    return params

def setupflowprofile(config,params):

    import funcs

    # flow profiles using duffy nfw 
    config.xL       = np.zeros(config.nm,dtype=jnp.float32)
    config.flowfunc = [None] * config.nm
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
        config.xL[i], config.flowfunc[i] = funcs.flowgen(flowparams)
        fa[i,:] = config.flowfunc[i](qa)
    params['na'] = na
    params['qa'] = jnp.asarray(qa)
    params['fa'] = jnp.asarray(fa)
    config.xL    = jnp.asarray(config.xL)
    return config, params

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

    ploss = abs(np.log(cl_dmg)-np.log(cl_hfl)) / cl_dmg * (1.-r_dh**2) / k 
    ploss *= heavileft(k,cen=kmax,soft=config.soft)

    loss = ploss.cumsum()[-1] / 1e3

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
        collkernel = collkernelm
    else:
        collkernel = collkernele
    deltasmooth = config.convolve(config.deltai, Rpix, wfunc=collkernel)
    deltasigma  = jnp.sqrt(deltasmooth.var())

     # halo-collapse-inspired thresholds using fits from Musso & Sheth 2021
    if config.filter == "matter":
        deltac = 1.56 + 0.63 * deltasigma # "matter" =~ r^2dr
    elif config.filter == "energy":
        deltac = 1.78 + 0.81 * deltasigma # "energy" =~ r^4dr
    deltac *= (config.cmass[i] / 10.**params['logM0'])**params['alpha']

    cfield  = jnp.array(heaviright(deltasmooth,cen=deltac,scale=1e-5*deltac),dtype=config.cftype)

    if config.masking:
        cfield *= mask

    if config.exclusion:
        cfield = cexclusion(config,cfield,deltasmooth,Rpix,deltac)

    cmask = config.convolve(cfield, Rpix, norm=False)
    mask *= jnp.array(heavileft(cmask,cen=0.1,soft=config.soft),dtype=config.masktype)

    cfields += jnp.array((tophat(cfield,cen=1,soft=config.soft)*(i+1)),dtype=config.cftype)

    return cfields, mask
getcfield = jax.jit(getcfield,static_argnums=[0,])

def particleflow(config,params,i,cfield,xf,yf,zf):

    RLagpix = config.fRLag[i] / config.dsub
    r0pix   = config.fRLag[i] / config.dsub / config.xL[i]
    Rsmooth = config.fRLag[i] * params['lptsigma']

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
    yh = y0 + dy                         ; del dy ; gc.collect()
    yf = yh * flowing + yf * (1-flowing) ; del yh ; gc.collect()

    dz = flow * (config.convolve(zs*cfield, RLagpix, norm=False) / count - z0)
    zh = z0 + dz                         ; del dz ; gc.collect()
    zf = zh * flowing + zf * (1-flowing) ; del zh ; gc.collect()
 
    return xf,yf,zf
particleflow = jax.jit(particleflow,static_argnums=[0,])

def scaleflow(config,params,i,cfield,xf,yf,zf):

    # remove cfield within specified distance of boundary to avoid wrapping artefacts
    # TBD do this in jax-friendly way inside of jitted particleflow function
    ntophat = int(params['lptsigma']*config.fRLag[i]/config.dsub)+1
    nbx = min(ntophat,config.sboxdims[0]//2-1)
    nby = min(ntophat,config.sboxdims[1]//2-1)
    nbz = min(ntophat,config.sboxdims[2]//2-1)
    cfield = cfield.at[:nbx ,:,:].set(0)
    cfield = cfield.at[-nbx:,:,:].set(0)
    cfield = cfield.at[:,:nby ,:].set(0)
    cfield = cfield.at[:,-nby:,:].set(0)
    cfield = cfield.at[:,:,: nbz].set(0)
    cfield = cfield.at[:,:,-nbz:].set(0)

    xf,yf,zf = particleflow(config,params,i,cfield,xf,yf,zf)

    return xf,yf,zf

def cfieldstep(config,params,i,cfields,mask):
    t0 = time()
    cfields, mask = getcfield(config,params,i,cfields,mask)
    cfield = tophat(cfields,cen=i+1,width=(i+1)/2,soft=config.soft)
    nc = int(cfield.sum())
    if config.verbose:
        print(f"threshold: {i+1:>4}/{config.nm:<4} nc={nc:<8} logM={np.log10(config.cmass[i]):<5.2f} "+
              f"dt={time()-t0:<6.3f} RLag={config.cRLag[i]:<6.3f}",end='\r')
    return cfields, mask

def flowstep(config,params,i,cfields,xf,yf,zf):
    t0 = time()
    ci = i+1 if config.ctdwn == config.ftdwn else config.nm-i
    cfield = tophat(cfields,cen=ci,width=ci/2,soft=config.soft)
    nc = int(cfield.sum())
    if int(cfield.sum())==0: return xf,yf,zf

    xf,yf,zf = scaleflow(config,params,i,cfield,xf,yf,zf)
    if config.verbose:
        print(f" dynamics: {i+1:>4}/{config.nm:<4} nh={nc:<8} logM={np.log10(config.fmass[i]):<5.2f} "+
              f"dt={time()-t0:<6.3f} RLag={config.fRLag[i]:<6.3f}",end='\r')
    return xf, yf, zf

def cfieldall(config,params,cfields,mask):

    # iterate over cfield scales
    for i in range(config.nm):
        cfields, mask = cfieldstep(config,params,i,cfields,mask)
    if config.verbose: print()

    return cfields, mask

def flowall(config,params,cfields,xf,yf,zf):

    # iterate over flow scales
    for i in range(config.nm):
        xf,yf,zf = flowstep(config,params,i,cfields,xf,yf,zf)
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

    # [xf,yf,zf] = nonlinear positions initially set to unsmoothed LPT positions
    xf = config.xl.copy()
    yf = config.yl.copy()
    zf = config.zl.copy()

    # config.ctdwn --> cfield field starting from smoothed on largest scales first [default]
    # config.ftdwn -->   flow field starting from smoothed on largest scales first [default]

    # if ordering between cfield and flow is different, run cfield and flow separately
    if config.ctdwn != config.ftdwn:
        cfields,mask = cfieldall(config,params,cfields,mask)
        xf,yf,zf = flowall(config,params,cfields,xf,yf,zf)
    else:
        xf,yf,zf,mask = cfieldflowall(config,params,cfields,mask,xf,yf,zf)

    return xf,yf,zf,mask

def flowloss(config,params):

    xf,yf,zf,mask = fullflow(config,params)

    loss, rhopfl = getloss(config,xf,yf,zf)

    return loss, [rhopfl,mask]

def initialize():

    args   = parsecommandline()

    config = configfromargs(args)
    params = paramsfromargs(args)

    config, params = setupflowprofile(config,params)

    return config, params