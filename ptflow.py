import numpy as np
import scipy
import jax
import jax.numpy as jnp
from time import time
from jax import vmap
import gc
from jax.scipy.special import erfc

from mutil import crosspower, crosspower2, powerspectrum, tophat, heavileft, heaviright, sortedinterp, convolve

def ptfinterp(params,x,xstring,fstring):

    f1 = params[fstring+'1']
    f2 = params[fstring+'2']
    p  = params[fstring+'p']

    x1 = params[xstring+'1']
    x2 = params[xstring+'2']

    w = ((x-x1)/(x2-x1))**p
    return f1 * (1-w) + f2 * w

def RofM(config,M):
    return (3*M/4/jnp.pi/config.rho)**(1./3.) * config.h # M in Msun; R in Mpc/h

def MofR(config,R):
    return 4/3*jnp.pi*config.rho*(R/config.h)**3         # M in Msun; R in Mpc/h

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
    fparams = {}
    pparams = {}
    params  = {}
    for param in allparams:
        if   param in pfd.allparams['cparams']:
            cparams[param] = allparams[param]
        elif param in pfd.allparams['fparams']:
            fparams[param] = allparams[param]
        elif param in pfd.allparams['pparams']:
            pparams[param]  = allparams[param]
        else:
            params[param] = allparams[param]
    params = params | (fparams | pparams)
    return cparams,fparams,pparams,params

def getRfuncs(config, params):

    delta = config.loadfield('deltai',scalegrowth=True).copy()
    R1 = RofM(config,10**config.logM1/2)
    R2 = RofM(config,10**config.logM2*2)
    nR   = 200
    Ri   = jnp.logspace(np.log10(R1),np.log10(R2),nR)
    sigmaR = [] #np.zeros(nR,dtype=jnp.float32)
    R = []
    sp = 0
    r0 = 1e-3
    for i in range(nR):
        s = jnp.sqrt(convolve(delta,Ri[i]/config.dsub).var())
        if i == 0 or abs(2*(sp-s)/(sp+s)) > r0:
            R.append(Ri[i])
            sigmaR.append(s)
        sp = s
    R = jnp.asarray(R)
    sigmaR = jnp.asarray(sigmaR)

    sigma = jnp.asarray(sigmaR,dtype=jnp.float32)
    lR    = jnp.log10(R)

    # store sigma at M1 and M2 in config
    lR1 = jnp.log10(RofM(config,10**config.logM1))
    lR2 = jnp.log10(RofM(config,10**config.logM2))

    sigma1 = sortedinterp(lR1, lR, sigma)
    sigma2 = sortedinterp(lR2, lR, sigma)

    # enforce params['sigma1'] < params['sigma2']
    if sigma1 > sigma2:
        params['sigma1']=sigma2
        params['sigma2']=sigma1
    else:
        params['sigma1']=sigma1
        params['sigma2']=sigma2

    # now that we have sigma bounds for dc function can generate fcoll function
    deltac = ptfinterp(params,sigma,'sigma','dt')
    nu    = deltac / sigmaR
    lF    = jnp.log10(erfc(nu/jnp.sqrt(2)))
    deltac1 = ptfinterp(params,sigma1,'sigma','dt')
    deltac2 = ptfinterp(params,sigma2,'sigma','dt')

    lFoflR = lambda lr: sortedinterp(lr,lR,lF)
    lRoflF = lambda lf: sortedinterp(lf,lF,lR)

    # store logF at M1 and M2 in config
    lfcoll1 = lFoflR(lR1)
    lfcoll2 = lFoflR(lR2)

    # enforce params['lfcoll1'] < params['lfcoll2']
    if lfcoll1 > lfcoll2:
        params['lfcoll1']=lfcoll2
        params['lfcoll2']=lfcoll1
    else:
        params['lfcoll1']=lfcoll1
        params['lfcoll2']=lfcoll2

    return lFoflR, lRoflF, params

def setscales(config, params):

    # ordered mass scales for covergence finding and particle flow

    lM1 = config.logM1
    lM2 = config.logM2

    lFoflR, lRoflF, params = getRfuncs(config, params)

    if config.spacing == "logM":
        mass = jnp.sort(jnp.logspace(lM1,lM2,config.nsc))
        params['cmass'] = jnp.flip(mass) if config.ctdwn else mass
        params['fmass'] = jnp.flip(mass) if config.ftdwn else mass

        params['cRLag']  = RofM(config,params['cmass'])
        params['fRLag']  = RofM(config,params['fmass'])

    if config.spacing == "logF":

        lR1 = jnp.log10(RofM(config,10.**lM1))
        lR2 = jnp.log10(RofM(config,10.**lM2))

        lF1 = lFoflR(lR1)
        lF2 = lFoflR(lR2)

        lF = jnp.linspace(lF1,lF2,config.nsc)
        lR = jnp.sort(lRoflF(lF))
        RLag  = 10.**(lR)
        params['cRLag'] = jnp.flip(RLag) if config.ctdwn else RLag
        params['fRLag'] = jnp.flip(RLag) if config.ftdwn else RLag

        params['cmass'] = MofR(config,params['cRLag'])
        params['fmass'] = MofR(config,params['fRLag'])

        delta = config.loadfield('deltai',scalegrowth=True).copy()

        sigma1 = jnp.sqrt(convolve(delta,RofM(config,10**config.logM1)/config.dsub).var())
        sigma2 = jnp.sqrt(convolve(delta,RofM(config,10**config.logM2)/config.dsub).var())

        # enforce params['sigma1'] < params['sigma2']
        if sigma1 > sigma2:
            params['sigma1']=sigma2
            params['sigma2']=sigma1
        else:
            params['sigma1']=sigma1
            params['sigma2']=sigma2

        deltac1 = ptfinterp(params,sigma1,'sigma','dt')
        deltac2 = ptfinterp(params,sigma2,'sigma','dt')

        nu1    = deltac1 / sigma1
        nu2    = deltac2 / sigma2

        lfcoll1 = jnp.log10(erfc(nu1/jnp.sqrt(2)))
        lfcoll2 = jnp.log10(erfc(nu2/jnp.sqrt(2)))

        # enforce params['lfcoll1'] < params['lfcoll2']
        if lfcoll1 > lfcoll2:
            params['lfcoll1']=lfcoll2
            params['lfcoll2']=lfcoll1
        else:
            params['lfcoll1']=lfcoll1
            params['lfcoll2']=lfcoll2

    config.Rbuff = -1e10
    # sbox boundary padding to avoid artefacts
    params['nbx'] = jnp.zeros((config.nsc,2),dtype=jnp.int32)
    params['nby'] = jnp.zeros((config.nsc,2),dtype=jnp.int32)
    params['nbz'] = jnp.zeros((config.nsc,2),dtype=jnp.int32)
    for i in range(config.nsc):
        RLag = params['fRLag'][i]
        Rbuff = 2 * RLag * ptfinterp(params,lFoflR(np.log10(RLag)),'lfcoll','ls')
        if Rbuff > config.Rbuff: config.Rbuff = Rbuff
        ntophat = (Rbuff/config.dsub).astype(int)+1
        params['nbx'] = params['nbx'].at[i,0].set(min(ntophat,config.sboxdims[0]//2-1))
        params['nby'] = params['nby'].at[i,0].set(min(ntophat,config.sboxdims[1]//2-1))
        params['nbz'] = params['nbz'].at[i,0].set(min(ntophat,config.sboxdims[2]//2-1))
        params['nbx'] = params['nbx'].at[i,1].set(max(config.sboxdims[0]-ntophat,params['nbx'][i,0]))
        params['nby'] = params['nby'].at[i,1].set(max(config.sboxdims[1]-ntophat,params['nby'][i,0]))
        params['nbz'] = params['nbz'].at[i,1].set(max(config.sboxdims[2]-ntophat,params['nbz'][i,0]))

    return config, params

def setupflowprofile(config,params):

    import funcs

    # flow profiles using duffy nfw 
    params['xL'] = np.zeros(config.nsc,dtype=jnp.float32)
    flowparams = {}
    na = 10000
    qa = np.logspace(-3,1,na)
    qa = np.insert(qa,0,0.0)
    fa = np.zeros((config.nsc,len(qa)))
    for i in range(config.nsc):
        z = 0.0
        M200m = params['fmass'][i] * config.h  # mass scale in Msun/h
        M200c = M200m * np.sqrt(config.omegam) # TBD CORRECT M200 CONVERSION
        flowparams['cnfw']  = funcs.duffycnfw(M200c,z)
        flowparams['beta']  = params['pe']
        flowparams['d0']    = params['d0']*100
        flowparams['gamma'] = params['fM']
        flowparams['inner'] = params['pi']
        flowparams['outer'] = params['po']
        params['xL'][i], flowfunc = funcs.flowgen(flowparams)
        fa[i,:] = flowfunc(qa)
    params['na'] = na
    params['qa'] = jnp.asarray(qa)
    params['fa'] = jnp.asarray(fa)
    params['xL'] = jnp.asarray(params['xL'])

    return params

def huber(r2,gamma=1.0):
    return gamma**2 * (jnp.sqrt(1+r2/gamma**2)-1)

def modhuber(r2,gamma=1.0):
    return huber(r2,gamma=gamma)#/huber(1+r2,gamma=gamma)

def particleloss(config,params,xfl,yfl,zfl):

    nbx0 = params['nbx'][0,0]
    nby0 = params['nby'][0,0]
    nbz0 = params['nbz'][0,0]
    nbx1 = params['nbx'][0,1]
    nby1 = params['nby'][0,1]
    nbz1 = params['nbz'][0,1]

    x = config.xyz[0] ; y = config.xyz[1] ; z = config.xyz[2]

    r0 = config.dsub
    dx  = (xfl-config.dmposx)/r0
    dy  = (yfl-config.dmposy)/r0
    dz  = (zfl-config.dmposz)/r0

    loss = modhuber(dx**2+dy**2+dz**2)*config.dmposd

    loss *= (heaviright(x-nbx0,soft=config.soft)*heaviright(y-nby0,soft=config.soft)
               *heaviright(z-nbz0,soft=config.soft))
    loss *= (heaviright(nbx1-x,soft=config.soft)*heaviright(nby1-y,soft=config.soft)
               *heaviright(nbz1-z,soft=config.soft))

    return loss.mean()

def getloss(config,params,xfl,yfl,zfl,rhopfl):

    if config.losstype == "pos":
        return particleloss(config,params,xfl,yfl,zfl)

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

    nx = i1-i0
    ny = j1-j0
    nz = k1-k0

    si = slice(i0,i1)
    sj = slice(j0,j1)
    sk = slice(k0,k1)

    rhopfl2 = rhopfl[si,sj,sk].mean(axis=0)
    rhodmg2 = config.rhodmg[si,sj,sk].mean(axis=0) * (config.N/2500)**3

    if config.losstype[-2:] == "1d":
        # get (cross) power spectra
        k, cl_dmg = powerspectrum(np.asarray(rhodmg2))
        k, cl_hfl = powerspectrum(np.asarray(rhopfl2))

        k, cl_dh  = crosspower(np.asarray(rhodmg2),np.asarray(rhopfl2))

        r_dh = cl_dh / np.sqrt(cl_dmg*cl_hfl)

        # convert k from pixel units [0:npixel] to wavenumbers h/Mpc
        k = k * 2*np.pi / dy

        loss = (1-r_dh)**2 * k**0.5 #*20
        if config.losstype == "p1d":
            loss *= abs(np.log(cl_dmg)-np.log(cl_hfl)) / cl_dmg / 1e3
        else:
            loss /= 1e1
        loss *= heavileft(k,cen=kmax,soft=config.soft)

        loss = loss.cumsum()[-1]
    else:
        ps_dmg = crosspower2(rhodmg2,rhodmg2)
        ps_hfl = crosspower2(rhopfl2,rhopfl2)
        ps_dh  = crosspower2(rhodmg2,rhopfl2)
        r_dh   = ps_dh / jnp.sqrt(ps_dmg*ps_hfl)

        # convert k from pixel units [0:npixel] to wavenumbers h/Mpc
        dk  = 2 * np.pi / dy
        kx, ky = jnp.meshgrid(jnp.arange(ny)-ny//2,jnp.arange(ny)-ny//2)
        k = jnp.sqrt(kx**2+ky**2) * dk

        # apply k cutoff
        r_dh *= heavileft(k,cen=kmax,soft=config.soft)

        loss  = (1-r_dh)**2 #/ k**1.5
        loss *= heavileft(k,cen=kmax,soft=config.soft)
        loss  = loss.sum()

    return loss

def collkernele(x):
    return x**2*heavileft(x,1,soft=True)

def collkernelm(x):
    return heavileft(x,1,soft=True)

def cexclusion(config,cfield,deltasmooth,R,deltac):
    epsilon = 1e-3

    iter = 0
    while iter < 5:
        iter += 1
        nlfield = (deltasmooth-deltac)**25
        counts    = convolve(cfield,R,norm=False)
        nlfield=jnp.array(cfield*nlfield).astype(jnp.float32)
        con = convolve(nlfield,R,norm=False) / (counts + epsilon)
        cfield *= jnp.heaviside(nlfield-con,1.0)

    return cfield

def getcfield(config, params, i, cfields, mask):

    Rpix0 = params['cRLag'][i] / config.dsub

    # matter and energy filters as in Musso & Sheth 2021
    if config.filter == "matter":
        collkernel = collkernelm # "matter" =~ int delta(r)r^2dr
    else:
        collkernel = collkernele # "energy" =~ int delta(r)r^4dr
    deltasmooth = convolve(config.deltai, Rpix0, wfunc=collkernel)
    sigma  = jnp.sqrt(deltasmooth.var())
    deltac = ptfinterp(params,sigma,'sigma','dt')
    #cfield  = jnp.array(heaviright(deltasmooth,cen=deltac,scale=1e-4,soft=config.soft),
    cfield  = jnp.array(heaviright(deltasmooth,cen=deltac,scale=1e-5*deltac,soft=config.soft),
                        dtype=config.cftype)

    if config.masking:
        cfield *= mask

    if config.exclusion:
        cfield = cexclusion(config,cfield,deltasmooth,Rpix0,deltac)

    Rpix = Rpix0 * ptfinterp(params,sigma,'sigma','ms')
    #Rpix = Rpix0
    cmask = convolve(cfield, Rpix, norm=False)
    mask *= jnp.array(heavileft(cmask,cen=0.1,soft=config.soft),dtype=config.masktype)

    cfields += (i+1+cfield) * heaviright(cfield,cen=0.1) * heavileft(cfields,cen=0.5)
    lfcoll = jnp.log10(erfc(deltac/sigma/jnp.sqrt(2)))

    cfields = jnp.asarray(cfields,dtype=config.cftype)
    mask = jnp.asarray(mask,dtype=config.masktype)

    return cfields, mask, lfcoll, deltac
getcfield = jax.jit(getcfield,static_argnums=[0,])

def particleflow(config,params,i,cfield,lfcolls,xf,yf,zf):

    RLagpix = params['fRLag'][i] / config.dsub
    r0pix   = params['fRLag'][i] / config.dsub / params['xL'][i] # config.xL[i]

    Rsmooth = params['fRLag'][i] * ptfinterp(params,lfcolls[i],'lfcoll','ls')

    # count number of convergence points within RLagpix pixels of each point and set flowing --> 1 when count > 1/2
    count  = convolve(cfield, RLagpix, norm=False)
    count *= heaviright(count,cen=0.0,soft=config.soft)
    count += 1e-10 # TBD jax-friendly don't divide by zero
    flowing = heaviright(count, 0.5, soft=config.soft)

    # displacements at smoothing scale
    sxc,syc,szc = config.getslpt(Rsmooth=Rsmooth)

    # convergence displacement field is mean of all overlapping convergence centers, otherwise zero
    sxc = flowing * convolve(sxc*cfield, RLagpix, norm=False) / count
    syc = flowing * convolve(syc*cfield, RLagpix, norm=False) / count
    szc = flowing * convolve(szc*cfield, RLagpix, norm=False) / count

    # final positions of convergence centers
    xs,ys,zs = config.advect([sxc,syc,szc])
    del sxc ; gc.collect()
    del syc ; gc.collect()
    del szc ; gc.collect()

    # flow field infalling towards cfield with profile in flowfunc
    qa = params['qa']
    fa = params['fa'][i]
    flowf = lambda q: sortedinterp(q,qa,fa)
    flow = convolve(cfield, r0pix, flowf, norm = False) * flowing
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

    flowweight = ptfinterp(params,lfcolls[i],'lfcoll','fw')

    # nonlinear displacement correction [dx,dy,dz] from convergence flow where flowing > 0
    con = convolve(xs*cfield, RLagpix, norm=False)
    dx = flow * (convolve(xs*cfield, RLagpix, norm=False) / count - x0) * flowweight
    xc = x0 + dx                         ; del dx ; gc.collect()
    xf = xc * flowing + xf * (1-flowing) ; del xc ; gc.collect()

    dy = flow * (convolve(ys*cfield, RLagpix, norm=False) / count - y0) * flowweight
    yc = y0 + dy                         ; del dy ; gc.collect()
    yf = yc * flowing + yf * (1-flowing) ; del yc ; gc.collect()

    dz = flow * (convolve(zs*cfield, RLagpix, norm=False) / count - z0) * flowweight
    zc = z0 + dz
    zf = zc * flowing + zf * (1-flowing) ; del zc ; gc.collect()
 
    return xf,yf,zf
particleflow = jax.jit(particleflow,static_argnums=[0,])

def scaleflow(config,params,i,cfield,lfcolls,xf,yf,zf,buffer=False):

    # remove cfield within specified distance of boundary to avoid wrapping artefacts
    nbx0 = params['nbx'][i,0]
    nby0 = params['nby'][i,0]
    nbz0 = params['nbz'][i,0]
    nbx1 = params['nbx'][i,1]
    nby1 = params['nby'][i,1]
    nbz1 = params['nbz'][i,1]

    x = config.xyz[0] ; y = config.xyz[1] ; z = config.xyz[2]

    if buffer:
        cfield *= (heaviright(x-nbx0,soft=config.soft)*heaviright(y-nby0,soft=config.soft)
                *heaviright(z-nbz0,soft=config.soft))
        cfield *= (heaviright(nbx1-x,soft=config.soft)*heaviright(nby1-y,soft=config.soft)
                *heaviright(nbz1-z,soft=config.soft))

    xf,yf,zf = particleflow(config,params,i,cfield,lfcolls,xf,yf,zf)

    return xf,yf,zf
scaleflow = jax.jit(scaleflow,static_argnums=[0,],static_argnames=["buffer",])

def cfieldstep(config,params,i,cfields,mask):
    t0 = time()
    cfieldsi = cfields.copy()
    maski = mask.copy()

    cfields, mask, lfcoll, deltac = getcfield(config,params,i,cfields,mask)

    cfield = tophat(cfields,cen=i+1,width=0.1,soft=config.soft)
    nc = cfield.sum().astype(np.float32)
    if config.verbose:
        print(f"  threshold: {i+1:>4}/{config.nsc:<4} nc={nc} logM={np.log10(params['cmass'][i]):<5.2f} "+
              f"dt={time()-t0:<6.3f} RLag={params['cRLag'][i]:<6.3f}",
              end='\r')
    return cfields, mask, lfcoll

def flowstep(config,params,i,cfields,lfcolls,xf,yf,zf):
    t0 = time()
    ci = i+1 if config.ctdwn == config.ftdwn else config.nsc-i
    cfield = (cfields - ci) * tophat(cfields,cen=ci+1,width=0.1,soft=config.soft)

    nc = cfield.sum().astype(np.float32)

    xf,yf,zf = scaleflow(config,params,i,cfield,lfcolls,xf,yf,zf)

    if config.verbose:
        ls = ptfinterp(params,lfcolls[i],'lfcoll','ls')
        print(f"   dynamics: {i+1:>4}/{config.nsc:<4} nh={nc} logM={np.log10(params['fmass'][i]):<5.2f} "+
              f"dt={time()-t0:<6.3f} RLag={params['fRLag'][i]:<6.3f}",
              end='\r')
    return xf,yf,zf

def cfieldall(config,params,cfields,mask,lfcolls):

    # iterate over cfield scales
    for i in range(config.nsc):
        cfields, mask, lfcoll = cfieldstep(config,params,i,cfields,mask)
        lfcolls.append(lfcoll)
    if config.verbose: print()

    return cfields, mask, lfcolls

def flowall(config,params,cfields,lfcolls,xf,yf,zf):

    # iterate over flow scales
    for i in range(config.nsc):
        xf,yf,zf = flowstep(config,params,i,cfields,lfcolls,xf,yf,zf)
    if config.verbose: print()

    return xf,yf,zf

def fullflow(config,params):

    # cfield = 1 --> convergence point
    #   mask = 0 --> already a convergence point from previous iterations
    mask    = jnp.array(jnp.ones( config.sboxdims),dtype=config.masktype)    
    cfields = jnp.array(jnp.zeros(config.sboxdims),dtype=config.cftype)

    lfcolls  = []
    # [xf,yf,zf] = nonlinear positions initially set to unsmoothed LPT positions
    xf = config.xl.copy()
    yf = config.yl.copy()
    zf = config.zl.copy()

    # config.ctdwn --> cfield field starting from smoothed on largest scales first [default]
    # config.ftdwn -->   flow field starting from smoothed on largest scales first [default]

    cfields,mask,lfcolls = cfieldall(config,params,cfields,mask,lfcolls)
    lfcolls = jnp.asarray(lfcolls)
    # if ordering between cfield and flow is different, reverse order of lfcolls
    if config.ctdwn != config.ftdwn:
        lfcolls = jnp.flip(lfcolls)
    xf,yf,zf = flowall(config,params,cfields,lfcolls,xf,yf,zf)

    rhopfl = config.binpoints(xf,yf,zf)

    loss = getloss(config,params,xf,yf,zf,rhopfl)

    return loss,rhopfl,xf,yf,zf,mask

def flowgrad(config,params,dparams):

    def fullflowaux(config,params,dparams):
        params = params | dparams
        loss, rhopfl, xf, yf, zf, mask = fullflow(config,params)
        return loss, [loss,rhopfl,xf,yf,zf,mask]
    lossgrad,[loss,rhopfl,xf,yf,zf,mask] = jax.grad(fullflowaux,argnums=2,has_aux=True)(config,params,dparams)
    return lossgrad,loss,rhopfl,xf,yf,zf,mask

def initialize():
    import config as ptc

    allparams = parsecommandline()

    cparams,fparams,pparams,params = parseallparams(allparams)

    config = ptc.PTflowConfig(**cparams)

    config, params = setscales(config,params)

    params = setupflowprofile(config,params)

    return config, params
