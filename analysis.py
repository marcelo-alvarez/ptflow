import numpy as np
import matplotlib.pyplot as plt
import pathlib
import pickle
import jax.numpy as jnp
import ptflow as ptf

from matplotlib import gridspec
from matplotlib.colors import LogNorm
from mutil import heavileft, crosspower, powerspectrum

def savedata(config,params,rhopfl,xf,yf,zf,mask,opt=False):

    paramtag = f'{config.N}_k{config.kmax}'
    datastring   = 'comp_' + paramtag
    datastring = config.runname

    if opt:
        datastring += "_opt"
    else:
        datastring += "_fid"

    datadir = "./data/"

    pathlib.Path(datadir).mkdir(parents=True, exist_ok=True)

    datapath = datadir + datastring

    print(f"\n    data saved to {datapath}.npz")
    np.savez(datapath,mask=mask,rhopfl=rhopfl,xf=xf,yf=yf,zf=zf)

    config.params = params
    print(f"  config saved to {datapath}.config")
    pickle.dump(config, open(f"{datapath}.config", "wb"))

    return datastring

def analyze(datastring,zoom=None,zoomx=None,bindm=False):

    # load cached data
    config = pickle.load(open(f"./data/{datastring}.config", "rb"))
    params = config.params

    data = np.load(f"./data/{datastring}.npz")
    rhopfl = data['rhopfl']
    mask   = data['mask']
    xf     = data['xf']
    yf     = data['yf']
    zf     = data['zf']

    fmin = fmax = None

    plt.rcParams['figure.figsize']    = [10, 5]
    plt.rcParams["contour.linewidth"] = 0.5

    if zoom is None:
        zoom = config.zoom
    if zoomx is None:
        zoomx = config.zoom

    kmax = config.kmax

    ncol=3; nrow=2
    fig, axes = plt.subplots(nrow,ncol,figsize=(8,8),gridspec_kw={'wspace': -0.3, 'hspace': 0.1})
    axes = axes.ravel()
    fig.set_size_inches(18.5, 10.5, forward=True)

    dsubx = (config.sbox[0][1]-config.sbox[0][0])
    dsuby = (config.sbox[1][1]-config.sbox[1][0])
    dsubz = (config.sbox[2][1]-config.sbox[2][0])

    dx = dsubx * zoomx
    dy = dsuby * zoom
    dz = dsubz * zoom

    # buffx = (1-zoomx)/2 * dsubx
    # buffy = (1-zoom )/2 * dsuby
    # buffz = (1-zoom )/2 * dsubz

    # print(config.Rbuff,buffx,buffy,buffz)
    # if buffx < 2 * config.Rbuff: dx = dsubx - 4 * config.Rbuff
    # if buffy < 2 * config.Rbuff: dy = dsuby - 4 * config.Rbuff
    # if buffz < 2 * config.Rbuff: dz = dsubz - 4 * config.Rbuff
    # print(dsubx,4*config.Rbuff,dx)
    xc = 0.5 * (config.sbox[0][0]+config.sbox[0][1])
    yc = 0.5 * (config.sbox[1][0]+config.sbox[1][1])
    zc = 0.5 * (config.sbox[2][0]+config.sbox[2][1])

    x0 = xc - dx / 2; x1 = xc + dx / 2
    y0 = yc - dy / 2; y1 = yc + dy / 2
    z0 = zc - dz / 2; z1 = zc + dz / 2

    extent=[z0,z1,y0,y1]

    i0 = int(x0/config.dsub) - config.sboxrange[0][0] ; i1 = int(x1/config.dsub) - config.sboxrange[0][0]
    j0 = int(y0/config.dsub) - config.sboxrange[1][0] ; j1 = int(y1/config.dsub) - config.sboxrange[1][0]
    k0 = int(z0/config.dsub) - config.sboxrange[2][0] ; k1 = int(z1/config.dsub) - config.sboxrange[2][0]

    i1 = max(i1,i0+1)
    j1 = max(j1,j0+1)
    k1 = max(k1,k0+1)

    si = slice(i0,i1)
    sj = slice(j0,j1)
    sk = slice(k0,k1)

    plt.rcParams["contour.linewidth"] = 0.5

    if bindm:
        rhodmg = config.binpoints(xd,yd,zd)
        rhodmg = rhodmg[si,sj,sk].mean(axis=0)
    else:
        rhodmg = config.loadfield('rhodmg',scalegrowth=True )
        rhodmg = rhodmg[si,sj,sk].mean(axis=0) * (config.N/2500)**3

    deltai = config.loadfield('deltai',scalegrowth=True )
    rholpt = config.rholpt

    xd = config.fields['dmposx']['data']
    yd = config.fields['dmposy']['data']
    zd = config.fields['dmposz']['data']

    deltai = deltai[si,sj,sk].mean(axis=0)
    mask   =   mask[si,sj,sk].mean(axis=0)
    rholpt = rholpt[si,sj,sk].mean(axis=0)
    rhopfl = rhopfl[si,sj,sk].mean(axis=0)

    xl = config.xl
    yl = config.yl
    zl = config.zl

    gridx = jnp.linspace(config.sbox[0,0]+config.dsub/2,config.sbox[0,1]-config.dsub/2,config.sboxdims[0]).astype(jnp.float32)
    gridy = jnp.linspace(config.sbox[1,0]+config.dsub/2,config.sbox[1,1]-config.dsub/2,config.sboxdims[1]).astype(jnp.float32)
    gridz = jnp.linspace(config.sbox[2,0]+config.dsub/2,config.sbox[2,1]-config.dsub/2,config.sboxdims[2]).astype(jnp.float32)
    [gridx,gridy,gridz]=jnp.meshgrid(gridx,gridy,gridz,indexing='ij')

    sxd = xd - gridx ; syd = yd - gridy ; szd = zd - gridz
    sxf = xf - gridx ; syf = yf - gridy ; szf = zf - gridz
    fd = False
    if fd:
        sxl = xl - gridx ; syl = yl - gridy ; szl = zl - gridz
    else:
        Rsmooth = 10.0
        #sxl,dum,dum = config.getslpt(Rsmooth=Rsmooth,dir=dir)
        #dum,szl,dum = config.getslpt()
        #dir=None ; dum,syl,dum = config.getslpt(Rsmooth=Rsmooth,dir=dir)
        #dir=None ; dum,dum,szl = config.getslpt(Rsmooth=Rsmooth,dir=dir)
        #sxl,syl,szl = config.getslpt(Rsmooth)
        sxl,syl,szl = config.getslpt()

    gxl = -np.gradient(sxl)[0]
    gyl = -np.gradient(syl)[1]
    gzl = -np.gradient(szl)[2]

    gxf = -np.gradient(sxf)[0]
    gyf = -np.gradient(syf)[1]
    gzf = -np.gradient(szf)[2]

    gxd = -np.gradient(sxd)[0]
    gyd = -np.gradient(syd)[1]
    gzd = -np.gradient(szd)[2]

    grads = False
    if grads:
        dyd = (gyd - gyl)[si,sj,sk].mean(axis=0)
        dyf = (gyf - gyl)[si,sj,sk].mean(axis=0)

        dzd = (gzd - gzl)[si,sj,sk].mean(axis=0)
        dzf = (gzf - gzl)[si,sj,sk].mean(axis=0)
    else:
        dyd = np.log10(1/abs(syd - syl)[si,sj,sk].mean(axis=0))
        dyf = np.log10(1/abs(syf - syl)[si,sj,sk].mean(axis=0))

        dzd = np.log10(1/abs(szd - szl)[si,sj,sk].mean(axis=0))
        dzf = np.log10(1/abs(szf - szl)[si,sj,sk].mean(axis=0))

    divl = (gxl + gyl + gzl)[si,sj,sk].mean(axis=0)
    divd = (gxd + gyd + gzd)[si,sj,sk].mean(axis=0)
    divf = (gxf + gyf + gzf)[si,sj,sk].mean(axis=0)

    gxl = gxl[si,sj,sk].mean(axis=0)
    gyl = gyl[si,sj,sk].mean(axis=0)
    gzl = gzl[si,sj,sk].mean(axis=0)

    gxf = gxf[si,sj,sk].mean(axis=0)
    gyf = gyf[si,sj,sk].mean(axis=0)
    gzf = gzf[si,sj,sk].mean(axis=0)

    gxd = gxd[si,sj,sk].mean(axis=0)
    gyd = gyd[si,sj,sk].mean(axis=0)
    gzd = gzd[si,sj,sk].mean(axis=0)

    #ddi = config.loadfield('deltai',scalegrowth=True )[si,sj,sk].mean(axis=0)
    rholpt = np.array(rholpt)
    rhopfl = np.array(rhopfl)

    rms = np.log10(rhodmg[rhodmg>0]).var()**0.5
    mean = np.log10(rhodmg[rhodmg>0]).mean()
    if fmax is None:
        fmax = 10.**(mean+4*rms)
    if fmin is None:
        fmin = 10.**(mean-1*rms)

    minmk=0;    maxmk=1    ; cmapmk = 'viridis'
    minrl=fmin; maxrl=fmax ; cmaprl = 'inferno'
    minrh=fmin; maxrh=fmax ; cmaprh = 'inferno'
    minrd=fmin; maxrd=fmax ; cmaprd = 'inferno'

    # get (cross) power spectra
    k, cl_dmg = powerspectrum(rhodmg)
    k, cl_lpt = powerspectrum(rholpt)
    k, cl_hfl = powerspectrum(rhopfl)

    k, cl_dl  = crosspower(rhodmg,rholpt)
    k, cl_dh  = crosspower(rhodmg,rhopfl)

    r_dl = cl_dl / np.sqrt(cl_dmg*cl_lpt)
    r_dh = cl_dh / np.sqrt(cl_dmg*cl_hfl)

    # convert k from pixel units [0:npixel] to wavenumbers h/Mpc
    k = k * 2*np.pi / dy

    rcax = 0
    clax = 3
    vfax = 1
    vdax = 2
    dfax = 4
    ddax = 5

    # cross-correlation coefficient
    axes[rcax].plot(k, r_dl,c='k',ls=':',lw=3,label="dm-lpt")
    axes[rcax].plot(k, r_dh,c='k',ls='-',lw=3,label="dm-hfl")

    axes[rcax].set_xscale('log')
    axes[rcax].set_ylim((-0.4,1.1))
    axes[rcax].set_xlim((0.05,40))
    axes[rcax].set_aspect(1.0/axes[rcax].get_data_ratio(),adjustable='box')

    axes[rcax].legend()

    # power spectra
    axes[clax].plot(k, cl_dmg,c='k',ls='-',lw=3,label="dm")
    axes[clax].plot(k, cl_hfl,c='r',ls='-',lw=3,label="hfl")
    axes[clax].plot(k, cl_lpt,c='r',ls=':',lw=3,label="lpt")

    ploss = abs(np.log(cl_dmg)-np.log(cl_hfl)) / cl_dmg * (1.-r_dh**2) / k**1.5 * 20
    ploss *= heavileft(k,cen=kmax,soft=config.soft)

    #axes[clax].plot(k,ploss.cumsum()/1e6)

    axes[clax].set_xscale('log')
    axes[clax].set_yscale('log')
    axes[clax].set_ylim((1e-6,5))
    axes[clax].set_xlim((0.05,40))
    axes[clax].set_aspect(1.0/axes[clax].get_data_ratio(),adjustable='box')

    axes[clax].legend()

    # images

    rholpt = np.array(rholpt)
    rhopfl = np.array(rhopfl)
    rhodmg = np.array(rhodmg)

    rholpt[rholpt<minrl]=minrl
    rhopfl[rhopfl<minrh]=minrh
    rhodmg[rhodmg<minrd]=minrd

    #axes[2].imshow(mask,               vmin=minmk,vmax=maxmk ,extent=extent,cmap=cmapmk)
    # axes[3].imshow(rholpt,norm=LogNorm(vmin=minrl,vmax=maxrl),extent=extent,cmap=cmaprl)

    fd0=-dyd.max()
    fd1=dyd.max()
    fg0=-1.0
    fg1=1.0
    cmaps = 'viridis'
    # axes[0].imshow(dyf,                vmin=fd0, vmax=fd1  ,extent=extent,cmap=cmaps)
    # axes[3].imshow(gyl,                vmin=fg0, vmax=fg1  ,extent=extent,cmap=cmaps)
    # axes[6].imshow(dyd,                vmin=fd0, vmax=fd1  ,extent=extent,cmap=cmaps)

    #axes[3].imshow(divl, vmin=fd0, vmax=fd1, extent=extent,cmap=cmaps)
    axes[vfax].imshow(divf, vmin=fd0, vmax=fd1, extent=extent,cmap=cmaps)
    axes[vdax].imshow(divd, vmin=fd0, vmax=fd1, extent=extent,cmap=cmaps)

    #axes[6].imshow(rholpt,norm=LogNorm(vmin=minrh,vmax=maxrh),extent=extent,cmap=cmaprh)
    axes[dfax].imshow(rhopfl,norm=LogNorm(vmin=minrh,vmax=maxrh),extent=extent,cmap=cmaprh)
    axes[ddax].imshow(rhodmg,norm=LogNorm(vmin=minrd,vmax=maxrd),extent=extent,cmap=cmaprd)
    figdir  = f"./figures/"

    pathlib.Path( figdir).mkdir(parents=True, exist_ok=True) 
    figfile  =  figdir + f"{datastring}.png"

    print(f"\n  figure saved to {figfile}")
    plt.savefig(figfile,bbox_inches='tight',dpi=100)

    plt.close('all')
