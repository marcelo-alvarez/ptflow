import numpy as np
import matplotlib.pyplot as plt
import pathlib

from matplotlib import gridspec
from matplotlib.colors import LogNorm
from mutil import heavileft, crosspower, powerspectrum

def analyze(config,params,rhopfl,mask,fmin=None,fmax=None,opt=False):

    plt.rcParams['figure.figsize']    = [10, 5]
    plt.rcParams["contour.linewidth"] = 0.5

    zoom = config.zoom
    kmax = config.kmax

    ncol=3; nrow=2
    fig, axes = plt.subplots(nrow,ncol,figsize=(8,8),gridspec_kw={'wspace': -0.3, 'hspace': 0.1})
    axes = axes.ravel()
    fig.set_size_inches(18.5, 10.5, forward=True)

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

    plt.rcParams["contour.linewidth"] = 0.5

    deltai = config.loadfield('deltai',scalegrowth=True )
    rhodmg = config.loadfield('rhodmg',scalegrowth=True )
    rholpt = config.rholpt

    deltai = deltai[si,sj,sk].mean(axis=0)
    mask   =   mask[si,sj,sk].mean(axis=0)
    rholpt = rholpt[si,sj,sk].mean(axis=0)
    rhopfl = rhopfl[si,sj,sk].mean(axis=0)
    rhodmg = rhodmg[si,sj,sk].mean(axis=0) * (config.N/2500)**3

    rholpt = np.array(rholpt)
    rhopfl = np.array(rhopfl)

    rms = np.log10(rhodmg[rhodmg>0]).var()**0.5
    mean = np.log10(rhodmg[rhodmg>0]).mean()
    if fmax is None:
        fmax = 10.**(mean+3*rms)
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

    # cross-correlation coefficient
    axes[0].plot(k, r_dl,c='k',ls=':',lw=3,label="dm-lpt")
    axes[0].plot(k, r_dh,c='k',ls='-',lw=3,label="dm-hfl")

    axes[0].set_xscale('log')
    axes[0].set_ylim((-0.4,1.1))
    axes[0].set_xlim((0.05,40))
    axes[0].set_aspect(1.0/axes[0].get_data_ratio(),adjustable='box')

    axes[0].legend()

    # power spectra
    axes[1].plot(k, cl_dmg,c='k',ls='-',lw=3,label="dm")
    axes[1].plot(k, cl_hfl,c='r',ls='-',lw=3,label="hfl")
    axes[1].plot(k, cl_lpt,c='r',ls=':',lw=3,label="lpt")

    ploss = abs(np.log(cl_dmg)-np.log(cl_hfl)) / cl_dmg * (1.-r_dh**2) / k 
    ploss *= heavileft(k,cen=kmax,soft=config.soft)

    axes[1].plot(k,ploss.cumsum()/1e6)

    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_ylim((1e-6,5))
    axes[1].set_xlim((0.05,40))
    axes[1].set_aspect(1.0/axes[1].get_data_ratio(),adjustable='box')

    axes[1].legend()

    # images

    rholpt = np.array(rholpt)
    rhopfl = np.array(rhopfl)
    rhodmg = np.array(rhodmg)

    rholpt[rholpt<minrl]=minrl
    rhopfl[rhopfl<minrh]=minrh
    rhodmg[rhodmg<minrd]=minrd

    axes[2].imshow(mask,               vmin=minmk,vmax=maxmk ,extent=extent,cmap=cmapmk)
    axes[3].imshow(rholpt,norm=LogNorm(vmin=minrl,vmax=maxrl),extent=extent,cmap=cmaprl)
    axes[4].imshow(rhopfl,norm=LogNorm(vmin=minrh,vmax=maxrh),extent=extent,cmap=cmaprh)
    axes[5].imshow(rhodmg,norm=LogNorm(vmin=minrd,vmax=maxrd),extent=extent,cmap=cmaprd)

    paramtag = f'{config.N}_d{params["d0"]}_n{config.nm}_k{config.kmax}'
    prefix   = 'comp_' + paramtag 

    if opt:
        prefix += "_opt"
    else:
        prefix += "_fid"

    figdir  = "./figures/"
    datadir = "./data/"

    pathlib.Path( figdir).mkdir(parents=True, exist_ok=True) 
    pathlib.Path(datadir).mkdir(parents=True, exist_ok=True) 

    figfile  =  figdir + prefix + ".png"
    datafile = datadir + prefix

    print(f"saving figure to file {figfile}")
    plt.savefig(figfile,bbox_inches='tight',dpi=100)

    print(f"saving data to file {datafile}.npz")
    np.savez(datafile,rholpt=rholpt,mask=mask,rhopfl=rhopfl,rhodmg=config.rhodmg)