import jax
import jax.numpy as jnp
import numpy as np

# 1d filtering
def tophat(x,cen=0,scale=1e-5,width=1e-2,soft=True):
    cenl = cen - width/2
    cenr = cen + width/2
    if soft:
        return jax.nn.sigmoid((x-cenl)/scale)*jax.nn.sigmoid((cenr-x)/scale)#*4.0
    return jnp.heaviside(x-cenl,1)*jnp.heaviside(cenr-x,1)
tophat = jax.jit(tophat,static_argnames=['soft',])

def heaviright(x,cen=0,scale=1e-5,soft=True):
    if soft: return jax.nn.sigmoid((x-cen)/scale)
    return jnp.heaviside(x-cen,1)
heaviright = jax.jit(heaviright,static_argnames=['soft',])

def heavileft(x,cen=0,scale=1e-5,soft=True):
    if soft: return jax.nn.sigmoid((cen-x)/scale)
    return jnp.heaviside(cen-x,1)
heavileft = jax.jit(heavileft,static_argnames=['soft',])

# sorted jnp.interp wrapper
def sortedinterp(x,xp,yp):
    dm = jnp.argsort(xp)
    return jnp.interp(x,xp[dm],yp[dm])

# convolution
def convolve(signal,win,wfunc=None,norm=True,winarr=False):
    def _kernel(scale,sigshape,wfunc=None,norm=True):
        nxr = sigshape[0]//2 ; nyr = sigshape[1]//2 ; nzr = sigshape[2]//2
        ax = jnp.arange(2*nxr) - nxr
        ay = jnp.arange(2*nyr) - nyr
        az = jnp.arange(2*nzr) - nzr
        x,y,z  = jnp.meshgrid(ax,ay,az,indexing='ij')
        r      = jnp.sqrt(x**2+y**2+z**2)
        if wfunc is not None:
            kernel = wfunc(r/scale)
        else:
            kernel = heaviright(scale-r,scale=0.1) # jnp.heaviside(scale-r,1.0)
        if norm:
            kernel /= kernel.sum()
        return jnp.roll(kernel,(-nxr,-nyr,-nzr),axis=(0,1,2))
    if not winarr:
        win = _kernel(win,jnp.shape(signal),wfunc=wfunc,norm=norm)
    return jnp.real(jnp.fft.irfftn(jnp.fft.rfftn(signal)*jnp.fft.rfftn(win),signal.shape))
convolve  = jax.jit(convolve,static_argnames=["wfunc","norm","winarr"])

def azimuthalAverage(image, center=None, stddev=False, returnradii=False, return_nr=False ):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fractional pixels).
    stddev - if specified, return the azimuthal standard deviation instead of the average
    returnradii - if specified, return (radii_array,radial_profile)
    return_nr   - if specified, return number of pixels per radius *and* radius

    adapted from code by Jessica R. Lu
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if center is None:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    deltar[-1] = 1                   # include outermost points
    rind = np.where(deltar)[0] + 1   # location of changed radius (minimum 1 point)
    nr = np.concatenate([rind[0:1], rind[1:] - rind[:-1]]) # number of radius bin
                                     # concatenate to include center pixel / innermost bin

    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = np.concatenate([csim[0:1], csim[rind[1:]] - csim[rind[:-1]]]) # include innermost bin

    radial_prof = tbin / nr

    if stddev:
        # find azimuthal standard deviation
        r_int[r_int==r_int.max()] = r_int.max() - 1  # set last bin equal to second-to-last bin
        rad_mean = radial_prof[r_int]
        rad_diffsum = np.cumsum( (i_sorted-rad_mean)**2 )
        rad_std = np.sqrt( ( np.concatenate([rad_diffsum[0:1], rad_diffsum[rind[1:]] - rad_diffsum[rind[:-1]]]) ) / nr )

        if returnradii:
            return r_int[rind],rad_std
        elif return_nr:
            return nr,r_int[rind],rad_std
        else:
            return rad_std

    else:
        if returnradii:
            return r_int[rind],radial_prof
        elif return_nr:
            return nr,r_int[rind],radial_prof
        else:
            return radial_prof

# power spectrum
def powerspectrum(f):

    ff = np.fft.fftshift(np.fft.fft2(f))

    ff = ff / ff.shape[0] / ff.shape[1]

    ps2 = np.abs(ff)**2

    r, ps = azimuthalAverage(ps2, returnradii = True)

    return r, ps

def crosspower(f1,f2):

    ff1 = np.fft.fftshift(np.fft.fft2(f1))
    ff2 = np.fft.fftshift(np.fft.fft2(f2))

    ff1 = ff1 / ff1.shape[0] / ff1.shape[1]
    ff2 = ff2 / ff2.shape[0] / ff2.shape[1]

    ps2 = np.real(ff1 * np.conjugate(ff2))

    r, ps = azimuthalAverage(ps2, returnradii = True)

    return r, ps   

def crosspower2(f1,f2):

    ff1 = jnp.fft.fftshift(jnp.fft.fft2(f1))
    ff2 = jnp.fft.fftshift(jnp.fft.fft2(f2))

    ff1 = ff1 / ff1.shape[0] / ff1.shape[1]
    ff2 = ff2 / ff2.shape[0] / ff2.shape[1]

    ps = jnp.real(ff1 * jnp.conjugate(ff2))

    return ps
