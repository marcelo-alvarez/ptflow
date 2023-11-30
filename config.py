import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import jax
import jax.numpy as jnp
import os.path
from mutil import convolve
import os

class PTflowConfig:
    '''PTflowConfig'''
    def __init__(self, **kwargs):
        # data location
        self.initdir  = kwargs.get('datadirICs', os.environ["PTFLOW_COMPARISON_DATA"]+'/ICs/')
        self.griddir = kwargs.get('griddir',     os.environ["PTFLOW_COMPARISON_DATA"]+'/griddata/')
        self.partdir = kwargs.get('griddir',     os.environ["PTFLOW_COMPARISON_DATA"]+'/particledata/')

        # simulation information     
        boxsize = kwargs.get('boxsize',205.0) # ICs box size in Mpc/h
        N       = kwargs.get('N',      625)   # ICs dimension

        # sub-box parameters
        nx = kwargs.get('nx', N//5) # sbox 1st grid dimension
        ny = kwargs.get('ny',    N) # sbox 2nd grid dimension
        nz = kwargs.get('nz',    N) # sbox 3rd grid dimension

        self.nm        = kwargs.get(   'nm', 20)
        self.logM1     = kwargs.get('logM1', 10.0)
        self.logM2     = kwargs.get('logM2', 15.0)
        self.zoom      = kwargs.get( 'zoom', 0.7)
        self.kmax      = kwargs.get( 'kmax', 7.0)
        self.filter    = kwargs.get( 'fltr', "matter")
        self.cftype    = kwargs.get('ctype', "int8")
        self.masktype  = kwargs.get('mtype', "int8")
        self.exclusion = kwargs.get( 'excl', False)
        self.masking   = kwargs.get( 'mask', True)
        self.soft      = kwargs.get( 'soft', False)

        report = kwargs.get('report', False)

        next = N

        self.masses = jnp.logspace(self.logM2,self.logM1,self.nm)

        if N % 2 == 0:
            i0=N//2-nx//2-1
            j0=N//2-ny//2-1
            k0=N//2-nz//2-1
        else:
            i0=N//2-nx//2
            j0=N//2-ny//2
            k0=N//2-nz//2
   
        nxe = int(next/N*nx)
        nye = int(next/N*ny)
        nze = int(next/N*nz)

        i0e = int(next/N*i0)
        j0e = int(next/N*j0)
        k0e = int(next/N*k0)

        # setup sub-box
        dsub      = boxsize / N
        i1        = i0 + nx                  ; j1 = j0 + ny                  ; k1 = k0 + nz 
        sx        = (i1-i0) / N * boxsize ; sy = (j1-j0) / N * boxsize ; sz = (k1-k0) / N * boxsize
        x0        = i0 * dsub                ; y0 = j0 * dsub                ; z0 = k0 * dsub
        x1        = x0 + sx                  ; y1 = y0 + sy                  ; z1 = z0 + sz

        # background cosmology
        self.omegam    =0.3
        self.h         =0.7
        self.rho       = 2.775e11*self.omegam*self.h**2
        self.D         = 0.8/0.0079692 # ICs growth factor; Dlinear(zICs = 127 | omegam=0.3089, h=0.6774) = 0.0079692 / 0.8

        # simulation box dimension
        self.boxsize   = boxsize
        self.N         = N

        # training and validation subbox
        self.dsub      = dsub
        self.sboxdims  = (nx,ny,nz)
        self.sbox      = np.asarray([[x0,x1],[y0,y1],[z0,z1]])
        self.sboxrange = np.asarray([[i0,i1],[j0,j1],[k0,k1]])
        self.pad       = 1e-3 * sx / nx
        self.offset = self.sboxrange[0,0] * N * N * 4
        self.count  =    self.sboxdims[0] * N * N 

        # fields
        sfields = ["sx1","sy1","sz1","sx2","sy2","sz2"]
        self.fields = {}
        self.fields['deltai'] = {}
        self.fields['rhodmg'] = {}

        # field filenames
        self.fields['deltai']['files'] = self.initdir+'delta'+'_'+str(N)
        for s in sfields:
            self.fields[s] = {} 
            self.fields[s]['files'] = self.initdir+s+'_s2lpt'+'_'+str(N)
        self.fields['rhodmg']['files'] = self.griddir+'dm_nx-'+str(next)+'_ngrid-'+str(next)+'.bin'

        # field data
        self.fields['deltai']['data'] = None
        for s in sfields:
            self.fields[s]['data'] = None
        self.fields['rhodmg']['data'] = None

        # convolution
        self.convolve = convolve

        # ics lpt order
        self.lpt = 2

        # load training data
        self.deltai = self.loadfield('deltai',scalegrowth=True ).copy()
        self.rhodmg = self.loadfield('rhodmg',scalegrowth=False).copy() 

        # logging on by default
        self.verbose = True

        if report:
            # print set up information
            print()
            print("PTFlow configuration")
            print()
            print(" grid dimensions: ",f"{self.sboxdims[0]:5d} {self.sboxdims[1]:5d} {self.sboxdims[2]:5d}")
            print()
            print(" grid resolution: ",f"{self.dsub:5.3f}")
            print()
            print("    subbox range: ",f"{self.sbox[0][0]:5.1f} {self.sbox[0][1]:5.1f}")
            print("                  ",f"{self.sbox[1][0]:5.1f} {self.sbox[1][1]:5.1f}")
            print("                  ",f"{self.sbox[2][0]:5.1f} {self.sbox[2][1]:5.1f}")
            print()
            print("grid DM filename: ",f"{self.fields['rhodmg']['files']}")
            print("    ICs filename: ",f"{self.fields['deltai']['files']}")
            print("    sx1 filename: ",f"{self.fields['sx1']['files']}")
            print()

    def fieldsbox(self,arr):

        slicey = slice(self.sboxrange[1,0],self.sboxrange[1,1])
        slicez = slice(self.sboxrange[2,0],self.sboxrange[2,1])
        arr = np.reshape(arr,(self.sboxdims[0],self.N,self.N))[:,slicey,slicez]

        return jnp.asarray(arr)

    def loadfield(self,field,scalegrowth=False):

        data     = self.fields[field]['data']
        filename = self.fields[field]['files']

        if data is None and os.path.isfile(filename):
            data = self.fieldsbox(np.fromfile(filename,count=self.count,dtype=np.float32,offset=self.offset))
            if scalegrowth:
                data *= self.D
            self.fields[field]['data'] = jnp.asarray(data)
        return self.fields[field]['data']

    def loadlpt(self):

        for sfield in ['sx1','sy1','sz1','sx2','sy2','sz2']:
            if self.fields[sfield]['data'] is None:
                self.fields[sfield]['data'] = self.loadfield(sfield)

        return
    
    def getslpt(self,Rsmooth=None,f=1.0):

        self.loadlpt()
        sx1 = self.fields['sx1']['data']; sy1 = self.fields['sy1']['data']; sz1 = self.fields['sz1']['data']
        sx2 = self.fields['sx2']['data']; sy2 = self.fields['sy2']['data']; sz2 = self.fields['sz2']['data']
        Df = self.D * f

        if Rsmooth is not None:
            sigmaL = Rsmooth / self.dsub
            sx1 = self.convolve(sx1,sigmaL)
            sy1 = self.convolve(sy1,sigmaL)
            sz1 = self.convolve(sz1,sigmaL)
        sx = Df * sx1
        sy = Df * sy1
        sz = Df * sz1
        if sx2 is not None:
            if Rsmooth is not None:
                sx2 = self.convolve(sx2,sigmaL)
                sy2 = self.convolve(sy2,sigmaL)
                sz2 = self.convolve(sz2,sigmaL)
            sx -= 3./7. * Df**2 * sx2
            sy -= 3./7. * Df**2 * sy2
            sz -= 3./7. * Df**2 * sz2 
        return [sx,sy,sz]

    def advect(self,s):
        b = self.boxsize

        gridx = jnp.linspace(self.sbox[0,0]+self.dsub/2,self.sbox[0,1]-self.dsub/2,self.sboxdims[0]).astype(jnp.float32)
        gridy = jnp.linspace(self.sbox[1,0]+self.dsub/2,self.sbox[1,1]-self.dsub/2,self.sboxdims[1]).astype(jnp.float32)
        gridz = jnp.linspace(self.sbox[2,0]+self.dsub/2,self.sbox[2,1]-self.dsub/2,self.sboxdims[2]).astype(jnp.float32)
        q=jnp.meshgrid(gridx,gridy,gridz,indexing='ij')

        x = q[0]+s[0] ; y = q[1]+s[1] ; z = q[2]+s[2]

        x -= jnp.heaviside(x-b,1.0) * b
        y -= jnp.heaviside(y-b,1.0) * b
        z -= jnp.heaviside(z-b,1.0) * b

        x += jnp.heaviside(-x,1.0) * b
        y += jnp.heaviside(-y,1.0) * b
        z += jnp.heaviside(-z,1.0) * b

        return [x,y,z]

    def binpoints(self,xp,yp,zp):
        xp = xp.flatten()
        yp = yp.flatten()
        zp = zp.flatten()

        bins  = self.sboxdims
        range = self.sbox
        field = jnp.histogramdd(jnp.array([xp,yp,zp]).transpose(),bins=bins,range=range)[0]
  
        return jnp.array(field)
    binpoints = jax.jit(binpoints,static_argnums=0)

    def getICs(self):
        # read / cache initial particle data (not currently used)
        import numpy as np
        import os.path

        datadir = self.partdir

        if(os.path.isfile('./dmICs.npz')):
            data = np.load('./dmICs.npz')
            self.xdm = data['x']
            self.ydm = data['y']
            self.zdm = data['z']
            return

        N        = 2500**3
        Ninchunk = 2500**2*250
        Nchunks  = N // Ninchunk

        self.xdm = np.asarray([],dtype=np.float32)
        self.ydm = np.asarray([],dtype=np.float32)
        self.zdm = np.asarray([],dtype=np.float32)
        for ichunk in range(Nchunks):

            print(ichunk,Nchunks)
            istart = ichunk * Ninchunk
            offset = istart * 4

            print('loading position x'); x = np.fromfile(datadir+'x.bin',count=Ninchunk,offset=offset,dtype=np.float32)/1e3           # kpc/h to Mpc/h
            print(x.min(),x.max())
            dmx = np.where((x >= self.sbox[0][0]) & (x <= self.sbox[0][1]))
            x  = x[dmx]
            print('loading position y'); y = np.fromfile(datadir+'y.bin',count=Ninchunk,offset=offset,dtype=np.float32)[dmx]/1e3      # kpc/h to Mpc/h
            dmy = np.where((y >= self.sbox[1][0]) & (y <= self.sbox[1][1]))
            y  = y[dmy]
            x  = x[dmy]
            print('loading position z'); z = np.fromfile(datadir+'z.bin',count=Ninchunk,offset=offset,dtype=np.float32)[dmx][dmy]/1e3 # kpc/h to Mpc/h
            dmz = np.where((z >= self.sbox[2][0]) & (z <= self.sbox[2][1]))
            z = z[dmz]
            y = y[dmz]
            x = x[dmz]

            self.xdm = np.append(self.xdm,x)
            self.ydm = np.append(self.ydm,y)
            self.zdm = np.append(self.zdm,z)
        
        np.savez('dmICs',x=self.xdm,y=self.ydm,z=self.zdm)        

        return