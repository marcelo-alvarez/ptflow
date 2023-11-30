import numpy     as np
import jax.numpy as jnp

# duffy et al. cnfw
def duffycnfw(M,z):
    # M in Msun/h
    a = 5.71
    b = -0.084
    c = -0.47
    m_pivot = 2e12
    return a * (M / m_pivot)**b * (1 + z)**c
# density profiles
def rhonfw(x,params):
    #
    # returns un-normalized NFW density profile rho
    #

    c = 15
    if params is not None:
        c = params['cnfw']
    rho = 1 / (x*(1+c*x)**2)

    return rho

def rhopl(x,params):
    #
    # returns un-normalized power-law profile rho
    #

    beta = -2.9
    if params is not None:
        beta = params['beta']
    rho   = x**beta

    return rho

# flow profile
def flowgen(params,rho1=rhonfw,rho2=rhopl):
    #
    # Returns a function for the weight, flow(q), as function  
    # as a function of Lagrangian radius, q, for transforming
    # to Eulerian coordinates according to 
    #   
    #   x(q) = x0 * flow(q) + q * [1-flow(q)]
    #
    # such that the density profile centered on x0, is 
    # proportional to rho (and setting x0=0),
    #
    #   rho(x) = rho1(x)  # x < 1 
    # and  
    #   rho(x) = rho2(x)  # 1 <= x < xL
    #
    # the overdensity is d0 within x=1, the overdensity is 1 
    # within x=xL, and the mass within x=xL is gamma times 
    # the mass within x=1. The density profile rho is 
    # determined by user-supplied functions rho1 and rho2,
    # with parameters in the dictionary params.
    # 
    # The Lagrangian radius is given by 
    #
    #   q(x)^3 / d0^3 = f1(x)     # x <= 1
    #   q(x)^3 / d0^3 = f2(x) + 1 # 1 < x < xL
    #   q(x)^3 / d0^3 = f3(x)     # x >= xL
    #
    # where
    #
    #   xL = (gamma*d0)^(1/3)
    #
    #   f1(x) = {int_0^x [r^2dr*rho(r)]} / {int_0^1  [r^2dr*rho(r)]}
    #   f2(x) = {int_1^x [r^2dr*rho(r)]} / {int_1^xL [r^2dr*rho(r)]} * (gamma-1)
    #   f3(x) = x^3 / d0^3
    #
    # The weight, flow(q), is determined by inverting q(x),
    # 
    #   flow(q) = 1 - x(q) / q
    #
    # Note that flow(q=0) = 1 and flow(q=xL) = 0, i.e. positions are
    # unperturbed at q > xL.
    #

    epsilon = 1e-10
    N       = 10000 # cumsum is used as approximation for integrals
    d0      = params['d0']
    gamma   = params['gamma']

    xL = (gamma*d0)**(1/3)

    xa   = jnp.logspace(-10,jnp.log10(xL*2),N)
    dlnx = xa[1]-xa[0]

    df1  = xa**3 * rho1(xa,params) * dlnx
    df2  = xa**3 * rho2(xa,params) * dlnx

    df1 *= jnp.heaviside(1-xa,1)
    df2 *= jnp.heaviside(xa-1,0) * jnp.heaviside(xL-xa,0)

    f1   = jnp.cumsum(df1) / jnp.sum(df1) 
    f2   = jnp.cumsum(df2) / jnp.sum(df2) * (gamma-1) + 1
    f3   = xa**3 / d0 # uniform for q >= xL

    f1  *= jnp.heaviside( 1-xa,1)
    f2  *= jnp.heaviside( xa-1,0) * jnp.heaviside(xL-xa,0)
    f3  *= jnp.heaviside(xa-xL,1)

    qa = (d0*(f1 + f2 + f3))**(1/3)

    fa = (1-xa/qa)*jnp.heaviside(xL-qa,1.0)
    qa = jnp.insert(qa,0,0.0)
    fa = jnp.insert(fa,0,1.0)
    flow = lambda q: jnp.interp(q,qa,fa)

    return xL, flow

def flowgenbp(d0=200,gamma=1.1,alpha=2.5,beta=2.9):
    # Returns a function for the Eulerian radius, x(q), 
    # as a function of Lagrangian radius, q, for a double 
    # power law density profile in Eulerian space of
    #
    #   rho/<rho> = A * x^-alpha    # x < 1 
    # and  
    #   rho/<rho> = A * B * x^-beta # 1 <= x < xL
    #
    # with mean overdensity of d0 at x=1, overdensity of 1 at x=xL, such that
    # the mass within x=xL is gamma times the mass within x=1.
    # 
    # The Lagrangian radius is given by 
    #
    #   q(x) = d0^(1/3) * x^[(3-alpha)/3] # x <= 1
    # and
    #   q(x) = d0^(1/3) * {1 + (gamma-1) * (x^[3-beta]-1) / (xL^[3-beta]-1)}^(1/3)
    # where
    #   xL = (gamma*d0)^(1/3)
    #
    import jax.numpy as jnp

    epsilon = 1e-10
    xL = (gamma*d0)**(1./3.)
    q0 = d0**(1./3.)
    def qofx(x):
        # x <= 1
        xle1 = jnp.heaviside( 1-x,1.0)
        qle1 = q0 * x**((3-alpha)/3)

        # x >= xL
        xgel = jnp.heaviside(x-xL,1.0)
        qgel = x

        # 1 < x < xL
        xbet = jnp.heaviside( x-1,0.0)*jnp.heaviside(xL-x,0.0)
        qbet = q0 * (1+(gamma-1)*(x**(3-beta)-1)/(xL**(3-beta)-1))**(1./3.)

        return qle1 * xle1 + qgel * xgel + qbet * xbet

    xa = jnp.logspace(-10,jnp.log10(xL*2),10000)
    qa = qofx(xa)

    fa = (1-xa/qa)*jnp.heaviside(xL-qa,1.0)
    qa = jnp.insert(qa,0,0.0)
    fa = jnp.insert(fa,0,1.0)

    flow = lambda q: jnp.interp(q,qa,fa)
    return xL, flow

def showprofs():
    import matplotlib.pyplot as plt
    plt.clf()

    xL,   flowf   = flowgen()
    xLbp, flowfbp = flowgenbp()

    q    = np.logspace(-4,np.log10(10*xL),100000)
    flow   = flowf(q)
    flowbp = flowfbp(q)

    plt.plot(q/xL,  flow,c='k',ls=':')
    plt.plot(q/xL,1-flow,c='k',ls='-')


    plt.plot(q/xLbp,  flowbp,c='r',ls=':')
    plt.plot(q/xLbp,1-flowbp,c='r',ls='-')

    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.show()

    return xL, flowf