"""A rotated Gaussian on an oscillatory equilibrium for testing local resolution."""
import numpy as np


def coordinates(r,z):
    x,y = r-1.30,z-0.72
    return (0.8*x+0.6*y)/0.08, (-0.6*x+0.8*y)/0.24


def manufactured_tilted(wave):
    def exact(r,z):
        u,v = coordinates(r,z)
        bump = 0.6*np.exp(-0.5*(u*u+v*v))
        radial = -(0.8*u/0.08-0.6*v/0.24)
        vertical = -(0.6*u/0.08+0.8*v/0.24)
        background = (1+r*r)*np.cos(wave*z)
        current = -wave**2*background + bump*(
            radial**2+vertical**2-0.08**-2-0.24**-2-radial/r)
        return (background+bump, 2*r*np.cos(wave*z)+radial*bump,
                -wave*(1+r*r)*np.sin(wave*z)+vertical*bump,current)
    return exact
