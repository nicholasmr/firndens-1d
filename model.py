#!/usr/bin/python3
# Nicholas M. Rathmann <rathmann@nbi.ku.dk>, 2021-2022

import copy, sys, code # code.interact(local=locals())
import numpy as np
from progress.bar import Bar

from constants import *
from dolfin import *

from mesh import *
from stokes import *
from density import *

# fenics verbosity
CRITICAL  = 50 # errors that may lead to data corruption and suchlike
ERROR     = 40 # things that go boom
WARNING   = 30 # things that may go boom later
INFO      = 20 # information of general interest
PROGRESS  = 16 # what's happening (broadly)
TRACE     = 13 # what's happening (in detail)
DBG       = 10 # sundry

set_log_level(WARNING)
#set_log_level(INFO)

def integrate_model(H=200, rho0=330, adot=0, Aglen=6.8e-26, nglen=3, dt=1*SecPerYear, Nt=1000, dHdt_tol=0.01, verbose=True):
    
    '''
    -----------
    PARAMETERS:
    -----------
    
        H        :: Initial firn column height (m)
        rho      :: Density of accumulation (kg/m^3)
        adot     :: Surface accumulation rate (m/s)
        nglen    :: Flow law exponent
        dt       :: Time step size (s)
        Nt       :: Maximum number of time steps before aborting integration
        dHdt_tol :: Rate-of-change of firn column height below which integration is aborted (steady-state assumed reached)
            
    ''' 
    
    # Initialize the mesh
    msh = Mesh(H)
    
    # Set initial density profile
    rho_init = Expression("rho0-(rho0-rhoi)*pow((H-x[0])/H,0.35)", H=H, rhoi=rhoi, rho0=rho0, degree=2)
    #rho_init = Expression("rhoi - (rhoi-rho0)*x[0]/H", H=H, rhoi=rhoi, rho0=rho0, degree=2)
    #rho_init = Expression("rhoi*0.80", H=H, rhoi=rhoi, rho0=rho0, degree=2)
    #rho_init = Expression("rhoi", H=H, rhoi=rhoi, rho0=rho0, degree=2)  # USE THIS TO TEST LAGRANGE PRESSURE MULTIPLIER FOR div(u) = 0 
    
    # Initialize the density solver
    dns = Density(msh, rho_init)
    
    # Initialize the Stokes solver
    stk = Stokes(Aglen, nglen)
    
    # Numerics
    tsteps = np.arange(0,Nt) # 0, 1, ..., Nt
    time   = tsteps*dt * s2yr # time
       
    # H(t) (column height)
    H = np.full((Nt), np.nan) # initialize with nan
    H[0] = msh.H
    
    # m(t) (total column mass)
    m = np.full((Nt), np.nan) # initialize with nan
    m[0] = assemble(dns.rho*dx) # = integral_0^H rho * dz
    
    #### Start integration ####
    
    abort = False # abort flag
    nt    = 0 # Number of integration (time) steps taken so far
    
    w_prev = None # Mixed function of solution from previous time step (mixed space depending on on problem being solved)
    w_prev_prev = None # for debugging

    rho_sol = [None, None] # init and final solution    
    uz_sol  = [None, None] # init and final solution
    p_sol   = [None, None] # init and final solution
    
    z_rho = [None, None] # init and final solution    
    z_uz  = [None, None] # init and final solution    
    z_p   = [None, None] # init and final solution    
    
    with Bar('Integrating with dt=%.3fyr and Nt=%i'%(dt*s2yr,Nt), max=Nt, fill='#', suffix='%(percent).1f%% - %(eta)ds') as bar:
        for ii in tsteps[1:]: # skip first step, it's reserved for the initial state
            
            # Set new mesh
            stk.set_geometry(msh)
            m[ii] = assemble(dns.rho*dx) # Total mass in domain
            
            # Solve stokes problem for current density profile
            toleranceMultiplier = 1 if ii==tsteps[0] else 1e2 # set stricter tolerence for nonlinear stokes solver if this is not first time step (where the initial guess might be very far from the solution)
            w = stk.solve(dns.rho, adot, rho0, w_guess=w_prev, tolfac=toleranceMultiplier, info={'tstep':ii, 'w_prev_prev':w_prev_prev})
            
            # Save solutions
            uz = project(w.sub(0), stk.U)
            p  = project(w.sub(1), stk.P)
    
            # Tolerence reached?
            dHdt = abs(H[ii-1]-H[ii-2])/(dt*s2yr)
            if ii>1 and  dHdt < dHdt_tol: 
                abort = True
                if verbose: print("\nSteady-state tolerence (dH/dt < %.2e m/yr) reached after %i time steps (%.1f yr)"%(dHdt_tol, ii-1,time[ii-1]))
                ii = tsteps[-1] # overwrite "ii" so that we correctly save the final solution at the end of the array(s).
            
            # First or last step? Save solutions in dedicated arrays.
            if ii == tsteps[1] or ii == tsteps[-1]:  
                I = 0 if ii == tsteps[1] else 1
                rho_sol[I], uz_sol[I], p_sol[I] = dns.rho.vector()[dns.Iz], uz.vector()[stk.Iz], p.vector()
                z_rho[I], z_uz[I], z_p[I] = dns.z[dns.Iz], stk.z_uz[stk.Iz], stk.z_p
                
            if abort: break
                
            # Calculate updated density profile
            rho_new = dns.solve(uz, rho0, dt)
            
            # Generate new mesh (adjusting the surface height)
            msh.update_surface(uz, adot, dt)
            msh.update_mesh()
            H[ii] = msh.H
                
            # Interpolate density profile to new mesh
            dns.change_mesh(msh)
            
            # Stored previous velocity solution for guess at next time step
            w_prev_prev = w_prev.copy() if w_prev != None else None
            w_prev = w.copy() 
            
            nt = ii # update number of completed integration steps
            bar.next() # update progress bar
        
    ### Done integrating! ###

    # Debug
    if 0:
        print('vertical strain rates are:')
        duzdz = project(uz.dx(0), stk.U)
        print(duzdz.vector()[stk.Iz])

    # Return solutions (numpy array), z coordinates of solutions, and aux vars.
    return (rho_sol, uz_sol, p_sol,   z_rho, z_uz, z_p,   H, m, time, nt)
            
