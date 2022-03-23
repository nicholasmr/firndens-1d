#!/usr/bin/python3
# Nicholas M. Rathmann <rathmann@nbi.ku.dk>, 2021-2022

import copy, sys, code # code.interact(local=locals())
import numpy as np

from constants import *
from dolfin import *

from mesh import *

g = Constant(gmag) # grav accel
ms2myrConst = Constant(ms2myr)
ms2kyrConst = Constant(ms2kyr)

#------------------
# a, b functions
#------------------

c1_a = 13.22240
c2_a = 15.78652

c1_b = 15.09371 
c2_b = 20.46489

f_a0 = lambda rhohat,n: (1+2/3*(1-rhohat))*rhohat**(-2*n/(n+1))
f_b0 = lambda rhohat,n: 3/4*((1-rhohat)**(1/n)/(n*(1-(1-rhohat)**(1/n))))**(2*n/(n+1))

f_a1 = lambda rhohat,n: exp(c1_a - c2_a*rhohat) 
f_b1 = lambda rhohat,n: exp(c1_b - c2_b*rhohat) # add regularization for nonlin problem?

# Crossover scaling
if 1:
    #c3 = 12
    #cexpo = 10
    c3 = 12
    cexpo = 12
else:
    c3 = 20
    cexpo = 1
#    c3 = 20
#    cexpo = 0
    
f_a1c = lambda rhohat,n: (1-exp(-c3*(1-rhohat)))**cexpo
f_b1c = lambda rhohat,n: (1-exp(-c3*(1-rhohat)))**cexpo

# These are a(rho/rhoi) and b(rho/rhoi)
#f_a = lambda rhohat,n: f_a0(rhohat,n) + f_a1(rhohat,n) * f_a1c(rhohat,n)
#f_b = lambda rhohat,n: f_b0(rhohat,n) + f_b1(rhohat,n) * f_b1c(rhohat,n)

# J&L solution
rhohat_thres = 0.81
f_a = lambda rhohat,n: conditional(rhohat <= rhohat_thres, f_a1(rhohat,n), f_a0(rhohat,n))
f_b = lambda rhohat,n: conditional(rhohat <= rhohat_thres, f_b1(rhohat,n), f_b0(rhohat,n))


#------------------
    
class Stokes():
    
    def __init__(self, Aglen, nglen, compressible_ice=1):

        self.compressible_ice = compressible_ice # see comments below for definition
        print('Stokes solver: Is ice/firn compressible when rho ~= rhoi? %s'%('Yes' if compressible_ice else 'No'))
        
        ### Flow law parameters
        self.nglen   = nglen
        self.Aglen   = Constant(Aglen) 
        Elin  = 5e+9 # Enhancement-factor (A -> Elin*A) for linear viscious problem to approximately match nonlinear-viscous solution
        self.Aglen_lin  = Constant(Elin) * self.Aglen  # A for linear problem        
        self.Aglen_nlin = self.Aglen # A for nonlinear problem
        
        ### Domain IDs
        self.domid_ice  = 0
        self.domid_firn = 1
        
    def set_geometry(self, msh):
        
        ### Mesh
        self.mesh       = msh.mesh
        self.boundaries = msh.boundaries
        self.ds         = msh.ds
        
        ### Function spaces
        self.Uele = FiniteElement("CG", self.mesh.ufl_cell(), 2) # ele: velocity
        self.Pele = FiniteElement("CG", self.mesh.ufl_cell(), 1) # ele: pressure
        self.U = FunctionSpace(self.mesh, self.Uele)
        self.P = FunctionSpace(self.mesh, self.Pele)
        
        self.MixedEle = MixedElement([self.Uele, self.Pele]) 
        self.W = FunctionSpace(self.mesh, self.MixedEle)
        
        ### Vertical coordinates
        self.z_uz = self.U.tabulate_dof_coordinates()[:,0] # velocity node coordinates
        self.z_p  = self.P.tabulate_dof_coordinates()[:,0] # pressure node coordinates
        
        ### Coordinate/node sorting
        self.Iz = np.argsort(self.z_uz) # Sorted list of indices for nodal coordinates used for e.g. plotting
      
    def set_subdomains(self, rho, tol=rhoi*1E-8):
        
        ### Firn nodes selected
        rho.set_allow_extrapolation(True)
        class Firn(SubDomain):
            def inside(self, x, on_boundary): return rho(x) < rhoi - tol
        self.subdom_firn = Firn()
        
        ### All nodes selected; full column
        class FullColumn(SubDomain):
            def inside(self, x, on_boundary): return 1 
        self.subdom_full = FullColumn()
        
    def sig_compressible(self, u,Aglen,nglen, fa,fb):

        ### The compressible flow law

        eps = u.dx(0) # = sym(grad(u))
        epsm = eps # = tr(eps)
        
        # auxiliary variables
        edot = eps - epsm/3*1 # = eps - tr(eps)/3*I
        gamma2 = 2*edot**2 # = 2*(edot:edot)
        epsD2 = gamma2/fa + epsm**2/fb
        Bn = 2*Aglen
        
        # stress and pressure
        common_factor = Bn**(-1/nglen) * epsD2**((1-nglen)/(2*nglen)) # note epsD = sqrt(epsD2)
        tau = +2/fa * common_factor * edot
        p   = -1/fb * common_factor * epsm
        
        # combine everything
        sig = tau - p*1 # tau - p*I
        return sig
        
        
    def get_weak_form(self, u,v, p,q, rho, Aglen,nglen):
        
        ### Coefficient functions a and b
        fa = f_a(rho/rhoi,nglen)
        fb = f_b(rho/rhoi,nglen)
        
        ### Weak form
        a = self.sig_compressible(u,Aglen,nglen, fa,fb) * v.dx(0) * dx # viscous stress divergence
        L = rho*g*v*dx # Body force (gravity)

        # Ice (rho ~= rhoi) is incompressible, this pressure Lagrange multiplier ensures that. 
        # If self.compressible_ice is True, then p will have forced (known) values of p=0 (hence is p not solved for).
        a += (-v.dx(0)*p + q*u.dx(0) )*dx
     
        return (a, L)

    def solve(self, rho, adot, rho0, w_guess=None, tolfac=1, relaxation=0.5, maxiter=100, info={'tstep':0, 'w_prev_prev':None}):
            
        (u0, p0) = TrialFunctions(self.W) # the unknowns for linear problem
        (v,q)    = TestFunctions(self.W) # the weight functions (Galerkin method)
        w_lin, w = Function(self.W), Function(self.W) # containers for solution
        (u, p)   = split(w) # the unknowns for nonlinear problem

        ### Boundary conditions
        
        # Velocity BC
        uz_out = -adot*rho0/rhoi # balance between massflux in and out of domain
        bc = [ DirichletBC(self.W.sub(0), Constant(uz_out), self.boundaries, bnd_bed_id) ] 
        
        # Pressure Lagrange multiplier
        #-----------------------------
        # If self.compressible_ice is True, then set p=0 everywhere, which might make ice (rho ~= rhoi) slightly compressible.
        # Else use DirichletBC() to enforce p=0 *exclusively in the firn* (i.e. p is not solved for in the firn), implying 
        #   the part of the column which is ice (rho ~= rhoi) will be made strictly incompressible.
        self.set_subdomains(rho)
        domain__dont_solve_for_p = self.subdom_full if self.compressible_ice else self.subdom_firn
        bc += [ DirichletBC(self.W.sub(1), Constant(0), domain__dont_solve_for_p, method='pointwise') ]
                
        ### Weak Stokes problem
        
        # Is problem linear or no initial guess passed? => solve linear problem
        if self.nglen == 1 or (w_guess == None):
#            print('*** NO guess...')
            (a_lin, L_lin) = self.get_weak_form(u0,v, p0,q, rho, self.Aglen_lin, 1)
            solve(a_lin==L_lin, w_lin, bc)
            w.assign(w_lin)
            
        # Interpolate guess onto currrent problem mesh; that is, the guess is allowed to be on another mesh.
        else:
#            print('*** using guess...')
            w_guess.set_allow_extrapolation(True) 
            LagrangeInterpolator.interpolate(w, w_guess) # init guess for nonlin solver

        # If n>1, solve the nonlinear problem using init guess stored in "w", else we are done at this point.    
        if self.nglen > 1:    
        
          #  try:
            
                print('rho: ', rho.vector()[self.Iz])

                fa = f_a(rho/rhoi,self.nglen)
                fb = f_b(rho/rhoi,self.nglen)
                print('a: ', project(fa,self.U).vector()[:])
                print('b: ', project(fb,self.U).vector()[:])
                
#                (ug, pg) = w.split()
#                print('uguess:', ug.vector()[self.Iz])
            
                (a_nlin, L_nlin) = self.get_weak_form(u,v, p,q, rho, self.Aglen_nlin, self.nglen)
                F = a_nlin-L_nlin
                solve(F == 0, w, bc, solver_parameters={"newton_solver":  {"relative_tolerance": tolfac*1e-6,"absolute_tolerance": tolfac*1e-5,"relaxation_parameter":relaxation, "maximum_iterations":maxiter}})
    #               w.assign(info['w_prev_prev'])      
                            
          #  except:
                if info['tstep'] % 1 == 0 and info['tstep']>1: # > 1:
                    print("\n*** Nonlinear solver failed at step %i. Plotting provided guess..."%(info['tstep']))
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(8,6))
                    
                    z_uz,z_p  = self.z_uz[self.Iz], self.z_p
                    z_epsxy = z_uz
                    R = self.U
                    z_rho, I_rho = z_uz, self.Iz
                    
                    ax = plt.subplot(141)
        #                ax.plot(project(info['w_prev_prev'].sub(0),self.U).vector()[self.Iz]*ms2myr, z_uz, 'k.-', label='u guess prev')
                    ax.plot(project(w.sub(0),self.U).vector()[self.Iz]*ms2myr, z_uz, 'r.-', label='u guess')
                    ax.set_xlabel('u (m/yr)')
                    ax.legend()
                    
                    ax = plt.subplot(143)
        #                ax.plot(project(info['w_prev_prev'].sub(1)*1e-3, self.P).vector()[:], z_p, 'k.-', label='p guess prev')
                    ax.plot(project(w.sub(1)*1e-3, self.P).vector()[:], z_p, 'r.-', label='p guess')
                    ax.legend()
                    ax.set_xlabel('p (kPa)')
                    
                    ax = plt.subplot(144)
                    ax.plot([rhoi,rhoi], [0,z_rho[-1]], ':', color='#99000d', lw=1.5, label=r'$\rho_\mathrm{i}$')
                    ax.plot(rho.vector()[I_rho], z_rho, 'r.-', label='rho')
                    ax.plot(self.rho_prev.vector()[I_rho], z_rho, 'k.-', label='rho prev')
                    ax.legend()
                    ax.set_xlabel('rho (kg/m^3)')
                    
                    plt.savefig('debug/state_%i.png'%(info['tstep']))
                    plt.close()

#                    raise Exception("...aborting") 

        ### Save previously used fields
        self.rho_prev = rho
        
        return w
