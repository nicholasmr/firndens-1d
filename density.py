#!/usr/bin/python3
# Nicholas M. Rathmann <rathmann@nbi.ku.dk>, 2021-2022

import copy, sys, code # code.interact(local=locals())
import numpy as np

from constants import *
from dolfin import *

from mesh import *
from stokes import *

####################
    
class Density():
    
    def __init__(self, msh, rho_init_expression):
        
        self.set_geometry(msh)
        self.rho = interpolate(rho_init_expression, self.R)
        
    def set_geometry(self, msh):
        
        self.mesh       = msh.mesh
        self.boundaries = msh.boundaries
        self.ds         = msh.ds
        
        # Function spaces
        deg = 2 # deg = 2 seems to give good mass conservation
        self.Rele = FiniteElement("CG", self.mesh.ufl_cell(), deg)
        self.R = FunctionSpace(self.mesh, self.Rele)
        
        # Coordinates
        self.z = self.R.tabulate_dof_coordinates()[:,0]
        self.Iz = np.argsort(self.z)
        
    def change_mesh(self, mesh_new):
        
        rho_old = self.rho.copy()
        self.set_geometry(mesh_new)
        rho_new = Function(self.R)
        rho_old.set_allow_extrapolation(True)
        LagrangeInterpolator.interpolate(rho_new , rho_old)  
        self.rho = rho_new
        
        # Hard limit on density 
        # test idea: a Lagrange multiplier of sorts should prevent rho>rhoi
        if 1:
            rhovec = self.rho.vector()[:] 
            rhovec[rhovec > rhoi] = rhoi
            self.rho.vector()[:] = rhovec
        
    def solve(self, uz, rho0, dt):
    
        rho = TrialFunction(self.R) # the unknown
        w = TestFunction(self.R) # the weight function
        rhosol = Function(self.R) # container for solution
                
        dt_ = Constant(dt)
        a = (rho/dt_ + rho*uz.dx(0) + rho.dx(0)*uz) * w * dx
        L = self.rho/dt_ * w * dx
        
        bc = []
        # No surface BC when rho0 = 0, i.e. no accumulation.
        if rho0 > 0: bc += [ DirichletBC(self.R, Constant(rho0), self.boundaries, bnd_sfc_id) ] 
        
        solve(a==L, rhosol, bc) 
        self.rho = rhosol
        
        return self.rho
        
