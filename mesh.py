#!/usr/bin/python3
# Nicholas M. Rathmann <rathmann@nbi.ku.dk>, 2021-2022

import copy, code # code.interact(local=locals())
import numpy as np

from constants import *
from dolfin import *

#-----------------

bnd_bed_id = 1
bnd_sfc_id = 2

class SfcBoundary(SubDomain): 
    def inside(self, x, on_boundary): return bool( near(x[0],1) and on_boundary)
    
class BedBoundary(SubDomain): 
    def inside(self, x, on_boundary): return bool( near(x[0],0) and on_boundary)
   
class Mesh():
    
    def __init__(self, H, vertres=25, DEBUG=False):

        self.H = H  # Model domain height
        self.RES_Z = vertres # Vertical nodal resolution (number of layers)
        self.z0 = 0 # Column bottom coordinate

        self.update_mesh() # set interior mesh
        self.usz = np.nan # surface velocity not yet tracked.
        
        if DEBUG:
            plt.figure()
            plot(self.mesh)
            plt.plot(self.mesh.coordinates()[:], 0*self.mesh.coordinates()[:])
       
    def update_surface(self, uz, adot, dt): 
        
        uz.set_allow_extrapolation(True) # should not be necessary (self.H is within the domain), but numerical errors might imply we need to extrapolate.
        uz_s = uz(self.H) # current surface velocity 
        uz_s_eff = uz_s + adot # effective surface velocity; adot is the *positive* accumulation rate
        self.H = self.H + uz_s_eff * dt
        
    def update_mesh(self):  

        vl = self.RES_Z - 1 # Number of vertical layers
        mesh0 = UnitIntervalMesh(vl); # Line mesh of unit length
        self.mesh,self.boundaries,self.ds = self._update_mesh(mesh0)
        
    def _update_mesh(self, mesh):
    
        boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0) #FacetFunction("size_t", mesh)
        boundaries.set_all(0)
        sfc, bed = SfcBoundary(), BedBoundary()
        sfc.mark(boundaries, bnd_sfc_id)
        bed.mark(boundaries, bnd_bed_id)
        ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
    
        # Extract z (x) coordinate of unit-length mesh and linearly rescale to fit height of domain.
        z = mesh.coordinates()[:]
        znew = self.z0 + z*(self.H - self.z0) # rescale
        self.meshcoords = znew 
        mesh.coordinates()[:] = znew
    
        return (mesh, boundaries, ds)
    
