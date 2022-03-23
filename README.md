# firndens-1d
FEniCS solver for coupled evolution of firn density and velocity in 1D (vertical column):
> <img src="https://render.githubusercontent.com/render/math?math=\large\displaystyle \frac{\partial \rho}{\partial t} %2B \nabla\cdot(\rho\bf{u})= 0">

> <img src="https://render.githubusercontent.com/render/math?math=\large\displaystyle \nabla\cdot[\bf{\tau}(\dot{\bf{\epsilon}}(\bf{u}))] - \nabla p = \rho \bf{g}">

where <img src="https://render.githubusercontent.com/render/math?math=\large\displaystyle \bf{\tau}(\dot{\bf{\epsilon}}(\bf{u}))"> is the compressible (porous) rheology as documented in Gagliardini and Meyssonnier (1997), Lüthi and Funk (2000,2001), Boutillier (2004), Zwinger et al. (2007).
