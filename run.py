#!/usr/bin/python3
# Nicholas M. Rathmann <rathmann@nbi.ku.dk>, 2021-2022

import copy, sys, code # code.interact(local=locals())
import numpy as np

from constants import *
from model import *
from mesh import *
from stokes import *
from density import *

#------------------
# Parameters
#------------------

H     = 200 # ice column height
nglen = 1   # flow-law exponent ***only n=1 is currently working***

#------------------
# Experiment
#------------------

expr_names = ['DYE3',]
EXPR_DYE3  = 0
#.... add more

# Select site to model
EXPR = EXPR_DYE3

#------------------
# Experiment definitions
#------------------

if EXPR == EXPR_DYE3:

    Aglen0 = 6.8e-26 # A(T=-25 deg.)
    rho0 = 345.3717043 # surface density at DYE3
    adot = (171/SecPerYear)/rho0 # accumulation rate

print('accum. rate [m/s] = ', adot)

#------------------
# Numerics
#------------------

fac = 0.2 # tune step size
dt  = 0.25 * 1/fac * SecPerYear
Nt  = 15 * 100 * fac 

#------------------
# Integrate model
#------------------

(rho, uz, p, z_rho, z_uz, z_p, H, m, time, nt) = \
    integrate_model(H=H, rho0=rho0, adot=adot, Aglen=Aglen0, nglen=nglen, dt=dt, Nt=int(Nt), dHdt_tol=1e-3)

#------------------
# Plot solution
#------------------

import matplotlib.pyplot as plt
from matplotlib import rcParams, rc
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec

import pandas as pd

FS = 12
rc('font',**{'family':'serif','sans-serif':['Times'],'size':FS})
rc('text', usetex=True)
rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{physics} \usepackage{txfonts} \usepackage{siunitx} \DeclareSIUnit\year{a}'

lw = 1.7
legkwargs = {'frameon':True, 'fancybox':False, 'edgecolor':'k', 'framealpha':1, 'ncol':1, 'handlelength':1.34, 'labelspacing':0.3}

scale = 0.75
fig = plt.figure(figsize=(12.5*scale,9*scale), constrained_layout=True)
gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.7])
ax_rho = fig.add_subplot(gs[0, 0])
ax_uz  = fig.add_subplot(gs[0, 1])
ax_p   = fig.add_subplot(gs[0, 2])
ax_H   = fig.add_subplot(gs[1, 0])
ax_dHdt= fig.add_subplot(gs[1, 1])
ax_m   = fig.add_subplot(gs[1, 2])

def setyaxis(ax):
    ax.set_yticks(np.arange(0,H[0]+1,50))
    ax.set_yticks(np.arange(0,H[0]+1,50/4), minor=True)
    ax.set_ylim([0,H[0]])
    ax.set_ylabel(r'$z$ (\SI{}{\metre})')

### rho ###

if EXPR == EXPR_DYE3:
    df = pd.read_csv(r'data/dye3.txt', header=2, dtype={'a': np.float32, 'b': np.float32}, sep='\s+')
    dye3 = df.to_numpy()
    z_dye3   = H[nt] - (dye3[:,0] - dye3[0,0])
    rho_dye3 = dye3[:,1]
    ax_rho.plot(rho_dye3, z_dye3, '.', c='0.6', label=r'DYE 3', zorder=1)

#----------

ax_rho.plot([rhoi,rhoi], [0,H[0]], ':', color='#99000d', lw=1.5, label=r'$\rho_\mathrm{i}$')
ax_rho.plot(rho[0],  z_rho[0],  'k:', lw=lw, label='$t=%.0f$\,yr'%time[0])
ax_rho.plot(rho[-1], z_rho[-1], 'k-', lw=lw, label='$t=%.0f$\,yr'%time[nt])

ax_rho.legend(loc=3, **legkwargs)
ax_rho.set_xticks(np.arange(300,1000,200))
ax_rho.set_xticks(np.arange(300,1000,50), minor=True)
ax_rho.set_xlim([300,950])
setyaxis(ax_rho)
ax_rho.grid()

ax_rho.set_xlabel(r'$\rho$ (\SI{}{\kilogram\per\metre\cubed})')

### uz ###

#ax_uz.plot([-adot*rho0/rhoi*ms2myr]*2, [0, H], 'k-')
ax_uz.plot(uz[0]*ms2myr,  z_uz[0],  'k:', lw=lw, label='$t=%.0f$\,yr'%time[0]) # first solution
ax_uz.plot(uz[-1]*ms2myr, z_uz[-1], 'k-', lw=lw, label='$t=%.0f$\,yr'%time[nt]) # last solution
setyaxis(ax_uz)
ax_uz.legend(loc=3, **legkwargs)
ax_uz.set_xlabel('$u_z$ (\SI{}{\metre\per\year})')
ax_uz.grid()

### AUX ###
xlims = time[[0,nt]]

ax_H.plot(time, H, 'k.-')
ax_H.set_yticks(np.arange(0,H[0]+1,5))
ax_H.set_yticks(np.arange(0,H[0]+1,1), minor=True)
ax_H.set_ylim([H[nt]*0.995,H[0]])
ax_H.set_ylabel('$H$ (\SI{}{\metre})')
ax_H.set_xlabel('$t$ (\SI{}{\year})')
ax_H.set_xlim(xlims)

ax_m.plot(time, (m/m[0]-1)*100, 'k.-')
ax_m.set_ylabel(r'$m/m_0 - 1$ (\%)')
ax_m.set_xlabel(r'$t$ (\SI{}{\year})')
ymax = 1
ax_m.set_yticks(np.arange(-ymax,ymax+1,0.5))
ax_m.set_yticks(np.arange(-ymax,ymax+1,0.1), minor=True)
ax_m.set_ylim([-ymax,ymax])
ax_m.set_xlim(xlims)

ax_dHdt.semilogy(time[:-1], -np.diff(H)/(dt*s2yr), 'k.-')
ax_dHdt.set_ylabel('$dH/dt$ (\SI{}{\metre\per\year})')
ax_dHdt.set_xlabel('$t$ (\SI{}{\year})')
ax_dHdt.set_ylim([8e-4,2e0])
ax_dHdt.set_xlim(xlims)

ax_p.plot(p[-1]*1e-3, z_p[-1], 'k.-')
ax_p.set_ylabel('$z$ (\SI{}{\metre})')
ax_p.set_xlabel('$p$ (\SI{}{\kilo\pascal})')
setyaxis(ax_p)

# Save figure
if 1: 
    fname = '%s.png'%(expr_names[EXPR])
    print('Saving output to %s'%(fname))
    plt.savefig(fname, bbox_inches='tight', dpi=300)
    
plt.close()

