import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.cm import bwr as cmap

from params import params

region = 'deju'
p = params(region)

realFile = os.path.join(p.segDir.replace('segments', 'metrics'), '.'.join([region, 'real', 'pkl']))
simsFile = os.path.join(p.segDir.replace('segments', 'metrics'), '.'.join([region, 'sims', 'pkl']))

real = pd.read_pickle(realFile)
sims = pd.read_pickle(simsFile)

cvFunc = lambda ρgρv, Ng, Nv: 100.*(1. / (1. + ρvρg*(Ng/Nv)))
ρvρgMin, ρvρgMax, ρvρgStep = 0.5, 1.5, 0.01
ρvρgBins = np.arange(ρvρgMin, ρvρgMax+ρvρgStep/2., ρvρgStep)
real['Nph'] = real.Ng + real.Nv
Nmin, Nmax, Nstep = 0, 250, 10
NphBins = np.arange(Nmin-Nstep/2, Nmax+Nstep, Nstep)
err = np.full((ρvρgBins.shape[0], NphBins.shape[0]), np.nan)
ρvρgErr = np.full(ρvρgBins.shape[0], np.nan)
for i in range(ρvρgBins.shape[0]):
  ρvρg = ρvρgBins[i]
  ρvρgErr[i] = np.nanmean(cvFunc(ρvρg, real.Ng, real.Nv)-real.cv_ALS)
  for j in range(NphBins.shape[0]):
    Nph = NphBins[j]
    df = real.loc[(real.Nph>=Nph)&(real.Nph<Nph+Nstep)]
    if df.shape[0] > 0:
      err[i,j] = np.nanmean(cvFunc(ρvρg, df.Ng, df.Nv)-df.cv_ALS)

x, y = np.meshgrid(ρvρgBins, NphBins)
plt.gca().set_facecolor("black")
plt.pcolormesh(ρvρgBins-ρvρgStep/2., NphBins, err.T,
               cmap=cmap, vmin=-10, vmax=10)
plt.xlabel('ρv/ρg'); plt.ylabel('Number of signal photons')
plt.colorbar(label='Canopy cover error (%, obs. ICESat-2 - ALS)')
plt.show()

plt.figure()
plt.plot(ρvρgBins, ρvρgErr)
plt.show()
