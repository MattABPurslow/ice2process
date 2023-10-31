
import os, pdb

import numpy as np

from params import params
from photons import photons

import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt
from matplotlib.cm import gist_earth as cmap
from matplotlib.ticker import ScalarFormatter

if __name__=='__main__':
  region = 'wref'
  date = '20200818'
  gtx = 'gt1l'
  ls = 100.
  p = params(region)
  ## Load photons
  phoDF = photons(region, date, gtx)
  phoDF.getZ()
  phoDF.getClass()
  phoDF.getBins(ls)
  phoDF = phoDF.data
  ## Identify interesting segments (slope + significant canopy)
  atbDF = phoDF.groupby('atb')
  zRange = atbDF.max().zCoG-atbDF.min().zCoG
  slopeATB = zRange.loc[zRange > 50].index.values
  cv = phoDF.groupby('atb').mean().cv_ALS
  cvATB = cv.loc[(cv>50)].index.values
  atbList = slopeATB[np.isin(slopeATB, cvATB)]
  print(atbList)
  ## Create vertical bins
  zMin, zMax, zStep = -100., 100., 1.
  zBinEdge = np.arange(zMin-zStep/2., zMax+zStep, zStep)
  zBin = zBinEdge[:-1]+zStep/2.
  ## Plot interesting segments
  for atb in atbList:
    df = phoDF.loc[phoDF.atb==atb]
    canopy = df.loc[df.canopy]
    ground = df.loc[df.ground]
    noise = df.loc[df.noise]
    fig, ax = plt.subplots(1,3, figsize=(8,3))
    zMin, zMax, zMean = df.z.min(), df.z.max(), df.zCoG.mean()
    zRange = zMax-zMin
    # Plot raw segment
    ax[0].plot(df.y, df.zCoG, c='k', lw=1, zorder=0, label='ALS ground')
    ax[0].scatter(noise.y, noise.z, color=cmap(.2), s=3, label='Noise')
    ax[0].scatter(canopy.y, canopy.z, color=cmap(.5), s=3, label='Canopy')
    ax[0].scatter(ground.y, ground.z, color=cmap(.8), s=3, label='Ground')
    ax[0].set_xticks(np.unique(np.round(df.y/50)*50))
    ax[0].ticklabel_format(useOffset=False, style='plain')
    ax[0].set_xlabel('Northing (m)')
    ax[0].set_ylabel('Height above WGS84 ellipsoid (m)')
    #ax[0].set_ylim(zMean-zRange/2., zMean+zRange/2.)
    # Plot segment relative to local height
    ax[1].axhline(0, c='k', lw=1, zorder=0)
    dat = lambda df: np.sqrt((df.x-df.x.iloc[0])**2. + (df.y-df.y.iloc[0])**2.)
    ax[1].scatter(dat(noise), noise.zRel, color=cmap(.2), s=3)
    ax[1].scatter(dat(canopy), canopy.zRel, color=cmap(.5), s=3)
    ax[1].scatter(dat(ground), ground.zRel, color=cmap(.8), s=3)
    ax[1].set_xticks(np.linspace(0, ls, 3)); ax[1].set_xlim(0, ls)
    ax[1].ticklabel_format(useOffset=False, style='plain')
    ax[1].set_xlabel('Distance along segment (m)')
    ax[1].set_ylabel('Height above ALS ground (m)')
    ax[1].set_ylim(np.floor(df.zRel.min()/50)*50, np.ceil(df.zRel.max()/50)*50)
    # Plot pseudo waveform
    N, _ = np.histogram(df.zRel, bins=zBinEdge)
    Ng, _ = np.histogram(ground.zRel, bins=zBinEdge)
    Nv, _ = np.histogram(canopy.zRel, bins=zBinEdge)
    ax[2].axhline(0, c='k', lw=1, zorder=0)
    ax[2].fill_betweenx(zBin, 0, Ng, facecolor=cmap(.8))
    ax[2].fill_betweenx(zBin, 0, Nv, facecolor=cmap(.5))
    ax[2].set_xlabel('Number of photons'); ax[2].set_xlim(0, np.max([Ng,Nv])+1)
    ax[2].set_ylabel('Height above ALS ground (m)')
    ax[2].set_ylim(np.floor((df.zMin-df.zCoG).min()/5)*5,
                   np.ceil((df.zMax-df.zCoG).max()/5)*5)
    for i in range(3):
      x0, x1 = ax[i].get_xlim(); y0, y1 = ax[i].get_ylim()
      ax[i].set_aspect((x1-x0)/(y1-y0))
    fig.legend(loc='upper center', ncol=4, fancybox=False, scatterpoints=3,
               edgecolor='none')
    fig.tight_layout()
    fn = 'plots/pseudowaveforms/%s.%s.%s.%.0f.withATL08.pdf' % (region, date, gtx, atb)
    fig.savefig(fn)
    plt.close()
