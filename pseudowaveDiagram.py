
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
  #for atb in atbList:
  for atb in [80]:
    df = phoDF.loc[phoDF.atb==atb]
    atl08canopy = df.loc[df.atl08canopy.astype(bool)]
    atl08ground = df.loc[df.atl08ground.astype(bool)]
    atl08noise = df.loc[df.atl08unclassified.astype(bool)|df.atl08noise.astype(bool)]
    canopy = df.loc[df.canopy]
    ground = df.loc[df.ground]
    noise = df.loc[df.noise]
    import matplotlib.gridspec as gs
    fig = plt.figure(figsize=(8,6))
    Nc = 5
    grid = gs.GridSpec(2, Nc, figure=fig)
    ax00 = fig.add_subplot(grid[0,:Nc//2])
    ax01 = fig.add_subplot(grid[0,Nc//2:Nc-1])
    ax10 = fig.add_subplot(grid[1,:Nc//2])
    ax11 = fig.add_subplot(grid[1,Nc//2:Nc-1])
    #fig, ax = plt.subplots(2,3, figsize=(8,6))
    zMin, zMax, zMean = df.z.min(), df.z.max(), df.zCoG.mean()
    zRange = zMax-zMin
    # Plot ATL08 classified segment
    ax00.plot(df.y, df.h_te_interp, 'k--', lw=1, zorder=0, label='ATL08 ground')
    ax00.scatter(atl08noise.y, atl08noise.z, color=cmap(.2), s=3)
    ax00.scatter(atl08canopy.y, atl08canopy.z, color=cmap(.5), s=3)
    ax00.scatter(atl08ground.y, atl08ground.z, color=cmap(.8), s=3)
    ax00.set_xticks(np.unique(np.round(df.y/50)*50))
    ax00.ticklabel_format(useOffset=False, style='plain')
    ax00.set_xlabel('Northing (m)')
    #ax00.tick_params(axis='both', which='major', labelsize=8)
    ax00.set_ylabel('Height above\nWGS84 ellipsoid (m)')
    # Plot ALS classified segment
    ax01.plot(df.y, df.zCoG, c='k', lw=1, zorder=0, label='ALS ground')
    ax01.scatter(noise.y, noise.z, color=cmap(.2), s=3, label='Noise')
    ax01.scatter(canopy.y, canopy.z, color=cmap(.5), s=3, label='Canopy')
    ax01.scatter(ground.y, ground.z, color=cmap(.8), s=3, label='Ground')
    ax01.set_xticks(np.unique(np.round(df.y/50)*50))
    ax01.ticklabel_format(useOffset=False, style='plain')
    ax01.set_xlabel('Northing (m)')
    #ax01.tick_params(axis='x', which='major', labelsize=8)
    ax01.set_ylabel('Height above\nWGS84 ellipsoid (m)')
    # Plot segment relative to local height
    ax10.axhline(0, c='k', lw=1, zorder=0)
    dat = lambda df: np.sqrt((df.x-df.x.iloc[0])**2. + (df.y-df.y.iloc[0])**2.)
    ax10.scatter(dat(noise), noise.zRel, color=cmap(.2), s=3)
    ax10.scatter(dat(canopy), canopy.zRel, color=cmap(.5), s=3)
    ax10.scatter(dat(ground), ground.zRel, color=cmap(.8), s=3)
    ax10.set_xticks(np.linspace(0, ls, 3))
    ax10.set_xlim(0, ls)
    ax10.ticklabel_format(useOffset=False, style='plain')
    ax10.set_xlabel('Distance along\nsegment (m)')
    ax10.set_ylabel('Height above\nALS ground (m)')
    ax10.set_ylim(np.floor((df.zMin-df.zCoG).min()/5)*5,
                  np.ceil((df.zMax-df.zCoG).max()/5)*5)

    # Plot pseudo waveform
    N, _ = np.histogram(df.zRel, bins=zBinEdge)
    Ng, _ = np.histogram(ground.zRel, bins=zBinEdge)
    Nv, _ = np.histogram(canopy.zRel, bins=zBinEdge)
    ax11.axhline(0, c='k', lw=1, zorder=0)
    ax11.fill_betweenx(zBin, 0, Ng, facecolor=cmap(.8))
    ax11.fill_betweenx(zBin, 0, Nv, facecolor=cmap(.5))
    ax11.set_xlabel('Number of\nphotons')
    ax11.set_xlim(0, np.max([Ng,Nv])+1)
    ax11.set_ylabel('Height above\nALS ground (m)')
    ax11.set_ylim(np.floor((df.zMin-df.zCoG).min()/5)*5,
                     np.ceil((df.zMax-df.zCoG).max()/5)*5)
    for ax in [ax00, ax01, ax10, ax11]:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_aspect((x1-x0)/(y1-y0))
    fig.legend(loc='center right', ncol=1, fancybox=False, scatterpoints=3,
               edgecolor='none')
    fig.tight_layout()
    fn = 'plots/pseudowaveforms/%s.%s.%s.%.0f.withATL08.pdf' % (region, date, gtx, atb)
    fig.savefig(fn)
    plt.close()
