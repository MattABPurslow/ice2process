import os, pdb

import numpy as np
import pandas as pd
import laspy

from geoid import geoid
from params import params
from photons import photons

import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def readWaves(region, date, gtx):
  import h5py; import xarray as xr
  h5 = h5py.File(os.path.join(params(region).waveDir, '.'.join([date, gtx, 'waves'])), 'r')
  ds = xr.Dataset()
  ds['waveID'] = (('waveID'), [int(''.join(wave.astype(str))) for wave in np.copy(h5['WAVEID'][:])])
  ds['z'] = (('waveID', 'zIndex'), np.array([np.linspace(h5['Z0'][i], h5['ZN'][i], h5['NBINS'][0])\
                                             for i in range(h5['NWAVES'][0])]))
  count = np.sum(h5['RXWAVECOUNT'][:], axis=1)
  ds['full'] = (('waveID', 'zIndex'), h5['RXWAVECOUNT'][:] / count[:, np.newaxis])
  ds['ground'] = (('waveID', 'zIndex'), h5['GRWAVECOUNT'][:] / count[:, np.newaxis])
  ds['canopy'] = ds.full - ds.ground
  ds['x'] = (('waveID'), h5['LON0'][:])
  ds['y'] = (('waveID'), h5['LAT0'][:])
  return ds

def findFootprint(x, y, p):
  σf = 11./4.
  tiles = pd.read_csv(p.tileList, sep=' ',
                      names=['filename',
                             'xMin', 'yMin', 'zMin',
                             'xMax', 'yMax', 'zMax'])
  mask = (tiles.xMin < x + 2*σf) & (tiles.xMax > x - 2*σf) & \
         (tiles.yMin < y + 2*σf) & (tiles.yMax > y - 2*σf)
  tiles = tiles.loc[mask]
  g = geoid(p)
  g.geoidCorrect(np.array([x]),np.array([y]))
  df = pd.DataFrame()
  for t in tiles.filename:
    dft = pd.DataFrame()
    las = laspy.file.File(t)
    mask = (las.x >= x - 2*σf) & (las.x <= x + 2*σf) & \
           (las.y >= y - 2*σf) & (las.y <= y + 2*σf)
    dft['x'] = las.x[mask]
    dft['y'] = las.y[mask]
    dft['z'] = las.z[mask] + g.correction
    dft['canopy'] = las.classification[mask] == 1
    dft['ground'] = las.classification[mask] == 2
    df = df.append(dft)
  return df

if __name__=='__main__':
  region = 'sonoma'
  date = '20181030'
  gtx = 'gt1l'
  zBuff = 5
  p = params(region)
  ## Load photons
  phoDF = photons(region, date, gtx)
  phoDF.getZ()
  phoDF.getClass()
  phoDF = phoDF.data
  ## Load waves
  wavesDF = readWaves(region, date, gtx)
  waveID = phoDF.groupby('waveID').min().index[(phoDF.groupby('waveID').sum()[['canopy']].sum(axis=1)>=2)&(phoDF.groupby('waveID').sum()[['ground']].sum(axis=1))].values[0]
  pho = phoDF.loc[waveID]
  waves = wavesDF.sel(waveID=waveID)
  waves.to_dataframe().to_pickle('plots/waveform.pkl')
  ## Load ALS pointcloud
  als = findFootprint(waves.x.values, waves.y.values, p)
  ## Get plot parameters
  αf = 0.3; αw = 0.8
  colors = {'a': cm.gist_earth(0.2),
            'b': cm.gist_earth(0.9),
            'g': cm.gist_earth(0.7),
            'v': cm.gist_earth(0.5)}
  phoClass = []
  for i in range(pho.shape[0]):
    p = pho.iloc[i]
    if p.z < p.zMin:
      phoClass.append('b')
    elif (p.z >= p.zMin) & (p.z <= p.zG):
      phoClass.append('g')
    elif (p.z > p.zG) & (p.z <= p.zMax):
      phoClass.append('v')
    else:
      phoClass.append('a')
  pho['color'] = [colors[c] for c in phoClass]
  xMax = np.array([waves.canopy.max(), waves.ground.max()]).max()*100.
  tMin = pho.tlmBase.mean(); tMax = pho.tlmTop.mean()
  zMin = pho.zMin.mean(); zMax = pho.zMax.mean(); zG = pho.zG.mean()
  yMin = np.floor(zMin/10)*10 - 10; yMax = np.ceil(zMax/10)*10 + 10
  ## Create figure
  fig = plt.figure(figsize=(8,4)); ax = fig.gca()
  ## Plot classification bands
  ax.fill_between([-xMax, 2*xMax], zMax+zBuff, tMax, color=colors['a'], alpha=αf, edgecolor='none')
  ax.fill_between([-xMax, 2*xMax], zG, zMax, color=colors['v'], alpha=αf, edgecolor='none')
  ax.fill_between([-xMax, 2*xMax], zMin, zG, color=colors['g'], alpha=αf, edgecolor='none')
  ax.fill_between([-xMax, 2*xMax], tMin, zMin-zBuff, color=colors['b'], alpha=αf, edgecolor='none')
  ## Plot ALS
  axALS = ax.twiny()
  axALS.scatter(als.loc[als.ground].y, als.loc[als.ground].z, c=colors['g'], s=1)
  axALS.scatter(als.loc[als.canopy].y, als.loc[als.canopy].z, c=colors['v'], s=1)
  axALS.set_xlim((als.y.min()//5)*5, ((als.y.min()//5)*5)+50)
  axALS.set_ylim(yMin, yMax)
  axALS.xaxis.tick_bottom()
  axALS.set_xticks(np.arange(5+(als.y.min()//5)*5, ((als.y.max()//5)*5)+1, 5))
  axALS.text(waves.y, yMin-5., 'Northing (m)', ha='center', va='center')
  axALS.ticklabel_format(useOffset=False, style='plain')
  ## Plot waveforms
  wMask = (waves.z>=zMin) & (waves.z<=zMax)
  ax.fill_betweenx(waves.z[wMask], 0, waves.canopy[wMask]*100.,
                   color=colors['v'], alpha=αw)
  ax.fill_betweenx(waves.z[wMask], 0, waves.ground[wMask]*100.,
                   color=colors['g'], alpha=αw)
  ## Plot ICESat-2 photons
  ax.scatter([1.25*xMax]*pho.shape[0], pho.z, s=20, c=pho.color)
  ## Label and format
  ax.axvline(0, color='k', lw=1, ls=':')
  ax.axvline(xMax, color='k', lw=1, ls=':')
  ax.text(-.5*xMax, zMax+2.5, 'ALS', ha='center', va='center', weight='bold')
  ax.text(1.25*xMax, zMax+2.5, 'ICESat-2', ha='center', va='center', weight='bold')
  ax.text(.5*xMax, zMax+2.5, 'Simulated waveform', ha='center', va='center', weight='bold')
  ax.text(1.9*xMax, yMax-2.5, 'Above', ha='right', va='center', weight='bold')
  ax.text(1.9*xMax, zG+(zMax-zG)/2., 'Canopy', ha='right', va='center', weight='bold')
  ax.text(1.9*xMax, zG-(zG-zMin)/2., 'Ground', ha='right', va='center', weight='bold')
  ax.text(1.9*xMax, yMin+2.5, 'Below', ha='right', va='center', weight='bold')
  ax.text(1.9*xMax, zMax+(zBuff/2.), 'Buffer', ha='right', va='center', style='italic')
  ax.text(1.9*xMax, zMin-(zBuff/2.), 'Buffer', ha='right', va='center', style='italic')
  ax.set_yticks(np.arange(yMin, yMax+1, 5))
  ax.set_xticks(np.arange(0, xMax, 1))
  ax.set_xlim(-xMax, 2*xMax); ax.set_ylim(yMin, yMax)
  ax.text(xMax/2., yMin-5., 'Waveform intensity (%)', ha='center', va='center')
  ax.set_ylabel('Height above WGS84 ellipsoid (m)')
  fig.tight_layout(); fig.show()
