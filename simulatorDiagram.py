import os, pdb

import numpy as np
import pandas as pd
import laspy
from scipy import spatial

from params import params
from photons import photons
from geoid import geoid

import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.gridspec import GridSpec

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
  gMask = (ds.ground.max(dim='waveID') > 0.)
  zCoG = np.full(ds.waveID.shape[0], np.nan)
  zCoG[gMask] = np.average(ds.z.values[gMask],
                           weights=ds.ground.values[gMask], axis=1)
  ds['zCoG'] = (('waveID'), zCoG)
  ds['x'] = (('waveID'), h5['LON0'][:])
  ds['y'] = (('waveID'), h5['LAT0'][:])
  ds['cv'] = (('waveID'), 100. * np.sum(canopy, axis=1) / np.sum(full, axis=1))
  ds['c_elev'] = ds
  return ds

def cylinder(x0, y0, z0, z1, r):
  z = np.arange(z0, z1, .01)
  θ = np.linspace(0, 2.*np.pi, 360)
  θg, zg = np.meshgrid(θ, z)
  xg = r*np.cos(θg)+x0
  yg = r*np.sin(θg)+y0
  return xg, yg, zg

if __name__=='__main__':
  region = 'sonoma'
  date = '20181030'
  gtx = 'gt1l'
  zBuff = 5
  colors = {'a': cm.gist_earth(0.2),
            'b': cm.gist_earth(0.9),
            'g': cm.gist_earth(0.7),
            'v': cm.gist_earth(0.5)}
  αf = 0.3; αw = 0.8
  p = params(region)
  geoid = geoid(p)
  ## Load photons
  phoDF = photons(region, date, gtx)
  phoDF.getZ()
  phoDF.getClass()
  phoDF = phoDF.data
  ## Load waves
  wavesDF = readWaves(region, date, gtx)
  geoid.geoidCorrect(wavesDF.x.values, wavesDF.y.values)
  wavesDF['geoid'] = ('waveID', geoid.correction)
  wavesDF['zc'] = wavesDF.z + wavesDF.geoid
  criteria = (phoDF.groupby('waveID').sum()[['canopy']].sum(axis=1)>=2) & \
             (phoDF.groupby('waveID').sum()[['ground']].sum(axis=1)>=1) & \
             ((phoDF.groupby('waveID').mean().zMax - phoDF.groupby('waveID').mean().zMin) > 45.)
  ## Select wave
  for waveID in phoDF.groupby('waveID').min().index[criteria].values[5:6]:
    pho = phoDF.loc[waveID]
    phoClass = []
    for i in range(pho.shape[0]):
      phoi = pho.iloc[i]
      if phoi.z < phoi.zMin:
        phoClass.append('b')
      elif (phoi.z >= phoi.zMin) & (phoi.z <= phoi.zG):
        phoClass.append('g')
      elif (phoi.z > phoi.zG) & (phoi.z <= phoi.zMax):
        phoClass.append('v')
      else:
        phoClass.append('a')
    pho['color'] = [colors[c] for c in phoClass]
    waves = wavesDF.sel(waveID=waveID)
    waves = waves.to_dataframe()
    waves.loc[waves.full<.00001] = np.nan
    ## Identify ALS tile
    alsList = pd.read_csv('alsinfo/%s.als.tiles' % region,  sep=' ',
                          names=['filename', 'xMin', 'yMin', 'zMin',
                                             'xMax', 'yMax', 'zMax']) 
    x, y = waves.x.mean(), waves.y.mean()
    mask = (x-2.*p.σf > alsList.xMin)&(x+2.*p.σf < alsList.xMax)&(y-2.*p.σf > alsList.yMin)&(y+2.*p.σf < alsList.yMax)
    alsTile = alsList.loc[mask].filename.values[0]
    ## Load ALS data
    als = laspy.file.File(alsTile)
    rgbMax = np.max([als.red, als.green, als.blue])
    waveGeoid = waves.geoid.mean(skipna=True)
    als = pd.DataFrame({'x':als.x, 'y':als.y, 'z':als.z + waveGeoid,
                        'ground':als.classification==2,
                        'canopy':als.classification==1,
                        'r':als.red/rgbMax,
                        'g':als.green/rgbMax,
                        'b':als.blue/rgbMax})
    als = als.loc[(als.x>=x-10.*p.σf)&(als.x<=x+10.*p.σf)&(als.y>=y-10.*p.σf)&(als.y<=y+10.*p.σf)]
    ## Interpolate ground
    ground = als.loc[als.ground]
    xArr = np.arange(np.floor(als.x.min()), np.ceil(als.x.max())+1, .1)
    yArr = np.arange(np.floor(als.y.min()), np.ceil(als.y.max())+1, .1)
    xGrid, yGrid = np.meshgrid(xArr, yArr)
    zGrid = np.full(xGrid.shape, np.nan)
    for i in range(zGrid.shape[0]):
      for j in range(zGrid.shape[1]):
        print(i*zGrid.shape[1]+j, '/', zGrid.shape[0]*zGrid.shape[1])
        xij, yij = xGrid[i,j], yGrid[i,j]
        idx = np.argmin((ground.x-xij)**2.+(ground.y-yij)**2.)
        zGrid[i,j] = np.round(ground.iloc[idx].z*100.)/100.
    ## Build beam
    xc, yc, zc = cylinder(x, y, zGrid.min(), als.z.max()+5, 2.*p.σf)
    xyGrid = np.array([xGrid.flatten(), yGrid.flatten()]).T
    xyc = np.array([xc.flatten(), yc.flatten()]).T
    dist, idx = spatial.KDTree(xyGrid).query(xyc)
    zcShape = zc.shape
    zcFlat = zc.flatten()
    zcFlat[zcFlat < zGrid.flatten()[idx]] = np.nan
    zc = zcFlat.reshape(zcShape)
    ## Remove ground outside beam
    zGrid[((xGrid-x)**2.+(yGrid-y)**2.)**.5>=2.*p.σf] = np.nan
    ## Create figure
    fig = plt.figure(figsize=(8,3))
    gs = GridSpec(1, 6, figure=fig)
    ## Plot raw ALS with footprint
    axRGB = fig.add_subplot(gs[:2], projection='3d')
    axRGB.view_init(elev=0, azim=0); axRGB.axis('off')
    axRGB.scatter(als.x, als.y, als.z, s=1, c=np.array([als.r, als.g, als.b]).T)
    axRGB.plot_surface(xc, yc, zc, color='g', alpha=0.25)
    axRGB.plot_surface(xGrid, yGrid, zGrid, color='g', alpha=0.25)
    axRGB.set_xlim(als.x.min(), als.x.max())
    axRGB.set_ylim(als.y.min(), als.y.max())
    axRGB.set_zlim(als.z.min(), als.z.max())
    ## Plot extracted, classified ALS
    als = als.loc[np.sqrt((als.x-x)**2.+(als.y-y)**2.)<=2.*p.σf]
    axALS = fig.add_subplot(gs[2:4], projection='3d')
    axALS.view_init(elev=0, azim=0); axALS.axis('off')
    axALS.scatter(als.loc[als.ground].x, als.loc[als.ground].y, als.loc[als.ground].z, c=colors['g'], s=1)
    axALS.scatter(als.loc[als.canopy].x, als.loc[als.canopy].y, als.loc[als.canopy].z, c=colors['v'], s=1)
    axALS.plot_surface(xc, yc, zc, color='g', alpha=0.1)
    axALS.plot_surface(xGrid, yGrid, zGrid, color='g', alpha=0.1)
    axALS.set_xlim(als.x.min(), als.x.max())
    axALS.set_ylim(als.y.min(), als.y.max())
    axALS.set_zlim(als.z.min(), als.z.max())
    ## Plot simulated waveform
    axWave = fig.add_subplot(gs[-2])
    axWave.fill_betweenx(waves.z, 0, waves.canopy*100.,
                         color=colors['v'], alpha=αw)
    axWave.fill_betweenx(waves.z, 0, waves.ground*100.,
                         color=colors['g'], alpha=αw)
    axWave.set_xlim(0, p.σf)
    axWave.set_aspect(.25)
    axWave.axis('off')
    ## Plot ICESat-2 points
    axICE = fig.add_subplot(gs[-1])
    axICE.scatter([0]*pho.shape[0], pho.z, c=pho.color)
    axICE.set_xlim(-2.*p.σf, 2.*p.σf)
    axICE.set_ylim(axWave.get_ylim())
    axICE.set_aspect('equal')
    axICE.axis('off')
    fig.tight_layout(pad=0., w_pad=0., h_pad=.0); fig.savefig('plots/simulatorDiagramNew.pdf')
  
