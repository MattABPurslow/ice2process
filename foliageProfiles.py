import os, pdb
import h5py

import numpy as np
import pandas as pd
import xarray as xr

from params import params
from geoid import geoid

region = 'sonoma'
waveDir = '/exports/csce/datastore/geos/users/s1503751/icesat2/sonoma/waves'
waveList = [os.path.join(waveDir, w) for w in np.sort(os.listdir(waveDir))]
geoid = geoid(params(region))

atl03List = np.array([f.split('/')[-1] for f in params(region).atl03List])
dates = np.array([f[6:14] for f in atl03List])

## Read wave file
for waveFile in waveList[:1]:
  print(waveFile)
  wave = h5py.File(waveFile, 'r')
  date, gtx, _ = waveFile.split('/')[-1].split('.')
  atl03File = str(atl03List[dates==date])
  ## Read photon file
  phoFile = os.path.join(waveDir.replace('waves', 'pickle'), \
                         '.'.join([date, gtx, 'photons', 'pkl']))
  pho = pd.read_pickle(phoFile)
  ## location and label
  lat = np.copy(wave['LAT0'])
  lon = np.copy(wave['LON0'])
  geoid.geoidCorrect(lon, lat)
  dg = geoid.correction
  wID = np.array([int(''.join(w.astype(str)))
                  for w in np.copy(wave['WAVEID'])])
  ## height info
  z0 = np.copy(wave['Z0']) + geoid.correction
  zN = np.copy(wave['ZN']) + geoid.correction
  zInd = wave['NBINS'][0]
  z = np.linspace(z0, zN, zInd).T
  ## wave info
  count = np.sum(wave['RXWAVECOUNT'][:], axis=1)
  full = np.copy(wave['RXWAVECOUNT'][:] / count[:, np.newaxis])
  ground = np.copy(wave['GRWAVECOUNT'][:] / count[:, np.newaxis])
  canopy = full - ground
  ## ground info
  pho.zATL08_local[pho.zATL08_local<-9998.] = np.nan
  zATL08 = pho.groupby('waveID').zATL08_local.mean().reindex(wID)
  zATL08 = zATL08.interpolate(limit_direction='both').values
  zG = np.copy(wave['ZG']) + geoid.correction
  zRel = np.array([np.round((z[i] - zATL08[i])*2.)/2. for i in range(wID.shape[0])])
  ## bin data
  zBin = np.arange(zRel.min(), zRel.max()+.5, .5)
  zBinIndex = np.digitize(zRel, zBin) - 1
  gFrac = np.full((wID.shape[0], zBin.shape[0]), 0.)
  cFrac = np.full((wID.shape[0], zBin.shape[0]), 0.)
  for i in range(wID.shape[0]):
    idx = zBinIndex[i,:]
    gFrac[i,idx] += ground[i,:]
    cFrac[i,idx] += canopy[i,:]
  ## Store in dataset
  ds = h5py.File(os.path.join(waveDir.replace('waves', 'nc'), \
                              '.'.join([date, gtx, 'canopy.h5'])), 'w')
  ds.attrs['atl03file'] = atl03File
  ds.attrs['date'] = date
  ds.attrs['gtx'] = gtx
  ds.attrs['epsg'] = params(region).epsg
  ds['waveID'] = wID.astype(np.int64)
  ds['x'] = lon
  ds['y'] = lat
  ds['z'] = zBin
  ds['canopy'] = cFrac
  ds['ground'] = gFrac
  ds.close()
