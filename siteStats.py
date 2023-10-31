import os, pdb
import numpy as np
import pandas as pd
import h5py
import xarray as xr

import matplotlib.pyplot as plt
from params import params

def readWaves(region, date, gtx):
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
  ds['zMin'] = (('waveID'), ds.z.where(ds.full>0).min(dim='zIndex'))
  zCoG = np.copy(ds.zMin)
  gMask = ds.ground.sum(dim='zIndex') > 0
  zCoG[gMask] = np.average(ds.z[gMask], weights=ds.ground[gMask], axis=1)
  ds['zCoG'] = (('waveID'), zCoG)
  z = np.ma.masked_array(ds.z, ds.canopy<=0)
  ds['ch'] = (('waveID'), z.max(axis=1) - ds.zCoG)
  ds['ch'] = ds.ch.fillna(0)
  ds['cv'] = 100. * ds.canopy.sum(dim='zIndex') / ds.full.sum(dim='zIndex')
  return ds

def getRegion():
  import argparse
  args = argparse.ArgumentParser()
  args.add_argument('-r', '--region', dest='region', type=str, default='sonoma',
                    help='Region of desired track')
  args = args.parse_args()
  return args.region

regions = ['deju', 'ornl', 'sodankyla', 'sonoma', 'wref']
for region in regions:
  p = params(region)
  rates = pd.read_csv(p.rateDir+'/%s.rates' % region)
  cv = []
  ch = []
  cnt = 1
  for date, gtx in zip(rates.date.astype(str), rates.gtx):
    print(region, cnt, '/', len(rates.date), end='\r')
    wavesDF = readWaves(region, date, gtx)
    for i in range(wavesDF.waveID.shape[0]):
      cv.append(wavesDF.cv.values[i])
      ch.append(wavesDF.ch.values[i])
    cnt += 1
  
  print(region)
  print('cv (%)', np.mean(cv))
  print('ch (m)', np.mean(ch))
