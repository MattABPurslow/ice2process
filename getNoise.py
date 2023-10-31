import os, pdb
import numpy as np
import pandas as pd
from params import params
from photons import photons

import matplotlib.pyplot as plt

region = 'sonoma'
p = params(region)
outDir = p.pklDir.replace('pickle', 'noise')
if os.path.exists(outDir)==False:
  os.mkdir(outDir)

for date, gtx in zip(p.dates, p.gtxs):
  print(date, gtx)
  outFile = os.path.join(outDir, '.'.join([date, gtx, 'noise.csv']))
  pho = photons(region, date, gtx)
  pho.getZ()
  pho.getClass(zBuff=5)
  sumGroup = pho.data.groupby('waveID').sum()
  meanGroup = pho.data.groupby('waveID').mean()
  minGroup = pho.data.groupby('waveID').min()
  df = pd.DataFrame()
  df['waveID'] = sumGroup.index.values
  df = df.set_index('waveID')
  df['Nph'] = (sumGroup.noise + sumGroup.ground + sumGroup.canopy).values
  df['Nn_als_buffer'] = sumGroup.noise.values
  df['tlmTop'] = meanGroup.tlmTop.values
  df['tlmBase'] = meanGroup.tlmBase.values
  df['dw'] = ((meanGroup.zMax - meanGroup.zMin) + 10).values
  df['dn'] = ((df.tlmTop - df.tlmBase) - df.dw).values
  df['delta_time'] = minGroup.delta_time
  df['night'] = minGroup.night.astype(bool)
  df['segID'] = minGroup.segment_id
  segmask = df.groupby('segID').sum().Nph > 0
  df['segmask'] = segmask.reindex(df.segID).values
  pho.getClass(zBuff=0)
  sumGroup = pho.data.groupby('waveID').sum()
  df['Nn_als_nobuffer'] = sumGroup.noise.values
  df['Nn_atl08'] = (sumGroup.atl08noise + sumGroup.atl08unclassified).astype(int).values
  outCols = ['delta_time', 'night', 'Nn_als_buffer', 'Nn_als_nobuffer', 'Nn_atl08', 'dn']
  df.loc[df.segmask, outCols].to_csv(outFile)
