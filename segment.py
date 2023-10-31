import os, pdb
from progress.bar import ChargingBar

import numpy as np
import pandas as pd
from scipy.constants import speed_of_light

from params import params
from photons import photons

class segments(object):
  def __init__(self, region, ls=100., zBuff=5., read=False):
    self.region = region
    self.p = params(self.region)
    self.ls = ls
    self.zBuff = zBuff
    self.getSegFile()
    if read:
      self.data = pd.read_pickle(self.segFile)
    else:
      self.createSegs()
 
  def createSegs(self):
    self.data = pd.DataFrame()
    bar = ChargingBar('Load segments', max=len(self.p.dates))
    for date, gtx in zip(self.p.dates, self.p.gtxs):
     if os.path.exists(os.path.join(self.p.zDir,'.'.join([date,gtx,'z.pkl']))):
      self.date = date
      self.gtx = gtx
      self.getPhotons()
      self.getSegs()
      self.data = self.data.append(self.seg)
      bar.next()
    bar.finish()
    self.date = self.gtx = self.pho = self.seg = None
    self.writePickle()
  
  def getSegFile(self):
    fnameParams = [self.region, str(int(self.ls))+'m', 'segs', 'pkl']
    self.segFile = os.path.join(self.p.segDir, '.'.join(fnameParams))
  
  def writePickle(self):
    self.data.to_pickle(self.segFile)
  
  def getPhotons(self):
    pho = photons(self.region, self.date, self.gtx)
    pho.getZ()
    pho.getClass(zBuff=self.zBuff)
    pho.getBins(self.ls)
    self.pho = pho.data.replace(-9999., np.nan)
  
  def getMeanNoise(self):
    Nn = self.pho.groupby('waveID').sum().noise
    group = self.pho.groupby('waveID').mean()
    dn = (group.tlmTop - group.zMax) + (group.zMin - group.tlmBase) - 2.*self.zBuff
    dt = (dn / speed_of_light) * 10**6
    ρ̅n_m = np.mean(Nn/dn)
    ρ̅n_s = np.mean(Nn/dt)
    return ρ̅n_m, ρ̅n_s
 
  def getSegs(self):
    sumGroup = self.pho.groupby('atb').sum()
    meanGroup = self.pho.groupby('waveID').mean().groupby('atb').mean()
    atb = sumGroup.index.values[np.isin(sumGroup.index.values,
                                        meanGroup.index.values)]
    sumGroup = sumGroup.loc[atb]
    meanGroup = meanGroup.loc[atb]
    self.seg = pd.DataFrame({'atb':atb})
    self.seg['ls'] = self.ls
    self.seg['date'] = self.date
    self.seg['gtx'] = self.gtx
    self.seg['Ns'] = [self.pho.loc[self.pho.atb==atbi].index.unique().shape[0] \
                      for atbi in atb]
    self.seg['Nn'] = sumGroup.noise.values
    self.seg['Ng'] = sumGroup.ground.values
    self.seg['Nv'] = sumGroup.canopy.values
    self.seg['dn'] = ((meanGroup.tlmTop - meanGroup.zMax) + \
                      (meanGroup.zMin - meanGroup.tlmBase)).values - 2.*self.zBuff
    self.seg['bhr'] = meanGroup.bhr
    self.seg['atl08noiseRate'] = meanGroup.bcr / meanGroup.bhr
    self.seg['dg'] = (meanGroup.zG - meanGroup.zMin).values
    self.seg['dv'] = (meanGroup.zMax - meanGroup.zG).values
    self.seg['ρ̅n_m'], self.seg['ρ̅n_s'] = self.getMeanNoise()
    self.seg['cv_ALS'] = meanGroup.cv_ALS.values
    self.seg['strong'] = meanGroup.strong.values.astype(bool)
    self.seg['night'] = meanGroup.night.values.astype(bool)
    self.seg['clean'] = (meanGroup.clean.values > .95).astype(bool)

  def writeSegs(self):
    self.data.to_pickle(self.segFile)

def getRegion():
  import argparse
  args = argparse.ArgumentParser()
  args.add_argument('-r', '--region', dest='region', type=str, default='sonoma',
                    help='Region of desired track')
  args = args.parse_args()
  return args.region

if __name__=='__main__':
  region = getRegion()
  segs = segments(region)
