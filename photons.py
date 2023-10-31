import os, pdb
import numpy as np
import pandas as pd
from params import params

class photons(object):
  def __init__(self, region, date, gtx):
    self.region = region
    self.date = date
    self.gtx = gtx
    self.p = params(self.region)
    pklDir = params(region).pklDir
    self.pklFile = os.path.join(pklDir, '.'.join([date, gtx, 'photons.pkl']))
    self.data = pd.read_pickle(self.pklFile)
    self.getClean()
  
  def getZ(self):
    z = pd.read_pickle(os.path.join(self.p.zDir,
                                    '.'.join([self.date, self.gtx, 'z.pkl'])))
    z = z.loc[z.index.duplicated()==False]
    for col in z.columns:
      self.data[col] = z[col]
    self.data['zRel'] = self.data.z - self.data.zCoG
    self.data.loc[self.data.dummy, 'zRel']=-9999.
    self.data = self.data.loc[((self.data.tlmTop>self.data.zMin)&
                               (self.data.tlmBase<self.data.zMax))|
                               self.data.dummy]
    wIDmin = self.data.loc[self.data.z>-9999].index.min()
    wIDmax = self.data.loc[self.data.z>-9999].index.max()
    self.data = self.data.loc[wIDmin:wIDmax]
    self.data = self.data.replace(-9999., np.nan)
    self.data = self.data.interpolate()
    self.data.loc[self.data.dummy, 'z'] = np.nan
  
  def getClass(self, zBuff=0):
    self.data['noise'] = ((self.data.dummy==False) &
                          ((self.data.z < self.data.zMin - zBuff) | \
                           (self.data.z > self.data.zMax + zBuff))).astype(bool)
    self.data['ground'] = ((self.data.z >= self.data.zMin) & \
                           (self.data.z <= self.data.zG)).astype(bool)
    self.data['canopy'] = ((self.data.z > self.data.zG) & \
                           (self.data.z <= self.data.zMax)).astype(bool)
  
  def getClean(self):
    lc = [1,2,3,4,5,6,7,8,9,10,12,14,16]
    self.data['clean'] = (self.data.watermask==0).astype(bool) & \
                         np.isin(self.data.landcover, lc).astype(bool) & \
                         (self.data.snowcover==1).astype(bool)# & \
                         #(self.data.cloudmask==1).astype(bool)# & \
                         #np.isin(self.data.mswmask, [-1, 0]).astype(bool)
    if 'saturated' in self.data.columns:
      self.data['clean'] = self.data.clean & (self.data.saturated==0)
    self.data['dummy'] = (self.data.z == -9999.).astype(bool)
  
  def getBins(self, ls):
    self.y0 = self.data.y.min()
    self.x0 = self.data.loc[self.data.y==self.y0].x.mean()
    alongdist = ((self.data.x-self.x0)**2+(self.data.y-self.y0)**2)**.5
    self.data['atb'] = np.floor(alongdist / ls).astype(int)

  def writePickle(self):
    self.data.to_pickle(self.pklFile)

def commands():
  import argparse
  args = argparse.ArgumentParser()
  args.add_argument('-r', '--region', dest='region', type=str, default='sonoma',
                    help='Region of desired track')
  args.add_argument('-d', '--date', dest='date', type=str, default='20190101',
                    help='Date of desired track')
  args.add_argument('-g', '--gtx', dest='gtx', type=str, default='gt1l',
                    help='Ground track ID of desired track')
  args = args.parse_args()
  return args.region, args.date, args.gtx

if __name__=='__main__':
  region, date, gtx = commands()
  pho = photons(region, date, gtx)
  pho.getZ()
  pho.getClass()
