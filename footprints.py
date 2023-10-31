import os, pdb
import numpy as np
import pandas as pd
from params import params
from photons import photons

def getTiles(self):
    ''' Read tile list and check file integrity '''
    print('Checking available ALS tiles')
    self.tiles = pd.read_csv(self.tileList, sep=' ',
                             names=['filename',
                                    'xMin', 'yMin', 'zMin',
                                    'xMax', 'yMax', 'zMax'])
    ## Remove missing tiles
    self.tiles['exists'] = [os.path.isfile(f) for f in self.tiles.filename]
    self.tiles = self.tiles.where(self.tiles.exists.astype(bool))\
                           .dropna().drop(columns='exists')
    ## Remove corrupted files
    self.tiles['m'] = [os.path.getsize(f) > 1024 for f in self.tiles.filename]
    self.tiles = self.tiles.where(self.tiles.m.astype(bool))\
                           .dropna().drop(columns='m')

def getIntersected(self, tol=50):
    ''' Identify tiles containing ICESat-2 footprints '''
    print('Identifying intersected ALS tiles')
    self.tol = tol
    self.tiles['intersected'] = False
    mask = (self.tiles.xMax >= self.data.x.min() - tol) & \
           (self.tiles.xMin <= self.data.x.max() + tol) & \
           (self.tiles.yMax >= self.data.y.min() - tol) & \
           (self.tiles.yMin <= self.data.y.max() + tol)
    self.tiles = self.tiles.loc[mask]
    for tile in self.tiles.index:
      x0, y0, x1, y1 = self.tiles.loc[tile,
                                      ['xMin', 'yMin', 'xMax', 'yMax']].values
      mask = (self.data.x >= x0 - tol) & (self.data.x <= x1 + tol) & \
             (self.data.y >= y0 - tol) & (self.data.y <= y1 + tol)
      if mask.sum() > 0:
        self.tiles.at[tile, 'intersected'] = True


class footprints(object):
  def __init__(self, region, date, gtx):
    self.region = region
    self.date = date
    self.gtx = gtx
    p = params(region)
    self.p = p
    self.tileList = p.tileList
    self.offsetFile = os.path.join(p.offsetDir, '.'.join([self.date, self.gtx, 'xyz']))
    print('Reading photon data')
    pho = photons(region, date, gtx)
    self.pklFile, self.pho = pho.pklFile, pho.data
    self.getFootprints()
    getTiles(self)
    getIntersected(self)
    self.cropAtl03()
    self.writeTileList()
    self.Ns = self.data.shape[0]
    self.maxPer = p.maxPer
    self.nBatch = int(self.Ns/self.maxPer)
    self.outRoot = os.path.join(p.coordDir, '.'.join([self.date, self.gtx, 'footprints']))
    if (self.Ns % self.maxPer) != 0:
      self.nBatch += 1
    self.writeFootprints()
  
  def getFootprints(self):
    ''' Locate footprints on ground '''
    print('Locating footprints')
    self.data = self.pho[['x','y']].where(self.pho.atl08ground==True).groupby('waveID').mean()
    self.data = self.data.interpolate().dropna()
    self.pho = self.pho.loc[(self.pho.index >= self.data.index.min()) & \
                            (self.pho.index <= self.data.index.max())]
    wavePho = self.pho.where(self.pho.lat==-9999.).dropna().index
    self.pho.loc[wavePho]['x'] = self.data.loc[wavePho]['x']
    self.pho.loc[wavePho]['y'] = self.data.loc[wavePho]['y']
    self.Ns = self.data.shape[0]
  
  def cropAtl03(self):
    ''' Crop ATL03 data to ALS '''
    print('Cropping ATL03 to ALS bounds')
    yMin = self.tiles.where(self.tiles.intersected).yMin.min(skipna=True)
    yMax = self.tiles.where(self.tiles.intersected).yMax.max(skipna=True)
    self.pho = self.pho.loc[(self.pho.y.values>=yMin-self.tol) & \
                            (self.pho.y.values<=yMax+self.tol)]
    self.pho.to_pickle(self.pklFile)
    self.data = self.data.loc[(self.data.y.values>=yMin-self.tol) & \
                              (self.data.y.values<=yMax+self.tol)]
  
  def writeTileList(self):
    ''' Write intersected ALS tiles to list '''
    print('Write intersected ALS tiles to list')
    tileOut = os.path.join(self.p.tileDir,
                           '.'.join([self.date, self.gtx, 'tiles']))
    df = self.tiles.filename.where(self.tiles.intersected).dropna()
    df.to_csv(tileOut, index=False, header=False)
  
  def writeFootprints(self):
    ''' Write ICESat-2 footprints to lists containing up to maxPer footprints'''
    print('Writing footprint coordinates to list')
    pad = len(str(self.nBatch))
    self.lists = []
    self.data['waveID'] = self.data.index
    for i in range(self.nBatch):
      print(i+1, '/', self.nBatch, '\r')
      df = self.data.iloc[i*self.maxPer:(i+1)*self.maxPer]
      fn = '.'.join([self.outRoot, str(i).zfill(pad)])
      df[['x', 'y', 'waveID']].to_csv(fn, index=False, header=False, sep=' ')
      self.lists.append(fn)

if __name__=='__main__':
  coords = footprints('sonoma', '20190101', 'gt1l')
  
