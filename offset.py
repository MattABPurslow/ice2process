import os, sys, pdb
sys.path.append('/home/s1503751/src/PhoREAL/source_code')

import h5py
import numpy as np
import pandas as pd

from pyproj import Transformer
from scipy.interpolate import interp2d

from getAtlMeasuredSwath_auto import getAtlMeasuredSwath
from getAtlTruthSwath_auto import getAtlTruthSwath
from getMeasurementError_auto import offsetsStruct, getMeasurementError
from icesatIO import writeLas, headerStruct, getTruthFilePaths, getTruthHeaders
from icesatReader import get_atl03_struct
from icesatUtils import (ismember, getRaster,
                         getIntersection2d, getCoordRotRev)

from params import params
from geoid import geoid

class offset(object):
  def __init__(self, region):
    self.region = region
    self.p = params(self.region)
    self.geoid = geoid(self.p)
    for atl03File in self.p.atl03List[36:]:
      self.atl03File = atl03File
      self.atl08File = self.atl03File.replace('atl03','atl08') \
                                     .replace('ATL03','ATL08')
      self.date = self.atl03File.split('/')[-1][6:14]
      for gtx in self.p.gtxList:
        self.gtx = gtx
        print('Loading points', self.date, self.gtx)
        self.offsetFile = os.path.join(self.p.offsetDir,
                                       '.'.join([self.date, self.gtx, 'xyz']))
        if np.isin('heights', list(h5py.File(self.atl03File, 'r')[gtx])):
          Npho = h5py.File(self.atl03File, 'r')['/'.join([gtx,'heights','lon_ph'])][:].shape[0]
        else:
          Npho = 0
        if (Npho > 0)&(os.path.exists(self.atl08File)):
          self.data = get_atl03_struct(self.atl03File, self.gtx).df
          self.getTiles()
          self.getIntersected()
          if self.tiles.intersected.sum() > 0:
            try:
              self.getOffset()
            except:
              self.zOffset = -999.9
            if self.zOffset > -999.:
              self.writeOffset()
  
  def getTiles(self):
    ''' Read tile list and check file integrity '''
    print('Checking available ALS tiles')
    self.tiles = pd.read_csv(self.p.tileList, sep=' ',
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
    ## Trim ICESat-2 to ALS
    mask = (self.data.northing > self.tiles.yMin.min() - tol) & \
           (self.data.northing < self.tiles.yMax.max() + tol)
    self.data = self.data.loc[mask]
    # Identify intersected
    self.tiles['intersected'] = False
    mask = (self.tiles.xMax >= self.data.easting.min() - tol) & \
           (self.tiles.xMin <= self.data.easting.max() + tol) & \
           (self.tiles.yMax >= self.data.northing.min() - tol) & \
           (self.tiles.yMin <= self.data.northing.max() + tol)
    self.tiles = self.tiles.loc[mask]
    for tile in self.tiles.index:
      x0, y0, x1, y1 = self.tiles.loc[tile,
                                      ['xMin', 'yMin', 'xMax', 'yMax']].values
      mask = (self.data.easting >= x0 - tol) & (self.data.easting <= x1 + tol) & \
             (self.data.northing >= y0 - tol) & (self.data.northing <= y1 + tol)
      if mask.sum() > 0:
        self.tiles.at[tile, 'intersected'] = True
    self.tiles = self.tiles.loc[self.tiles.intersected]
  
  def getOffset(self):
    atl03FilePath = self.atl03File
    atl08FilePath = self.atl08File
    outFilePath = '~'
    gtNum = self.gtx
    trimInfo = 'auto'
    createAtl03LasFile = createAtl03KmlFile = createAtl08KmlFile = False
    createAtl03CsvFile = createAtl08CsvFile = False
    atlMeasuredOut = getAtlMeasuredSwath(atl03FilePath, atl08FilePath,
                                         outFilePath, gtNum, trimInfo,
                                         createAtl03LasFile,
                                         createAtl03KmlFile,
                                         createAtl08KmlFile,
                                         createAtl03CsvFile,
                                         createAtl08CsvFile)
    atl03Data, atl08Data, rotationData = atlMeasuredOut
    truthFileType = '.las'
    truthFilePaths = self.tiles.filename.values
    truthHeaderDF = pd.DataFrame({'epsg':'epsg:'+self.p.epsg,
                                  'xmin':self.tiles.xMin.values,
                                  'xmax':self.tiles.xMax.values,
                                  'ymin':self.tiles.yMin.values,
                                  'ymax':self.tiles.yMax.values,
                                  'fileName':self.tiles.filename.values})
    buffer = 50
    outFilePath = '~'
    createTruthFile = False
    useExistingTruth = False
    atlTruthData = getAtlTruthSwath(atl03Data, rotationData,
                                    truthHeaderDF, truthFilePaths,
                                    buffer, outFilePath, createTruthFile,
                                    truthFileType, useExistingTruth,
                                    logFileID=False)
    if atlTruthData.z.shape[0] > 0:
      self.geoid.geoidCorrect(atlTruthData.lon.T[0], atlTruthData.lat.T[0],
                              lonlat=True)
      atlTruthData.z += np.array([self.geoid.correction]).T
      refHeightType = 'HAE'
      outFilePath = '~'
      useMeasSigConf = False
      filterData = 2 
      offsets = offsetsStruct(np.array([-50,50]), np.array([-50,50]),
                              np.array([8, 4, 2, 1]), False, 0)
      createMeasCorrFile = False
      makePlots = False
      showPlots = False
      atlCorrections = getMeasurementError(atl03Data, atlTruthData,
                                           refHeightType, rotationData,
                                           outFilePath, useMeasSigConf,
                                           filterData, offsets,
                                           createMeasCorrFile, makePlots,
                                           showPlots, logFileID=False)
      self.xOffset = np.float64(atlCorrections.easting)
      self.yOffset = np.float64(atlCorrections.northing)
      self.zOffset = np.float64(atlCorrections.z)
    else:
      self.xOffset = -999.
      self.yOffset = -999.
      self.zOffset = -999.
  
  def writeOffset(self):
    xyz = np.array([self.xOffset, self.yOffset, self.zOffset])
    np.savetxt(self.offsetFile, xyz, fmt='%.3f')

def getRegion():
  import argparse
  args = argparse.ArgumentParser()
  args.add_argument('-r', '--region', dest='region', type=str, default='sonoma',
                    help='Region of desired track')
  args = args.parse_args()
  return args.region

if __name__=='__main__':
  o = offset(getRegion())
