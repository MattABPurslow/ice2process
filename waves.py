import os, pdb
import h5py
import numpy as np
import pandas as pd
import xarray as xr

from scipy.interpolate import interp2d
from scipy.io import loadmat

from pyproj import Transformer
from params import params
from geoid import geoid
from multiprocessing import Pool
import subprocess


class waves(object):
  def __init__(self, region, date, gtx):
    self.region = region
    self.p = params(region)
    self.geoid = geoid(self.p)
    self.date = date
    self.gtx = gtx
    self.coordDir = self.p.coordDir
    self.coordList = np.sort([os.path.join(self.p.coordDir, f) \
                              for f in os.listdir(self.p.coordDir) \
                              if ((self.date in f) & (self.gtx in f))])
    self.tileList = os.path.join(self.p.tileDir,
                                 '.'.join([self.date, self.gtx, 'tiles']))
    self.waveOut = os.path.join(self.p.waveDir, 
                                '.'.join([self.date, self.gtx, 'waves']))
    self.zFile = os.path.join(self.p.zDir,
                              '.'.join([self.date, self.gtx, 'z.pkl']))
    self.pulseFile = os.path.join(self.p.pulseDir,
                                  '.'.join([self.date, self.gtx, 'pulse']))
  
  def createWaveFile(self, i):
    coordFile = self.coordList[i]
    outFile = coordFile.replace('coords', 'waves') \
                       .replace('footprints', 'waves')
    command = ["gediRat",
               "-inList %s" % self.tileList,
               "-output %s" % outFile, 
               "-listCoord %s" % coordFile,
               "-readPulse %s" % self.pulseFile,
               "-fSigma %.3f" % self.p.σf,
               #"-maxScanAng %.3f" % 5,
               "-ground",
               "-hdf"]
    print(' '.join(command))
    subprocess.run(' '.join(command), shell=True)
    return outFile
  
  def createWaves(self):
    with Pool(self.p.CPUs) as p:
      self.waveFiles = p.map(self.createWaveFile,
                             range(self.coordList.shape[0]))
  
  def readWave(self, waveFile):
    return h5py.File(waveFile, 'r')
  
  def combineWaves(self):
    self.data = {}
    nWaves = 0
    for waveFile in self.waveFiles:
      if os.path.exists(waveFile):
        print('Reading', waveFile)
        h5 = self.readWave(waveFile)
        if self.data == {}:
          for k in list(h5):
            if k=='WAVEID':
              waveID = np.copy(h5[k])
              for i in range(waveID.shape[0]):
                for j in range(waveID.shape[1]-1):
                  if waveID[i,j]==b'':
                    waveID[i,j] = b'0'
              self.data[k] = waveID
            else:
              self.data[k] = np.copy(h5[k])
        else:
          for k in list(h5):
            if k=='WAVEID':
              waveID = np.copy(h5[k])
              for i in range(waveID.shape[0]):
                for j in range(waveID.shape[1]-1):
                  if waveID[i,j]==b'':
                    waveID[i,j] = b'0'
              diff = waveID.shape[1] - self.data[k][:].shape[1]
              if diff > 0:
                self.data[k] = np.array([np.append([b'0']*diff, wID) for wID in self.data[k][:]])
                self.data[k] = np.append(self.data[k], waveID, axis=0)
              elif diff < 0:
                waveID = np.array([np.append([b'0']*abs(diff), wID) for wID in waveID])
                self.data[k] = np.append(self.data[k], waveID, axis=0)
              else:
                self.data[k] = np.append(self.data[k], waveID, axis=0)
            else:
              self.data[k] = np.append(self.data[k], np.copy(h5[k]), axis=0)
        nWaves += h5['NWAVES'][0]
        h5.close()
      else:
        print(waveFile, 'missing')
    self.data['FSIGMA'] = [self.data['FSIGMA'][0]]
    self.data['IDLENGTH'] = [self.data['IDLENGTH'][0]]
    self.data['NBINS'] = [self.data['NBINS'][0]]
    self.data['PRES'] = [self.data['PRES'][0]]
    self.data['NPBINS'] = [self.data['NPBINS'][0]]
    self.data['PULSE'] = self.data['PULSE'][:self.data['NPBINS'][0]]
    self.data['NTYPEWAVES'] = [self.data['NTYPEWAVES'][0]]
    self.data['NWAVES'] = [nWaves]
    self.data['PSIGMA'] = [self.data['PSIGMA'][0]]
    h5 = h5py.File(self.waveOut, 'w')
    self.geoid.geoidCorrect(self.data['LON0'][:], self.data['LAT0'][:])
    for k in ['Z0', 'ZN']:
      self.data[k] += self.geoid.correction
    for k in self.data.keys():
      h5[k] = self.data[k]
    self.heightBounds(h5)
    h5.close()
    for waveFile in self.waveFiles:
      if os.path.exists(waveFile):
        os.remove(waveFile)  

  def heightBounds(self, h5, thresh=.0001):
    print('Calculating height bounds')
    count = np.sum(h5['RXWAVECOUNT'][:], axis=1)
    full = h5['RXWAVECOUNT'][:] / count[:, np.newaxis]
    ground = h5['GRWAVECOUNT'][:] / count[:, np.newaxis]
    canopy = full - ground
    cumsum = np.cumsum(full, axis=1)
    z = np.array([np.linspace(h5['Z0'][i], h5['ZN'][i], h5['NBINS'][0]) \
                  for i in range(h5['NWAVES'][0])])
    mask = (cumsum >= thresh) & (cumsum <= 1.-thresh)
    zMin = np.array([zi[maski].min() for zi, maski in zip(z, mask)])
    zMax = np.array([zi[maski].max() for zi, maski in zip(z, mask)])
    groundMask = mask & (ground > canopy)
    zG = np.copy(zMin); zCoG = np.copy(zMin)
    for i, m in zip(range(zG.shape[0]), groundMask):
      if m.sum()>0:
        zG[i] = z[i,m].max()
    gMask = (ground.max(axis=1) > 0.)
    zCoG[gMask] = np.average(z[gMask], weights=ground[gMask], axis=1)
    self.z = pd.DataFrame()
    waveID = [int(''.join(wID.astype(str))) for wID in np.copy(h5['WAVEID'][:])]
    self.z['waveID'] = waveID
    self.z['zMin'] = zMin
    self.z['zMax'] = zMax
    self.z['zG'] = zG
    self.z['zCoG'] = zCoG
    self.z['cv_ALS'] = 100. * np.sum(canopy, axis=1) / np.sum(full, axis=1)
    self.z = self.z.set_index('waveID')
  
  def writeZ(self):
    print('Saving height bound file')
    self.z.to_pickle(self.zFile)

if __name__=='__main__':
  w = waves('sonoma', '20190101', 'gt1l')
  w.createWaves()
  w.combineWaves()
  w.writeZ()
  print(w.z.zCoG)
