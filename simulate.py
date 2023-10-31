import os, pdb
import subprocess
import numpy as np
import pandas as pd
from scipy.constants import speed_of_light
import h5py
from params import params

class simulate(object):
  def __init__(self, region, date, gtx, simNoise=True, reclassAll=True):
    self.p = params(region)
    self.date = date
    self.gtx = gtx
    self.reclassAll = reclassAll
    self.waveFile = os.path.join(self.p.waveDir,
                                 '.'.join([date, gtx, 'waves']))
    self.rateFile = os.path.join(self.p.rateDir,
                                 '.'.join([date, gtx, 'rates']))
    self.simFile = os.path.join(self.p.simDir,
                                 '.'.join([date, gtx, 'sim']))
    self.rateFile = os.path.join(self.p.rateDir,
                                 '.'.join([region, 'rates']))
    rates = pd.read_csv(self.rateFile)
    rates = rates.loc[(rates.date.astype(str)==date)&(rates.gtx==gtx)]
    self.ρg, self.ρv, self.ρn = rates[['ρg', 'ρv', 'ρn']].values[0]
    if simNoise == False:
      self.ρn = 0
    self.run()
    self.read()
    self.getZ()
    self.getClass()
    self.writePickle()

  def run(self):
    print('Simulating %s %s' % (self.date, self.gtx))
    command = ['gediMetric', ## run gediMetric
               '-input %s' % self.waveFile, ## for this waveFile
               ## create an ICESat-2 track
               '-ground -readHDFgedi -photonCount -photHDF',
               ## with/without noise
               '-noiseMult %.6f' % (self.ρn),
               ## Within a window this tall
               '-photonWind %.0f' % 500,
               ## with these photon rates
               '-nPhotC %.6f' % (self.ρv),
               '-nPhotG %.6f' % (self.ρg),
               ## Seed random sampling
               '-seed %00d' % np.random.randint(low=1, high=32001),
               ## and save it here
               '-outRoot %s' % self.simFile]
    print(' '.join(command))
    self.success = subprocess.run(' '.join(command), shell=True) == 0
    self.simFile += '.pts'
  
  def read(self):
    print('Reading simulation')
    cols = ['x', 'y', 'z', 'minht', 'zGCoG',
            'RH50_ALS', 'RH60_ALS', 'RH75_ALS', 'RH90_ALS', 'RH95_ALS',
            'ch_ALS', 'cv_ALS', 'shotNo', 'photonNo', 'iterationNo',
            'refDEM', 'noiseInt', 'signal', 'ground']
    if self.simFile[-3:] != 'pts':
      self.simFile += '.pts'
    self.data = pd.read_csv(self.simFile, sep=' ', names=cols, skiprows=1)
  
  def getZ(self):
    print('Adding wave bounds to simulation')
    h5 = h5py.File(self.waveFile, 'r')
    waveID = np.array([int(''.join(wave.astype(str))) \
                       for wave in np.copy(h5['WAVEID'][:])])
    self.data['waveID'] = waveID[self.data.shotNo]
    z = pd.read_pickle(os.path.join(self.p.zDir,
                                    '.'.join([self.date, self.gtx, 'z.pkl'])))
    z = z.loc[z.index.duplicated()==False]
    for col in z.columns:
      self.data[col] = z[col].loc[self.data.waveID].values
    self.data['zRel'] = self.data.z - self.data.zCoG
  
  def getClass(self):
    print('Classifying simulated photons')
    self.data['noise'] = ((self.data.z < self.data.zMin) | \
                          (self.data.z > self.data.zMax)).astype(bool)
    if self.reclassAll:
      self.data['ground'] = ((self.data.z >= self.data.zMin) & \
                             (self.data.z <= self.data.zG)).astype(bool)
      self.data['canopy'] = ((self.data.z > self.data.zG) & \
                             (self.data.z <= self.data.zMax)).astype(bool)
    else:
      self.data['canopy'] = self.data.signal & (self.data.ground == False)
      iNoise = (self.data.signal == False)
      self.data.loc[iNoise, 'ground'] = ((self.data.loc[iNoise].z >= self.data.loc[iNoise].zMin) & \
                                         (self.data.loc[iNoise].z <= self.data.loc[iNoise].zG)).astype(bool)
      self.data.loc[iNoise, 'canopy'] = ((self.data.loc[iNoise].z > self.data.loc[iNoise].zG) & \
                                         (self.data.loc[iNoise].z <= self.data.loc[iNoise].zMax)).astype(bool)
  
  def printSimNoise(self):
    Nn = self.data.groupby('waveID').sum().noise
    dn = 500. - (self.data.groupby('waveID').mean().zMax - self.data.groupby('waveID').mean().zMin)
    dt = (dn / speed_of_light) * 10**6 ## one-way travel time in μs
    print(np.mean(Nn / dt))
  
  def printSimRates(self):
    Ng = self.data.groupby('waveID').sum().noise

  def writePickle(self):
    self.data.to_pickle(self.simFile.replace('pts', 'pkl'))
  
if __name__=='__main__':
  sim = simulate('sonoma', '20191001', 'gt3l')

