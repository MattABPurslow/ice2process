import os, pdb
import h5py
import numpy as np
import matplotlib.pyplot as plt

from scipy.constants import speed_of_light

from params import params

class pulse(object):
  def __init__(self, region):
    self.p = params(region)
    for atl03File in self.p.atl03List:
      self.atl03File = atl03File
      self.date = self.atl03File.split('/')[-1][6:14]
      self.readATL03()
      for gtx in self.p.gtxList:
        self.gtx = gtx
        print('Running', self.date, self.gtx)
        self.getPulse()
        self.writePulse()
      self.atl03.close()
  
  def readATL03(self):
    self.atl03 = h5py.File(self.atl03File, 'r')
    self.validSpot = self.atl03['/ancillary_data/tep/tep_valid_spot']
    self.tpe = {}
    for gtx, spot in zip(self.p.gtxList, self.validSpot):
      if spot==1:
        self.tpe[gtx] = 'pce1_spot1'
      else:
        self.tpe[gtx] = 'pce2_spot3'
    
  def getPulse(self):
    group = '/'.join(['atlas_impulse_response', self.tpe[self.gtx], 'tep_histogram'])
    self.tep_hist = np.copy(self.atl03[group+'/tep_hist'][:])
    self.tep_time = np.copy(self.atl03[group+'/tep_hist_time'][:])
    iMin = np.argmax(self.tep_hist).min()
    while self.tep_hist[iMin] > 1e-4:
      iMin -= 1
    iMax = np.argmax(self.tep_hist).min()
    while self.tep_hist[iMax] > 1e-4:
      iMax += 1
    self.tep_hist = self.tep_hist[iMin:iMax+1]
    self.tep_time = self.tep_time[iMin:iMax+1] -  self.tep_time[iMin]
    self.tep_dist = self.tep_time * speed_of_light
  
  def getFWHM(self):
    halfMax = self.tep_hist.max() / 2.
    mask = self.tep_hist > halfMax
    minT = self.tep_time[mask].min() * 10.**9.
    maxT = self.tep_time[mask].max() * 10.**9.
    print('FWHM (ns) = ', maxT-minT)
    minD = self.tep_dist[mask].min()
    maxD = self.tep_dist[mask].max()
    print('FWHM (m) = ', maxD-minD)
    mean = np.average(self.tep_time, weights=self.tep_hist)
    σp = np.average((self.tep_time-mean)**2, weights=self.tep_hist)**.5
    print('σp (ns) = ', σp * 10.**9.)
    print('σp (m) = ', σp * speed_of_light)
  
  def writePulse(self):
    pulseFile = os.path.join(self.p.pulseDir,
                             '.'.join([self.date, self.gtx, 'pulse']))
    np.savetxt(pulseFile, np.array([self.tep_dist, self.tep_hist]).T, fmt='%.9f')
  
  def plotPulse(self):
    plt.plot(self.tep_dist, self.tep_hist)
    plt.xlabel('Range (m)')
    plt.ylabel('Normalised count')
    plt.savefig('pulses/'+'.'.join([self.date, self.gtx, 'pulse', 'png']))
    plt.close()

def getRegion():
  import argparse
  args = argparse.ArgumentParser()
  args.add_argument('-r', '--region', dest='region', type=str, default='sonoma',
                    help='Region of desired track')
  args = args.parse_args()
  return args.region

if __name__=='__main__':
  pulse(getRegion())
