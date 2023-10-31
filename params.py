import os, pdb
import numpy as np
from matplotlib import rcParams
rcParams['font.family'] = 'Arial'

class params(object):
  def __init__(self, region):
    self.region = region.lower()
    self.σp = 0.1
    self.σf = 11./4.
    self.maxPer = 1000
    self.CPUs = 16
    self.gtxList = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
    self.getRegionSpecific()
    self.makeDirs()
    
  def getRegionSpecific(self):
    self.tileList = 'alsinfo/%s.als.tiles' % self.region
    dataDir = '/exports/csce/datastore/geos/groups/3d_env/data/purslowm/icesat2/%s' % self.region #'/exports/csce/datastore/geos/users/s1503751/icesat2/%s' % self.region
    self.atl03Dir = '/scratch/local/purslowm/icesat-2_v004/%s/atl03' % self.region
    self.atl08Dir = '/scratch/local/purslowm/icesat-2_v004/%s/atl08' % self.region
    if os.path.exists(dataDir)==False:
      os.mkdir(dataDir)
      os.mkdir(self.atl03Dir)
      os.mkdir(self.atl08Dir)
    self.atl03List = [os.path.join(self.atl03Dir, f) \
                      for f in np.sort(os.listdir(self.atl03Dir))]
    self.offsetDir = os.path.join(dataDir, 'offset')
    self.pklDir = os.path.join(dataDir, 'pickle')
    self.pulseDir = os.path.join(dataDir, 'pulse')
    self.ncDir = os.path.join(dataDir, 'nc')
    self.waveDir = os.path.join(dataDir, 'waves')
    self.coordDir = os.path.join(dataDir, 'coords')
    self.rateDir = os.path.join(dataDir, 'rates')
    self.segDir = os.path.join(dataDir, 'segments')
    self.simDir = os.path.join(dataDir, 'sim')
    self.zDir = os.path.join(dataDir, 'z')
    self.tileDir = os.path.join(dataDir, 'tiles')
    for d in [self.offsetDir, self.pklDir, self.pulseDir, self.ncDir, self.waveDir, self.rateDir, \
              self.coordDir, self.simDir, self.zDir, self.tileDir, self.segDir]:
      if os.path.exists(d)==False:
        os.mkdir(d)
    self.dates = [f.split('.')[0] for f in np.sort(os.listdir(self.pklDir))]
    self.gtxs = [f.split('.')[1] for f in np.sort(os.listdir(self.pklDir))]
    alsAllDir = '/exports/csce/datastore/geos/groups/3d_env/data/ALS/raw/'
    if self.region=='davos':
      self.epsg = '32632'
      self.geoidFile = None
      self.latBounds = [46.5, 47.]
      self.alsDir = os.path.join(alsAllDir, 'slf_engadine/EUCHEA20170829')
      self.ρvρg = 1.
    elif self.region=='bart':
      self.epsg = '32619'
      self.geoidFile = 'geoid/g2012a_conus.gtx'
      self.latBounds = [44., 45.]
      self.alsDir = os.path.join(alsAllDir, 'neon_bart2018/ALS_ground')
      self.ρvρg = 1.
    elif self.region=='deju':
      self.epsg = '32606'
      self.geoidFile = 'geoid/g2012a_ak.tif' 
      self.latBounds = [63.75, 64.]
      self.alsDir = os.path.join(alsAllDir, 'neon_deju2019/ALS_ground')
      self.ρvρg = 1.3
    elif self.region=='laselva':
      self.epsg = '32616'
      self.geoidFile = None
      self.latBounds = [5, 15]
      self.alsDir = os.path.join(alsAllDir, 'nasa_laselva_2009/ALS_ground')
      self.ρvρg = 1.
    elif self.region=='ornl':
      self.epsg = '32616'
      self.geoidFile = 'geoid/g2012a_conus.gtx'
      self.latBounds = [35., 37.]
      self.alsDir = os.path.join(alsAllDir, 'neon_ornl/ALS_ground')
      self.ρvρg = .75
    elif self.region=='sodankyla':
      self.epsg = '32635'
      self.geoidFile = 'geoid/geoidFin2005N00_latlon.mat'
      self.latBounds = [67., 68.]
      self.alsDir = os.path.join(alsAllDir, 'nls_sodankyla')
      self.ρvρg = 1.
    elif self.region=='sonoma':
      self.epsg = '32610'
      self.geoidFile = 'geoid/g2012a_conus.gtx'
      self.latBounds = [38., 39.5]
      self.alsDir = os.path.join(alsAllDir, 'nasa_sonoma_whole/ALS_ground')
      self.ρvρg = .7
    elif self.region=='wref':
      self.epsg = '32610'
      self.geoidFile = 'geoid/g2012a_conus.gtx'
      self.latBounds = [45., 47.]
      self.alsDir = os.path.join(alsAllDir, 'neon_wref/ALS_ground')
      self.ρvρg = 1.1
    else:
      ValueError('Unknown region!')

  def makeDirs(self):
    dirs = [self.alsDir, self.offsetDir, self.pklDir, self.ncDir,\
            self.waveDir, self.coordDir, self.simDir, self.tileDir]
    for d in dirs:
      if os.path.exists(d)==False:
        os.mkdir(d)
