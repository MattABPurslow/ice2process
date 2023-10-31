import os, pdb
import h5py
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

class tracks(object):
  def __init__(self,region):
    self.region = region
    atl03Dir = '/exports/csce/datastore/geos/users/s1503751/icesat-2_v003/' + \
               '%s/atl03' % self.region
    self.atl03List = np.array([os.path.join(atl03Dir, f)
                               for f in np.sort(os.listdir(atl03Dir))
                               if (f[-3:]=='.h5')&(f[:5]=='ATL03')])
    self.gtxs = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
    self.alsShape()
    self.atl03()
    self.writeList()
  
  def alsShape(self):
    self.alsDF = gpd.read_file('alsinfo/%s.als.tiles.shp' % self.region)
    if self.alsDF.geometry.unary_union.type=='MultiPolygon':
      lon, lat = np.empty(0), np.empty(0)
      for poly in self.alsDF.geometry.unary_union:
        lon = np.append(lon, poly.exterior.coords.xy[0])
        lat = np.append(lat, poly.exterior.coords.xy[1]) 
    elif self.alsDF.geometry.unary_union.type=='Polygon':
      lon, lat = self.alsDF.geometry.unary_union.exterior.coords.xy
    else:
      print('Unknown shape')
      exit()
    self.lon0, self.lat0, self.lon1, self.lat1 = np.min(lon), np.min(lat), np.max(lon), np.max(lat)
  
  def atl03(self):
    self.useful = []
    for atl03File in self.atl03List:
      print(atl03File)
      self.atl03File = atl03File
      try:
        atl03 = h5py.File(atl03File, 'r')
      except:
        print('%s not complete' % atl03File)
        exit()
      self.atlDF = pd.DataFrame()
      for gtx in self.gtxs:
        df = pd.DataFrame({'lon': np.copy(atl03['%s/heights/lon_ph' % gtx][:]),
                           'lat': np.copy(atl03['%s/heights/lat_ph' % gtx][:]),
                           'gtx': gtx})
        self.atlDF = self.atlDF.append(df)
      atl03.close()
      self.checkUse()

  def checkUse(self):
    self.atlDF = self.atlDF.loc[(self.atlDF.lon >= self.lon0) & \
                                (self.atlDF.lon <= self.lon1) & \
                                (self.atlDF.lat >= self.lat0) & \
                                (self.atlDF.lat <= self.lat1)]
    if self.atlDF.shape[0] > 0:
      fig = plt.figure(); ax = fig.gca()
      self.alsDF.plot(ax=ax)
      ax.scatter(self.atlDF.lon, self.atlDF.lat, c='r')
      ax.set_title(self.atl03File)
      pltDir = 'tracks/%s/' % self.region
      pltFile = self.atl03File.split('/')[-1].replace('.h5', '.png')
      plt.savefig(os.path.join(pltDir, pltFile))
      plt.close()
      self.useful.append(True) 
    else:
      os.system('rm %s %s' % (self.atl03File,
                              self.atl03File.replace('atl03','atl08') \
                                            .replace('ATL03','ATL08')))
      self.useful.append(False)
  
  def writeList(self):
    np.savetxt('atl03info/%s.atl03.list' % self.region,
               self.atl03List[self.useful], fmt='%s')

def getRegion():
  import argparse
  args = argparse.ArgumentParser()
  args.add_argument('--region', '-r', dest='region', type=str,
                    default='sonoma', help='Region name')
  return args.parse_args().region

if __name__=='__main__':
  tracks(getRegion())
