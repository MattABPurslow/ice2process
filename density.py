import argparse
from progress.bar import ChargingBar
import numpy as np
import pandas as pd
import laspy
from params import params

class density(object):
  def __init__(self, region):
    """ Read in ALS tiles for region and calculate point density. """
    self.p = params(region)
    self.tiles = pd.read_csv(self.p.tileList, sep=' ', header=None,
                             names=['filename', 'xMin', 'yMin', 'zMin',
                                                'xMax', 'yMax', 'zMax'])
    self.getDensity()
  
  def getDensity(self):
    """ Calculate average density for all tiles. """
    self.N = np.empty(0)
    bar = ChargingBar('Reading tiles', max=self.tiles.shape[0])
    ## Load las files into memory
    for lasFile in self.tiles.filename:
      self.readLas(lasFile)
      bar.next()
    bar.finish()
    print(self.p.region)
    print('m²:', self.N.shape[0])
    print('pts/m²:', self.N.mean())

  def readLas(self, lasFile):
    """ Read x, y, z positions of points in las file and add count for each m**2 to array """
    las = laspy.file.File(lasFile, mode='r')
    df = pd.DataFrame()
    df['x'] = np.round(las.x)
    df['y'] = np.round(las.y)
    df['z'] = las.z
    las.close()
    ## Append count for each m**2 to array
    self.N = np.append(self.N, df.groupby(['x', 'y']).count().z.values).astype(int).flatten()

def getRegion():
  """ Read in region from command line"""
  args = argparse.ArgumentParser()
  args.add_argument('--region', dest='region', type=str, default='sonoma',
                    help='Region name')
  args = args.parse_args()
  return args.region

if __name__=='__main__':
  region = getRegion()
  density(region)
    
