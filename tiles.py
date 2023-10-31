import os, pdb
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from laspy.file import File

from params import params

class tiles(object):
  def __init__(self, region):
    self.p = params(region)
    self.alsDir = self.p.alsDir
    self.tileList = self.p.tileList
    self.identifyTiles()
    self.getBounds()
    self.writeList()
    self.writeShape()
  
  def identifyTiles(self):
    """ Identify useful ALS tiles (i.e. size > 1kb) """
    ## List tiles
    tiles = np.sort(os.listdir(self.alsDir))
    ## Include directory in strings
    tiles = np.array([os.path.join(self.alsDir, t) for t in tiles])
    ## Only keep LAS files
    tiles = np.array([t for t in tiles if '.las' in t.lower()])
    ## Check file sizes
    tiles = np.array([t for t in tiles if os.path.getsize(t) > 1023])
    ## Save to dataframe
    self.tiles = pd.DataFrame({'filename':tiles})
  
  def getBounds(self):
    """ Identify ALS tile bounds """
    ## Create arrays to store bounds
    xMin, yMin, zMin, xMax, yMax, zMax = [np.full(self.tiles.shape[0], np.nan)
                                          for i in range(6)]
    ## Loop over tiles
    for i in range(self.tiles.shape[0]):
      print(self.tiles.filename.iloc[i])
      ## Saving bounds to arrays
      try:
        las = File(self.tiles.filename.iloc[i])
        xMin[i], yMin[i], zMin[i] = las.header.min
        xMax[i], yMax[i], zMax[i] = las.header.max
        las.close()
      except:
        print(':(')
    ## Add bounds columns to dataframe
    self.tiles['xMin'], self.tiles['yMin'], self.tiles['zMin'] = xMin,yMin,zMin
    self.tiles['xMax'], self.tiles['yMax'], self.tiles['zMax'] = xMax,yMax,zMax
  
  def writeShape(self):
    boxes = [Polygon([(xMin, yMin), (xMin, yMax), (xMax, yMax), (xMax, yMin)]) \
             for xMin, yMin, xMax, yMax in zip(self.tiles.xMin, \
                                               self.tiles.yMin, \
                                               self.tiles.xMax, \
                                               self.tiles.yMax)]
    gdf = gpd.GeoDataFrame(self.tiles, geometry=boxes, crs='EPSG:%s' % self.p.epsg)
    gdf['loc'] = self.p.region
    #gdf.dissolve('loc').to_crs(epsg=4326).to_file(self.tileList+'.shp')
    gdf.to_file(self.tileList+'.shp')
  
  def writeList(self):
    """ Write filenames with bounds to file"""
    ## Write to space-delimited text file
    self.tiles.to_csv(self.tileList, sep=' ', index=False, header=False,
                      float_format='%.2f')

if __name__=='__main__':
  import argparse
  args = argparse.ArgumentParser()
  args.add_argument('-r', '--region', dest='region', type=str, default='sonoma',
                    help='Region name')
  args = args.parse_args()
  tiles(args.region.lower())
