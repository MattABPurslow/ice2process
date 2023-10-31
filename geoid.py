import pdb
import numpy as np
import xarray as xr
from pyproj import Transformer
from scipy.interpolate import interp2d, griddata, SmoothBivariateSpline
from scipy.io import loadmat

class geoid(object):
  def __init__(self, params):
    self.p = params

  def getGeoid(self):
    if self.p.geoidFile:
      if self.p.geoidFile[-3:] == 'mat':
        geoid = xr.Dataset()
        xyz = loadmat(self.p.geoidFile)
        geoid['x'] = ('x', xyz['geoid']['lons'][0][0][0])
        geoid['y'] = ('y', xyz['geoid']['lats'][0][0][0])
        geoid['z'] = (('y', 'x'), xyz['geoid']['geoidalHeight'][0][0])
        geoid = geoid.z
      else:
        geoid = xr.open_rasterio(self.p.geoidFile)
        if geoid.x.max() > 180.:
          geoid = geoid.assign_coords({'x':geoid.x.values-360.})
      geoid = geoid.sortby('y')
      self.geoid = geoid
    else:
      self.geoid = False
  
  def geoidCorrect(self, x, y, lonlat=False):
    if lonlat==False:
      t = Transformer.from_crs('EPSG:%s' % self.p.epsg, 'EPSG:4326',
                               always_xy=True)
      lon, lat = t.transform(x, y)
    else:
      lon, lat = x, y
    self.getGeoid()
    if self.p.geoidFile:
      self.geoid = self.geoid.sel(x=slice(lon.min()-.1, lon.max()+.1),
                                  y=slice(lat.min()-.1, lat.max()+.1))
      xg, yg = np.meshgrid(self.geoid.x, self.geoid.y)
      gFunc = SmoothBivariateSpline(xg.flatten(), yg.flatten(), self.geoid.values.flatten())
      self.correction = gFunc(lon, lat, grid=False)
      #gFunc = interp2d(self.geoid.x, self.geoid.y, self.geoid)
      #self.correction = np.full(lon.shape[0], 0.)
      #for i in range(lon.shape[0]):
      #    self.correction[i]] = gFunc(lon[i], lat[i])
    else:
      self.correction = np.full(lon.shape, 0.)

if __name__=='__main__':
  from params import params
  p = params('deju')
  g = geoid(p)
  g.getGeoid()
