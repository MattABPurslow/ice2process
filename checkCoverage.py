import os, pdb
from progress.bar import ChargingBar
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from rasterio import merge
from rasterio.transform import Affine
from rasterio.crs import CRS
import laspy
import matplotlib.pyplot as plt
from params import params

region = 'sonoma'

p = params(region)
epsg = np.int64(p.epsg)
alsList = np.sort([os.path.join(p.alsDir, f) for f in  os.listdir(p.alsDir)])
alsList = alsList[[f[-4:]=='.las' for f in alsList]]

def getTiff(idx):
  tiffDir = '/exports/csce/datastore/geos/users/s1503751/%s' % region
  if os.path.exists(tiffDir)==False:
    os.mkdir(tiffDir)
  return os.path.join(tiffDir, '%s.%06d.tif' % (region, idx))

def writeTiff(x, y, z, idx, epsg):
  X, Y, Z = x, y[::-1], z.T[::-1]
  res = 1
  transform = Affine.translation(X[0] + res/2, Y[0] - res/2) * Affine.scale(res, -res)
  fname = getTiff(idx)
  raster = rasterio.open(fname, 'w',
                         driver='GTiff',
                         height=Z.shape[0],
                         width=Z.shape[1],
                         count=1,
                         dtype=Z.dtype,
                         nodata=99,
                         crs=CRS.from_epsg(epsg),
                         transform=transform)
  raster.write(Z, 1)
  raster.close()

Ntiles = alsList.shape[0]
bar = ChargingBar('Creating rasters for %d tiles' % Ntiles, max=Ntiles)
for idx in range(Ntiles):
  alsFile = alsList[idx]
  try:
    las = laspy.file.File(alsFile)
  except:
    las = pd.DataFrame({'x':[]})
  if las.x.shape[0] > 0:
    df = pd.DataFrame({'x':np.int64(las.x),
                       'y':np.int64(las.y),
                       'sa': np.abs(las.scan_angle_rank)})
    xMin, xMax = df.x.min(), df.x.max()+1
    yMin, yMax = df.y.min(), df.y.max()+1
    x, y = np.arange(xMin, xMax, 1), np.arange(yMin, yMax, 1)
    sa = np.full((x.shape[0], y.shape[0]), 99).astype(np.int8)
    df['i'], df['j'] = df.x - xMin, df.y - yMin
    df = df.groupby(['i', 'j']).min().sa
    idxAdd, saAdd = df.index.values, df.values
    for i, s in zip(idxAdd, saAdd):
      sa[i] = s
    writeTiff(x, y, sa, idx, epsg)
    las.close()
  bar.next()
bar.finish()

bar = ChargingBar('Merging rasters', max=Ntiles-1)
outTiff = getTiff(0).replace('000000', 'merged')
os.system('mv %s %s' % (getTiff(0), outTiff))
for idx in range(1, Ntiles):
  if os.path.exists(getTiff(idx)):
    r1 = rasterio.open(outTiff)
    r2 = rasterio.open(getTiff(idx))
    dest, trans = merge.merge([r1, r2], method='min')
    out_meta = r1.meta.copy()
    out_meta.update({'driver': 'GTiff',
                     'height': dest.shape[1],
                     'width': dest.shape[2],
                     'transform': trans})
    r1.close(); r2.close()
    with rasterio.open(outTiff, 'w', **out_meta) as out:
      out.write(dest); out.close()
    os.system('rm %s' % getTiff(idx))
  bar.next()
bar.finish()

ds = xr.open_rasterio(outTiff).sel(band=1)
ds = ds.where(ds != 99, np.nan)
ds = ds[::-1]
ds = ds.interpolate_na(dim='x', method='linear')
ds = ds.interpolate_na(dim='y', method='linear')
side = 5000
for i in range((ds.x.shape[0]//side)):
  for j in range((ds.y.shape[0]//side)):
    dsPlot = ds.isel(x=slice(i*side, (i+1)*side),
                     y=slice(j*side, (j+1)*side))
    fig = plt.figure(figsize=(16,9)); ax = fig.gca()
    cf = ax.contourf(dsPlot.x, dsPlot.y, dsPlot, np.arange(0, 46, 5))
    fig.colorbar(cf, label='Minimum scan angle from vertical (°)')
    plotID = 'x.%03d.y.%03d' % (i,j)
    fig.savefig(outTiff.replace('tif', plotID+'.pdf'))
    plt.close()
