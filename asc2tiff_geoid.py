import numpy as np
import rasterio
from rasterio.transform import Affine

ascIn = 'geoid/g2012a00.asc'
tifOut = 'geoid/g2012a_ak.tif'

file = open(ascIn, 'r')
header = file.readline()
zstr = []
while True:
  next = file.readline()
  if not next:
    break
  zstr.append(next)

header = [float(h) for h in header.split(' ') if h != '']
minLat, minLon, dLat, dLon, nLat, nLon, _ = header
nLat, nLon = int(nLat), int(nLon)
latitude = np.arange(minLat, minLat+(dLat*nLat), dLat)
longitude = np.arange(minLon, minLon+(dLon*nLon), dLon)

z = []
for zl in zstr:
  zs = [float(z) for z in zl.split(' ') if z != '']
  for zi in zs:
    z.append(zi)

z = np.array(z)

geoid = np.empty((nLat, nLon))
c = 0
for i in range(nLat):
  for j in range(nLon):
    geoid[i,j] = z[c]
    c += 1

transform = Affine.translation(minLon - dLon / 2, minLat - dLat / 2) * Affine.scale(dLon, dLat)
tiff = rasterio.open(tifOut,
                     'w',
                     driver='GTiff',
                     height=geoid.shape[0],
                     width=geoid.shape[1],
                     count=1,
                     nodata=-88.8888,
                     dtype=geoid.dtype,
                     crs='+proj=latlong',
                     transform=transform)

tiff.write(geoid, 1)
tiff.close()

                            
