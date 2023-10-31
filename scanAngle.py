import os, pdb
from progress.bar import ChargingBar
import numpy as np
import pandas as pd
import laspy
import matplotlib.pyplot as plt
from params import params

fig = plt.figure(); ax = fig.gca()
regions = ['bart', 'davos', 'ornl', 'sodankyla', 'sonoma', 'wref']

df = pd.DataFrame({'scan_angle': np.arange(-90., 90.1, 1)}).set_index('scan_angle')
for region in regions:
  p = params(region)
  alsList = np.sort([os.path.join(p.alsDir, f) for f in  os.listdir(p.alsDir)])
  alsList = alsList[[f[-4:]=='.las' for f in alsList]]
  bins = np.arange(-90.5, 90.6, 1)
  freq = np.zeros(bins.shape[0]-1)
  bar = ChargingBar(region,max=alsList.shape[0])
  for alsFile in alsList:
    try:
      las = laspy.file.File(alsFile)
      N, _ = np.histogram(las.scan_angle_rank, bins=bins)
      freq += N 
      las.close(); bar.next()
    except:
      bar.next()
  bar.finish()
  df[region] = 100. * freq / freq.sum()
  df.to_pickle('scan_angle.pkl')

df.plot(ax=ax)
ax.legend(loc='upper left', edgecolor='none', fancybox=False)
ax.set_xlim(-90, 90)
ax.set_xlabel('Scan angle rank (°)')
ax.set_ylabel('% of points')
fig.savefig('scanAngle.pdf')
