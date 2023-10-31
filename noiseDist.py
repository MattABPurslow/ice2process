import pdb
import pandas as pd
from scipy.stats import linregress

from params import params
from segment import segments

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.family'] = 'Arial'
from matplotlib.cm import gist_earth as cmap

def getRegionName(region):
  if region=='deju':
    return 'Delta Junction'
  if region=='ornl':
    return 'Oak Ridge'
  if region=='sodankyla':
    return 'Sodankylä'
  if region=='sonoma':
    return 'Sonoma'
  if region=='wref':
    return 'Wind River'

for region in ['deju', 'ornl', 'sodankyla', 'sonoma', 'wref']:
  p = params(region)
  rate = pd.read_csv(p.rateDir+'/%s.rates' % region)
  rate = rate.sort_values('ρg')
  segs = segments(region, read=True).data
  Ntrack = rate.shape[0]
  slope, intercept, r, p, se = linregress(segs.cv_ALS, segs.ρn)
  print(region)
  print('m:', slope)
  print('R²:', r**2)
  fig = plt.figure(figsize=(4,3.5)); ax = fig.gca()
  s = segs#segs.loc[segs.date+segs.gtx==rate.track.values[t]]
  c = cmap(.5)
  ax.scatter(s.cv_ALS*100., s.ρn, color=c, s=1)
  ax.set_xlabel('ALS Canopy Cover (%)')
  ax.set_ylabel('$ρ_{n}$ (m$^{-}$¹shot$^{-}$¹)')
  ax.set_xlim(0, 100)
  ax.set_xticks(np.arange(0,100.1,25))
  ax.set_title(getRegionName(region))
  fig.tight_layout()
  fig.show()
