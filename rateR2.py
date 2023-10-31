import pdb
import pandas as pd

from params import params
from segment import segments

import numpy as np
from scipy.stats import linregress
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

def orthodist(ρg, ρv, m, c):
  d = np.abs((m*ρg)-ρv+c) / np.sqrt((m**2.)+1.)
  return d

def residual(ρg, ρv, m, c):
  return ρv - ((m*ρg)+c)

for region in ['deju', 'ornl', 'sodankyla', 'sonoma', 'wref']:
  p = params(region)
  rate = pd.read_csv(p.rateDir+'/%s.rates' % region)
  rate = rate.sort_values('ρg')
  segs = segments(region, read=True).data
  fig = plt.figure(figsize=(4,3.5)); ax = fig.gca()
  Ntrack = rate.shape[0]
  r2 = []
  for t in range(Ntrack):
    s = segs.loc[segs.date+segs.gtx==rate.track.values[t]]
    r = rate.loc[rate.track==rate.track.values[t]]
    slope, intercept, r, p, se = linregress(s.ρg_c, s.ρv_c)
    r2.append(r**2)
  print(region, 'R²:', np.mean(r2))
    
