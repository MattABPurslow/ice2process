import pdb
import pandas as pd

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
  fig = plt.figure(figsize=(4,3.5)); ax = fig.gca()
  Ntrack = rate.shape[0]
  for t in range(Ntrack):
    s = segs.loc[segs.date+segs.gtx==rate.track.values[t]]
    r = rate.loc[rate.track==rate.track.values[t]]
    c = cmap(t/(1.1*Ntrack))
    ax.scatter(s.ρg_c, s.ρv_c, color=c, s=1)
    ax.plot([0, r.ρg], [r.ρv, 0], color=c, lw=1)
  ax.set_xlabel('$ρ_{gc}$ (shot$^{-}$¹)')
  ax.set_ylabel('$ρ_{vc}$ (shot$^{-}$¹)')
  ax.set_aspect('equal'); ax.set_xlim(0, 2); ax.set_ylim(0, 2)
  ax.set_xticks(np.arange(0,2.1,.5)); ax.set_yticks(np.arange(0,2.1,.5))
  ax.set_title(getRegionName(region))
  #fig.tight_layout()
  #fig.savefig('plots/%s.ρvρg.combined.fixed.pdf' % region)
  #plt.close()
