import os, pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import gist_earth as cmap
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'
import seaborn as sns
from scipy.constants import speed_of_light
from scipy.optimize import least_squares
from scipy.stats import linregress
from scipy import odr
import random
from params import params
from photons import photons
from segment import segments

class rates(object):
  def __init__(self, region, ls=100., removeNoise=True, cvMin=-1, cvMax=101, useATL08=False):
    self.region = region
    self.p = params(region)
    self.ls = ls
    self.removeNoise = removeNoise
    self.useATL08 = useATL08
    seg = segments(self.region, read=True)
    self.seg = seg.data
    self.segFile = seg.segFile
    self.getRates()
  
  def getRates(self):
    self.getRawRates()
    self.getCorrectedRates()
    self.seg.to_pickle(self.segFile)
    self.seg = self.seg.loc[self.seg.clean]
  
  def getRawRates(self):
    self.seg['ρn'] = (self.seg.Nn / self.seg.Ns) / self.seg.dn
    self.seg['ρg_r'] = self.seg.Ng / self.seg.Ns
    self.seg['ρv_r'] = self.seg.Nv / self.seg.Ns
  
  def getCorrectedRates(self):
    self.seg['ρg_c'] = self.seg.ρg_r - (self.seg.ρ̅n_m * self.seg.dg)
    self.seg['ρv_c'] = self.seg.ρv_r - (self.seg.ρ̅n_m * self.seg.dv)
  
  def writeRates(self):
    self.rateFile = os.path.join(self.p.rateDir,
                                 '.'.join([self.region, 'rates']))
    self.data.to_csv(self.rateFile, index=False)

  def orthodist(self, ρg, ρv, m, c):
    d = np.abs((m*ρg)-ρv+c) / np.sqrt((m**2.)+1.)
    if ρv < ((m*ρg) + c):
      d = -d
    x0 = (-1.*((-1.*ρg) - (m*ρv)) - (m*c)) / (m**2 + 1)
    y0 = (m*x0) + c
    return d, x0, y0

  def line(self, C, x):
    return C[0]*x + C[1]
  
  def getWeights(self, weight=1):
    self.seg['weight'] = 0.
    for track in np.unique(self.seg.track.values):
      self.seg.loc[self.seg.track==track, 'Nseg'] = np.sum(self.seg.track==track)
      if weight==1:
        self.seg['weight'] = 1.
      else:
        R = self.seg.loc[self.seg.track==track][['ρg_c', 'ρv_c']].corr().values[0,1]
        if R < 0.:
          if weight=='|R|':
            self.seg.loc[self.seg.track==track, 'weight'] = np.abs(R)
          elif weight=='R²':
            self.seg.loc[self.seg.track==track, 'weight'] = R**2.
          else:
            self.seg.loc[self.seg.track==track, 'weight'] = 1.
        else:
          self.seg.loc[self.seg.track==track, 'weight'] = 0.

  def fit(self, weight='1', manual=False):
    self.seg['track'] = [str(d)+str(g) for d, g in zip(self.seg.date, self.seg.gtx)]
    self.getWeights(weight=weight)
    tracks = np.unique(self.seg.track.values)
    keep = [self.seg.loc[self.seg.track==t].shape[0] > 1 for t in tracks]
    tracks = tracks[keep]
    self.seg = self.seg.loc[np.isin(self.seg.track, tracks)]
    iTrack = np.full(self.seg.shape[0], -999)
    for t in range(tracks.shape[0]):
      i = np.argwhere(self.seg.track.values==tracks[t])
      iTrack[i] = t
    self.seg['trackIndex'] = iTrack
    if manual:
      self.m = -self.p.ρvρg
      self.manualFit(tracks)
    else:
      self.getODR(tracks)
  
  def func(self, mc, ρg, ρv, s, t, w, split):
    res = []
    huber = False
    if huber:
      for ti in np.unique(t):
        mask = (t==ti); Nseg = mask.sum()
        ε = 1.35
        d = np.empty(Nseg); i = 0
        ρgt = ρg[mask].values; ρvt = ρv[mask].values; wt = w[mask]
        for i in range(Nseg):
          d[i], x0, y0 = self.orthodist(ρgt[i], ρvt[i], mc[-1], mc[ti])
          i += 1
        σ = np.std(d) * 3.
        for i in range(Nseg):
          a = d[i] / σ
          if a > ε:
            wi = ε*(np.abs(a) - .5*ε)
          else:
            wi = .5*(a**2.)
          if d[i] > 0.:
            res.append(d[i]/wi)
          else:
            res.append(0)
    else:
      for ρgi, ρvi, si, ti, wi in zip(ρg, ρv, s, t, w):
        d, x0, y0 = self.orthodist(ρgi, ρvi, mc[-1], mc[ti]) 
        res.append(d*wi)
    return np.array(res)
  
  def getODR(self, tracks):
    Nt = tracks.shape[0]
    mc_init = np.full(Nt+1, 1.)
    mc_init[-1:] = -1.
    m_est = []
    for i in range(Nt):
      segs = self.seg.loc[self.seg.track==tracks[i]]
      ρv_est = segs.ρv_c.max()
      ρg_est = segs.ρg_c.max()
      #ρv_est = np.median(segs.ρv_c.loc[segs.ρg_c<np.quantile(segs.ρg_c, 0.1)])
      #ρg_est = np.median(segs.ρg_c.loc[segs.ρv_c<np.quantile(segs.ρv_c, 0.1)])
      m_est.append(-(ρv_est/ρg_est))
      mc_init[i] = ρv_est
    mc_init[-1] = np.mean(m_est)
    result = least_squares(self.func, mc_init, method='lm',
                           args=(self.seg.ρg_c, self.seg.ρv_c, self.seg.strong,
                                 self.seg.trackIndex, self.seg.weight, False))
    self.residuals = result.fun
    m_fit = result.x[-1]
    c_fit = result.x[:-1]
    rate = pd.DataFrame()
    rate['track'] = tracks
    rate['date'] = [t[:8] for t in rate.track.values]
    rate['gtx'] = [t[8:] for t in rate.track.values]
    rate['strong'] = [self.seg.loc[self.seg.track==t].strong.max() \
                      for t in rate.track.values]
    rate['night'] = [self.seg.loc[self.seg.track==t].night.max() \
                     for t in rate.track.values]
    rate['ρv'] = c_fit
    rate['ρg'] = -rate.ρv / m_fit
    rate['ρn'] = self.seg.groupby('track').mean().ρ̅n_s.loc[rate.track.values].values
    rate['weight'] = self.seg.groupby('track').mean().weight.loc[rate.track.values].values
    rate['ρvρg'] = rate.ρv / rate.ρg
    self.data = rate
    self.seg['residual'] = self.func(result.x, self.seg.ρg_c, self.seg.ρv_c, self.seg.strong,
                                     self.seg.trackIndex, self.seg.weight, False)    
 
  def manualFit(self, tracks):
    ρv_fit = np.full(tracks.shape[0], -1.)
    for i in range(tracks.shape[0]):
      track = tracks[i]
      seg = self.seg.loc[self.seg.track==track]
      Nseg = seg.shape[0]
      Ndiff = Nseg
      ρv = np.arange(0., 3., 0.01)
      Ndiff = np.full(ρv.shape[0], 1e6)
      for j in range(ρv.shape[0]):
        res, x0, y0 = self.orthodist(seg.ρg_c, seg.ρv_c, self.m, ρv[j])
        #Ngt = np.sum(seg.ρv_c > ((self.m * seg.ρg_c) + ρv[j]))
        #Nlt = np.sum(seg.ρv_c < ((self.m * seg.ρg_c) + ρv[j]))
        #Ndiff[j] = abs(Ngt-Nlt)
        Ndiff[j] = np.sum(res)
      ρv_fit[i] = ρv[Ndiff==Ndiff.min()].mean()
    rate = pd.DataFrame()
    rate['track'] = tracks
    rate['date'] = [t[:8] for t in rate.track.values]
    rate['gtx'] = [t[8:] for t in rate.track.values]
    rate['strong'] = [self.seg.loc[self.seg.track==t].strong.max() \
                      for t in rate.track.values]
    rate['night'] = [self.seg.loc[self.seg.track==t].night.max() \
                     for t in rate.track.values]
    rate['ρv'] = ρv_fit
    rate['ρg'] = -rate.ρv / self.m
    rate['ρn'] = self.seg.groupby('track').mean().ρ̅n_s.loc[rate.track.values].values
    rate['weight'] = self.seg.groupby('track').mean().weight.loc[rate.track.values].values
    rate['ρvρg'] = rate.ρv / rate.ρg
    self.data = rate
    res = self.func(np.append(self.data.ρv, self.m),
                                     self.seg.ρg_c, self.seg.ρv_c,
                                     self.seg.strong, self.seg.trackIndex,
                                     self.seg.weight, False)
  
  def dropBadFit(self):
    qc = np.full(self.data.shape[0], False, dtype=bool)
    for i in range(self.data.track.shape[0]):
      track = self.data.track.values[i]
      rate = self.data.loc[self.data.track==track]
      seg = self.seg.loc[self.seg.track==track]
      Nseg = seg.shape[0]
      Ngt = np.sum(seg.ρv_c > (-(rate.ρvρg.values * seg.ρg_c) + rate.ρv.values))
      Nlt = np.sum(seg.ρv_c < (-(rate.ρvρg.values * seg.ρg_c) + rate.ρv.values))
      if abs(Ngt-Nlt) < .25 * Nseg:
        qc[i] = True
    self.data['goodFit'] = qc
    self.data = self.data.loc[self.data.goodFit]
    self.seg = self.seg.loc[np.isin(self.seg.track, self.data.track)]
  
  def plotAll(self):
    fig = plt.figure(figsize=(32,18))
    Nax = self.data.track.shape[0]
    Ny = int(np.ceil(np.sqrt(Nax))+2)
    Nx = int(np.ceil(Nax/Ny))
    for i in range(Nax):
      ax = fig.add_subplot(Nx, Ny, i+1)
      track = self.data.track.values[i]
      date = self.data.date.values[i]; gtx = self.data.gtx.values[i]
      seg = self.seg.loc[self.seg.track==track]
      data = self.data.loc[self.data.track==track]
      ax.scatter(seg.ρg_c, seg.ρv_c, s=1, color=cmap(.5))
      ax.plot([0, data.ρg], [data.ρv, 0], 'k', lw=1)
      axLim = np.ceil(np.max([seg.ρg_c.max(), seg.ρv_c.max()]))
      ax.set_xlim(0, axLim)
      ax.set_xticks(np.arange(np.ceil(seg.ρg_c.max())+1))
      ax.set_ylim(0, axLim)
      ax.set_title(date+' '+gtx)
      ax.set_yticks(np.arange(np.ceil(seg.ρv_c.max())+1))
      ax.set_aspect('equal')
    fig.tight_layout()
    #fig.show()
    fig.savefig('plots/'+self.region+'.ρvρg.pdf')
  
def getRegion():
  import argparse
  args = argparse.ArgumentParser()
  args.add_argument('-r', '--region', dest='region', type=str, default='sonoma',
                    help='Region of desired track')
  args = args.parse_args()
  return args.region

if __name__=='__main__':
  regions = ['deju', 'ornl', 'sodankyla', 'sonoma', 'wref']
  for i in range(len(regions)):
    ## Calculate segment rates
    rate = rates(regions[i])
    ## Remove negative rates
    rate.seg = rate.seg.loc[(rate.seg.ρg_c>0.)&(rate.seg.ρv_c>0.)]
    ## Reduce to just summer passes
    summer = np.array([(int(d[4:6])>5)&(int(d[4:6])<10) for d in rate.seg.date])
    rate.seg = rate.seg.loc[summer]
    ## Remove asymptote segments
    rate.seg = rate.seg.loc[(rate.seg.ρv_c>=.1*rate.seg.ρg_c)&\
                            (rate.seg.ρv_c<=9.*rate.seg.ρg_c)]
    ## Fit line of best fit to rates
    rate.fit(weight='R²')
    ## Plot fit for all tracks 
    ρgScaled, ρvScaled = np.empty(rate.seg.shape[0]), np.empty(rate.seg.shape[0])
    for t in range(rate.seg.shape[0]):
      df = rate.seg.iloc[t]
      ρg, ρv = rate.data.loc[rate.data.track==df.track][['ρg', 'ρv']].values[0]
      ρgScaled[t], ρvScaled[t] = df.ρg_c/ρg, df.ρv_c/ρv
    from scipy.stats import pearsonr
    print(regions[i], 'r, p=', pearsonr(ρgScaled, ρvScaled))
