import os, pdb
import numpy as np
import pandas as pd
from params import params
from photons import photons
from scipy.constants import speed_of_light
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.cm import gist_earth

class poisson(object):
  def __init__(self, region):
    self.p = params(region)
    self.region = region
    self.rateFile = os.path.join(self.p.rateDir,
                                 '.'.join([self.region, 'rates']))
    self.rates = pd.read_csv(self.rateFile)
    self.dates = self.rates.date.values.astype(str)
    self.gtxs = self.rates.gtx.values.astype(str)
    self.cv = np.linspace(0,100,21)
    self.Nbin = np.arange(-.5,10.6,1.)
    self.getPho()
 
  def getPho(self):
    self.data = pd.DataFrame()
    self.Ns = 0
    for date, gtx in zip(self.dates, self.gtxs):
      pho = photons(self.region, date, gtx) 
      pho.getZ()
      pho.getClass()
      df = pho.data[['noise','ground','canopy']].groupby('waveID').sum()
      wID = df.index.values
      wIDcont = np.arange(wID.min(), wID.max(), 1)
      wIDcont = np.array([w for w in wIDcont \
                          if (int(str(w)[-3:])>0) & ((int(str(w)[-3:])<201))])
      self.Ns += wIDcont.shape[0]
      df = df.reindex(wIDcont).fillna(0)
      means = pho.data.groupby('waveID').mean()
      df['dg'] = means.zG - means.zMin
      df['dv'] = means.zMax - means.zG
      df['cv'] = means.cv_ALS *100.
      df = df.loc[df.cv.isna()==False]
      df['date'] = date
      df['gtx'] = gtx
      ρg, ρv, ρn = self.rates.loc[(self.rates.date.astype(int)==int(date))&(self.rates.gtx==gtx)][['ρg', 'ρv', 'ρn']].values[0]
      df['ρg'], df['ρv'], df['ρn'] = ρg, ρv, ρn
      df.canopy -= df.dv*df.ρn; df.ground -= df.dg*df.ρn
      df.loc[df.canopy<0,'canopy'] = 0
      df.loc[df.ground<0,'ground'] = 0
      df = df.set_index(['date', 'gtx'], append=True)
      self.data = self.data.append(df)
      
  def count(self):
    self.Ng_obs, _ = np.histogram(self.data.ground, bins=self.Nbin)
    self.Nv_obs, _ = np.histogram(self.data.canopy, bins=self.Nbin)
    Ns = self.data.shape[0]
    self.Ng_obs = self.Ng_obs/Ns; self.Nv_obs = self.Nv_obs/Ns
    self.Ng_exp = np.full(self.Nbin.shape[0]-1, 0.)
    self.Nv_exp = np.full(self.Nbin.shape[0]-1, 0.)
    Ns_exp = 0
    for date, gtx in zip(self.dates, self.gtxs):
      df = self.data.loc[:, date, gtx]
      for i in range(self.cv.shape[0]-1):
        mask = (df.cv > self.cv[i]) & (df.cv < self.cv[i+1])
        Ns = mask.sum()
        if Ns > 0:
          Nv = df.loc[mask].canopy.mean(); Ng = df.loc[mask].ground.mean()
          self.Nv_exp += Ns*stats.poisson.pmf(self.Nbin[:-1]+.5, mu=Nv)
          self.Ng_exp += Ns*stats.poisson.pmf(self.Nbin[:-1]+.5, mu=Ng)
        Ns_exp += Ns
    self.Ng_exp = self.Ng_exp / Ns_exp; self.Nv_exp = self.Nv_exp / Ns_exp
  
  def chi_square(self, ls=100.):
    Ns = (self.Ns / (ls/.7))
    self.X2g, self.pg = stats.chisquare(self.Ng_obs*Ns,f_exp=self.Ng_exp*Ns)
    self.X2v, self.pv = stats.chisquare(self.Nv_obs*Ns,f_exp=self.Nv_exp*Ns)

  def hist(self):
    fig, ax = plt.subplots(1,2, figsize=(6,3))
    fig.suptitle(self.region)
    ax[0].bar(self.Nbin[:6]+.5, self.Nv_obs[:6]*100., label='Observed',
              color=gist_earth(0.5), zorder=1)
    ax[0].step(self.Nbin[:7], self.Nv_exp[:7]*100., label='Expected', zorder=2,
               where='post', color=gist_earth(0.2))
    ax[0].set_xlabel('Canopy photons per shot')
    ax[1].bar(self.Nbin[:6]+.5, self.Ng_obs[:6]*100., label='Observed',
              color=gist_earth(0.5), zorder=1)
    ax[1].step(self.Nbin[:7], self.Ng_exp[:7]*100., label='Expected', zorder=2,
               where='post', color=gist_earth(0.2))
    ax[1].set_xlabel('Ground photons per shot')
    for i in [0, 1]:
      ax[i].set_xlim(-1,6); ax[i].set_ylabel('Probability (%)')
      ax[i].set_ylim(0, 100); ax[i].set_aspect(0.07)
      ax[i].set_xticks(np.arange(0,6,1))
    if self.region=='deju':
      ax[1].legend(loc='upper right', edgecolor='none')
    fig.tight_layout(); fig.show()#fig.savefig('plots/%s.poisson.pdf' % region)
  
  def chi_plot(self):
    fig = plt.figure(); ax = fig.gca()
    ls = np.arange(10,10001,10)
    pg = np.empty(ls.shape); pv = np.empty(ls.shape)
    for i in range(ls.shape[0]):
      self.chi_square(ls=ls[i])
      pg[i] = self.pg; pv[i] = self.pv
    ax.plot(ls, pg, c=gist_earth(.8), label='Ground')
    ax.plot(ls, pv, c=gist_earth(.5), label='Canopy')
    ax.set_xlabel('Segment length (m)'); ax.set_ylabel('P(Poisson)'); ax.legend(loc='best')
    fig.savefig('plots/'+self.region+'.poisson.chi.pdf')

def getRegion():
  import argparse
  args = argparse.ArgumentParser()
  args.add_argument('-r', '--region', dest='region', type=str, default='sonoma',
                    help='Region of desired track')
  args = args.parse_args()
  return args.region
    
if __name__=='__main__':
  region = getRegion()
  fish = poisson(region)
  fish.count()
  fish.hist()
  fish.chi_plot()
  print(region, 'Pv=', fish.pv, 'Pg=', fish.pg)
