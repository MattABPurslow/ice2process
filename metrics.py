import os, pdb
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d
from params import params
from photons import photons
from segment import segments

def rhInfo():
  ''' Get RH values and labels '''
  rh = np.append(np.arange(10,91,10), 98)
  rhLabels = np.array(['RH%d' % r for r in rh])
  return rh, rhLabels

class metrics(object):
  def __init__(self, region, date, gtx, pho, seg):
    self.p = params(region)
    self.pho = pho
    self.seg = seg
    self.rateFile = os.path.join(self.p.rateDir,
                                 '.'.join([region, 'rates']))
    rates = pd.read_csv(self.rateFile)
    rates = rates.loc[(rates.date.astype(str)==date)&(rates.gtx==gtx)]
    self.date = date
    self.gtx = gtx
    if rates[['ρg', 'ρv', 'ρn']].values.shape == (1,3):
      self.ρg, self.ρv, self.ρn = rates[['ρg', 'ρv', 'ρn']].values[0]
      self.ch()
      self.cv()
      self.rh()
      self.lai()
    else:
      self.ds = xr.Dataset()
  
  def getResiduals(self):
    m = -self.ρv/self.ρg; c = self.ρv
    self.seg['ρg_r'] = self.seg.Ng / self.seg.Ns
    self.seg['ρv_r'] = self.seg.Nv / self.seg.Ns
    self.seg['ρg_c'] = self.seg.ρg_r - (self.seg.ρ̅n_m * self.seg.dg)
    self.seg['ρv_c'] = self.seg.ρv_r - (self.seg.ρ̅n_m * self.seg.dv)
    ρg = self.seg.ρg_c; self.seg.ρv_c
    self.seg['residual'] =  np.abs((m*ρg)-ρv+c) / np.sqrt((m**2.)+1.)

  def ch(self):
    ''' Calculate canopy heights (RH98 of canopy photons) '''
    print('Calculating canopy heights')
    zMin, zMax, zStep = -10., 100., 0.1
    zBinEdge = np.arange(zMin-zStep/2., zMax+zStep, zStep)
    zBin = np.arange(zMin, zMax+zStep, zStep)
    for atb in self.seg.index:
      pho = self.pho.loc[self.pho.atb==atb]
      pho = pho.loc[pho.canopy]
      if pho.shape[0] > 0:
        Np_n, _ = np.histogram(pho.zRel, bins=zBinEdge,
                               weights=100.*np.ones(pho.zRel.count())/pho.zRel.count())
        idx = np.argwhere(Np_n>0).flatten()
        cumsum = Np_n[idx].cumsum(); zBin_atb = zBin[idx]
        cumsum = np.append(np.append(0, cumsum), 100)
        zBin_atb = np.append(np.append(zBin_atb.min()-zStep, zBin_atb),
                             zBin_atb.max()+zStep)
        rh_n = interp1d(cumsum, zBin_atb)
        self.seg.at[atb,'ch'] = rh_n(98)
      else:
        self.seg.at[atb,'ch'] = 0

  def cv(self):
    ''' Calculate canopy cover using ρv/ρg method '''
    print('Calculating canopy cover')
    cvFunc = lambda ρg, ρv, Ng, Nv: 100.*(1. / (1. + (ρv*Ng)/(ρg*Nv)))
    self.seg['cv'] = cvFunc(self.ρg, self.ρv, self.seg.Ng, self.seg.Nv)
  
  def rh(self):
    ''' Calculate RH metrics (including ground photons, as GEDI) '''
    print('Calculating RH metrics')
    rhVals, rhLabels = rhInfo()
    zMin, zMax, zStep = -10., 100., 0.1
    zBinEdge = np.arange(zMin-zStep/2., zMax+zStep, zStep)
    zBin = np.arange(zMin, zMax+zStep, zStep)
    for atb in self.seg.index:
      pho = self.pho.loc[self.pho.atb==atb]
      Np_n, _ = np.histogram(pho.zRel, bins=zBinEdge,
                             weights=100.*np.ones(pho.zRel.count())/pho.zRel.count())
      if Np_n.max()>.000001:
        idx = np.argwhere(Np_n>0).flatten()
        cumsum = Np_n[idx].cumsum(); zBin_atb = zBin[idx]
        cumsum = np.append(np.append(0, cumsum), 100)
        zBin_atb = np.append(np.append(zBin_atb.min()-zStep, zBin_atb),
                             zBin_atb.max()+zStep)
        rh_n = interp1d(cumsum, zBin_atb)
        for rh, rhLabel in zip(rhVals, rhLabels):
          self.seg.at[atb,rhLabel] = rh_n(rh)
      else:
        for rh, rhLabel in zip(rhVals, rhLabels):
          self.seg.at[atb,rhLabel] = np.nan
  
  def lai(self):
    '''Calculate LAI profiles for each segment. Based on findLAI from
       testFoliage.py in the gedisimulator package (Hancock, 2021). '''
    print('Calculating LAI profiles')
    self.ds = xr.Dataset()
    zMin, zMax, zStep = -10., 100., .1
    laiRes = 5
    zBinEdge = np.arange(zMin-zStep/2., zMax+zStep, zStep)
    zBin = np.arange(zMin, zMax+zStep, zStep)[::-1]
    laiBins = np.arange(zMin, zMax+laiRes/2., laiRes)
    G = 0.6 # angular distribution
    ρvρg = self.ρv / self.ρg
    for atb in self.seg.index:
      pho = self.pho.loc[self.pho.atb==atb]
      pho = pho.loc[pho.ground|pho.canopy]
      zMax = np.ceil((pho.zMax - pho.zG).max()/laiRes)*laiRes
      zMin = np.floor((pho.zMin - pho.zG).min()/laiRes)*laiRes
      Ng = pho.ground.sum()
      Nv = pho.canopy.sum()
      Wv, _ = np.histogram(pho.zRel.loc[pho.canopy], bins=zBinEdge)
      Wv = Wv[::-1]
      if Wv.max()>.000001:
        cumsum = np.cumsum(Wv,0)
        gap = 1. - (cumsum/(Nv + ρvρg*Ng))
        lngap = np.zeros(gap.shape, dtype=float)
        lngap[gap>0.0] = np.log(gap[gap>0.0])
        rawLAI = np.zeros(zBin.shape, dtype=float)
        for j in range(0, zBin.shape[0]-1):
          rawLAI[j] = (-1. * (lngap[j+1]-lngap[j])) / G
        lai = np.zeros(laiBins.shape[0], dtype=float)
        for j in range(0, zBin.shape[0]):
          tBin = np.searchsorted(laiBins, zBin[j], side='left')
          if (tBin >= 0) & (tBin < laiBins.shape[0]):
            lai[tBin] += rawLAI[j]
        ds = xr.Dataset()
        ds['date'] = ('date', [self.date])
        ds['gtx'] = ('gtx', [self.gtx])
        ds['atb'] = ('atb', [atb])
        ds['z'] = ('z', laiBins)
        ds['lai'] = (('date', 'gtx', 'atb', 'z'), [[[lai]]])
        ds = ds.where((ds.z <= zMax)&(ds.z >= zMin))
        ds['zMax'] = (('date', 'gtx', 'atb'), [[[zMax]]])
        ds['zMin'] = (('date', 'gtx', 'atb'), [[[zMin]]])
        self.ds = xr.merge([self.ds, ds])
      else:
        ds = xr.Dataset()
        ds['date'] = ('date', [self.date])
        ds['gtx'] = ('gtx', [self.gtx])
        ds['atb'] = ('atb', [atb])
        ds['z'] = ('z', laiBins)
        ds['lai'] = (('date', 'gtx', 'atb', 'z'), [[[np.full(laiBins.shape[0],
                                                              np.nan)]]])
        ds['zMax'] = (('date', 'gtx', 'atb'), [[[np.nan]]])
        ds['zMin'] = (('date', 'gtx', 'atb'), [[[np.nan]]])
        self.ds = xr.merge([self.ds, ds])
   
  def rh_ds(self):
    import xarray as xr
    zMin, zMax, zStep = -10., 100., 0.1
    zBinEdge = np.arange(zMin-zStep/2., zMax+zStep, zStep)
    zBin = np.arange(zMin, zMax+zStep, zStep)
    rhVals = np.arange(0, 101, 1)
    zRH = np.full(rhVals.shape[0], np.nan)
    self.ds = xr.Dataset()
    for atb in self.seg.index:
      pho = self.pho.loc[self.pho.atb==atb]
      Np_n, _ = np.histogram(pho.zRel, bins=zBinEdge,
                             weights=100.*np.ones(pho.zRel.count())/pho.zRel.count())
      if Np_n.max()>.000001:
        idx = np.argwhere(Np_n>0).flatten()
        cumsum = Np_n[idx].cumsum(); zBin_atb = zBin[idx]
        cumsum = np.append(np.append(0, cumsum), 100)
        zBin_atb = np.append(np.append(zBin_atb.min()-zStep, zBin_atb),
                             zBin_atb.max()+zStep)
        rh_n = interp1d(cumsum, zBin_atb)
        for i in range(zRH.shape[0]):
          zRH[i] = rh_n(rhVals[i])
        ds = xr.Dataset()
        ds['date'] = ('date', [self.date])
        ds['gtx'] = ('gtx', [self.gtx])
        ds['atb'] = ('atb', [atb])
        ds['rh'] = ('rh', rhVals)
        ds['z'] = (('date', 'gtx', 'atb', 'rh'), [[[zRH]]])
        self.ds = xr.merge([self.ds, ds])
      else:
        ds = xr.Dataset()
        ds['date'] = ('date', [self.date])
        ds['gtx'] = ('gtx', [self.gtx])
        ds['atb'] = ('atb', [atb])
        ds['rh'] = ('rh', rhVals)
        ds['z'] = (('date', 'gtx', 'atb', 'rh'), [[[np.full(rhVals.shape[0], np.nan)]]])

class atl(object):
  ''' Read real ICESat-2 photons '''
  def __init__(self, region, date, gtx, ls=100.):
    print('Reading real photons')
    self.p = params(region)
    self.seg = segments(region, read=True).data
    self.seg = self.seg.loc[(self.seg.date.astype(str)==str(date))&(self.seg.gtx==gtx)]
    self.pho = photons(region, date, gtx)
    self.pho.getZ()
    self.pho.getClass()
    self.ls = ls 
    self.pho.getBins(self.ls)
    self.x0, self.y0 = self.pho.x0, self.pho.y0
    self.pho = self.pho.data.loc[(self.pho.data.dummy==False) & \
                                 (self.pho.data.canopy | self.pho.data.ground)]
    self.seg['strong'] = bool(self.pho.strong.max())
    self.rateFile = os.path.join(self.p.rateDir,
                                 '.'.join([region, 'rates']))
    rates = pd.read_csv(self.rateFile)
    rates = rates.loc[(rates.date.astype(str)==date)&(rates.gtx==gtx)]
    if rates[['ρg', 'ρv', 'ρn']].values.shape == (1,3):
      self.ρg, self.ρv, self.ρn = rates[['ρg', 'ρv', 'ρn']].values[0]

class sim(object):
  ''' Read simulated ICESat-2 photons '''
  def __init__(self, region, date, gtx):
    print('Reading simulated photons')
    self.p = params(region)
    self.simFile = os.path.join(self.p.simDir,
                                 '.'.join([date, gtx,'sim.pkl']))
    self.pho = pd.read_pickle(self.simFile)
    self.pho = self.pho.loc[(self.pho.canopy | self.pho.ground)]

class comparison(object):
  def __init__(self, region, date, gtx):
    self.region = region
    self.date = date
    self.gtx = gtx
    self.real = atl(region, date, gtx)
    self.sims = sim(region, date, gtx)
    self.getSimBins()
    self.getSimSegs()
    self.getCoords()
    self.getMetrics()
  
  def getSimBins(self):
    ''' Bin simulated photons into same bins as real data '''
    print('Binning simulated photons into segments')
    self.sims.pho['atb'] = (((self.sims.pho.x-self.real.x0)**2 + \
                            (self.sims.pho.y-self.real.y0)**2)**.5) \
                           // self.real.ls
    self.sims.pho['atb'] = self.sims.pho.atb.astype(int)

  def getSimSegs(self):
    ''' Calculate segment stats for simulations '''
    print('Calculating simulated segment stats')
    self.sims.seg = pd.DataFrame({'atb':self.sims.pho.atb.unique()})
    self.sims.seg['ls'] = self.real.ls
    self.sims.seg = self.sims.seg.set_index('atb')
    self.real.seg['atb'] = self.real.seg.atb.astype(int)
    self.real.seg = self.real.seg.set_index('atb')
    self.sims.seg['Ng'] = self.sims.pho[['atb', 'ground']].groupby('atb').sum()
    self.sims.seg['Nv'] = self.sims.pho[['atb', 'canopy']].groupby('atb').sum()
    cols = ['waveID', 'atb', 'zMin', 'zG', 'zMax', 'cv_ALS']
    group = self.sims.pho[cols].groupby('waveID').mean().groupby('atb').mean()
    self.sims.seg['dg'] = (group.zG - group.zMin)
    self.sims.seg['dv'] = (group.zMax - group.zG)
    self.sims.seg['cv_ALS'] = group.cv_ALS
  
  def getCoords(self):
    group = self.real.pho.groupby('atb')[['lon', 'lat']].median()
    self.real.seg['lon'] = group.lon.reindex(self.real.seg.index)
    self.real.seg['lat'] = group.lat.reindex(self.real.seg.index)
    
  def getPairs(self):
    ''' Reduce data to where both simulated and real data available '''
    print('Pairing real and simulated data')
    self.sims.seg = self.sims.seg.loc[np.isin(self.sims.seg.index, self.real.seg.index)]
    self.real.seg = self.real.seg.loc[np.isin(self.real.seg.index, self.sims.seg.index)]
  
  def getMetrics(self):
    ''' Calculate ch, cv and rh metrics for simulated and real data '''
    print('Getting canopy metrics')
    sims = metrics(self.region, self.date, self.gtx, self.sims.pho, self.sims.seg)
    self.simsLAI = sims.ds
    real = metrics(self.region, self.date, self.gtx, self.real.pho, self.real.seg)
    self.realLAI = real.ds
    self.getPairs()
    #self.getLAIquantiles()
  
  def getLAIquantiles(self):
    for atb in self.real.seg.index:
      realLAI = self.realLAI.sel(atb=atb, date=self.date, gtx=self.gtx)
      ch = self.real.seg.loc[atb].ch
      res = np.abs(self.realLAI.z[1]-self.realLAI.z[0]).values
      lq = np.round((ch/4.)/res)*res; mq, uq = 2.*lq, 3.*lq
      self.real.seg.at[atb,'lai25'] = realLAI.sel(z=lq).lai.values
      self.real.seg.at[atb,'lai50'] = realLAI.sel(z=mq).lai.values
      self.real.seg.at[atb,'lai75'] = realLAI.sel(z=uq).lai.values
      self.real.seg.at[atb,'lai'] = realLAI.sum(dim='z').lai.values
      if np.isin(atb, self.simsLAI.atb):
        simsLAI = self.simsLAI.sel(atb=atb, date=self.date, gtx=self.gtx)
        self.sims.seg.at[atb,'lai25'] = simsLAI.sel(z=lq).lai.values
        self.sims.seg.at[atb,'lai50'] = simsLAI.sel(z=mq).lai.values
        self.sims.seg.at[atb,'lai75'] = simsLAI.sel(z=uq).lai.values
        self.sims.seg.at[atb,'lai'] = simsLAI.sum(dim='z').lai.values

if __name__=='__main__':
  self = comparison('sonoma', '20190101', 'gt1l')
