import os, pdb
import matplotlib.pyplot as plt
from matplotlib.cm import gist_earth as cmap
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
from scipy.stats import linregress

from metrics import comparison, rhInfo
from params import params
from rates import rates
from simulate import simulate

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

class plot(object):
  def __init__(self, region, dates, gtxs):
    self.region = region
    self.real = pd.DataFrame()
    self.sims = pd.DataFrame()
    self.realLAI = xr.Dataset()
    self.simsLAI = xr.Dataset()
    for date, gtx in zip(dates, gtxs):
      comp = comparison(region, date, gtx)
      comp.real.seg['date'] = date; comp.real.seg['gtx'] = gtx
      comp.real.seg.set_index(['date', 'gtx'], append=True, inplace=True)
      self.real = self.real.append(comp.real.seg)
      comp.sims.seg['date'] = date; comp.sims.seg['gtx'] = gtx
      comp.sims.seg.set_index(['date', 'gtx'], append=True, inplace=True)
      self.sims = self.sims.append(comp.sims.seg)
      self.realLAI = xr.merge([self.realLAI, comp.realLAI])
      self.simsLAI = xr.merge([self.simsLAI, comp.simsLAI])
  
  def orthodist(self, ρg, ρv, m, c):
    d = np.abs((m*ρg)-ρv+c) / np.sqrt((m**2.)+1.) 
    return d
  
  def dropOutliers(self, rates):
    from scipy.interpolate import interp1d
    self.real['d'] = np.nan
    for i in range(rates.shape[0]):
      rate = rates.iloc[i]
      ρg = rate.ρg; c = rate.ρv
      m = -c / ρg
      date, gtx = str(rate.date), str(rate.gtx)
      atbArr, dateArr, gtxArr = np.array([i for i in self.real.index]).T
      idx = np.argwhere((dateArr==date)&(gtxArr==gtx)).flatten()
      real = self.real.iloc[idx]
      d = self.orthodist(real.ρg_c, real.ρv_c, m, c)
      d0 = self.orthodist(0, 0, m, c)
      self.real.loc[real.index, 'd'] = d
    self.real = self.real.loc[self.real.d < 0.05]
    self.sims = self.sims.loc[self.real.index]
  
  def ch(self, save=False):
    ''' Plot simulated against real canopy height '''
    print('Plotting ch')
    fig = plt.figure(figsize=(4,3.5))
    gs = GridSpec(1, 22)
    ax = fig.add_subplot(gs[:,0:20])
    ax.set_title(getRegionName(self.region))
    cax = fig.add_subplot(gs[:,-2])
    self.chPlot(self.real, self.sims, ax, cax)
    fig.subplots_adjust(bottom=.15)
    if save==False:
      fig.show()
    else:
      fig.savefig(save)
      plt.close()
  
  def chPlot(self, real, sims, ax, cax):
    hst = sns.histplot(x=real.ch, y=sims.ch, ax=ax,
                       cmap='mako_r', binrange=(0,50), binwidth=1,
                       vmin=1, vmax=100, norm=LogNorm(vmin=1.1, vmax=100),
                       cbar=True, cbar_ax=cax,
                       cbar_kws={'label':'Number of segments',
                                 'orientation':'vertical'})
    _, _, r, _, _ = linregress(real.ch, sims.ch)
    diff = sims.ch - real.ch
    bias = diff.mean()
    rmse = (diff**2.).mean()**.5
    ax.text(.05, .95, 'R² = %.2f' % r**2, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left')
    ax.text(.05, .875, 'RMSE = %.1f' % rmse+' m', transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left')
    ax.text(.05, .8, 'Bias = %.1f' % bias+' m', transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left')
    ax.plot([0,50], [0,50], color='k', lw=1)
    ax.set_xlim(0, 50); ax.set_ylim(0, 50); ax.set_aspect('equal')
    ax.set_xlabel('Observed ICESat-2 canopy height (m)')
    ax.set_ylabel('Simulated ICESat-2 canopy height (m)')
   
  def cv(self, save=False):
    ''' Plot simulated against real canopy height '''
    print('Plotting cv')
    fig = plt.figure(figsize=(4,3.5))
    gs = GridSpec(1, 22)
    ax = fig.add_subplot(gs[:,0:20])
    ax.set_title(getRegionName(self.region))
    cax = fig.add_subplot(gs[:,-2])
    real = self.real.loc[self.real.strong]
    sims = self.sims.loc[self.real.strong]
    beam = True
    self.cvPlot(real, sims, ax, beam, cax)
    fig.subplots_adjust(bottom=.15)
    if save==False:
      fig.show()
    else:
      fig.savefig(save)
      plt.close()
 
  def cvPlot(self, real, sims, ax, beam, cax):
    hst = sns.histplot(x=real.cv, y=sims.cv,
                       ax=ax, cbar_ax=cax,
                       cmap='mako_r', binrange=(0,100), binwidth=5,
                       vmin=1, vmax=100, norm=LogNorm(vmin=1.1, vmax=100),
                       cbar=True, cbar_kws={'label':'Number of segments',
                                            'orientation':'vertical'})
    _, _, r, _, _ = linregress(real.cv.loc[sims.index], sims.cv.loc[sims.index])
    diff = sims.cv.loc[sims.index] - real.cv.loc[sims.index]
    bias = np.nanmean(diff)
    rmse = np.nanmean(diff**2.)**.5
    ax.text(.05, .95, 'R² = %.2f' % r**2, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left')
    ax.text(.05, .875, 'RMSE = %.1f' % rmse+'%', transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left')
    ax.text(.05, .8, 'Bias = %.1f' % bias+'%', transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left')
    ax.plot([0,100], [0,100], color='k', lw=1)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect('equal')
    ax.set_xlabel('Observed ICESat-2 canopy cover (%)')
    ax.set_ylabel('Simulated ICESat-2 canopy cover (%)')
  
  def rh(self, save=False):
    ''' Plot simulated against real canopy height '''
    print('Plotting rh')
    fig = plt.figure(figsize=(4.4,3.3))
    gs = GridSpec(10, 1)
    ax = [fig.add_subplot(gs[:4,0]), fig.add_subplot(gs[4:7,0]), fig.add_subplot(gs[7:,0])]
    ax[0].set_title(getRegionName(self.region))
    self.rhPlot(self.real, self.sims, ax)
    fig.tight_layout(rect=[-.01,-.01,1.01,.97], h_pad=.9)
    if save==False:
      fig.show()
    else:
      fig.savefig(save)
      plt.close()
  
  def rhPlot(self, real, sims, ax):
    rhVals, rhLabels = rhInfo()
    diffs = sims[rhLabels] - real[rhLabels]
    bias = np.nanmean(diffs, axis=0)
    rmse = np.nanmean(diffs**2, axis=0)**.5
    ## Plot
    sns.violinplot(data=diffs, ax=ax[0], color=cmap(.2),
                   linewidth=0., scale='width')
    ax[1].bar(rhLabels, rmse, color=cmap(.8))
    ax[2].bar(rhLabels, bias, color=cmap(.5))
    ## Format axes
    ax[0].axhline(0, c='k', lw=1)
    ax[1].axhline(0, c='k', lw=1)
    ax[2].axhline(0, c='k', lw=1)
    ax[0].set_ylim(-10, 10)
    ax[0].set_yticks(np.arange(-10,11,5))
    ax[1].set_ylim(0, 5)
    ax[1].set_yticks(np.arange(0,6,2.5))
    ax[2].set_ylim(-1., 1.1)
    ax[2].set_yticks(np.arange(-1.,1.1,.5))
    for a in ax:
      a.spines['right'].set_visible(False)
      a.spines['top'].set_visible(False)
    ## Label axes
    ax[0].spines['bottom'].set_visible(False)
    for i in [0,1]:
      ax[i].xaxis.set_tick_params(size=0)
      ax[i].set_xticklabels([])
    ax[2].set_xticklabels(rhVals)
    ax[2].set_xlabel('Relative height metric (%)')
    ax[0].set_ylabel('Height difference \n (sim. - obs.) (m)')
    ax[1].set_ylabel('RMSE (m)')
    ax[2].set_ylabel('Bias (m)')
  
  def lai(self, save=False):
    ''' Plot simulated against real lai '''
    print('Plotting lai')
    for label in ['lai','lai25', 'lai50', 'lai75']:
      fig = plt.figure(figsize=(4,3.5))
      gs = GridSpec(1, 22)
      ax = fig.add_subplot(gs[:,0:20])
      ax.set_title(getRegionName(self.region))
      cax = fig.add_subplot(gs[:,-2])
      real = self.real.loc[self.real.strong]
      sims = self.sims.loc[self.real.strong]
      beam = True
      self.laiPlot(real, sims, ax, beam, cax, label)
      fig.subplots_adjust(bottom=.15)
      if save==False:
        fig.show()
      else:
        fig.savefig(save)
        plt.close()
  
  def laiPlot(self, real, sims, ax, beam, cax, label):
    hst = sns.histplot(x=real[label], y=sims[label],
                       ax=ax, cbar_ax=cax,
                       cmap='mako_r', binrange=(0,10), binwidth=.1,
                       vmin=1, vmax=100, norm=LogNorm(vmin=1.1, vmax=100),
                       cbar=True, cbar_kws={'label':'Number of segments',
                                            'orientation':'vertical'})
    _, _, r, _, _ = linregress(real[label].loc[sims.index], sims[label].loc[sims.index])
    diff = sims[label].loc[sims.index] - real[label].loc[sims.index]
    bias = np.nanmean(diff)
    rmse = np.nanmean(diff**2.)**.5
    ax.text(.05, .95, 'R² = %.2f' % r**2, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left')
    ax.text(.05, .875, 'RMSE = %.1f' % rmse+' m²/m²', transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left')
    ax.text(.05, .8, 'Bias = %.1f' % bias+' m²/m²', transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left')
    ax.plot([0,10], [0,10], color='k', lw=1)
    if label=='lai':
      ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect('equal')
      ax.set_xlabel('Observed ICESat-2 total LAI (m²/m²)')
      ax.set_ylabel('Simulated ICESat-2 total LAI (m²/m²)')
    else:
      ax.set_xlim(0, 2.5); ax.set_ylim(0, 2.5); ax.set_aspect('equal')
      ax.set_xlabel('Observed ICESat-2 LAI \n at '+label[-2:]+'% of canopy height (m²/m²)')
      ax.set_ylabel('Simulated ICESat-2 LAI \n at '+label[-2:]+'% of canopy height (m²/m²)')
  
  def laiz(self, save=False):
    ''' Plot simulated against real canopy height '''
    print('Plotting lai')
    fig = plt.figure(figsize=(4.4,3.3))
    gs = GridSpec(10, 1)
    ax = [fig.add_subplot(gs[:4,0]), fig.add_subplot(gs[4:7,0]), fig.add_subplot(gs[7:,0])]
    ax[0].set_title(getRegionName(self.region))
    self.laizPlot(ax)
    fig.tight_layout(rect=[-.01,-.01,1.01,.97], h_pad=.9)
    if save==False:
      fig.show()
    else:
      fig.savefig(save)
      plt.close()

  def laizPlot(self, ax):
    zMin = float(self.realLAI.zMin.min())
    zMax = float(self.realLAI.zMax.max())
    self.realLAI = self.realLAI.sel(z=slice(zMin, zMax))
    self.simsLAI = self.simsLAI.sel(z=slice(zMin, zMax))
    diffs = self.simsLAI.lai - self.realLAI.lai
    df = pd.DataFrame()
    for z in diffs.z.values:
      df[z] = diffs.sel(z=z).to_dataframe().lai
    zArr = diffs.z.values
    bias = df.mean(skipna=True).values
    rmse = (df**2.).mean(skipna=True).values**.5
    ## Plot
    sns.violinplot(data=df, ax=ax[0], color=cmap(.2),
                   linewidth=0., scale='width')
    ax[1].bar(range(df.shape[1]), rmse, color=cmap(.8))
    ax[2].bar(range(df.shape[1]), bias, color=cmap(.5))
    iMin = int(np.argwhere(zArr==0))
    if zArr.max()>50:
      iMax = int(np.argwhere(zArr==50))
    else:
      iMax = int(np.argwhere(zArr==zMax))
    ## Format axes
    ax[0].axhline(0, c='k', lw=.75)
    ax[1].axhline(0, c='k', lw=.75)
    ax[2].axhline(0, c='k', lw=.75)
    ax[0].set_xlim(float(iMin)-.5, float(iMax)+.5)
    ax[1].set_xlim(ax[0].get_xlim())
    ax[2].set_xlim(ax[0].get_xlim())
    ax[0].set_ylim(-5, 5)
    ax[1].set_ylim(0, 1); ax[1].set_yticks([0, 0.5, 1])
    ax[2].set_ylim(-.1, .1)
    for a in ax:
      a.spines['right'].set_visible(False)
      a.spines['top'].set_visible(False)
    ## Label axes
    ax[0].spines['bottom'].set_visible(False)
    for i in [0,1]:
      ax[i].xaxis.set_tick_params(size=0)
      ax[i].set_xticklabels([])
    ax[2].set_xticks(range(iMin, iMax+1))
    ax[2].set_xticklabels(zArr[iMin:iMax+1].astype(int))
    ax[2].set_xlabel('Height (m)')
    ax[0].set_ylabel('PAD difference \n (sim. - obs.) \n (m²/m³)')
    ax[1].set_ylabel('RMSE \n (m²/m³)')
    ax[2].set_ylabel('Bias \n (m²/m³)')

def bool2str(n):
  if n:
    return 'true'
  else:
    return 'false'
    
def getRegion():
  import argparse
  args = argparse.ArgumentParser()
  args.add_argument('-r', '--region', dest='region', type=str, default='sonoma',
                    help='Region of desired track')
  args = args.parse_args()
  return args.region

if __name__=='__main__':
  region = getRegion()
  p = params(region)
  dates = np.array(p.dates); gtxs = np.array(p.gtxs)
  summer = np.array([(int(d[4:6])>5)&(int(d[4:6])<10) for d in dates])
  dates = dates[summer]; gtxs = gtxs[summer]
  rates = pd.read_csv(os.path.join(p.rateDir, region+'.rates'))
  #for date, gtx in zip(dates, gtxs):
  #  zFile = os.path.join(p.zDir, '.'.join([date,gtx,'z','pkl']))
  #  if os.path.exists(zFile):
  #    trackMask = (rates.date.astype(str)==date)&(rates.gtx==gtx)
  #    if trackMask.sum() == 1:
  #      r = rates.loc[trackMask]
  #    else:
  #      r = pd.DataFrame({'ρg':[0], 'ρv':[0]})
  #    check = (r.ρg.values[0] > .25) & (r.ρv.values[0] > .25)
  #    if check:
  #      sim = simulate(region, date, gtx)
  dateSim, gtxSim = np.array([[d, g] for d, g in zip(dates, gtxs)\
                              if os.path.exists(os.path.join(p.simDir,
                                              '.'.join([d, g, 'sim.pkl'])))]).T
  comp = plot(region, dateSim, gtxSim)
  comp.dropOutliers(rates)
  cutdown = (comp.real.ch < .1)&(comp.sims.ch > 1)
  comp.real = comp.real.loc[cutdown==False]
  comp.sims = comp.sims.loc[cutdown==False]
  comp.real.to_pickle(os.path.join(p.segDir.replace('segments', 'metrics'), '.'.join([region, 'real', 'pkl'])))
  comp.sims.to_pickle(os.path.join(p.segDir.replace('segments', 'metrics'), '.'.join([region, 'sims', 'pkl'])))
  comp.realLAI.to_netcdf(os.path.join(p.segDir.replace('segments', 'nc'), '.'.join([region, 'real', 'nc'])))
  comp.simsLAI.to_netcdf(os.path.join(p.segDir.replace('segments', 'nc'), '.'.join([region, 'sims', 'nc'])))
  #comp.ch(save='plots/%s.ch.pdf' % region)
  #comp.cv(save='plots/%s.cv.pdf' % region)
  #comp.rh(save='plots/%s.rh.pdf' % region)
  #comp.lai()
  #comp.laiz(save='plots/%s.lai.pdf' % region)
  
  #for date in comp.realLAI.date:
  #  dateDS = comp.realLAI.sel(date=date)
  #  for gtx in dateDS.gtx:
  #    dateGT = dateDS.sel(gtx=gtx)
  #    for atb in dateGT.atb:
  #      real = comp.realLAI.sel(date=date, gtx=gtx, atb=atb)
  #      sims = comp.simsLAI.sel(date=date, gtx=gtx, atb=atb)
  #      if (np.isnan(sims.lai.max())==0)&(np.isnan(real.lai.max())==0):
  #        plt.plot(sims.lai, sims.z, label='Sim.')
  #        plt.plot(real.lai, real.z, label='Obs.')
  #        plt.legend(loc='lower right')
  #        plt.xlabel('Effective PAI (m²/m²)')
  #        plt.ylabel('Height above ground (m)')
  #        plt.show()
  #        pdb.set_trace()
  #        plt.close()
  
  #fig = plt.figure(); ax = fig.gca()
  #for date in comp.realLAI.date:
  #  real = comp.realLAI.sel(date=date)
  #  sims = comp.simsLAI.sel(date=date)
  #  for gtx in real.gtx:
  #    realtrack = real.sel(gtx=gtx).dropna(dim='atb')
  #    simstrack = sims.sel(gtx=gtx).dropna(dim='atb')
  #    for atb in realtrack.atb:
  #      if np.isin(atb, simstrack.atb):
  #        realseg = realtrack.sel(atb=atb)
  #        simsseg = simstrack.sel(atb=atb)
  #        ax.plot(simsseg.lai-realseg.lai, realseg.z, c=cmap(.5), alpha=.05)
  #ax.set_xlabel('LAI difference (m²/m²)')
  #ax.set_ylabel('Height above local ground (m)')
  #ax.axvline(0, lw=1, color='k')
  #ax.set_xlim(-5, 5)
  #fig.show()
  #
  #simsArr = []
  #realArr = []
  #fig = plt.figure(); ax = fig.gca()
  #for date in realz.date:
  #  real = comp.realLAI.sum(dim='z').sel(date=date)
  #  sims = comp.simsLAI.sum(dim='z').sel(date=date)
  #  for gtx in real.gtx:
  #    realtrack = real.sel(gtx=gtx).dropna(dim='atb')
  #    simstrack = sims.sel(gtx=gtx).dropna(dim='atb')
  #    for atb in realtrack.atb:
  #      if np.isin(atb, simstrack.atb):
  #        realArr.append(realtrack.sel(atb=atb).lai.values)
  #        simsAr.append(simstrack.sel(atb=atb).lai.values)
  #ax.scatter(realArr, simsArr, color=cmap(.5), s=5)
  #ax.set_xlabel('Observed LAI (m²/m²)')
  #ax.set_ylabel('Simulated LAI (m²/m²)')
  #ax.set_aspect('equal')
  #fig.show()
