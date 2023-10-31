import os, sys, pdb
import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from pyproj import Transformer
from params import params
from offset import offset

class atl03(object):
  def __init__(self, atl03File, gtx):
    """ Load & label ICESat-2 photons"""
    ## Name input files
    self.atl03File = atl03File
    self.atl08File = self.atl03File.replace('atl03','atl08')\
                         .replace('ATL03','ATL08')
    ## Choose track
    self.gtx = gtx
    ## Identify track date
    self.date = self.atl03File.split('/')[-1][6:14]
    ## Get region
    self.region = self.atl03File.split('/')[-3].lower()
    p = params(self.region)
    self.latBounds = p.latBounds
    ## Get offset file
    self.offsetFile = os.path.join(p.offsetDir,
                                   '.'.join([self.date, self.gtx, 'xyz']))
    ## Name EPSG code and region
    self.epsg = p.epsg
    self.photonOut = os.path.join(p.pklDir, '%s.%s.photons.pkl' \
                                             % (self.date, self.gtx))
    ## Load photons
    self.makeDF()
  
  def makeDF(self):
    ''' Load ICESat-2 photons into DataFrame '''
    print('Loading ATL03 data')
    print(self.date, self.gtx)
    h5 = h5py.File(self.atl03File, 'r')
    hgts = h5['/'.join([self.gtx, 'heights'])]
    idx = np.argwhere((hgts['lat_ph'][:]>=self.latBounds[0]) & \
                      (hgts['lat_ph'][:]<=self.latBounds[1])).flatten()
    self.data = pd.DataFrame({'cnt':        hgts['ph_id_count'][:][idx],
                              'lat':        hgts['lat_ph'][:][idx],
                              'alongtrack': hgts['dist_ph_along'][:][idx],
                              'pulse':      hgts['ph_id_pulse'][:][idx],
                              'z':          hgts['h_ph'][:][idx],
                              'delta_time': hgts['delta_time'][:][idx],
                              'channel':    hgts['ph_id_channel'][:][idx],
                              'lon':        hgts['lon_ph'][:][idx],
                              'dt':         hgts['delta_time'][:][idx],
                              'pce_mframe': hgts['pce_mframe_cnt'][:][idx],
                              'acrosstrack':hgts['dist_ph_across'][:][idx],
                              'conf':       hgts['signal_conf_ph'][:,0][idx]})
    geo = h5['/'.join([self.gtx, 'geolocation'])]
    segID = []; segCnt = []
    segID = [np.full(N, segID) for N, segID in zip(geo['segment_ph_cnt'][:],
                                                   geo['segment_id'][:])]
    segCnt = [np.arange(1, N+1, 1) for N in geo['segment_ph_cnt'][:]]
    self.data['segment_id'] = np.concatenate(segID)[idx]
    self.data['segment_cnt'] = np.concatenate(segCnt)[idx]
    self.coordTransform()
    self.getLandCoverClass(geo)
    self.getTelemetry(h5)
    self.getBeamStrength(h5)
    self.getAtl08()
    self.getOffset()
    self.getShotIDs()
  
  def getShotIDs(self):
    ''' Label shots with unique identifiers,
        including shots with zero returns '''
    print('Labeling shots')
    ## Combine counts into waveID IDs
    self.data['waveID'] = 1000 * self.data.pce_mframe.astype(np.int64) \
                               + self.data.pulse.astype(np.int64)
    ## Set wave ID as index
    self.data = self.data.set_index('waveID')
    ## Get array of all possible wave IDs
    pceMin = self.data.pce_mframe.values.astype(int).min()
    pceMax = self.data.pce_mframe.values.astype(int).max()
    pce = np.arange(pceMin, pceMax+1, 1)*1000
    pulse = np.arange(1,201,1)
    pceArr = np.full((pulse.shape[0], pce.shape[0]), pce).T.flatten()
    pulseArr = np.full((pce.shape[0], pulse.shape[0]), pulse).flatten()
    waveArr = pceArr + pulseArr
    waveArr = waveArr[(waveArr >= self.data.index.min()) & \
                      (waveArr <= self.data.index.max())]
    ## Create NaN dataframe for missing waves
    noPho = waveArr[np.in1d(waveArr, self.data.index)==False]
    noPho = pd.DataFrame({'waveID':noPho}).set_index('waveID')
    for l in self.data.columns:
      noPho.loc[:,self.data.columns] = -9999.
    ## Add missing waves to main dataframe
    self.data = self.data.append(noPho).sort_values('waveID')

  def getBeamStrength(self, h5):
    ''' Identify if strong or weak beam '''
    print('Identifying beam strength')
    if h5['orbit_info/sc_orient'][0] == 0:
      self.strong = self.gtx[-1]=='l'
    elif h5['orbit_info/sc_orient'][0] == 1:
      self.strong = self.gtx[-1]=='r'
    else:
      self.strong = 'N/A'
    self.data['strong'] = bool(self.strong)

  def getTelemetry(self, h5):
    ''' Get information describing top and bottom of telemetry window '''
    print('Retrieving telemetry window elevations')
    temp = h5['/'.join([self.gtx,'bckgrd_atlas'])]
    ## Locate telemetry window top and bottom
    tlm = pd.DataFrame({'pce_mframe': temp['pce_mframe_cnt'][:],
                        'bcr': temp['bckgrd_counts_reduced'][:],
                        'bhr': temp['bckgrd_int_height_reduced'][:],
                        'top1': temp['tlm_top_band1'][:],
                        'hgt1': temp['tlm_height_band1'][:],
                        'top2': temp['tlm_top_band2'][:],
                        'hgt2': temp['tlm_height_band2'][:]})
    tlm['base1'] = tlm.top1-tlm.hgt1
    tlm['base2'] = tlm.top2-tlm.hgt2
    mask1 = tlm.hgt1 < .01 ## True if band 1 not used
    mask2 = tlm.hgt2 < .01 ## True if band 2 not used
    maska = (mask1==False) & mask2 ## True if band 1 only
    maskb = mask1 & (mask2==False) ## True if band 2 only
    maskc = (mask1==False) & (mask2==False) ## True if both bands
    maskd = mask1 & mask2 ## True if neither
    ## Create empty arrays to store telemetry top & bottom
    top = np.empty(tlm.shape[0]); base = np.empty(tlm.shape[0])
    ## Where only band 1
    top[maska] = tlm.top1.values[maska]
    base[maska] = tlm.base1.values[maska]
    ## Where only band 2
    top[maskb] = tlm.top2.values[maskb]
    base[maskb] = tlm.base2.values[maskb]
    ## Where both
    top[maskc] = tlm[['top1','top2']].max(axis=1).values[maskc]
    base[maskc] = tlm[['base1','base2']].min(axis=1).values[maskc]
    ## Where neither
    top[maskd] = np.nan; base[maskd] = np.nan
    ## Put in dataframe
    tlm['top'] = top; tlm['base'] = base
    ## Reduce dataframe to single value per pce_mframe
    tlm = tlm.groupby('pce_mframe').mean()
    ## Reindex dataframe with pce_mframe for each photon
    tlm = tlm.reindex(index=self.data.pce_mframe)
    self.data['tlmTop'] = tlm.top.values
    self.data['tlmBase'] = tlm.base.values
    self.data['bcr'] = tlm.bcr.values
    self.data['bhr'] = tlm.bhr.values
  
  def getLandCoverClass(self, geo):
    ''' Identify if over ice- and water-free land '''
    print('Getting land / non-land mask')
    ## Get ATL03 land cover class (20m resolution)
    lcc = pd.DataFrame({'segment_id':geo['segment_id'][:], 'land':geo['surf_type'][:,0]==True})
    lcc = lcc.set_index('segment_id')
    self.data['land'] = lcc.loc[self.data.segment_id].land.values

  def getAtl08(self):
    ''' Load additional information from ATL08 data product '''
    h5 = h5py.File(self.atl08File, 'r')
    self.getMasks(h5['/'.join([self.gtx,'land_segments'])])
    self.getSegments(h5['/'.join([self.gtx, 'land_segments'])])
    self.getClassification(h5['/'.join([self.gtx,'signal_photons'])])
    h5.close()
  
  def getMasks(self, ls):
    ''' Load ATL08 masks '''
    print('Reading mask data')
    ## Get ATL08 masks and flags
    df = pd.DataFrame({'seg_id_beg': ls['segment_id_beg'][:],
                       'watermask': ls['segment_watermask'][:],
                       'msw_flag': ls['msw_flag'][:],
                       'layer_flag': ls['layer_flag'][:],
                       'cloud_flag': ls['cloud_flag_atm'][:],
                       'cloud_fold': ls['cloud_fold_flag'][:],
                       'landcover': ls['segment_landcover'][:],
                       'snowcover': ls['segment_snowcover'][:],
                       'night': ls['night_flag'][:],
                       'sat': ls['sat_flag'][:]})
    ## Interpolate to get masks as function of segment ID
    watermask = interp1d(df.seg_id_beg, df.watermask,
                         kind='previous', fill_value='extrapolate')
    mswmask = interp1d(df.seg_id_beg, df.msw_flag,
                       kind='previous', fill_value='extrapolate')
    cloudmask = interp1d(df.seg_id_beg, df.layer_flag,
                         kind='previous', fill_value='extrapolate')
    cloudlayers = interp1d(df.seg_id_beg, df.cloud_flag,
                         kind='previous', fill_value='extrapolate')
    cloudfold = interp1d(df.seg_id_beg, df.cloud_fold,
                         kind='previous', fill_value='extrapolate')
    landcover = interp1d(df.seg_id_beg, df.landcover,
                         kind='previous', fill_value='extrapolate')
    snowcover = interp1d(df.seg_id_beg, df.snowcover,
                         kind='previous', fill_value='extrapolate')
    night = interp1d(df.seg_id_beg, df.night,
                         kind='previous', fill_value='extrapolate')
    sat = interp1d(df.seg_id_beg, df.sat,
                   kind='previous', fill_value='extrapolate')
    ## Save to dataframe
    self.data['watermask'] = watermask(self.data.segment_id)
    self.data['mswmask'] = mswmask(self.data.segment_id)
    self.data['cloudmask'] = cloudmask(self.data.segment_id)
    self.data['cloudlayers'] = cloudlayers(self.data.segment_id)
    self.data['cloudfold'] = cloudfold(self.data.segment_id)
    self.data['landcover'] = landcover(self.data.segment_id)
    self.data['snowcover'] = snowcover(self.data.segment_id)
    self.data['night'] = night(self.data.segment_id)
    self.data['saturated'] = sat(self.data.segment_id)
  
  def getSegments(self, ls):
    print('Reading segment terrain info')
    sID = np.copy(ls['segment_id_beg'][:])
    seg = interp1d(sID, np.arange(len(ls['segment_id_beg'][:])).astype(int),
                   kind='previous', fill_value='extrapolate')
    self.data['atl08segment'] = seg(self.data.segment_id)
    for k in list(ls['terrain']):
      if k == 'subset_te_flag':
        fit = interp1d(sID, np.min(ls['terrain/'+k][:], axis=1),
                       kind='previous', fill_value='extrapolate')
      else:
        fit = interp1d(sID, np.copy(ls['terrain/'+k][:]),
                       kind='previous', fill_value='extrapolate')
      self.data[k] = fit(self.data.segment_id)
  
  def getClassification(self, sp):
    ''' Load ATL08 photon classifications '''
    print('Classifying photons')
    df = pd.DataFrame({'classed_pc_flag': sp['classed_pc_flag'][:], 'ph_h': sp['ph_h'][:]},
                      index=pd.MultiIndex.from_arrays([sp['ph_segment_id'][:],
                                                       sp['classed_pc_indx'][:]],
                                                       names=['segment_id', 'segment_cnt']))
    self.data = self.data.set_index(pd.MultiIndex.from_frame(self.data[['segment_id', 'segment_cnt']]))
    self.data['zRelATL08'] = df.ph_h
    self.data['zATL08_local'] = self.data.z-self.data.zRelATL08
    self.data['classification'] = df.classed_pc_flag
    self.data['classification'] = self.data.classification.fillna(-1).astype(int)
    self.data['atl08unclassified'] = (self.data.classification == -1).astype(bool)
    self.data['atl08noise'] = (self.data.classification == 0).astype(bool)
    self.data['atl08ground'] = (self.data.classification == 1).astype(bool)
    self.data['atl08canopy'] = (self.data.classification == 2).astype(bool)
    self.data['atl08toc'] = (self.data.classification == 3).astype(bool)
  
  def coordTransform(self):
    self.reproj = Transformer.from_crs('EPSG:4326', 'EPSG:%s' % self.epsg, always_xy=True)
    self.data['x'], self.data['y'] = self.reproj.transform(self.data.lon.values,
                                                           self.data.lat.values)
  
  def getOffset(self):
    self.dx, self.dy, self.dz = np.loadtxt(self.offsetFile)
    self.data.x += self.dx; self.data.y += self.dy; self.data.z += self.dz
    self.data['lon'], self.data['lat'] = self.reproj.transform(self.data.x.values,
                                                               self.data.y.values,
                                                               direction='inverse')
  
  def writePickle(self):
    self.data.to_pickle(self.photonOut)

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
  for atl03File in p.atl03List:
    for gtx in p.gtxList:
      date = atl03File.split('/')[-1][6:14]
      offsetFile = os.path.join(p.offsetDir, '.'.join([date, gtx, 'xyz']))
      if os.path.exists(offsetFile):
        atl = atl03(atl03File, gtx)
        atl.writePickle()
 
