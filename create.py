import os, pdb
import numpy as np
from params import params
from atl03 import atl03
from footprints import footprints
from waves import waves

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
      if os.path.exists(offsetFile)&(int(str(date)[4:6])>5)&(int(str(date)[4:6])<10):
        atl = atl03(atl03File, gtx)
        atl.writePickle()
        del atl
        coords = footprints(region, date, gtx)
        del coords
        w = waves(region, date, gtx)
        w.createWaves()
        nFiles = np.sum([os.path.exists(w) for w in w.waveFiles])
        if nFiles > 0:
          w.combineWaves()
          w.writeZ()
        del w
