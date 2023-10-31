import os
regions = ['deju', 'ornl', 'sodankyla', 'sonoma', 'wref']

for region in regions:
  #os.system('python3 segment.py -r %s && python3 rates.py -r %s && python3 compare.py -r %s' \
  #          % (region, region, region))
  os.system('python3 plot.py -r %s' % (region))
