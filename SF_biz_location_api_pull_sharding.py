import time
import pandas as pd #pandas (0.22.0)
import numpy as np #numpy (1.14.1)
import geopy as gp
import datetime as dt
import re
from datetime import datetime
from os import listdir
from os.path import isfile, join

### initialize and settings
start_time = time.time()
print '\n\n\t', 'Start Time: ', start_time, '\n'
# Set ipython's max row display to
pd.set_option('display.max_row', 24)
# Set iPython's max column width to:
pd.set_option('display.max_columns', 50)

mypath = r'/users/joebuty/documents/joe/TuneIn/'
shard_file_path = r'/users/joebuty/documents/joe/TuneIn/shards/'

# zipcodes that are located in SF. 
o_df = pd.read_csv(mypath + 'sf business dataset.csv')
# print df['Business Location']

# see if there are shards of the converted address dataframe.
all_files = [f for f in listdir(shard_file_path) if isfile(join(shard_file_path, f))]
all_shard_files = sorted([f for f in all_files if f.find('.csv') == len(f) - 4])
current_shard = len(all_shard_files)

## Break apart the business location (source Zillow)
#### https://www.zillow.com/browse/homes/ca/san-francisco-county/
## this seems to be the best choice since the list(df['City'].unique()) shows a very
## interpretive set of spellings. I ASSUME that zips are more accurate based on intuition.
sf_zips = [94102,
	94104,
	94103,
	94105,
	94108,
	94107,
	94110,
	94109,
	94112,
	94111,
	94115,
	94114,
	94117,
	94116,
	94118,
	94121,
	94123,
	94122,
	94124,
	94127,
	94126,
	94129,
	94131,
	94133,
	94132,
	94134,
	94139,
	94143,
	94151,
	94159,
	94158,
	94188,
	94177]
df = o_df[np.isfinite(o_df['Source Zipcode'])].copy()
df['Source Zipcode'] = df['Source Zipcode'].astype('int64')
df = df[df['Source Zipcode'].isin(sf_zips)]
df_loc = df['Business Location'].str.split(r'\r', expand=True)
df.ix[:,'Business Location'] = df_loc[0] + r'\r' + df_loc[1]
df['Business coordinates'] = df_loc[2]
# print list(df['Source Zipcode'].unique())
df['loc_addr'] = df['Street Address'] + ' ' + df['City']
df.to_csv(mypath + 'df_sf_biz.csv')
locations = pd.DataFrame()
locations = df[['loc_addr']]

# print 
uni_loc = pd.DataFrame(locations['loc_addr'].unique())
print 'SHAPE SIZE Oringinal:\n', o_df.shape, '\n'
print 'NEW SIZE: SF zips and unique addresses: \n', uni_loc.shape, '\n'*2
uni_loc.columns = ['Location']

geolocator = gp.geocoders.Nominatim()

step_size = 10
def find_coordinates(idf):
	try:
		idf['coordinates'] = idf['Location'].apply(geolocator.geocode)
		idf['coordinates'] = \
			idf['coordinates'].apply(lambda x: np.nan \
								if x is None else(x.latitude, x.longitude))
	except gp.exc.GeocoderTimedOut as e:
	    print 'Error: geocode failed on input message: {}'.format(e)	
	except gp.exc.GeocoderServiceError as e:
		print 'Error: geocode failed on input message: {}'.format(e)	
	return idf

uni_loc['coordinates'] = np.nan

if current_shard >0:
	init_idx = 0
	print 'Shards already found:'
	print all_shard_files
	for shard_fname in all_shard_files:
		shard_idx =  [int(s) for s in re.findall(r'\d+', shard_fname)]
		if shard_idx[1] > init_idx:
			init_idx = max(shard_idx[1], init_idx)
# 			print 'Index calculated to: ', init_idx
else:
	init_idx = 0

start_shard = current_shard
shard_count = (uni_loc.shape[0] - init_idx)/step_size
print 'Initial Index: ', init_idx
print 'This part is very slow like a second per line... Go to make coffee and shower.'
print 'Running:'


for i in range(0, 10):
	start_idx = (i * step_size) + 1
	print 'shard {} of {}'.format(i, shard_count)
	if init_idx > 0: # if shards have been previously created, add those to the index
# 		print i, step_size, start_idx
		start_idx += init_idx
	df_shard = uni_loc[start_idx : start_idx + step_size].copy()
	print 'call index {}:{}'.format(str(start_idx), str(start_idx + step_size))
	df_shard = pd.concat([df_shard, find_coordinates(df_shard)])
	if i % 3 == 0:
		current_shard +=1
		shard_file_name = shard_file_path+str('shard_{}_end_idx_{}'.format(current_shard, start_idx + step_size - 1)+'.csv')
		print 'write file: ', shard_file_name
		df_shard.to_csv(shard_file_name)
	else:
		pass
		print i, '\t', i, shard_count, i % shard_count
print '\n'

### 429 Error Too many requests. For more on usage see below link.

print '\n', 'Total Runtime:', '\n'
print("--- %s seconds ---" % (time.time() - start_time))