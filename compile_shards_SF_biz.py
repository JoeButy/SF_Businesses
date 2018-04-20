import time
import pandas as pd #pandas (0.22.0)
import numpy as np #numpy (1.14.1)
import datetime as dt
import re
from os import listdir
from os.path import isfile, join
# Set ipython's max row display to
pd.set_option('display.max_row', 6)
# Set iPython's max column width to:
pd.set_option('display.max_columns', 6)

mypath = r'/users/joebuty/GitHub/SF_Businesses/'
shard_file_path = r'/users/joebuty/GitHub/SF_Businesses/shards/'
df = pd.read_csv(mypath + 'df_sf_biz.csv')

##############################
#######	COMPILE SHARDS #######
##############################
finished_shards = [f for f in listdir(shard_file_path) if isfile(join(shard_file_path, f))]
finished_shard_files = sorted([f for f in finished_shards if f.find('.csv') == len(f) - 4])

final_loc_df = pd.DataFrame()
for shard in finished_shard_files:
	temp_df = pd.read_csv(shard_file_path + shard)
	final_loc_df = pd.concat([temp_df, final_loc_df])

print final_loc_df.shape
final_loc_df.columns = ['index', 'Street Location', 'coordinates']
final_loc_df = final_loc_df.set_index('index', drop=True)
final_loc_df = final_loc_df.sort_index()
# print list(final_loc_df)
# print final_loc_df
print '\n' *2 
print 'Location coordinates found.'

df_loc = pd.concat([final_loc_df, df], axis=1, join='inner')

print list(df_loc)

df_loc = df_loc[['DBA Name', 
	'Street Address',
	'Business Start Date', 
	'Business End Date',
	'Location Start Date',
	'Location End Date',
	'Supervisor District',
	'Neighborhoods - Analysis Boundaries',
	'Business Location',
	'Street Location',
	'coordinates']]

df_loc = df_loc[~df_loc.index.duplicated(keep='first')]
df_loc.to_csv(mypath + 'formatted_sf_biz.csv')

print df_loc

