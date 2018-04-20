import time
import pandas as pd #pandas (0.22.0)
import numpy as np #numpy (1.14.1)
import matplotlib.pyplot as plt
import matplotlib as mpl
import geopy as gp
import datetime as dt
import re
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from copy import deepcopy
from mpl_toolkits.basemap import Basemap
from datetime import datetime
from os import listdir
from os.path import isfile, join

# Set ipython's max row display to
pd.set_option('display.max_row', 12)
# Set iPython's max column width to:
pd.set_option('display.max_columns', 6)
start_time = time.time()
# matplot settings
plt.rcParams['figure.figsize'] = (9, 16)
mpl.style.use('ggplot')

mypath = r'/users/joebuty/documents/joe/TuneIn/'

#read csv with door location data
o_df = pd.read_csv(mypath + 'formatted_sf_biz.csv')
loc_df = o_df.copy()
# print list(loc_df)
# Slice dataframe
loc_df = loc_df.ix[:,['coordinates', 'Unnamed: 0', 'Street Location']]
# print  loc_df

loc_df.columns = ['coordinates', 'index', 'location_address']
loc_df = loc_df.set_index('index')
crd_df = loc_df['coordinates'].str.strip('()') \
                   .str.split(', ', expand=True) \
                   .rename(columns={0:'Latitude', 1:'Longitude'})
crd_df = crd_df.dropna()
crd_df['Latitude'] = crd_df['Latitude'].values.astype(float)
loc_df['Latitude'] = crd_df['Latitude']
crd_df['Longitude'] = crd_df['Longitude'].values.astype(float)
loc_df['Longitude'] = crd_df['Longitude']

#remove any outlying points from geopy call.
crd_df = crd_df[(abs(crd_df['Latitude'] - 37) < 1) & (abs(crd_df['Longitude'] + 122) < 1)]

lat = crd_df['Latitude']
lon = crd_df['Longitude']

# determine range to print based on min, max lat and lon of the data
margin = .01 # buffer to add to the range
lat_min = min(lat) - margin
lat_max = max(lat) + margin
lon_min = min(lon) - margin
lon_max = max(lon) + margin

def build_basemap():
	"""
	Return a basemap object for plotting.
	"""
	plt.figure(figsize=(12,6))
	sf = Basemap(projection='merc',
				llcrnrlon = lon_min,     # lower-left corner longitude
				llcrnrlat = lat_min,       # lower-left corner latitude
				urcrnrlon = lon_max,      # upper-right corner longitude
				urcrnrlat = lat_max,       # upper-right corner latitude
				resolution = 'h',
				area_thresh = 100000.0)
	sf.drawmapboundary(fill_color='#46bcec')
	sf.fillcontinents(color='white', lake_color='#46bcec', zorder=1)
	sf.drawcoastlines()
	return sf
sf_1 = build_basemap()
# Set the coords for business locations
datax, datay = np.array(crd_df['Longitude']), np.array(crd_df['Latitude'])
xy = np.vstack([datax, datay])
biz_lon, biz_lat = sf_1(datax, datay)

# Build plot
#################################################################################
sf_1.scatter(biz_lon, biz_lat, s=1, marker='o', color = 'b', zorder=2)
plt.title('Business Locations in San Francisco',
          fontsize=18)
#################################################################################
# Number of clusters
k = 10
# X coordinates of random centroids
C_x = np.random.uniform(lon_min, lon_max, size=k)
# Y coordinates of random centroids
C_y = np.random.uniform(lat_min, lat_max, size=k)
start_coords = map(list, zip(*[C_x,C_y]))
start_coords = np.array(start_coords)
def KMeansGeo(df, n_clusters):
    """
    Calculates KMeans for geographic coordinates.
    """
    print start_coords.shape
    km = KMeans(n_clusters=n_clusters)#, init=start_coords)
    return km.fit(df[['Latitude', 'Longitude']])

# Set the coords for kmeans centroids
model = KMeansGeo(crd_df, k)
lat, lon = zip(*model.cluster_centers_)
km_lon, km_lat = sf_1(lon, lat)
closest_clusters = model.predict(crd_df[['Latitude', 'Longitude']])
sf_2 = build_basemap()
sf_2.scatter(biz_lon, biz_lat, s=5, marker='o', c = closest_clusters, alpha=0.75, \
			cmap='rainbow', zorder=5, label='Business Location')
sf_2.scatter(km_lon, km_lat, s=200, marker='o', color="r", alpha=0.75, label='Cluster Centroid', zorder=4)
plt.title('Business Clusters', fontdict={'fontsize': 18})
plt.legend()
#################################################################################
sf_3 = build_basemap()
density = gaussian_kde(xy)(xy)

sf_3.hexbin(biz_lon, biz_lat, C=density, gridsize=45, cmap=plt.cm.jet, zorder=6)
plt.title('Business Location Density in San Francisco',
          fontdict={'fontsize': 18})
cb = plt.colorbar(sf_3)
cb.set_label('counts')

print '\n', 'Total Runtime:', '\n'
print("--- %s seconds ---" % (time.time() - start_time))
plt.show()
