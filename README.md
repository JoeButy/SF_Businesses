# SF_Businesses

This repo is meant to analyse Business locations in San Francisco CA, data set [https://data.sfgov.org/Economy-and-Community/Registered-Business-Locations-San-Francisco/g8m3-pdis].

Conver the street addresses given in the data set to geolocations (latitude, longitude) with the file 'SF_biz_location_api_pull_sharding.py'
Since the API was running slowly, the a sharding method was employed to avoid data loss due to time out, or over requesting. These shards are writen to a csv.

Compiling the csvs that have been obtained is the next step. These CSVs have the index of the original data set so they can be checked for completeness, and a recursive API call should be additionally implelented when index is missing. This file can actually be run while the API pull is still sharding the data since only finished .csv filetypes are compiled.
To compile the data run the 'compile_shards_SF_biz.py' file.

Once a compiled CSV has been created run the data viz 'SF_biz_locations_viz.py' file.
