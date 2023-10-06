#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 16:00:14 2023

@author: Ted
"""

import json
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

from glob import glob
from tqdm import tqdm
from pathlib import Path
from datetime import timedelta
from shapely.geometry import Point, LineString
from math import radians, sin, cos, sqrt, atan2

#spherical distance
def haversine_distance(lat1_d,lon1_d,lat2_d,lon2_d):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = radians(lat1_d), radians(lon1_d)
    lat2, lon2 = radians(lat2_d), radians(lon2_d)

    # Radius of the Earth in kilometers
    earth_radius_km = 6371.0

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance_km = earth_radius_km * c

    return distance_km

def point_polyline_distance(point: Point, line: LineString) -> int:
    coords = line.coords
    distances = [haversine_distance(point.y, point.x, y, x) for x,y in coords]
    # Return the 2nd closest distance. for a group of continuous polylines, 
    # the 1st closest distance will be shared by two lines.
    return sorted(distances)[1]

#dataframe connecting different hydrography datasets
matchups = gpd.read_file('./hydrography_matchups.csv',
                         GEOM_POSSIBLE_NAMES='nhd_geom',
                         KEEP_GEOM_COLUMNS='NO')
matchups.crs = 'EPSG:4326'

gage_loc = pd.read_csv('./gage_location/gage_location.csv')
gage_loc['USGS_CODE'] = gage_loc['SOURCE_FEA'].apply(lambda x: (f'USGS-{x}'))
gage_loc['geometry'] = [Point(lon, lat) for lon, lat in zip(gage_loc['X'], gage_loc['Y'])]




# %% RSR

columns = ['ID',
           'datetime',
           'landsatID',
           'Aerosol',
           'sd_Aerosol',
           'Surface_temp_kelvin',
           'pixel_qa',
           'clouds',
           'dswe',
           'hillShadow',
           'pCount_dswe1',
           'pCount_shadow',
           'IMAGE_QUALITY',
           'CLOUD_COVER',
           'SUN_ELEVATION',
           'SUN_AZIMUTH',
           'sat',
           'Red_raw',
           'Green_raw',
           'Blue_raw',
           'Nir_raw',
           'Swir1_raw',
           'Swir2_raw',
           'Red',
           'Green',
           'Blue',
           'Nir',
           'Swir1',
           'Swir2',
           'hue',
           'saturation',
           'bright',
           'bright_tot',
           'dw']

rsr = pd.read_parquet('./river_sr/LC02_polygons.parquet', columns=columns)
rsr['date'] = rsr['datetime'].dt.normalize()
rsr = rsr.set_index('date')
rsr_features = columns[3:]


# %% WQP

# Get dict of non-standard timezone UTC offsets
timezone_df = pd.read_csv('./wqp/wqp_time_zones.csv')
timezones = timezone_df.set_index("Code")["Offset"].to_dict()

def parse_dates(row):
    if row['TimeZoneCode'] in list(timezones.keys()):
        offset = timezones[row['TimeZoneCode']]
    else:
        return None
    date_col = str(row['SampleDate'])
    time_col = str(row['SampleTime'])
    datetime_str = f"{date_col} {time_col}"
    datetime_value = pd.to_datetime(datetime_str, utc=True)+timedelta(hours=offset)
    return datetime_value

#WQP comes with clunky column names
column_mapping = {
    'OrganizationIdentifier':'Organization',
    'ActivityMediaSubdivisionName':'Media',
    'ActivityStartDate':'SampleDate',
    'ActivityStartTime/Time':'SampleTime',
    'ActivityStartTime/TimeZoneCode':'TimeZoneCode',
    'MonitoringLocationIdentifier':'LocationID',
    'CharacteristicName':'SampleType',
    'ResultSampleFractionText':'Fraction',
    'ResultMeasureValue':'Value',
    'ResultMeasure/MeasureUnitCode':'Units',
    'USGSPCode':'USGSPCode'
    }
    
    
sample_dict = {"ssc (mg/l)":
                   {"SampleType":["Fixed suspended solids",
                                  "Sediment", 
                                  "Total solids",
                                  "Total suspended solids", 
                                  "Suspended Sediment Concentration (SSC)"],
                    "Units":["mg/l","mg/L"]},
               "flux (tons/day)":
                   {"SampleType":["Suspended Sediment Discharge"],
                    "Units":["tons/day"]},
                
               "turbidity (NTU)":
                   {"SampleType":["Turbidity",
                                       "Turbidity Field"],
                    "Units":["NTU","FNU","FTU"]}}

def load_wqp(file):
    tmp = pd.read_csv(file, usecols=list(column_mapping.keys()), dtype=str)
    tmp = tmp.rename(columns = column_mapping)
    
    for measure, values in sample_dict.items():
        sampleMask = tmp["SampleType"].isin(values["SampleType"])
        unitMask = tmp["Units"].isin(values["Units"])
        tmp[measure] = pd.to_numeric(tmp[sampleMask & unitMask]["Value"],errors='coerce')
    measureMask = tmp[sample_dict.keys()].notna().any(axis=1)   
    tmp = tmp[measureMask]
    
    # Drop locations that have less than a set number of observations
    nObs = tmp.LocationID.value_counts()
    valid_locations = (nObs[nObs>100]).index.values
    tmp = tmp[tmp["LocationID"].isin(valid_locations)]
    
    return tmp

wqp_files = glob("./wqp/raw/*.csv")
wqp = pd.DataFrame()
for file in tqdm(wqp_files, desc="Loading WQP"):
    wqp = pd.concat([wqp,load_wqp(file)], axis=0, ignore_index=True)

# Calculate timezone correct date and add day.
tqdm.pandas(desc="Parsing dates")
wqp['datetime'] = wqp.progress_apply(parse_dates, axis=1)

# Drop invalid dates
n_invalid = np.sum(wqp.datetime.isna())
print(f"Dropping {n_invalid} ({n_invalid/len(wqp):0.2f}%) observations due to invalid time")
wqp = wqp.dropna(subset=['datetime'])

# tmp = tmp[tmp['datetime'].dt.year <= 2019]
wqp['date'] = wqp['datetime'].dt.normalize()
wqp = wqp.set_index('date')


#read in wqp station data and convert to geodataframe
wqp_locs = pd.read_csv('./wqp/station.csv').rename(columns={"MonitoringLocationIdentifier":"LocationID"})
wqp_site_list = np.unique(wqp.LocationID)
wqp_locs = wqp_locs[wqp_locs['LocationID'].isin(wqp_site_list)]


wqp_locs['geometry'] = [Point(lon, lat) for lat, lon in zip(wqp_locs['LatitudeMeasure'], wqp_locs['LongitudeMeasure'])]
wqp_locs = gpd.GeoDataFrame(wqp_locs, geometry='geometry')
valid_geometry = wqp_locs['geometry'].is_valid
wqp_locs = wqp_locs[valid_geometry]
wqp_locs.crs = 'EPSG:4326'

wqp_locs_buff = wqp_locs.copy()[['LocationID','geometry']]
wqp_locs_buff['geometry'] = wqp_locs_buff.geometry.buffer(0.05)
matchups_filt = gpd.sjoin(matchups[['nhd_comid',
                                    'nhd_reachcode',
                                    'riversr_id',
                                    'merit_comid',
                                    'hybas_id',
                                    'geometry']],wqp_locs_buff,how='inner')

#deals with inconsistent types in the matchup df
def s2id(s):
    if s == '':
        return np.nan
    return int(float(s))

# Merge all of our locations references together.
locs = pd.DataFrame({'wqp_id':np.unique(matchups_filt['LocationID'])})
for row in tqdm(locs.itertuples(), total=len(locs), desc='Merging WQP and NHD'):
    matchups_ix = matchups_filt[matchups_filt['LocationID']==row.wqp_id]
    wqp_ix = wqp_locs[wqp_locs.LocationID == row.wqp_id]
    
    distances = [point_polyline_distance(wqp_ix.geometry, line.geometry) for line in matchups_ix.itertuples()]
    min_idx = distances.index(min(distances))
    
    locs.at[row.Index,"nhd_comid"] = s2id(matchups_ix.iloc[min_idx]['nhd_comid'])
    locs.at[row.Index,"nhd_reachcode"] = s2id(matchups_ix.iloc[min_idx]['nhd_reachcode'])
    locs.at[row.Index,"riversr_id"] = s2id(matchups_ix.iloc[min_idx]['riversr_id'])
    locs.at[row.Index,"merit_comid"] = s2id(matchups_ix.iloc[min_idx]['merit_comid'])
    locs.at[row.Index,"hybas_id"] = s2id(matchups_ix.iloc[min_idx]['hybas_id'])
    locs.at[row.Index,"geometry"] = matchups_ix.iloc[min_idx]['geometry']
    
# %% Get full hydrobasin outlines for each site
hybas_file = "./hydrobasins/hybas_na_lev12_v1c.shp"
hybas = gpd.read_file(hybas_file)

upstream_ids = dict()
upstream = gpd.GeoDataFrame(locs["wqp_id"]).set_index("wqp_id")
for row in tqdm(locs.itertuples(), total=len(locs), desc='Finding upstream HydroBasins'):
    #initial value is at the sample location
    all_up = hybas[hybas.HYBAS_ID == row.hybas_id]
    
    # all_up = gpd.GeoDataFrame(all_up, geometry="geometry")
    next_up = hybas[hybas["NEXT_DOWN"] == row.hybas_id]
    while not next_up.empty:
        all_up = pd.concat([all_up, next_up], ignore_index=True)
        #find the new next_up
        next_up = hybas[hybas["NEXT_DOWN"].isin(next_up["HYBAS_ID"])]

    upstream.at[row.wqp_id,"geometry"] = all_up.unary_union
    upstream_ids[row.wqp_id] = all_up["HYBAS_ID"].to_list()

        
# Save for use with Caravan scripts.
save_path = "./preprocessed/metadata/basin_polygons/"
upstream = upstream.reset_index().rename(columns={"wqp_id":"gauge_id"})
upstream.crs = 'EPSG:4326'
upstream.to_file(save_path + "upstream_basin_polygons.shp")

# Dump the upstream id dict into json.
with open(save_path+"upstream_ids.json", "w") as file:
    json.dump(upstream_ids, file)

# %%
t1 = "1979-01-01"
t2 = "2019-12-31"
date_range = pd.date_range(start=t1, end=t2, freq='D', tz='UTC')

grfr_path = './grfr/output_pfaf_07_1979-2019.nc'
grfr = xr.open_dataset(grfr_path) 

save_dir = Path("./preprocessed")

#write location HYBAS IDs out to a txt file.
site_file_path = save_dir / "metadata" / "site_list.txt"
# Load existing site_list from a text file if it exists
if site_file_path.is_file():
    with open(site_file_path, 'r') as file:
        site_list = [line.strip() for line in file]
else:
    site_list = []

# trim to data we will use
wqp = wqp[wqp['LocationID'].isin(locs.wqp_id)]
rsr = rsr[rsr['ID'].isin(locs.riversr_id)]

for row in tqdm(locs.itertuples(),total=len(locs), desc="Writing files"):
    if np.isnan(row.merit_comid) or np.isnan(row.hybas_id):
        continue
    
    # Check if the current site ID is already in site_list
    if row.wqp_id not in site_list:
        # Write the current site ID to the text file
        site_list.append(row.wqp_id)
        with open(site_file_path, 'a') as file:
            file.write(str(row.wqp_id) + '\n')

    
    # Check that we have not already processed this basin.
    pickle_file_path = save_dir / "dataframe" / f"{row.wqp_id}.pickle"
    nc_file_path = save_dir / "netcdf" / f"{row.wqp_id}.nc"
    if pickle_file_path.is_file(): #& nc_file_path.is_file():
        continue
    
    
    # Filter our datasets to this observation location.
    wqp_subset = wqp[wqp['LocationID']==row.wqp_id][list(sample_dict.keys())]
    
    rsr_subset = rsr[rsr['ID']==row.riversr_id][rsr_features]
    
    grfr_subset_xr = grfr.sel(time=slice(t1,t2),rivid=row.merit_comid)
    grfr_subset = grfr_subset_xr.to_dataframe()['Qout'].tz_localize('UTC')  
    
    # Merge all of the filtered datasets together. 
    df = pd.DataFrame(index=date_range)
    df = pd.merge(df, rsr_subset, left_index=True, right_index=True, how='left')
    df = pd.merge(df, grfr_subset, left_index=True, right_index=True, how='left')
    df = pd.merge(df, wqp_subset, left_index=True, right_index=True, how='left')  
    
    # Multiple obs per day can cause repeated days in index, which breaks the infer_freq method.
    df['day'] = df.index.date
    df_daily = df.groupby('day').mean(numeric_only=True)
    
    # Convert the index to a DatetimeIndex and set the timezone to UTC
    df_daily.index = pd.to_datetime(df_daily.index).tz_localize('UTC')
    
    if pd.infer_freq(df_daily.index) !=  "D":
        raise RuntimeError("Non-daily time freq found")
    
    # Save dataframe
    if not pickle_file_path.is_file():
        df_daily.to_pickle(pickle_file_path)
    
    # Save netcdf
    # if not nc_file_path.is_file():
    #     ds_daily = xr.Dataset.from_dataframe(df_daily)
    #     ds_daily.to_netcdf(nc_file_path)
    
grfr.close()


#write full location data out as a shapefile
locs_gdf = gpd.GeoDataFrame(locs[locs.wqp_id.isin(site_list)],geometry='geometry')
locs_gdf.to_file("./preprocessed/metadata/basin_list.shp")


