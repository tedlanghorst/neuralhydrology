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

save_dir = Path("./preprocessed/hybas")


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
    
    
sample_dict = {"ssc_(mg/l)":
                   {"SampleType":["Fixed suspended solids",
                                  "Sediment", 
                                  "Total solids",
                                  "Total suspended solids", 
                                  "Suspended Sediment Concentration (SSC)"],
                    "Units":["mg/l","mg/L"]},
               "flux_(tons/day)":
                   {"SampleType":["Suspended Sediment Discharge"],
                    "Units":["tons/day"]},
                
               "turbidity_(NTU)":
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
    locs.at[row.Index,"geometry"] = wqp_ix.geometry.values[0]
    
#write full location data out as a shapefile
locs_gdf = gpd.GeoDataFrame(locs,geometry='geometry')
locs_gdf.to_file(save_dir / "metadata" / "basin_matchups.shp")
    
# %% Get full hydrobasin outlines for each site
hybas_file = "./hydrobasins/hybas_na_lev12_v1c.shp"
hybas = gpd.read_file(hybas_file)

upstream_ids = dict()
unique_basins = np.unique(locs["hybas_id"].dropna()).astype(int)
metadata = gpd.GeoDataFrame(index=unique_basins)
for idx in tqdm(metadata.index, desc='Finding upstream HydroBasins'):
    if np.isnan(idx):
        continue
    #initial value is at the sample location
    outlet = hybas[hybas.HYBAS_ID == idx]
    all_up = outlet.copy()
    
    next_up = hybas[hybas["NEXT_DOWN"] == idx]
    while not next_up.empty:
        all_up = pd.concat([all_up, next_up], ignore_index=True)
        #find the new next_up
        next_up = hybas[hybas["NEXT_DOWN"].isin(next_up["HYBAS_ID"])]

    upstream_ids[idx] = all_up.HYBAS_ID.to_list()
    metadata.at[idx,"geometry"] = all_up.unary_union
    metadata.at[idx,"n_bas_up"] = all_up.size
    metadata.at[idx,"area"] = outlet["UP_AREA"].item()
    metadata.at[idx,"order"] = outlet["ORDER"].item()

save_path = "./preprocessed/metadata/basin_polygons/"

# Dump the metadata id dict into json.
with open(save_dir / "metadata" / "basin_polygons" / "upstream_ids.json", "w") as file:
    json.dump(upstream_ids, file)

# %%
t1 = "1979-01-01"
t2 = "2019-12-31"
date_range = pd.date_range(start=t1, end=t2, freq='D', tz='UTC')

grfr_path = './grfr/output_pfaf_07_1979-2019.nc'
grfr = xr.open_dataset(grfr_path) 

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

for basin in tqdm(unique_basins, desc="Writing files"):
    #all observation matchups in basin
    locs_subset = locs[locs.hybas_id == basin]
    
    #check if we failed to pair these locations
    missing_ids = [it[1].isna().all() for it in locs_subset.items()]
    if any(missing_ids):
        continue
    
    # Check if the current site ID is already in site_list
    if basin not in site_list:
        # Write the current site ID to the text file
        site_list.append(basin)
        with open(site_file_path, 'a') as file:
            file.write(str(basin) + '\n')

    
    # Check that we have not already processed this basin.
    pickle_file_path = save_dir / "dataframe" / f"{basin}.pickle"
    nc_file_path = save_dir / "netcdf" / f"{basin}.nc"
    if pickle_file_path.is_file():# & nc_file_path.is_file():
        continue
    
    # Filter our datasets to this observation location.
    wqp_mask = wqp['LocationID'].isin(locs_subset["wqp_id"])
    wqp_subset = wqp[wqp_mask][list(sample_dict.keys())]
    
    rsr_mask = rsr['ID'].isin(locs_subset["riversr_id"].astype(int))
    rsr_subset = rsr[rsr_mask][rsr_features]
    
    rivid_list = np.unique(locs_subset.merit_comid.values)
    rivid_list = rivid_list[~np.isnan(rivid_list)]
    grfr_subset_xr = grfr.sel(time=slice(t1,t2),rivid=rivid_list)
    grfr_subset = grfr_subset_xr.to_dataframe().reset_index('rivid')['Qout'].tz_localize('UTC')
    grfr_subset.name = "grfr_q"
    
    #load the gauge file if it exists
    wqp_id = locs_subset.wqp_id.iloc[0]
    gauge_file = Path(f"./usgs_q/{wqp_id}.pickle")
    if gauge_file.is_file():
        gauge = pd.read_pickle(gauge_file).astype('float64')
        gauge = gauge * (.3048**3) # ft3 to m3
        gauge.name = "usgs_q"
    else:
        gauge = pd.Series(name="usgs_q", dtype="float64")
    
    # Merge all of the filtered datasets together. 
    df = pd.DataFrame(index=date_range)
    df = pd.merge(df, rsr_subset, left_index=True, right_index=True, how='left')
    df = pd.merge(df, grfr_subset, left_index=True, right_index=True, how='left')
    df = pd.merge(df, gauge, left_index=True, right_index=True, how='left')
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



# %%

target_fields = {"turb":"turbidity (NTU)",
                   "ssc":"ssc (mg/l)",
                   "flux":"flux (tons/day)"}
feature_fields = {"sim":"grfr_q",
                  "obs":"usgs_q",
                  "red":"Red",
                  "nir":"Nir"}

#Calculate some more metdata for subsetting basins later
#Loop through the 3 target variables
for basin in tqdm(metadata.index, desc="Calculating metadata"):
    pickle_file_path = save_dir / "dataframe" / f"{basin}.pickle"
    if not pickle_file_path.is_file():
        continue
    data = pd.read_pickle(pickle_file_path)
    
    for t_key, target in target_fields.items():
        metadata.at[basin,f"n_{t_key}"] = np.sum(~np.isnan(data[target]))
        metadata.at[basin,f"min_{t_key}"] = data[target].min()
        metadata.at[basin,f"range_{t_key}"] = data[target].max() - data[target].min()
        
        for f_key, feat in feature_fields.items():
            #number of matchups
            n_matchups = np.sum(~data[[target,feat]].isna().any(axis=1))
            metadata.at[basin,f"n_{f_key}_{t_key}"] = n_matchups
            if n_matchups > 1: 
                metadata.at[basin,f"c_{f_key}_{t_key}"] = data[target].corr(data[feat])
            else:
                metadata.at[basin,f"c_{f_key}_{t_key}"] = np.NaN
  
metadata.crs = 'EPSG:4326'
metadata.sort_values(by='n_turb',ascending=False,inplace=True)
metadata = metadata.reset_index().rename(columns={"index":"hybas"})
metadata.to_file(save_dir / "metadata" / "basin_polygons" / "metadata_basin_polygons.shp")


# %% analyze metadata and make site lists

# metadata = gpd.read_file(save_dir / "metadata" / "basin_polygons" / "metadata_basin_polygons.shp")

area = 10
feature = "ssc"
feature_range = 100
n_sat_obs = 0
q_corr = 0

mask = ((metadata.area>=area) & 
        (metadata[f"range_{feature}"]>=feature_range) &
        (metadata[f"n_red_{feature}"]>= n_sat_obs) &
        (metadata[f"c_obs_{feature}"]>=q_corr))

file_path = f"sites_{feature}_area{area}_range{feature_range}_qcorr{q_corr}_n{n_sat_obs}.txt"

print(np.sum(mask))

# Save the Series to a text file with no delimiters or headers
# metadata.hybas[mask].to_csv(file_path, header=False, index=False)




# %% bandaid to rename columns in exisitng files


new_names = {"ssc_(mg/l)":"ssc (mg/l)","turbidity_(NTU)":"turbidity (NTU)","flux_(tons/day)":"flux (tons/day)"}

pickle_path = save_dir / "dataframe"
all_pickles = list(pickle_path.glob("*.pickle"))

for pickle_fp in tqdm(all_pickles):
    tmp = pd.read_pickle(pickle_fp)
    
    #if we have the old format, rename and save
    if any(tmp.columns == "flux_(tons/day)"):
        tmp = tmp.rename(columns=new_names)
        tmp.to_pickle(pickle_fp)
        
































