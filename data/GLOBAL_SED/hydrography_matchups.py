#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge identifiers and metadata from NHD, MERIT, HydroATLAS and USGS sites

@author: Ted
"""

from tqdm import tqdm
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon, box

nhd_file = "./river_sr/nhd_grwl_full_20191002.shp"
nhd_cols = {'COMID':'nhd_comid',
            'REACHCODE':'nhd_reachcode',
            'ID':'riversr_id',
            'TotDASqKM':'nhd_area',
            'geometry':'geometry'}
nhd = gpd.read_file(nhd_file)[list(nhd_cols.keys())].rename(columns=nhd_cols)

#only basin 7 (NA) for now
merit_file = "./merit_basins/riv_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp"
merit = gpd.read_file(merit_file).rename(columns={'COMID':'MERIT_COMID'})

#only NA for now
hybas_file = "./hydrobasins/hybas_na_lev12_v1c.shp"
hybas = gpd.read_file(hybas_file)


# %%
BUFFER = 0.05 #degrees
AREA_TOL = 0.2 #normalized diff

# Create an empty DataFrame to store results
matchups = nhd.copy()
matchups['merit_comid'] = np.nan
matchups['merit_area'] = np.nan
matchups['merit_geom'] = np.nan
matchups['hybas_id'] = np.nan
matchups['hybas_area'] = np.nan
matchups['hybas_geom'] = np.nan

# Throws a warning becuase buffering in degrees is bad, but this is good enough for us.
nhd_buff = nhd.copy()
nhd_buff['geometry'] = nhd_buff.geometry.buffer(BUFFER)

#function here would be better ,but ever forward.

# Iterate over each centerline segment. clunky
with tqdm(total=len(nhd),desc="Merging MERIT") as pbar:
    merit_filt = gpd.sjoin(merit,nhd_buff,how='inner')
    for idx,row in nhd.iterrows():
        #intersections
        merit_ix = merit_filt[merit_filt.nhd_comid==row.nhd_comid]
        if len(merit_ix)>0:
            area_diff = np.abs((merit_ix['uparea'] - row['nhd_area']).values)
            #check that the upstream area difference is less than 10%
            if min(area_diff) < row.nhd_area*AREA_TOL:
                merit_best = merit_ix[area_diff==min(area_diff)]
                matchups.at[idx,'merit_comid'] = merit_best['MERIT_COMID']
                matchups.at[idx,'merit_area'] = merit_best['uparea']
                matchups.at[idx,'merit_geom'] = merit_best.geometry.values[0]
        pbar.update(1)
      
# Iterate over each centerline segment. clunky
with tqdm(total=len(nhd),desc="Merging HYBAS") as pbar:
    hybas_filt = gpd.sjoin(hybas,nhd_buff,how='inner')
    for idx,row in nhd.iterrows():
        #intersections
        hybas_ix = hybas_filt[hybas_filt.nhd_comid==row.nhd_comid]
        if len(hybas_ix)>0:
            area_diff = np.abs((hybas_ix['UP_AREA'] - row['nhd_area']).values)
            #check that the upstream area difference is less than 10%
            if min(area_diff) < row.nhd_area*AREA_TOL:
                hybas_best = hybas_ix[area_diff==min(area_diff)]
                matchups.at[idx,'hybas_id'] = hybas_best['HYBAS_ID']
                matchups.at[idx,'hybas_area'] = hybas_best['UP_AREA']
                matchups.at[idx,'hybas_geom'] = hybas_best.geometry.values[0]
        pbar.update(1)
        

matchups = pd.DataFrame(matchups).rename(columns={'geometry':'nhd_geom'})
matchups.to_csv('./hydrography_matchups.csv')


# %%
import matplotlib.pyplot as plt
plt.close('all')


fig, ax = plt.subplots(1, 1, figsize=(10, 6))


matchups.plot(ax=ax,color='red')
matchups.plot(column="merit_area", cmap='viridis', ax=ax, legend=True)

# Show the plot
plt.show()















