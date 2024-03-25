#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 11:18:36 2023

@author: Ted
"""

import json
import numpy as np
import pandas as pd
import geopandas as gpd
import fiona

from tqdm import tqdm


with open("./preprocessed/hybas/metadata/basin_polygons/upstream_ids.json", "r") as file:
    upstream_ids = json.load(file)

# Read in hydroatlas
gdb_path = "./hydroatlas/BasinATLAS_v10.gdb"
layer_name = "BasinATLAS_v10_lev12"

hydroatlas = gpd.read_file(gdb_path, layer=layer_name)


# %%
# Open the GDB file and the specified layer using fiona
with fiona.open(gdb_path, layer=layer_name) as src:
    # Get the schema of the layer (which includes field information)
    schema = src.schema
    
    # Print the field names (attributes)
    property_names = schema['properties'].keys()
    print("Available Attributes:")
    for field_name in property_names:
        print(field_name)


# List of features for which we take the area-weighted majority vote
majority_properties = ['clz_cl_smj', # climate zones (18 classes)
                       'cls_cl_smj', # climate strata (125 classes)
                       'glc_cl_smj', # land cover (22 classes)
                       'pnv_cl_smj', # potential natural vegetation (15 classes)
                       'wet_cl_smj', # wetland (12 classes)
                       'tbi_cl_smj', # terrestrial biomes (14 classes)
                       'tec_cl_smj', # Terrestrial Ecoregions (846 classes)
                       'fmh_cl_smj', # Freshwater Major Habitat Types (13 classes)
                       'fec_cl_smj', # Freshwater Ecoregions (426 classes)
                       'lit_cl_smj', # Lithological classes (16 classes)
                      ]

# List of features for which we take the value of the most downstream polygon
pour_point_properties = ['dis_m3_pmn', # natural discharge annual mean
                         'dis_m3_pmx', # natural discharge annual max
                         'dis_m3_pyr', # natural discharge annual min
                         'lkv_mc_usu', # Lake Volume
                         'rev_mc_usu', # reservoir volume
                         'ria_ha_usu', # River area
                         'riv_tc_usu', # River volumne
                         'pop_ct_usu', # Population count in upstream area
                         'dor_pc_pva', # Degree of regulation in upstream area
                        ]

# These HydroSHEDS/RIVERS features are ignored                    
ignore_properties = ['system:index',
                     'COAST', 
                     'DIST_MAIN', 
                     'DIST_SINK', 
                     'ENDO', 
                     'MAIN_BAS', 
                     'NEXT_SINK', 
                     'ORDER_', 
                     'PFAF_ID', 
                     'SORT', 
                     ]

# These features are required to find the most downstream gauge and will be removed later.
additional_properties = ['HYBAS_ID',
                          'NEXT_DOWN',
                          'SUB_AREA',
                          'UP_AREA'
                        ]

# These HydroATLAS features are ignored, mostly because they have a per-polygon 
# counterpart that we include.
upstream_properties = ['aet_mm_uyr', # Actual evapotranspiration
                       'ari_ix_uav', # Global aridity index
                       'cly_pc_uav', # clay fraction soil
                       'cmi_ix_uyr', # Climate Moisture Index
                       'crp_pc_use', # Cropland Extent
                       'ele_mt_uav', # Elevtion
                       'ero_kh_uav', # Soil erosion
                       'for_pc_use', # Forest cover extent
                       'gdp_ud_usu', # Gross Domestic Product
                       'gla_pc_use', # Glacier Extent
                       'glc_pc_u01', # Land cover extent percent per class (22)
                       'glc_pc_u02',
                       'glc_pc_u03',
                       'glc_pc_u04',
                       'glc_pc_u05',
                       'glc_pc_u06',
                       'glc_pc_u07',
                       'glc_pc_u08',
                       'glc_pc_u09',
                       'glc_pc_u10',
                       'glc_pc_u11',
                       'glc_pc_u12',
                       'glc_pc_u13',
                       'glc_pc_u14',
                       'glc_pc_u15',
                       'glc_pc_u16',
                       'glc_pc_u17',
                       'glc_pc_u18',
                       'glc_pc_u19',
                       'glc_pc_u20',
                       'glc_pc_u21',
                       'glc_pc_u22',
                       'hft_ix_u09', # Human Footprint 2009
                       'hft_ix_u93', # Human Footprint 1993
                       'inu_pc_ult', # inundation extent long-term maximum
                       'inu_pc_umn', # inundation extent annual minimum
                       'inu_pc_umx', # inundation extent annual maximum
                       'ire_pc_use', # Irrigated Area Extent (Equipped)
                       'kar_pc_use', # Karst Area Extent
                       'lka_pc_use', # Limnicity (Percent Lake Area)
                       'nli_ix_uav', # Nighttime Lights
                       'pac_pc_use', # Protected Area Extent
                       'pet_mm_uyr', # Potential evapotranspiration
                       'pnv_pc_u01', # potential natural vegetation (15 classes)
                       'pnv_pc_u02',
                       'pnv_pc_u03',
                       'pnv_pc_u04',
                       'pnv_pc_u05',
                       'pnv_pc_u06',
                       'pnv_pc_u07',
                       'pnv_pc_u08',
                       'pnv_pc_u09',
                       'pnv_pc_u10',
                       'pnv_pc_u11',
                       'pnv_pc_u12',
                       'pnv_pc_u13',
                       'pnv_pc_u14',
                       'pnv_pc_u15',
                       'pop_ct_ssu', # population count
                       'ppd_pk_uav', # population density
                       'pre_mm_uyr', # precipitation
                       'prm_pc_use', # Permafrost extent
                       'pst_pc_use', # Pasture extent
                       'ria_ha_ssu', # river area in sub polygon
                       'riv_tc_ssu', # River volumne in sub polygon
                       'rdd_mk_uav', # Road density
                       'slp_dg_uav', # slope degree
                       'slt_pc_uav', # silt fraction
                       'snd_pc_uav', # sand fraction
                       'snw_pc_uyr', # snow cover percent
                       'soc_th_uav', # organic carbon content in soil
                       'swc_pc_uyr', # soil water content
                       'tmp_dc_uyr', # air temperature
                       'urb_pc_use', # urban extent
                       'wet_pc_u01', # wetland classes percent (9 classes)
                       'wet_pc_u02',
                       'wet_pc_u03',
                       'wet_pc_u04',
                       'wet_pc_u05',
                       'wet_pc_u06',
                       'wet_pc_u07',
                       'wet_pc_u08',
                       'wet_pc_u09',
                       'wet_pc_ug1', #  wetland classes percent by class grouping (2 classes)
                       'wet_pc_ug2',
                       'gad_id_smj', # global administrative areas (country id's)
                       ]


use_properties = [p for p in property_names if p not in ignore_properties + upstream_properties]

hydroatlas = hydroatlas[use_properties]
                
# %%

# Create an empty DataFrame with columns named after hydroatlas columns
basin_attributes = pd.DataFrame(columns=use_properties, index=upstream_ids.keys())
basin_attributes.drop(additional_properties, axis=1, inplace=True)

# Iterate through basins
for basin in tqdm(upstream_ids.keys()):
    hydroatlas_subset = hydroatlas[hydroatlas['HYBAS_ID'].isin(upstream_ids[basin])]

    # Initialize a dictionary to store computed values for each property
    basin_data = {}
    
    for prop in hydroatlas_subset.columns:
        # Ignore these auxiliary properties
        if prop in additional_properties:
            continue
        
        values = hydroatlas_subset[prop].to_numpy()
        
        if values.size == 0:
            basin_data[prop] = np.nan
            continue
        
        # Take the outlet value for these properties
        if prop in pour_point_properties:
            basin_data[prop] = values[0]
        
        # For wetland classes, define 'no wetland' as a new class.
        # In the dataset, this value is -999, here we set it to 13
        if prop == 'wet_cl_smj':
            values[values == -9999] = 13
            
        # If all values are -999, set value to NaN
        if np.all(values == -9999):
            basin_data[prop] = np.nan
        else:
            # Majority vote for the class properties
            if prop in majority_properties:
                basin_data[prop] = np.bincount(values[values > -999]).argmax()

            # Averaging for all remaining properties
            else:
                basin_data[prop] = np.average(values[values > -999])

    # Convert the basin_data dictionary to a Series and assign it to the DataFrame
    basin_attributes.loc[basin] = pd.Series(basin_data)
    
basin_attributes.to_pickle("./preprocessed/hybas/metadata/basin_attributes.pickle")









