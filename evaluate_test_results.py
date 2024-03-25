#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 09:24:21 2023

@author: Ted
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

data_path = "./configs/selected_basins/runs/test_run_2010_131408/test/model_epoch050/test_results.p"

# Load the Xarray dataset
dataset = pd.read_pickle(data_path)

# %%

for key,value in dataset.items():
    obs = value['1D']['xr'].variables['turbidity (NTU)_obs'].values
    sim = value['1D']['xr'].variables['turbidity (NTU)_sim'].values
    break
