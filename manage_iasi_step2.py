import numpy as np
import xarray as xr
import datetime as dt

import os
import glob
import sys
sys.path.insert(0, "/mnt/f/Studium_NIM/work/Codes/MOSAiC/")
from data_tools import *
from import_data import import_iasi_step1

import pdb

"""
	Follow-up script to manage_iasi.py to finish the processing of IASI data for Polarstern track.
	Import step-1-processed IASI data and compute averages over the second dimension (n_hits). 
	- import IASI data (processed with manage_iasi.py)
	- eliminate n_hits dimension
	- export IASI data to file
"""

# paths:
path_data = {	'iasi': "/mnt/f/heavy_data/IASI/mosaic_step1/",		# subfolders may exist
				}
path_output = "/mnt/f/heavy_data/IASI/mosaic_step2/"

# additional settings:
set_dict = {'with_std': True,					# also computes std dev. for reasonable variables (iwv, temp prof, ...)
			}

path_output_dir = os.path.dirname(path_output)
if not os.path.exists(path_output_dir):
	os.makedirs(path_output_dir)


# import IASI data (processed with manage_iasi.py): 
print("Importing IASI data processed with manage_iasi.py....")
IASI_DS = import_iasi_step1(path_data['iasi'])


# eliminate the second (n_hits) dimension for the following variables by averaging:
# don't forget to carry on the attributes. For record_start_time (record_stop_time), the minimum (maximum)
# will be computed.
vars_for_avg = ['fg_atmospheric_temperature', 'fg_atmospheric_water_vapor', 'atmospheric_temperature', 
				'atmospheric_water_vapor', 'iwv', 'surface_pressure', 'surface_z', 'lat', 'lon',
				'record_start_time', 'record_stop_time']
n_time = len(IASI_DS.time)		# needed for averaging of record_start_time, record_stop_time
fill_time = np.datetime64("1970-01-01T00:00:00")

print("Eliminating the second dimension (n_hits) for some variables....")
for vfa in vars_for_avg:

	if vfa not in['record_start_time', 'record_stop_time']:
		IASI_DS[vfa + "_mean"] = IASI_DS[vfa].mean('n_hits')
		IASI_DS[vfa + "_mean"].attrs = IASI_DS[vfa].attrs
		IASI_DS[vfa + "_mean"].attrs['comment_processing'] = "Averaged over pixels within spatio-temporal constraints (see global attribute)."

	elif vfa == 'record_start_time':

		# needs extra handling because fill values aren't nans:
		vfa_min = np.full((n_time,), np.datetime64("1970-01-01T00:00:00"))
		for k in range(n_time):
			vfa_min[k] = IASI_DS[vfa].values[k,np.where(IASI_DS[vfa].values[k,:] > fill_time)].min()

		IASI_DS[vfa + "_min"] = xr.DataArray(vfa_min, dims=['time'])
		IASI_DS[vfa + "_min"].attrs['long_name'] = "Record start time from the Generic Record Header of the MDR"
		IASI_DS[vfa + "_min"].attrs['comment_processing'] = "Minimum over pixels within spatio-temporal constraints (see global attribute)."

	elif vfa == 'record_stop_time':

		# needs extra handling because fill values aren't nans:
		vfa_max = np.full((n_time,), np.datetime64("1970-01-01T00:00:00"))
		for k in range(n_time):
			vfa_max[k] = IASI_DS[vfa].values[k,np.where(IASI_DS[vfa].values[k,:] > fill_time)].max()

		IASI_DS[vfa + "_max"] = xr.DataArray(vfa_max, dims=['time'])
		IASI_DS[vfa + "_max"].attrs['long_name'] = "Record end time from the Generic Record Header of the MDR"
		IASI_DS[vfa + "_max"].attrs['comment_processing'] = "Maximum over pixels within spatio-temporal constraints (see global attribute)."


# also compute standard deviation for subset of vars_for_avg if desired:
if set_dict['with_std']:
	exclude_for_std = ['lat', 'lon', 'record_start_time', 'record_stop_time']
	vars_for_std = [vfa for vfa in vars_for_avg if vfa not in exclude_for_std]

	for vfa in vars_for_std:
		IASI_DS[vfa + "_std"] = IASI_DS[vfa].std('n_hits')
		IASI_DS[vfa + "_std"].attrs = IASI_DS[vfa].attrs
		IASI_DS[vfa + "_std"].attrs['comment_processing'] = "Standard deviation over pixels within spatio-temporal constraints (see global attribute)."


# Remove the non-averaged variables within the vars_for_avg list to save space:
IASI_DS = IASI_DS.drop_vars(vars_for_avg)


# Update some attributes:
datetime_utc = dt.datetime.utcnow()
IASI_DS.attrs['processing_date'] = datetime_utc.strftime("%Y-%m-%d %H:%M:%S")
IASI_DS.attrs['comments2'] = ("In this processing step, various data variables (with _mean) have been averaged over the n_hits dimension. " +
								"Flag values have not been touched. They remain original.")

# Export daily files:
print(f"Exporting processed-processed IASI data as daily files to {path_output}....")
time_range = np.arange(IASI_DS.time.values.min(), IASI_DS.time.values.max(), np.timedelta64(1, "D")).astype("datetime64[D]")
for day in time_range:
	IASI_day_DS = IASI_DS.sel(time=str(day))

	# time encoding:
	IASI_day_DS['time'].encoding['units'] = 'seconds since 1970-01-01 00:00:00'
	IASI_day_DS['time'].encoding['dtype'] = 'double'
	IASI_day_DS['record_start_time_min'].encoding['units'] = 'seconds since 1970-01-01 00:00:00'
	IASI_day_DS['record_start_time_min'].encoding['dtype'] = 'double'
	IASI_day_DS['record_stop_time_max'].encoding['units'] = 'seconds since 1970-01-01 00:00:00'
	IASI_day_DS['record_stop_time_max'].encoding['dtype'] = 'double'


	IASI_day_DS.to_netcdf(path_output + f"MOSAiC_IASI_Polarstern_overlap_{str(day).replace('-','')}.nc", mode='w', format='NETCDF4')
	IASI_day_DS.close()
	del IASI_day_DS
