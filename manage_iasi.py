import numpy as np
import xarray as xr
import geopy
import geopy.distance
import datetime as dt

import os
import glob
import sys
sys.path.insert(0, "/mnt/f/Studium_NIM/work/Codes/MOSAiC/")
from data_tools import *
from import_data import import_iasi_nc, import_PS_mastertrack

import pdb


"""
	Script to prepare IASI data for Polarstern track. Import IASI and Polarstern track data, cut
	the data and export IASI data to a new, smaller file. No averages will be computed, all found 
	pixels will be saved. Call this script via "python3 manage_iasi.py" or append a digit 
	between 0 and 19 for the IASI subfolders, i.e.,	"python3 manage_iasi.py 14".
	- import data (iasi, polarstern track)
	- find spatio-temporal overlap
	- export IASI data to file
"""


# paths:
path_data = {	'iasi': "/mnt/f/heavy_data/IASI/",		# subfolders may exist
				'ps_track': "/mnt/f/heavy_data/polarstern_track/"}
path_output = "/mnt/f/heavy_data/IASI_mosaic/"

# additional settings:
set_dict = {'max_dist': 50.0,							# max distance in km
			}

path_output_dir = os.path.dirname(path_output)
if not os.path.exists(path_output_dir):
	os.makedirs(path_output_dir)


# import polarstern track:
files = sorted(glob.glob(path_data['ps_track'] + "PS122_3_link-to-mastertrack_V2.nc"))
PS_DS = import_PS_mastertrack(files, return_DS=True)


# import iasi data: consider each subfolder step by step:
# chose subfolder of IASI data if given in the system arguments:
subfolders = sorted(glob.glob(path_data['iasi'] + "*"))
if len(sys.argv) == 2:
	subfolders = subfolders[int(sys.argv[1]):int(sys.argv[1])+1]

for idx_folder, subfolder in enumerate(subfolders):

	# loop through subfolders
	files = sorted(glob.glob(subfolder + "/*.nc"))
	IASI_DS = import_iasi_nc(files)

	# adapt PS_DS time range according to IASI swath times:
	min_time_iasi = IASI_DS.record_start_time.values.min()
	max_time_iasi = IASI_DS.record_stop_time.values.max()
	PS_DS = PS_DS.sel(time=slice(min_time_iasi-np.timedelta64(7200,"s"), max_time_iasi+np.timedelta64(7200,"s")))

	# half of max temporal diff of PS track time as threshold for IASI time overlap:
	set_dict['max_dt'] = np.max(np.diff(PS_DS.time.values)).astype("timedelta64[s]")*0.5
	if set_dict['max_dt'] > np.timedelta64(1800, "s"):
		set_dict['max_dt'] = np.timedelta64(1800, "s")


	# find overlap of IASI and Polarstern for each track data point: reduce IASI data to 1D arrays: variables with only along_track
	# dimension must be brought to (along, across) first of all
	iasi_lon_f = IASI_DS.lon.values.ravel()
	iasi_lat_f = IASI_DS.lat.values.ravel()
	iasi_r_start_f = np.repeat(np.reshape(IASI_DS.record_start_time.values, (len(IASI_DS.along_track),1)), len(IASI_DS.across_track), axis=1).ravel()
	iasi_r_stop_f = np.repeat(np.reshape(IASI_DS.record_stop_time.values, (len(IASI_DS.along_track),1)), len(IASI_DS.across_track), axis=1).ravel()
	len_iasi_f = len(iasi_lon_f)		# length of flattened IASI data

	# first, filter for latitudes close to those seen in the Polarstern track: this reduces the computation time of the loop below
	lat_mask = np.full((len_iasi_f,), False)
	lat_mask_idx = np.where((iasi_lat_f >= (np.floor(PS_DS.Latitude.values.min()) - 1.0)) & (iasi_lat_f <= (np.ceil(PS_DS.Latitude.values.max()) + 1.0)))[0]
	lat_mask[lat_mask_idx] = True		# where this array is True, latitudes are within the PS track expectations
	n_lat_mask_idx = len(lat_mask_idx)

	# reduce flattened IASI data:
	iasi_lon_f_masked = iasi_lon_f[lat_mask]
	iasi_lat_f_masked = iasi_lat_f[lat_mask]

	# create variables and arrays to save IASI data that fulfills the spatio-temporal overlap constraints
	list_idx_masked = list()
	n_time_ps_track = len(PS_DS.time.values)
	n_hgt_temp = len(IASI_DS.pressure_levels_temp)
	n_hgt_wv = len(IASI_DS.pressure_levels_humidity)
	iasi_ps_keys = [varr for varr in IASI_DS.data_vars]
	iasi_ps_keys += ['lat', 'lon', 'pressure_levels_temp', 'pressure_levels_humidity']

	# create empty arrays:
	iasi_ps_dict = dict()
	for key in ['pressure_levels_temp', 'pressure_levels_humidity']: iasi_ps_dict[key] = IASI_DS[key].values
	for key in iasi_ps_keys:
		# save variables for all found pixels: first dimension: time, 
		# second dimension: all found pixels for that time (assumption that no more than 70 pixels will be recognised
		if key in ['pressure_levels_temp', 'pressure_levels_humidity']:
			continue
		elif key in ['fg_atmospheric_temperature', 'atmospheric_temperature']:
			iasi_ps_dict[key] = np.full((n_time_ps_track, 70, n_hgt_temp), np.nan)
		elif key in ['fg_atmospheric_water_vapor', 'atmospheric_water_vapor']:
			iasi_ps_dict[key] = np.full((n_time_ps_track, 70, n_hgt_wv), np.nan)
		elif key in ['record_start_time', 'record_stop_time']:
			iasi_ps_dict[key] = np.zeros((n_time_ps_track,70)).astype('datetime64[s]')
		else:
			iasi_ps_dict[key] = np.full((n_time_ps_track, 70), np.nan)


	# loop through PS track time:
	IASI_2d_shape = IASI_DS.lat.shape
	for k, ps_time in enumerate(PS_DS.time.values):

		if k%10 == 0: print(f"{k} of {n_time_ps_track}")

		# first, finde indices of IASI_DS that are within the spatial distance defined
		# in the settings:
		circ_centre = [PS_DS.Latitude.values[k], PS_DS.Longitude.values[k]]

		# loop through lat-masked IASI coordinates to find where distance to Polarstern is less than the threshold:
		iasi_ps_dist = np.ones((n_lat_mask_idx,))*(-1.0)
		for kk in range(n_lat_mask_idx):
			iasi_ps_dist[kk] = geopy.distance.distance((iasi_lat_f_masked[kk], iasi_lon_f_masked[kk]), (circ_centre[0], circ_centre[1])).km

		distance_mask = iasi_ps_dist <= set_dict['max_dist']	# True for data fulfilling the distance criterion
		iasi_idx_masked = lat_mask_idx[distance_mask]			# yields the indices where the flattened IASI data fulfills lat and distance masks

		# check for temporal overlap: first, apply the filters on the record start/end times for performance:
		iasi_r_start_f_masked = iasi_r_start_f[iasi_idx_masked]
		iasi_r_stop_f_masked = iasi_r_stop_f[iasi_idx_masked]
		iasi_mean_record_time_masked = iasi_r_start_f_masked + 0.5*(iasi_r_stop_f_masked - iasi_r_start_f_masked)

		# mean between record start and stop time will be checked for temporal overlap:
		# nearest_neighbour = iasi_idx_masked[np.argmin(np.abs(iasi_mean_record_time_masked - ps_time))]
		# nearest_neighbour = np.asarray([np.argmin(np.abs(iasi_mean_record_time_masked - ps_time))])
		# nearest_neighbour = nearest_neighbour[np.where(np.abs(iasi_mean_record_time_masked[nearest_neighbour] - ps_time) <= set_dict['max_dt'])]
		iasi_time_space_mask = iasi_idx_masked[np.where(np.abs(iasi_mean_record_time_masked - ps_time) <= set_dict['max_dt'])[0]]
		iasi_time_space_mask_2d = np.unravel_index(iasi_time_space_mask, IASI_2d_shape)		# tuple: 1st entry = along_track; 2nd entry: across_track
		n_iasi_left = len(iasi_time_space_mask)


		# save data to the dictionary id there is data to be saved:
		if n_iasi_left > 0:

			# avoid doubling of time steps by setting iasi_record_start_time to some value var away for these detected indices:
			iasi_r_start_f[iasi_time_space_mask] = np.datetime64("1970-01-01T00:00:00")
			iasi_r_stop_f[iasi_time_space_mask] = np.datetime64("1970-01-01T00:00:00")

			for key in iasi_ps_keys:

				if key in ['pressure_levels_temp', 'pressure_levels_humidity']:
					continue
				elif key in ['fg_atmospheric_temperature', 'atmospheric_temperature']:
					iasi_ps_dict[key][k,:n_iasi_left,:] = IASI_DS[key].values[iasi_time_space_mask_2d[0], iasi_time_space_mask_2d[1],:]
				elif key in ['fg_atmospheric_water_vapor', 'atmospheric_water_vapor']:
					iasi_ps_dict[key][k,:n_iasi_left,:] = IASI_DS[key].values[iasi_time_space_mask_2d[0], iasi_time_space_mask_2d[1],:]
				elif key in ['record_start_time', 'record_stop_time', 'degraded_proc_MDR', 'degraded_ins_MDR']:	# must first be expanded to 2D
					iasi_ps_dict[key][k,:n_iasi_left] = np.repeat(np.reshape(IASI_DS[key].values, (len(IASI_DS.along_track),1)), 
																	len(IASI_DS.across_track), axis=1).ravel()[iasi_time_space_mask]
				else:
					iasi_ps_dict[key][k,:n_iasi_left] = IASI_DS[key].values.ravel()[iasi_time_space_mask]


	# save data dict to xarray dataset, then to netCDF:
	IASI_PS_DS = xr.Dataset(coords={'time': (['time'], PS_DS.time.values),
									'n_hits': (['n_hits'], np.arange(70)),
									'pressure_levels_temp': (['nlt'], iasi_ps_dict['pressure_levels_temp']),
									'pressure_levels_humidity': (['nlq'], iasi_ps_dict['pressure_levels_humidity'])})

	for key in iasi_ps_keys:

		if key in ['pressure_levels_temp', 'pressure_levels_humidity']:
			continue
		elif key in ['fg_atmospheric_temperature', 'atmospheric_temperature']:
			IASI_PS_DS[key] = xr.DataArray(iasi_ps_dict[key], dims=['time', 'n_hits', 'nlt'], attrs=IASI_DS[key].attrs)
		elif key in ['fg_atmospheric_water_vapor', 'atmospheric_water_vapor']:
			IASI_PS_DS[key] = xr.DataArray(iasi_ps_dict[key], dims=['time', 'n_hits', 'nlq'], attrs=IASI_DS[key].attrs)
		elif key in ['record_start_time', 'record_stop_time']:
			IASI_PS_DS[key] = xr.DataArray(iasi_ps_dict[key], dims=['time', 'n_hits'])
		else:
			IASI_PS_DS[key] = xr.DataArray(iasi_ps_dict[key], dims=['time', 'n_hits'], attrs=IASI_DS[key].attrs)


	# Attributes to dimensions:
	IASI_PS_DS['pressure_levels_temp'].attrs = IASI_DS['pressure_levels_temp'].attrs
	IASI_PS_DS['pressure_levels_humidity'].attrs = IASI_DS['pressure_levels_humidity'].attrs
	IASI_PS_DS['n_hits'].attrs = {'comment': "Number of IASI (along_track, across_track) pixels fulfilling the spatio-temporal sampling."}

	# GLOBAL ATTRIBUTES:
	IASI_PS_DS.attrs['title'] = "IASI 2b the Infrared Atmospheric Sounding Interferometer, processed"
	IASI_PS_DS.attrs['title_short_name'] = "IASI L2"
	IASI_PS_DS.attrs['original_creator'] = "EUMETSAT, http://eumetsat.int, ops@eumetsat.int"
	IASI_PS_DS.attrs['source'] = IASI_DS.attrs['source']
	IASI_PS_DS.attrs['processed_by'] = "Andreas Walbroel (a.walbroel@uni-koeln.de), Institute for Geophysics and Meteorology, University of Cologne, Cologne, Germany"
	IASI_PS_DS.attrs['comments'] = (f"The processing ensures a temporal overlap of {set_dict['max_dist']} km and temporal " +
									f"accuracy of at least {set_dict['max_dt']}. For a given Polarstern track time stamp, " +
									"the existence of a spatial overlap is inquired. If spatial overlap has been detected, temporal overlap is tested. " +
									"If temporal overlap has been identified, all pixels fulfilling the spatio-temporal overlap constraints will be saved " +
									"to that Polarstern track time stamp. Afterwards, these detected pixels are excluded from further overlap search. " +
									f"Note that there might be a time difference up to {set_dict['max_dt']} between Polarstern track time " +
									"(main time axis of this dataset) and IASI (record_start_time + record_stop_time)*0.5.")
	IASI_PS_DS.attrs['conventions'] = "CF-1.7"
	IASI_PS_DS.attrs['python_version'] = f"python version: {sys.version}"
	IASI_PS_DS.attrs['python_packages'] = (f"numpy: {np.__version__}, xarray: {xr.__version__}, " +
											f"geopy: {geopy.__version__}")


	# further attributes:
	attr_list = ['platform_type', 'platform_long_name', 'sensor', 'processor_major_version', 
				'product_minor_version', 'format_major_version', 'format_minor_version']
	for attr_ in attr_list: IASI_PS_DS.attrs[attr_] = IASI_DS.attrs[attr_]

	datetime_utc = dt.datetime.utcnow()
	IASI_PS_DS.attrs['processing_date'] = datetime_utc.strftime("%Y-%m-%d %H:%M:%S")

	# time encoding
	IASI_PS_DS['time'] = PS_DS.time.values.astype("datetime64[s]").astype(np.float64)
	IASI_PS_DS['time'].attrs['units'] = "seconds since 1970-01-01 00:00:00"
	IASI_PS_DS['time'].attrs['comment'] = "time axis is based on Polarstern track: " + PS_DS.Citation
	IASI_PS_DS['time'].encoding['units'] = 'seconds since 1970-01-01 00:00:00'
	IASI_PS_DS['time'].encoding['dtype'] = 'double'

	# Limit dataset to data-only time steps:
	IASI_PS_DS = IASI_PS_DS.isel(time=(~np.isnan(IASI_PS_DS.lat.values[:,0])))

	IASI_PS_DS.to_netcdf(path_output + f"MOSAiC_IASI_Polarstern_overlap_step1_{int(sys.argv[1]):02}.nc", mode='w', format='NETCDF4')
	IASI_PS_DS.close()

	# clear memory:
	del IASI_PS_DS
