import pdb
import glob
import copy
import datetime as dt
import gc
import os
import sys
import subprocess
from copy import deepcopy

wdir = os.getcwd() + "/"
remote = ((("/net/blanc/" in wdir) | ("/work/awalbroe/" in wdir)) and ("/mnt/f/" not in wdir))		# identify if the code is executed on the blanc computer or at home

import numpy as np
import matplotlib as mpl
if not remote: mpl.use("WebAgg")
mpl.rcParams.update({'font.family': 'monospace'})
from matplotlib.ticker import PercentFormatter
import yaml
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd

sys.path.insert(0, os.path.dirname(wdir[:-1]) + "/")
from import_data import *
from my_classes import radiosondes, radiometers, era_i, era5
from info_content import info_content
from data_tools import *

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow

from sklearn.model_selection import KFold

ssstart = dt.datetime.utcnow()


def load_geoinfo_MOSAiC_polarstern(aux_i):

	"""
	Load Polarstern track information (lat, lon).

	Parameters:
	-----------
	aux_i : dict
		Dictionary containing additional information.
	"""

	# Data paths:
	path_ps_track = aux_i['path_ps_track']

	# Specify date range:
	date_start = "2019-09-01"
	date_end = "2020-10-12"			# default: "2020-10-12"

	# Import and concatenate polarstern track data: Cycle through all PS track files
	# and concatenate them
	ps_track_files = sorted(glob.glob(path_ps_track + "*.nc"))
	ps_track_dict = {'lat': np.array([]), 'lon': np.array([]), 'time': np.array([])}
	ps_keys = ['Latitude', 'Longitude', 'time']
	for pf_file in ps_track_files:
		ps_track_dict_temp = import_PS_mastertrack(pf_file, ps_keys)
		
		# concatenate ps_track_dict_temp data:
		ps_track_dict['lat'] = np.concatenate((ps_track_dict['lat'], ps_track_dict_temp['Latitude']), axis=0)
		ps_track_dict['lon'] = np.concatenate((ps_track_dict['lon'], ps_track_dict_temp['Longitude']), axis=0)
		ps_track_dict['time'] = np.concatenate((ps_track_dict['time'], ps_track_dict_temp['time']), axis=0)

	# sort by time just to make sure we have ascending time:
	ps_sort_idx = np.argsort(ps_track_dict['time'])
	for key in ps_track_dict.keys(): ps_track_dict[key] = ps_track_dict[key][ps_sort_idx]

	# arrange in an xarray dataset:
	ps_track_DS = xr.Dataset(coords={'time': 	(ps_track_dict['time'].astype('datetime64[s]'))})
	ps_track_DS['lat'] = xr.DataArray(ps_track_dict['lat'].astype(np.float32), dims=['time'], 
										attrs={'long_name': "latitude of the RV Polarstern",
												'standard_name': "latitude",
												'units': "degree_north"})
	ps_track_DS['lon'] = xr.DataArray(ps_track_dict['lon'].astype(np.float32), dims=['time'],
										attrs={'long_name': "longitude of the RV Polarstern",
												'standard_name': "longitude",
												'units': "degree_east"})

	return ps_track_DS


def mosaic_tb_offset_correction(
	DS,
	path_offsets,
	instr_label):

	"""
	Corrects TB offsets for microwave radiometers ('hatpro', 'mirac-p', specified by instr_label) 
	during the MOSAiC expedition. 

	Parameters:
	-----------
	DS : xarray dataset
		Dataset containing TB data as variable 'tb' on a (time,frequency) 2D-array. Time (frequency) 
		dimension and variable name must be called 'time' ('freq').
	path_offsets : str
		Path where the netCDF containing the offsets, slopes and biases are located. The files are
		the output of PAMTRA_fwd_sim_v2.py.
	instr_label : str
		Label to identify the instrument to apply offset correction to. Options: 'hatpro', 
		'mirac-p'
	"""

	try:
		OFF_DS = xr.open_dataset(path_offsets + f"MOSAiC_{instr_label}_radiometer_clear_sky_offset_correction.nc")
	except FileNotFoundError:
		print("WARNING! TB offset correction has been attempted, but file for offset correction has not been found!")
		return DS

	# linear fit mask indicating for each calibration period and frequency if linear fit is to be
	# applied for correction or if bias correction only is applied:
	# 1, if lin fit is used for correction, 0 for bias only, -1 if no offset available
	n_c_p = len(OFF_DS.time)	# number of calibration periods
	lin_fit_mask = np.zeros((n_c_p,len(DS.freq)))		# 1, if lin fit is used for correction, 0 for bias only or no offset available
	if instr_label == 'hatpro':
		lin_fit_mask[1,:] = np.array([0,1,1,0,0,0,0,0,0,1,1,1,1,0])
		lin_fit_mask[2,:] = np.array([1,1,0,0,0,0,0,1,1,1,1,1,1,1])
		lin_fit_mask[3,:] = np.array([1,1,1,0,0,0,0,1,1,1,1,1,1,1])
		lin_fit_mask[4,:] = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0])
		lin_fit_mask[5,:] = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1])
		lin_fit_mask[6,:] = np.array([1,1,1,1,1,1,1,0,1,1,1,1,1,1])
	elif instr_label == 'mirac-p':
		lin_fit_mask[2,:] = np.array([1,1,1,1,1,1,1,1])
		lin_fit_mask[3,:] = np.array([1,1,1,1,1,1,1,1])
		lin_fit_mask[4,:] = np.array([1,1,1,1,1,1,1,1])

	# compute offset corrected TBs for each calibration period and each freq
	# loop over calibration periods
	DS['tb_cor'] = deepcopy(DS['tb'])
	for c_p_i in range(n_c_p):
		# mask MWR time overlap with calibration period:
		c_p_mask = ((DS.time >= OFF_DS.calibration_period_start.values[c_p_i]) &
					(DS.time < OFF_DS.calibration_period_end.values[c_p_i]))

		if np.any(c_p_mask): # otherwise, MWR time is outside any calib period:
			# decide if linear fit or bias cor:
			lfm_temp = lin_fit_mask[c_p_i]
			lfm_bin = lfm_temp.astype('bool')
			if np.any(~lfm_bin):
				DS['tb_cor'][c_p_mask,~lfm_bin] -= OFF_DS.bias[c_p_i,~lfm_bin].values
			if np.any(lfm_bin):
				DS['tb_cor'][c_p_mask, lfm_bin] = (OFF_DS.slope[c_p_i,lfm_bin].values*DS['tb_cor'][c_p_mask,lfm_bin] + OFF_DS.offset[c_p_i,lfm_bin].values)


	# replace old TBs by corrected ones:
	OFF_DS.close()
	del DS['tb'], OFF_DS
	DS = DS.rename({'tb_cor': 'tb'})

	return DS


def reduce_dimensions(
	data, 
	check_dims_vars):

	"""
	This function reduces dimensions of variables in data defined in check_dims_vars.keys()
	that exceed the dimension number given as value in check_dims_vars. It is expected to
	have the time axis == first axis, height axis == last axis. Furthermore, it's assumed
	that each variable apart for some separately handled ones have the same dimension.

	Parameters:
	-----------
	data : class radiometers, era5
		Data whose attributes are checked for number of dimensions. If exceeding the dimension
		number given in check_dims_vars, the dimensions will be concatenated.
	check_dims_vars : dict
		Contains variable names as keys and their expected dimension number as values.
	"""

	# find out how the shape of the other dimensions are that will be concatenated into one:
	other_dims = ()
	for dv in check_dims_vars.keys():
		if (dv in data.__dict__.keys() and check_dims_vars[dv] == 1 
			and data.__dict__[dv].ndim > 1):

			other_dim_new = data.__dict__[dv].shape
			if other_dims and other_dim_new != other_dims:
				raise ValueError("Some other dimensions seem to exist")
			elif not other_dims:
				other_dims = other_dim_new

	# loop over variables and change (flatten) dimensions
	for dim_var in check_dims_vars.keys():

		if (dim_var in data.__dict__.keys() and 
			data.__dict__[dim_var].ndim > check_dims_vars[dim_var]):

			# just flatten arrays if dim should be 1; else, handle other dimensions:
			if check_dims_vars[dim_var] == 1 and dim_var not in ['launch_time', 'time']:
				# .flatten() reduces a [time,x,y] array to [y*x*time] --> reverses order
				# so that former first dimension changes slowest
				data.__dict__[dim_var] = data.__dict__[dim_var].flatten()

			elif check_dims_vars[dim_var] == 2:
				# flatten the array for each index of the last dimension; 
				# last dimension could be i.e., height or frequency!
				old_shape = data.__dict__[dim_var].shape
				new = np.full((np.prod(old_shape[:-1]), old_shape[-1]), np.nan)
				for i_z in range(old_shape[-1]):
					new[:,i_z] = data.__dict__[dim_var][...,i_z].flatten()

				data.__dict__[dim_var] = new
				del new, old_shape

		elif (dim_var in data.__dict__.keys() and check_dims_vars[dim_var] == 1
				and dim_var in ['launch_time', 'time']):
			# must be dealt with separately: check for other dimension sizes:
			data.__dict__[dim_var] = np.repeat(data.__dict__[dim_var], np.prod(other_dims[1:]))
			
	return data


def apply_sea_mask(
	data, 
	sfc_mask, 
	check_dims_vars):

	"""
	Reduce variables given in check_dims_vars of data to sea only grid cells as specified in
	sfc_mask.

	Parameters:
	-----------
	data : class radiometers, era5
		Data whose variables will be reduced to sea only regions given by sfc_mask.
	sfc_mask : array
		Surface mask depicting the land fraction of data. Must have the same shape as non-height
		dependend (or other secondary dimensions) variables of data. 
	check_dims_vars : dict
		Contains variable names as keys and their expected dimension number as values.
	"""

	# loop through variables and reduce them to indices where sfc_mask is True.
	for dim_var in check_dims_vars:
		if dim_var in data.__dict__.keys():
			if check_dims_vars[dim_var] == 1 and dim_var not in ['freq', 'freq_bl']:
				data.__dict__[dim_var] = data.__dict__[dim_var][sfc_mask]

			elif check_dims_vars[dim_var] == 2:
				data.__dict__[dim_var] = data.__dict__[dim_var][sfc_mask,:]

	return data


def interp_to_new_hgt_grd(
	data, 
	new_height, 
	height_vars,
	aux_i):

	"""
	Interpolate variables listed in height_vars in the data to the new height grid 
	specified by new_height.

	Parameters:
	-----------
	data : class era5
		Data of which the height depended variables will be interpolated to a new grid.
		The height variable of data must be called 'height'.
	height_vars : list
		Contains variable names as keys and their expected dimension number as values.
	aux_i : dict
		Dictionary containing additional information.
	"""

	# lopp through all variables:
	for height_var in height_vars:
		if height_var in data.__dict__.keys():
			for kk in range(data.__dict__[height_var].shape[0]):
				data.__dict__[height_var][kk,:aux_i['n_height']] = np.interp(new_height, 
																			data.__dict__['height'][kk,:],
																			data.__dict__[height_var][kk,:])

			# truncate above new_height:
			data.__dict__[height_var] = data.__dict__[height_var][:,:aux_i['n_height']]

	
	data.height = np.repeat(np.reshape(new_height, (1, aux_i['n_height'])), 
											len(data.launch_time), axis=0)

	return data


def build_input_vector(
	predictor, 
	specifier,
	aux_i):

	"""
	Builds the input vector with the TBs being on top and other variables following below it. aux_i indicates 
	which predictors exist. The predictor as radiometer class must be given.

	Parameters:
	-----------
	predictor : class radiometers
		Instance of the class radiometers containing TBs and other variables.
	specifier : str
		String that indicates if this is now the predictor of training or test data. Valid options:
		'training', 'test'
	aux_i : dict
		Dictionary containing additional information.
	"""

	predictor.input = predictor.TB

	if ("TBs" not in aux_i['predictors']) and ('tb_bl' in aux_i['predictors']):
		predictor.input = predictor.TB_BL

	elif ("TBs" in aux_i['predictors']) and ('tb_bl' in aux_i['predictors']):
		predictor.input = np.concatenate((predictor.input,
											predictor.TB_BL), axis=1)

	# other predictors:
	if "pres_sfc" in aux_i['predictors']:
		predictor.input = np.concatenate((predictor.input,
											np.reshape(predictor.pres, (aux_i[f'n_{specifier}'],1))),
											axis=1)

	if "CF" in aux_i['predictors']:
		predictor.input = np.concatenate((predictor.input, 
											np.reshape(predictor.CF, (aux_i[f'n_{specifier}'],1))),
											axis=1)

	if "iwv" in aux_i['predictors']:
		predictor.input = np.concatenate((predictor.input,
											np.reshape(predictor.iwv, (aux_i[f'n_{specifier}'],1))),
											axis=1)

	if "t2m" in aux_i['predictors']:
		predictor.input = np.concatenate((predictor.input,
											np.reshape(predictor.t2m, (aux_i[f'n_{specifier}'],1))),
											axis=1)

	# Compute Day of Year in radians if the sin and cos of it shall also be used in input vector:
	if ("DOY_1" in aux_i['predictors']) and ("DOY_2" not in aux_i['predictors']):
		predictor.DOY_1, predictor.DOY_2 = compute_DOY(predictor.time, return_dt=False, reshape=True)

		predictor.input = np.concatenate((predictor.input, 
											predictor.DOY_1), axis=1)

	elif ("DOY_2" in aux_i['predictors']) and ("DOY_1" not in aux_i['predictors']):
		predictor.DOY_1, predictor.DOY_2 = compute_DOY(predictor.time, return_dt=False, reshape=True)

		predictor.input = np.concatenate((predictor.input, 
											predictor.DOY_2), axis=1)

	elif ("DOY_1" in aux_i['predictors']) and ("DOY_2" in aux_i['predictors']):
		predictor.DOY_1, predictor.DOY_2 = compute_DOY(predictor.time, return_dt=False, reshape=True)

		predictor.input = np.concatenate((predictor.input, 
											predictor.DOY_1,
											predictor.DOY_2), axis=1)

	return predictor


def compute_error_stats(
	prediction, 
	predictand, 
	predictand_id,
	height=np.array([])):

	"""
	Compute error statistics (Root Mean Squared Error (rmse), bias, Standard Deviation (stddev))
	between prediction and (test data) predictand. Height must be provided if prediction or respective
	predictand is a profile. The prediction_id describes the prediction and predictand and must also
	be forwarded to the function.

	Parameters:
	-----------
	prediction : array of floats
		Predicted variables also available in predictand, predicted by the Neural Network.
	predictand : array of floats
		Predictand data as array, used as evaluation reference. Likely equals the attribute 
		'output' of the predictand class object.
	predictand_id : str
		String indicating which output variable is forwarded to the function.
	height : array of floats
		Height array for respective predictand or predictand profiles (of i.e., temperature or 
		humidity). Can be a 1D or 2D array (latter must be of shape (n_training,n_height)).
	"""

	error_dict = dict()

	# on x axis: reference; y axis: prediction
	x_stuff = predictand
	y_stuff = prediction

	# Compute statistics:
	if predictand_id in ['iwv', 'lwp']:
		# remove redundant dimension:
		x_stuff = x_stuff.squeeze()
		y_stuff = y_stuff.squeeze()
		stats_dict = compute_retrieval_statistics(x_stuff.squeeze(), y_stuff.squeeze(), compute_stddev=True)

		# For entire range:
		error_dict['rmse_tot'] = stats_dict['rmse']
		error_dict['stddev'] = stats_dict['stddev']
		error_dict['bias_tot'] = stats_dict['bias']

		# also compute rmse and bias for specific ranges only:
		# 'bias': np.nanmean(y_stuff - x_stuff),
		# 'rmse': np.sqrt(np.nanmean((x_stuff - y_stuff)**2)),
		range_dict = dict()
		if predictand_id == 'iwv':	# in mm
			range_dict['bot'] = [0,5]
			range_dict['mid'] = [5,10]
			range_dict['top'] = [10,100]
		elif predictand_id == 'lwp': # in kg m^-2
			range_dict['bot'] = [0,0.025]
			range_dict['mid'] = [0.025,0.100]
			range_dict['top'] = [0.100, 1e+06]

		mask_range = dict()
		x_stuff_range = dict()
		y_stuff_range = dict()
		stats_dict_range = dict()
		for range_id in range_dict.keys():
			mask_range[range_id] = ((x_stuff >= range_dict[range_id][0]) & (x_stuff < range_dict[range_id][1]))
			x_stuff_range[range_id] = x_stuff[mask_range[range_id]]
			y_stuff_range[range_id] = y_stuff[mask_range[range_id]]

			# compute retrieval stats for the respective ranges:
			stats_dict_range[range_id] = compute_retrieval_statistics(x_stuff_range[range_id], y_stuff_range[range_id], compute_stddev=True)
			error_dict[f"rmse_{range_id}"] = stats_dict_range[range_id]['rmse']
			error_dict[f"stddev_{range_id}"] = stats_dict_range[range_id]['stddev']
			error_dict[f"bias_{range_id}"] = stats_dict_range[range_id]['bias']


	elif predictand_id in ['temp', 'q']:

		if len(height) == 0:
			raise ValueError("Please specify a height variable to estimate error statistics for profiles.")

		# Compute statistics for entire profile:
		error_dict['rmse_tot'] = compute_RMSE_profile(y_stuff, x_stuff, which_axis=0)
		error_dict['bias_tot'] = np.nanmean(y_stuff - x_stuff, axis=0)
		error_dict['stddev'] = compute_RMSE_profile(y_stuff - error_dict['bias_tot'], x_stuff, which_axis=0)

		# Don't only compute bias, stddev and rmse for entire profile, but also give summary for 
		# bottom, mid and top range (height related in this case):
		range_dict = {	'bot': [0., 1500.0],
						'mid': [1500.0, 5000.0],
						'top': [5000.0, 15000.0]}
		if height.ndim == 2:
			height = height[0,:]

		mask_range = dict()
		for range_id in range_dict.keys():
			mask_range[range_id] = ((height >= range_dict[range_id][0]) & (height < range_dict[range_id][1]))
			error_dict[f"rmse_{range_id}"] = np.nanmean(error_dict['rmse_tot'][mask_range[range_id]])
			error_dict[f"stddev_{range_id}"] = np.nanmean(error_dict['stddev'][mask_range[range_id]])
			error_dict[f"bias_{range_id}"] = np.nanmean(error_dict['bias_tot'][mask_range[range_id]])

	return error_dict


def visualize_evaluation(
	prediction, 
	predictand, 
	predictand_id,
	ret_stats_dict,
	aux_i,
	height=np.array([])):

	"""
	Visualize the evaluation of the Neural Network prediction against a predictand (i.e., test data).
	Depending on the predicted variable (specified by predictand_id), different plots will be created:
	IWV: scatter plot, LWP: scatter plot, temperature profile: standard deviation and bias profile,
	specific humidity profile: standard deviation and bias profile

	Parameters:
	-----------
	prediction : array of floats
		Predicted variables also available in predictand_class.output, predicted by the Neural Network.
	predictand : array of floats
		Predictand (i.e., of test data) data as array, used as evaluation reference. Likely equals the attribute 
		'output' of the predictand class object.
	predictand_id : str
		String indicating which output variable is forwarded to the function.
	ret_stats_dict : dict
		Dictionary which has got several retrieval statistics as values and their names as keys. Output of function
		compute_error_stats.
	aux_i : dict
		Dictionary containing additional information.
	height : array of floats
		Height array for respective predictand or predictand profiles (of i.e., temperature or 
		humidity). Can be a 1D or 2D array (latter must be of shape (n_training,n_height)).
	"""

	if predictand_id in ['temp', 'q'] and len(height) == 0:
			raise ValueError("Please specify a height variable to estimate error statistics for profiles.")


	# create output path if not existing:
	plotpath_dir = os.path.dirname(aux_i['path_plots'] + f"{predictand_id}/")
	if not os.path.exists(plotpath_dir):
		os.makedirs(plotpath_dir)

	# visualize:
	fs = 26
	fs_small = fs - 2
	fs_dwarf = fs_small - 2
	fs_micro = fs_dwarf - 2
	msize = 7.0

	c_H = (0.7,0,0)

	# IWV scatter plot:
	if predictand_id == 'iwv':

		predictand = predictand[:,0]
		prediction = prediction[:,0]

		# again have to compute retrieval stats for N and R:
		ret_stats_temp = compute_retrieval_statistics(predictand, prediction)

		f1 = plt.figure(figsize=(9,9))
		a1 = plt.axes()

		ax_lims = np.asarray([0.0, 35.0])

		# plotting:
		a1.plot(predictand, prediction, linestyle='none', color=c_H, marker='.', markersize=msize,
				markeredgecolor=(0,0,0), label='Prediction')

		# generate a linear fit with least squares approach: notes, p.2:
		# filter nan values:
		nonnan_idx = np.argwhere(~np.isnan(prediction) & ~np.isnan(predictand)).flatten()
		y_fit = prediction[nonnan_idx]
		x_fit = predictand[nonnan_idx]

		# there must be at least 2 measurements to create a linear fit:
		if (len(y_fit) > 1) and (len(x_fit) > 1):
			G_fit = np.array([x_fit, np.ones((len(x_fit),))]).T		# must be transposed because of python's strange conventions
			m_fit = np.matmul(np.matmul(np.linalg.inv(np.matmul(G_fit.T, G_fit)), G_fit.T), y_fit)	# least squares solution
			a = m_fit[0]
			b = m_fit[1]

			ds_fit = a1.plot(ax_lims, a*ax_lims + b, color=c_H, linewidth=1.2, label="Best fit")

		# plot a line for orientation which would represent a perfect fit:
		a1.plot(ax_lims, ax_lims, color=(0,0,0), linewidth=1.2, alpha=0.5, label="Theoretical perfect fit")


		# add statistics:
		a1.text(0.99, 0.01, f"N = {ret_stats_temp['N']}\nMean = {np.mean(np.concatenate((x_fit, y_fit), axis=0)):.2f}\n" +
				f"bias = {ret_stats_dict['bias_tot']:.2f}\nrmse = {ret_stats_dict['rmse_tot']:.2f}\n" +
				f"std. = {ret_stats_dict['stddev']:.2f}\nR = {ret_stats_temp['R']:.3f}", 
				ha='right', va='bottom', transform=a1.transAxes, fontsize=fs_dwarf)


		# Legends:
		lh, ll = a1.get_legend_handles_labels()
		a1.legend(handles=lh, labels=ll, loc='upper left', bbox_to_anchor=(0.05, 1.00), fontsize=fs_micro-4,
					framealpha=0.5)

		# set axis limits:
		a1.set_ylim(bottom=ax_lims[0], top=ax_lims[1])
		a1.set_xlim(left=ax_lims[0], right=ax_lims[1])

		# set axis ticks, ticklabels and tick parameters:
		a1.minorticks_on()
		a1.tick_params(axis='both', labelsize=fs_micro-4)

		# aspect ratio:
		a1.set_aspect('equal')

		# grid:
		a1.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

		# labels:
		a1.set_ylabel("IWV$_{\mathrm{prediction}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)
		a1.set_xlabel("IWV$_{\mathrm{reference}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)
		a1.set_title(f"{aux_i['file_descr']}", fontsize=fs)

		if aux_i['save_figures']:
			plotname = f"NN_syn_ret_eval_{predictand_id}_scatterplot_{aux_i['file_descr']}"
			f1.savefig(aux_i['path_plots'] + f"{predictand_id}/" + plotname + ".png", dpi=300, bbox_inches='tight')
		else:
			plt.show()

		plt.close()


		# error diff composit: Generate bins and compute RMSE, Bias for each bin:
		val_max = 34.0
		val_bins = np.array([np.arange(0., val_max-2.+0.001, 2.), np.arange(2., val_max+0.001, 2.)]).T

		# compute errors for each bin
		RMSE_bins = np.full((val_bins.shape[0],), np.nan)
		BIAS_bins = np.full((val_bins.shape[0],), np.nan)
		N_bins = np.zeros((val_bins.shape[0],))		# number of matches for each bin
		for ibi, val_bin in enumerate(val_bins):
			# find indices for the respective bin (based on the reference (==truth)):
			idx_bin = np.where((predictand >= val_bin[0]) & (predictand < val_bin[1]))[0]
			N_bins[ibi] = len(idx_bin)

			# compute errors:
			RMSE_bins[ibi] = np.sqrt(np.nanmean((prediction[idx_bin] - predictand[idx_bin])**2))
			BIAS_bins[ibi] = np.nanmean(prediction[idx_bin] - predictand[idx_bin])


		# visualize:
		f1 = plt.figure(figsize=(11,7))
		a1 = plt.axes()

		# deactivate some spines:
		a1.spines[['right', 'top']].set_visible(False)

		ax_lims = np.asarray([0.0, val_max])
		er_lims = np.asarray([-1.5, 1.5])

		# plotting:
		# thin lines indicating RELATIVE errors:
		rel_err_contours = np.array([1.0,2.0,5.0,10.0,20.0])
		rel_err_range = np.arange(0.0, val_max+0.0001, 0.01)
		rel_err_curves = np.zeros((len(rel_err_contours), len(rel_err_range)))
		for i_r, r_e_c in enumerate(rel_err_contours):
			rel_err_curves[i_r,:] = rel_err_range*r_e_c / 100.0
			a1.plot(rel_err_range, rel_err_curves[i_r,:], color=(0,0,0,0.5), linewidth=0.75, linestyle='dotted')
			a1.plot(rel_err_range, -1.0*rel_err_curves[i_r,:], color=(0,0,0,0.5), linewidth=0.75, linestyle='dotted')

			# add annotation (label) to rel error curve:
			rel_err_label_pos_x = er_lims[1] * 100. / r_e_c
			if rel_err_label_pos_x > val_max:
				a1.text(ax_lims[1], ax_lims[1]*r_e_c / 100., f"{int(r_e_c)} %",
					color=(0,0,0,0.5), ha='left', va='center', transform=a1.transData, fontsize=fs_micro-6)
			else:
				a1.text(rel_err_label_pos_x, er_lims[1], f"{int(r_e_c)} %", 
					color=(0,0,0,0.5), ha='left', va='bottom', transform=a1.transData, fontsize=fs_micro-6)

		val_bins_plot = (val_bins[:,1] - val_bins[:,0])*0.5 + val_bins[:,0]
		a1.plot(ax_lims, [0,0], color=(0,0,0))
		a1.plot(val_bins_plot, RMSE_bins, color=(0.11,0.46,0.70), linewidth=1.2, label='RMSE')
		a1.plot(val_bins_plot, BIAS_bins, color=(0.11,0.46,0.70), linewidth=1.2, linestyle='dashed', label='Bias')

		
		# Legends:
		lh, ll = a1.get_legend_handles_labels()
		a1.legend(handles=lh, labels=ll, loc='lower left', bbox_to_anchor=(0.02, 0.00), fontsize=fs_micro-4,
					framealpha=0.5)

		# set axis limits:
		a1.set_ylim(bottom=er_lims[0], top=er_lims[1])
		a1.set_xlim(left=ax_lims[0], right=ax_lims[1])

		# set axis ticks, ticklabels and tick parameters:
		a1.minorticks_on()
		a1.tick_params(axis='both', labelsize=fs_micro-4)

		# grid:
		a1.grid(which='major', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

		# labels:
		a1.set_ylabel("Error: Predicted - reference IWV ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs_micro-2)
		a1.set_xlabel("Reference IWV ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs_micro-2)
		a1.set_title(f"{aux_i['file_descr']}", fontsize=fs_micro)

		if aux_i['save_figures']:
			plotname = f"NN_syn_ret_eval_{predictand_id}_err_diff_comp_{aux_i['file_descr']}"
			f1.savefig(aux_i['path_plots'] + f"{predictand_id}/" + plotname + ".png", dpi=300, bbox_inches='tight')
		else:
			plt.show()

		plt.close()


	elif predictand_id == 'lwp':

		predictand = predictand[:,0]
		prediction = prediction[:,0]

		# again have to compute retrieval stats for N and R:
		ret_stats_temp = compute_retrieval_statistics(predictand, prediction)

		f1 = plt.figure(figsize=(18,9))
		a1 = plt.subplot2grid((1,2), (0,0), fig=f1)
		a2 = plt.subplot2grid((1,2), (0,1), fig=f1)

		ax_lims = np.asarray([-50.0, 1000.0])		# g m-2
		ax_lims2 = np.asarray([0.0, 250.0])		# g m-2

		# plotting:
		a1.plot(predictand*1000.0, prediction*1000.0, linestyle='none', color=c_H, marker='.', markersize=msize,
				markeredgecolor=(0,0,0), label='Prediction')
		a2.plot(predictand*1000.0, prediction*1000.0, linestyle='none', color=c_H, marker='.', markersize=msize,
				markeredgecolor=(0,0,0))

		# generate a linear fit with least squares approach: notes, p.2:
		# filter nan values:
		nonnan_idx = np.argwhere(~np.isnan(prediction) & ~np.isnan(predictand)).flatten()
		y_fit = prediction[nonnan_idx]*1000.0
		x_fit = predictand[nonnan_idx]*1000.0

		# there must be at least 2 measurements to create a linear fit:
		if (len(y_fit) > 1) and (len(x_fit) > 1):
			G_fit = np.array([x_fit, np.ones((len(x_fit),))]).T		# must be transposed because of python's strange conventions
			m_fit = np.matmul(np.matmul(np.linalg.inv(np.matmul(G_fit.T, G_fit)), G_fit.T), y_fit)	# least squares solution
			a = m_fit[0]
			b = m_fit[1]

			ds_fit = a1.plot(ax_lims, a*ax_lims + b, color=c_H, linewidth=1.2, label="Best fit")
			ds_fit = a2.plot(ax_lims2, a*ax_lims2 + b, color=c_H, linewidth=1.2)

		# plot a line for orientation which would represent a perfect fit:
		a1.plot(ax_lims, ax_lims, color=(0,0,0), linewidth=1.2, alpha=0.5, label="Theoretical perfect fit")
		a2.plot(ax_lims2, ax_lims2, color=(0,0,0), linewidth=1.2, alpha=0.5)


		# add statistics:
		a1.text(0.99, 0.01, f"N = {ret_stats_temp['N']}\nMean = {np.mean(np.concatenate((x_fit, y_fit), axis=0)):.2f}\n" +
				f"bias = {1000.0*ret_stats_dict['bias_tot']:.2f}\nrmse = {1000.0*ret_stats_dict['rmse_tot']:.2f}\n" +
				f"std. = {1000.0*ret_stats_dict['stddev']:.2f}\nR = {ret_stats_temp['R']:.3f}", 
				ha='right', va='bottom', transform=a1.transAxes, fontsize=fs_dwarf)


		# Legends:
		lh, ll = a1.get_legend_handles_labels()
		a1.legend(handles=lh, labels=ll, loc='upper left', bbox_to_anchor=(0.05, 1.00), fontsize=fs_dwarf,
					framealpha=0.5)

		# set axis limits:
		a1.set_ylim(bottom=ax_lims[0], top=ax_lims[1])
		a1.set_xlim(left=ax_lims[0], right=ax_lims[1])
		a2.set_ylim(bottom=ax_lims2[0], top=ax_lims2[1])
		a2.set_xlim(left=ax_lims2[0], right=ax_lims2[1])

		# set axis ticks, ticklabels and tick parameters:
		a1.minorticks_on()
		a1.tick_params(axis='both', labelsize=fs_dwarf)
		a2.minorticks_on()
		a2.tick_params(axis='both', labelsize=fs_dwarf)

		# aspect ratio:
		a1.set_aspect('equal')
		a2.set_aspect('equal')

		# grid:
		a1.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)
		a2.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

		# labels:
		a1.set_ylabel("LWP$_{\mathrm{prediction}}$ ($\mathrm{g}\,\mathrm{m}^{-2}$)", fontsize=fs)
		a2.set_ylabel("LWP$_{\mathrm{prediction}}$ ($\mathrm{g}\,\mathrm{m}^{-2}$)", fontsize=fs)
		a1.set_xlabel("LWP$_{\mathrm{reference}}$ ($\mathrm{g}\,\mathrm{m}^{-2}$)", fontsize=fs)
		a2.set_xlabel("LWP$_{\mathrm{reference}}$ ($\mathrm{g}\,\mathrm{m}^{-2}$)", fontsize=fs)
		a1.set_title(f"{aux_i['file_descr']}", fontsize=fs)

		if aux_i['save_figures']:
			plotname = f"NN_syn_ret_eval_{predictand_id}_scatterplot_{aux_i['file_descr']}"
			f1.savefig(aux_i['path_plots'] + f"{predictand_id}/" + plotname + ".png", dpi=300, bbox_inches='tight')
		else:
			plt.show()

		plt.close()


		# error diff composit: Generate bins and compute RMSE, Bias for each bin:
		predictand *= 1000.0	# convert to g m-2
		prediction *= 1000.0	# convert to g m-2
		val_max = 1000.0		# in g m-2
		val_bins_log = np.array([np.arange(0.0, 3.001, 0.1), np.arange(0.1, 3.101, 0.1)]).T		# in log10 of g m-2 scale
		val_bins = 10**val_bins_log		# in g m-2


		# compute errors for each bin
		RMSE_bins = np.full((val_bins.shape[0],), np.nan)		# in g m-2
		BIAS_bins = np.full((val_bins.shape[0],), np.nan)		# in g m-2
		N_bins = np.zeros((val_bins.shape[0],))		# number of matches for each bin
		for ibi, val_bin in enumerate(val_bins):
			# find indices for the respective bin (based on the reference (==truth)):
			idx_bin = np.where((predictand >= val_bin[0]) & (predictand < val_bin[1]))[0]
			N_bins[ibi] = len(idx_bin)

			# compute errors:
			RMSE_bins[ibi] = np.sqrt(np.nanmean((prediction[idx_bin] - predictand[idx_bin])**2))
			BIAS_bins[ibi] = np.nanmean(prediction[idx_bin] - predictand[idx_bin])


		# visualize:
		f1 = plt.figure(figsize=(11,7))
		a1 = plt.axes()

		# deactivate some spines:
		a1.spines[['right', 'top']].set_visible(False)

		ax_lims = np.asarray([1.0, val_max])	# in g m-2
		er_lims = np.asarray([-80, 80])			# in g m-2

		# plotting:
		# thin lines indicating RELATIVE errors:
		rel_err_contours = np.array([10.,20.,50.,100.])
		rel_err_range = np.arange(0.0, val_max+0.0001, 0.01)
		rel_err_curves = np.zeros((len(rel_err_contours), len(rel_err_range)))
		for i_r, r_e_c in enumerate(rel_err_contours):
			rel_err_curves[i_r,:] = rel_err_range*r_e_c / 100.0
			a1.plot(rel_err_range, rel_err_curves[i_r,:], color=(0,0,0,0.5), linewidth=0.75, linestyle='dotted')
			a1.plot(rel_err_range, -1.0*rel_err_curves[i_r,:], color=(0,0,0,0.5), linewidth=0.75, linestyle='dotted')

			# add annotation (label) to rel error curve:
			rel_err_label_pos_x = er_lims[1] * 100. / r_e_c
			if rel_err_label_pos_x > val_max:
				a1.text(ax_lims[1], ax_lims[1]*r_e_c / 100., f"{int(r_e_c)} %",
					color=(0,0,0,0.5), ha='left', va='center', transform=a1.transData, fontsize=fs_micro-6)
			else:
				a1.text(rel_err_label_pos_x, er_lims[1], f"{int(r_e_c)} %", 
					color=(0,0,0,0.5), ha='left', va='bottom', transform=a1.transData, fontsize=fs_micro-6)

		val_bins_plot = 10**((val_bins_log[:,1] - val_bins_log[:,0])*0.5 + val_bins_log[:,0])
		a1.plot(ax_lims, [0,0], color=(0,0,0))
		a1.plot(val_bins_plot, RMSE_bins, color=(0.11,0.46,0.70), linewidth=1.2, label='RMSE')
		a1.plot(val_bins_plot, BIAS_bins, color=(0.11,0.46,0.70), linewidth=1.2, linestyle='dashed', label='Bias')

		
		# Legends:
		lh, ll = a1.get_legend_handles_labels()
		a1.legend(handles=lh, labels=ll, loc='lower left', bbox_to_anchor=(0.02, 0.00), fontsize=fs_micro-4,
					framealpha=0.5)

		# set axis limits:
		a1.set_ylim(bottom=er_lims[0], top=er_lims[1])
		a1.set_xlim(left=ax_lims[0], right=ax_lims[1])
		a1.set_xscale('log')

		# set axis ticks, ticklabels and tick parameters:
		a1.minorticks_on()
		a1.tick_params(axis='both', labelsize=fs_micro-4)

		# grid:
		a1.grid(which='major', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

		# labels:
		a1.set_ylabel("Error: Predicted - reference LWP ($\mathrm{g}\,\mathrm{m}^{-2}$)", fontsize=fs_micro-2)
		a1.set_xlabel("Reference LWP ($\mathrm{g}\,\mathrm{m}^{-2}$)", fontsize=fs_micro-2)
		a1.set_title(f"{aux_i['file_descr']}", fontsize=fs_micro)

		if aux_i['save_figures']:
			plotname = f"NN_syn_ret_eval_{predictand_id}_err_diff_comp_{aux_i['file_descr']}"
			f1.savefig(aux_i['path_plots'] + f"{predictand_id}/" + plotname + ".png", dpi=300, bbox_inches='tight')
		else:
			plt.show()

		plt.close()


	elif predictand_id == 'temp':

		# reduce unnecessary dimensions of height:
		if height.ndim == 2:
			height = height[0,:]

		RMSE_pred = ret_stats_dict['rmse_tot']
		STD_pred = ret_stats_dict['stddev']
		BIAS_pred = ret_stats_dict['bias_tot']


		f1 = plt.figure(figsize=(16,14))
		ax_bias = plt.subplot2grid((1,2), (0,0))			# bias profile
		ax_std = plt.subplot2grid((1,2), (0,1))				# std dev profile

		y_lim = np.array([0.0, height.max()])
		x_lim_std = np.array([0.0, 6.0])			# in K


		# bias profiles:
		ax_bias.plot(BIAS_pred, height, color=c_H, linewidth=1.5)
		ax_bias.plot(np.full_like(height, 0.0), height, color=(0,0,0), linewidth=1.0)


		# std dev profiles:
		ax_std.plot(STD_pred, height, color=c_H, linewidth=1.5, label='Std. ($\sigma$)')
		ax_std.plot(RMSE_pred, height, color=c_H, linewidth=1.5, linestyle='dashed', label='RMSE')


		# add figure identifier of subplots: a), b), ...
		ax_bias.text(0.05, 0.98, "a)", fontsize=fs_small, fontweight='bold', ha='left', va='top', transform=ax_bias.transAxes)
		ax_std.text(0.05, 0.98, "b)", fontsize=fs_small, fontweight='bold', ha='left', va='top', transform=ax_std.transAxes)

		# legends:
		lh, ll = ax_std.get_legend_handles_labels()
		ax_std.legend(lh, ll, loc='lower right', bbox_to_anchor=(0.98, 0.01), fontsize=fs_dwarf, framealpha=0.5)

		# axis lims:
		ax_bias.set_ylim(bottom=y_lim[0], top=y_lim[1])
		ax_bias.set_xlim(left=-1.5, right=1.5)
		ax_std.set_ylim(bottom=y_lim[0], top=y_lim[1])
		ax_std.set_xlim(left=x_lim_std[0], right=x_lim_std[1])

		# set axis ticks, ticklabels and tick parameters:
		ax_bias.minorticks_on()
		# # # # # # # # # # # # # # # # ax_bias.tick_params(axis='x', pad=7)
		ax_bias.tick_params(axis='both', labelsize=fs_dwarf)
		ax_std.minorticks_on()
		ax_std.tick_params(axis='both', labelsize=fs_dwarf)
		ax_std.yaxis.set_ticklabels([])

		# grid:
		ax_bias.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)
		ax_std.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

		# labels:
		ax_bias.set_xlabel("$\mathrm{T}_{\mathrm{prediction}} - \mathrm{T}_{\mathrm{reference}}$ (K)", fontsize=fs)
		ax_bias.set_ylabel("Height (m)", fontsize=fs)
		ax_std.set_xlabel("$\sigma_{\mathrm{T}}$, RMSE (K)", fontsize=fs)
		ax_bias.set_title(f"Bias, {aux_i['file_descr']}", fontsize=fs)
		ax_std.set_title(f"Standard dev., {aux_i['file_descr']}", fontsize=fs)


		# Limit axis spacing:
		plt.subplots_adjust(wspace=0.0)			# removes space between subplots

		# # # # # # # # # # adjust axis positions:
		# # # # # # # # # ax_pos = ax_bias.get_position().bounds
		# # # # # # # # # ax_bias.set_position([ax_pos[0]+0.05*ax_pos[0], ax_pos[1], 0.95*ax_pos[2], ax_pos[3]*0.95])
		# # # # # # # # # ax_pos = ax_std.get_position().bounds
		# # # # # # # # # ax_std.set_position([ax_pos[0]+0.05*ax_pos[0], ax_pos[1], 0.95*ax_pos[2], ax_pos[3]*0.95])

		if aux_i['save_figures']:
			plotname = f"NN_syn_ret_eval_{predictand_id}_bias_std_profile_{aux_i['file_descr']}"
			f1.savefig(aux_i['path_plots'] + f"{predictand_id}/" + plotname + ".png", dpi=300, bbox_inches='tight')
		else:
			plt.show()
		plt.close()


	elif predictand_id == 'q':

		# reduce unnecessary dimensions of height:
		if height.ndim == 2:
			height = height[0,:]

		STD_pred = ret_stats_dict['stddev']			# in g kg-1
		BIAS_pred = ret_stats_dict['bias_tot']		# in g kg-1
		RMSE_pred = ret_stats_dict['rmse_tot']		# in g kg-1


		f1 = plt.figure(figsize=(16,14))
		ax_bias = plt.subplot2grid((1,2), (0,0))			# bias profile
		ax_std = plt.subplot2grid((1,2), (0,1))				# std dev profile

		y_lim = np.array([0.0, height.max()])


		# bias profiles:
		ax_bias.plot(BIAS_pred, height, color=c_H, linewidth=1.5)
		ax_bias.plot(np.full_like(height, 0.0), height, color=(0,0,0), linewidth=1.0)


		# std dev profiles:
		ax_std.plot(STD_pred, height, color=c_H, linewidth=1.5, label='Std. ($\sigma$)')
		ax_std.plot(RMSE_pred, height, color=c_H, linewidth=1.5, linestyle='dashed', label='RMSE')


		# add figure identifier of subplots: a), b), ...
		ax_bias.text(0.05, 0.98, "a)", fontsize=fs_small, fontweight='bold', ha='left', va='top', transform=ax_bias.transAxes)
		ax_std.text(0.05, 0.98, "b)", fontsize=fs_small, fontweight='bold', ha='left', va='top', transform=ax_std.transAxes)

		# legends:
		lh, ll = ax_std.get_legend_handles_labels()
		ax_std.legend(lh, ll, loc='lower right', bbox_to_anchor=(0.98, 0.01), fontsize=fs_dwarf, framealpha=0.5)

		# axis lims:
		ax_bias.set_ylim(bottom=y_lim[0], top=y_lim[1])
		ax_bias.set_xlim(left=-0.3, right=0.3)
		ax_std.set_ylim(bottom=y_lim[0], top=y_lim[1])
		ax_std.set_xlim(left=0, right=0.9)

		# set axis ticks, ticklabels and tick parameters:
		ax_bias.minorticks_on()
		# # # # # # # # # # # # # # # # ax_bias.tick_params(axis='x', pad=7)
		ax_bias.tick_params(axis='both', labelsize=fs_dwarf)
		ax_std.minorticks_on()
		ax_std.tick_params(axis='both', labelsize=fs_dwarf)
		ax_std.yaxis.set_ticklabels([])

		# grid:
		ax_bias.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)
		ax_std.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

		# labels:
		ax_bias.set_xlabel("$\mathrm{q}_{\mathrm{prediction}} - \mathrm{q}_{\mathrm{reference}}$ ($\mathrm{g}\,\mathrm{kg}^{-1}$)", fontsize=fs)
		ax_bias.set_ylabel("Height (m)", fontsize=fs)
		ax_std.set_xlabel("$\sigma_{\mathrm{q}}$, RMSE ($\mathrm{g}\,\mathrm{kg}^{-1}$)", fontsize=fs)
		ax_bias.set_title(f"Bias, {aux_i['file_descr']}", fontsize=fs)
		ax_std.set_title(f"RMSE, {aux_i['file_descr']}", fontsize=fs)


		# Limit axis spacing:
		plt.subplots_adjust(wspace=0.0)			# removes space between subplots

		if aux_i['save_figures']:
			plotname = f"NN_syn_ret_eval_{predictand_id}_bias_std_profile_{aux_i['file_descr']}"
			f1.savefig(aux_i['path_plots'] + f"{predictand_id}/" + plotname + ".png", dpi=300, bbox_inches='tight')
		else:
			plt.show()

		plt.close()

	# plt.clf()
	gc.collect()


def save_prediction_and_reference(
	prediction, 
	predictand,
	predictand_id,
	aux_i,
	height=np.array([])):

	"""
	Saves the predicted and reference (i.e., test data) values to a netCDF file for further processing.
	Currently only implemented for IWV and q.

	Parameters:
	-----------
	prediction : array of floats
		Predicted variables also available in predictand_class.output, predicted by the Neural Network.
	predictand : array of floats
		Predictand (or reference, i.e., of test data) data as array, used as evaluation reference. Likely equals the attribute 
		'output' of the predictand class object.
	predictand_id : str
		String indicating which output variable is forwarded to the function.
	aux_i : dict
		Dictionary containing additional information.
	height : array of floats
		Height array for respective predictand or predictand profiles (of i.e., temperature or 
		humidity). Can be a 1D or 2D array (latter must be of shape (n_training,n_height)).
	"""

	# check if output path exists: if it doesn't, create it:
	path_output_dir = os.path.dirname(aux_i['path_output_pred_ref'])
	if not os.path.exists(path_output_dir):
		os.makedirs(path_output_dir)

	if predictand_id not in ['iwv', 'q']:
		raise ValueError("Function save_prediction_and_reference is only for IWV at the moment.")


	# create xarray Dataset:
	DS = xr.Dataset(coords={'n_s': 	(['n_s'], np.arange(aux_i['n_test']), {'long_name': "Number of data samples"})})

	# save data into it
	if predictand_id == 'q':
		DS['height'] = xr.DataArray(height, dims=['n_s', 'n_height'], attrs={'long_name': "Height grid", 'units': "m"})
		pdb.set_trace() # check units of prediction  and predictand if q:
		DS['prediction'] = xr.DataArray(prediction*0.001, dims=['n_s', 'n_height'], 
									attrs={	'long_name': f"Predicted {predictand_id}", 'units': "SI units"})
		DS['reference'] = xr.DataArray(predictand*0.001, dims=['n_s', 'n_height'],
									attrs={'long_name': f"Reference {predictand_id}", 'units': "SI units"})

	else:
		# remove redundant dimension:
		prediction = prediction[:,0]
		predictand = predictand[:,0]

		DS['prediction'] = xr.DataArray(prediction, dims=['n_s'], 
										attrs={	'long_name': f"Predicted {predictand_id}", 'units': "SI units"})
		DS['reference'] = xr.DataArray(predictand, dims=['n_s'],
										attrs={'long_name': f"Reference {predictand_id}", 'units': "SI units"})


	# GLOBAL ATTRIBUTES:
	DS.attrs['title'] = f"Predicted and reference{predictand_id}"
	DS.attrs['author'] = "Andreas Walbroel (a.walbroel@uni-koeln.de), Institute for Geophysics and Meteorology, University of Cologne, Cologne, Germany"
	DS.attrs['predictor_TBs'] = aux_i['predictor_TBs']
	DS.attrs['setup_id'] = aux_i['file_descr']
	datetime_utc = dt.datetime.utcnow()
	DS.attrs['processing_date'] = datetime_utc.strftime("%Y-%m-%d %H:%M:%S")

	# export to netCDF:
	save_filename = aux_i['path_output_pred_ref'] + f"MOSAiC_synergetic_ret_prediction_and_reference_{predictand_id}_{aux_i['file_descr']}.nc"
	DS.to_netcdf(save_filename, mode='w', format='NETCDF4')
	DS.close()
	print(f"Saved {save_filename}")


def save_obs_predictions(
	path_output,
	prediction_ds,
	predictand_id,
	now_date,
	aux_i,
	ps_track_data,
	height=np.array([])):

	"""
	Save the Neural Network prediction to a netCDF file. Variables to be included:
	time, flag, output variable (prediction), standard error (std. dev. (bias corrected!),
	lat, lon, zsl (altitude above mean sea level),

	Parameters:
	-----------
	path_output : str
		Path where output is saved to.
	prediction_ds : xarray dataset
		Variables predicted by the Neural Network and additional information (flags, input tbs).
	predictand_id : str
		String indicating which output variable is forwarded to the function.
	now_date : str
		String idicating the currently processed date (in yyyy-mm-dd).
	aux_i : dict
		Dictionary containing additional information.
	ps_track_data : xarray dataset
		Dictionary containing Polarstern track data (lat, lon, time) for the MOSAiC expecition.
	height : array of floats
		Height array for respective predictand or predictand profiles (of i.e., temperature or 
		humidity). Can be a 1D or 2D array (latter must be of shape (n_training,n_height)).
	"""

	prediction = prediction_ds['output']

	path_output_l1 = path_output + "l1/"
	path_output_l2 = path_output + "l2/"


	# Add geoinfo data: Interpolate it on the MWR time axis, set the right attribute (source of
	# information):
	ps_track_data = ps_track_data.interp({'time': prediction_ds.time})


	# MOSAiC Legs to identify the correct Polarstern Track file:
	MOSAiC_legs = {'leg1': [dt.datetime(2019,9,20), dt.datetime(2019,12,13)],
				'leg2': [dt.datetime(2019,12,13), dt.datetime(2020,2,24)],
				'leg3': [dt.datetime(2020,2,24), dt.datetime(2020,6,4)],
				'leg4': [dt.datetime(2020,6,4), dt.datetime(2020,8,12)],
				'leg5': [dt.datetime(2020,8,12), dt.datetime(2020,10,12)]}

	# Add source of Polarstern track information as global attribute:
	source_PS_track = {'leg1': "Rex, Markus (2020): Links to master tracks in different resolutions of " +
								"POLARSTERN cruise PS122/1, Tromsø - Arctic Ocean, 2019-09-20 - 2019-12-13 " +
								"(Version 2). Alfred Wegener Institute, Helmholtz Centre for Polar and Marine Research, " +
								"Bremerhaven, PANGAEA, https://doi.org/10.1594/PANGAEA.924668",
						'leg2': "Haas, Christian (2020): Links to master tracks in different resolutions of " +
								"POLARSTERN cruise PS122/2, Arctic Ocean - Arctic Ocean, 2019-12-13 - 2020-02-24 " +
								"(Version 2). Alfred Wegener Institute, Helmholtz Centre for Polar and Marine Research, " +
								"Bremerhaven, PANGAEA, https://doi.org/10.1594/PANGAEA.924674",
						'leg3': "Kanzow, Torsten (2020): Links to master tracks in different resolutions of " +
								"POLARSTERN cruise PS122/3, Arctic Ocean - Longyearbyen, 2020-02-24 - 2020-06-04 " +
								"(Version 2). Alfred Wegener Institute, Helmholtz Centre for Polar and Marine Research, " +
								"Bremerhaven, PANGAEA, https://doi.org/10.1594/PANGAEA.924681",
						'leg4': "Rex, Markus (2021): Master tracks in different resolutions of " +
								"POLARSTERN cruise PS122/4, Longyearbyen - Arctic Ocean, 2020-06-04 - 2020-08-12. " +
								"Alfred Wegener Institute, Helmholtz Centre for Polar and Marine Research, " +
								"Bremerhaven, PANGAEA, https://doi.org/10.1594/PANGAEA.926829",
						'leg5': "Rex, Markus (2021): Master tracks in different resolutions of " +
								"POLARSTERN cruise PS122/5, Arctic Ocean - Bremerhaven, 2020-08-12 - 2020-10-12. " +
								"Alfred Wegener Institute, Helmholtz Centre for Polar and Marine Research, " +
								"Bremerhaven, PANGAEA, https://doi.org/10.1594/PANGAEA.926910"}


	# Save the data on a daily basis:
	# Also set flag bits to 1024 (which means adding 1024) when retrieved quantity is beyond thresholds:
	import sklearn
	import netCDF4 as nc
	l1_var = 'tb'
	l1_var_units = "K"
	l1_version = "v00"
	l2_version = "v00"
	if predictand_id == 'iwv':
		output_var = 'prw'
		output_units = "kg m-2"

		prediction_thresh = [0, 100]		# kg m-2
		idx_beyond = np.where((prediction < prediction_thresh[0]) | (prediction > prediction_thresh[1]))[0]
		prediction_ds['flag_h'][idx_beyond] += 1024
		prediction_ds['flag_m'][idx_beyond] += 1024

	elif predictand_id == 'lwp':
		output_var = 'clwvi'
		output_units = "kg m-2"

		prediction_thresh = [-0.2, 3.0]		# kg m-2
		idx_beyond = np.where((prediction < prediction_thresh[0]) | (prediction > prediction_thresh[1]))[0]
		prediction_ds['flag_h'][idx_beyond] += 1024
		prediction_ds['flag_m'][idx_beyond] += 1024

	elif predictand_id == 'temp':
		output_var = 'temp'
		if 'tb_bl' in aux_i['predictors']: output_var += '_bl'
		output_units = "K"

		prediction_thresh = [180.0, 310.0]		# K
		idx_beyond = np.where((prediction < prediction_thresh[0]) | (prediction > prediction_thresh[1]))[0]
		prediction_ds['flag_h'][idx_beyond] += 1024
		prediction_ds['flag_m'][idx_beyond] += 1024

	elif predictand_id == 'q':
		output_var = 'q'
		output_units = "kg kg-1"

		prediction_thresh = [0.0, 0.06]		# kg kg-1
		prediction *= 0.001	# to convert back to kg kg-1
		idx_beyond = np.where((prediction < prediction_thresh[0]) | (prediction > prediction_thresh[1]))[0]
		prediction_ds['flag_h'][idx_beyond] += 1024
		prediction_ds['flag_m'][idx_beyond] += 1024


	now_date = dt.datetime.strptime(now_date, "%Y-%m-%d")
	# path_addition = f"{now_date.year:04}/{now_date.month:02}/{now_date.day:02}/"
	path_addition = ""

	# check if path exists:
	path_output_dir = os.path.dirname(path_output_l1 + path_addition)
	if not os.path.exists(path_output_dir):
		os.makedirs(path_output_dir)
	path_output_dir = os.path.dirname(path_output_l2 + path_addition)
	if not os.path.exists(path_output_dir):
		os.makedirs(path_output_dir)


	# set the global attribute:
	if (now_date >= MOSAiC_legs['leg1'][0]) and (now_date < MOSAiC_legs['leg1'][1]):
		globat = source_PS_track['leg1']
	elif (now_date >= MOSAiC_legs['leg2'][0]) and (now_date < MOSAiC_legs['leg2'][1]):
		globat = source_PS_track['leg2']
	elif (now_date >= MOSAiC_legs['leg3'][0]) and (now_date < MOSAiC_legs['leg3'][1]):
		globat = source_PS_track['leg3']
	elif (now_date >= MOSAiC_legs['leg4'][0]) and (now_date < MOSAiC_legs['leg4'][1]):
		globat = source_PS_track['leg4']
	elif (now_date >= MOSAiC_legs['leg5'][0]) and (now_date <= MOSAiC_legs['leg5'][1]):
		globat = source_PS_track['leg5']


	# Save predictions (level 2) to xarray dataset, then to netcdf:
	nc_output_name = f"MOSAiC_uoc_hatpro_lhumpro-243-340_l2_{output_var}_{l2_version}_{now_date.strftime('%Y%m%d%H%M%S')}"

	# create Dataset:
	if predictand_id in ['iwv', 'lwp']:
		DS = xr.Dataset(coords=	{'time': (['time'], prediction_ds.time.values.astype("datetime64[s]").astype(np.float64),
								{'units': "seconds since 1970-01-01 00:00:00 UTC",
								'standard_name': "time"})})
	else:
		DS = xr.Dataset(coords=	{'time': 	(['time'], prediction_ds.time.values.astype("datetime64[s]").astype(np.float64),
											{'units': "seconds since 1970-01-01 00:00:00 UTC",
											'standard_name': "time"}),
								'height': 	(['height'], height[0,:],
											{'standard_name': "height",
											'long_name': "height above sea level", 
											'units': "m"})})

	DS['lat'] = xr.DataArray(ps_track_data['lat'].values.astype(np.float32), dims=['time'],
								attrs={'units': "degree_north",
									'standard_name': "latitude",
									'long_name': "latitude of the RV Polarstern"})
	DS['lon'] = xr.DataArray(ps_track_data['lon'].values.astype(np.float32), dims=['time'],
								attrs={'units': "degree_east",
									'standard_name': "longitude",
									'long_name': "longitude of the RV Polarstern"})
	DS['zsl'] = xr.DataArray(np.full_like(ps_track_data['lat'].values, 21.0).astype(np.float32), dims=['time'],
								attrs={'units': "m",
									'standard_name': "altitude",
									'long_name': "altitude above mean sea level"})
	DS['ele'] = xr.DataArray(np.full_like(ps_track_data['lat'].values, 90.0).astype(np.float32), dims=['time'],
								attrs={'units': "degree",
									'standard_name': "elevation_angle",
									'long_name': "observation elevation angle"})
	DS['flag_m'] = xr.DataArray(prediction_ds['flag_m'].values.astype(np.short), dims=['time'],
								attrs={'long_name': "quality control flags for MiRAC-P (LHUMPRO-243-340)",
									'flag_masks': np.array([1,2,4,8,16,32,64,128,256,512,1024], dtype=np.short),
									'flag_meanings': ("visual_inspection_filter_band_1 visual_inspection_filter_band2 visual_inspection_filter_band3 " +
														"rain_flag sanity_receiver_band1 sanity_receiver_band1 sun_in_beam unused " +
														"unused tb_threshold_band1 retrieved_quantity_threshold"),
									'comment': ("Flags indicate data that the user should only use with care. In cases of doubt, please refer " +
												"to the contact person. A Fillvalue of 0 means that data has not been flagged. " +
												"Bands refer to the measurement ranges (if applicable) of the microwave radiometer; " +
												"i.e band 1: all lhumpro frequencies (170-200, 243, and 340 GHz); tb valid range: " +
												f"[  2.70, 330.00] in K; retrieved quantity valid range: {str(prediction_thresh)} in {output_units}; ")})
	DS['flag_h'] = xr.DataArray(prediction_ds['flag_h'].values.astype(np.short), dims=['time'],
								attrs={'long_name': "quality control flags for HATPRO G5",
									'flag_masks': np.array([1,2,4,8,16,32,64,128,256,512,1024], dtype=np.short),
									'flag_meanings': ("visual_inspection_filter_band_1 visual_inspection_filter_band2 visual_inspection_filter_band3 " +
														"rain_flag sanity_receiver_band1 sanity_receiver_band2 sun_in_beam tb_threshold_band1 " +
														"tb_threshold_band2 tb_threshold_band3 retrieved_quantity_threshold"),
									'comment': ("Flags indicate data that the user should only use with care. In cases of doubt, please refer " +
												"to the contact person. A Fillvalue of 0 means that data has not been flagged. " +
												"Bands refer to the measurement ranges (if applicable) of the microwave radiometer; " +
												"i.e band 1: 20-35 GHz, band 2: 50-60 GHz, band 3: 90 GHz; tb valid range: " +
												f"[  2.70, 330.00] in K; retrieved quantity valid range: {str(prediction_thresh)} in {output_units}; ")})


	if predictand_id == 'iwv':
		DS[output_var] = xr.DataArray(prediction.values.flatten().astype(np.float32), dims=['time'],
										attrs={
										'units': output_units,
										'standard_name': "atmosphere_mass_content_of_water_vapor",
										'long_name': "integrated water vapor or precipitable water",
										'comment': ("These values denote the vertically integrated amount of water vapor from the surface to TOA. " +
													"The (bias corrected) standard error of atmosphere mass content of water vapor is " +
													f"?.?? {output_units}. More precisely, the " +											################################# add error stats
													f"standard errors of {output_var} in the ranges [0, 5), [5, 10), [10, 100) {output_units} are " +
													f"?.??, ?.??, ?.?? {output_units}.")})		###################	#############		#########################################

	if predictand_id == 'lwp':
		DS[output_var] = xr.DataArray(prediction.values.flatten().astype(np.float32), dims=['time'],
										attrs={
										'units': output_units,
										'standard_name': "atmosphere_mass_content_of_cloud_liquid_water_content",
										'long_name': "liquid water path",
										'comment': ("These values denote the vertically integrated amount of condensed water from the surface to TOA. " +
													"The (bias corrected) standard error of atmosphere mass content of cloud liquid water content is " +
													f"?.?? {output_units}. More precisely, the ................................"									################################# add error stats
													)})		###################	#############		#########################################)

	if predictand_id == 'temp':
		DS[predictand_id] = xr.DataArray(prediction.values.astype(np.float32), dims=['time', 'height'],
										attrs={
										'units': output_units,
										'standard_name': "air_temperature",
										'long_name': "air temperature",
										'comment': ("The (bias corrected) standard error of air temperature is " +
													f"?.?? {output_units}. More precisely, the standard errors for height ranges " +								################################# add error stats
													f"[0, 1500), [1500, 5000), [5000, top) m are ?.??, ?.??, ?.?? {output_units}")})

	if predictand_id == 'q':
		DS[output_var] = xr.DataArray(prediction.values.astype(np.float32), dims=['time', 'height'],
										attrs={
										'units': output_units,
										'standard_name': 'specific_humidity',
										'long_name': 'specific humidity',
										'comment': ("The (bias corrected) standard error of specific humidity is " +
													f"?.?? {output_units}. More precisely, the standard errors for height ranges " +								################################# add error stats
													f"[0, 1500), [1500, 5000), [5000, top) m are ?.??, ?.??, ?.?? {output_units}")})



	# adapt fill values:
	# Make sure that _FillValue is not added to certain variables:
	exclude_vars_fill_value = ['time', 'lat', 'lon', 'zsl']
	for kk in exclude_vars_fill_value:
		if kk in DS.variables:
			DS[kk].encoding["_FillValue"] = None

	# add fill values to remaining variables:
	vars_fill_value = ['ele', 'flag_h', 'flag_m', 'prw', 'clwvi', 'clwvi_offset', 'temp', 'q']
	for kk in vars_fill_value:
		if kk in DS.variables:
			if kk not in ['flag_h', 'flag_m']:
				DS[kk].encoding["_FillValue"] = float(-999.)
			else:
				DS[kk].encoding["_FillValue"] = np.array([0]).astype(np.short)[0]

	DS.attrs['Title'] = f"Microwave radiometer retrieved {output_var}"
	DS.attrs['Institution'] = "Institute for Geophysics and Meteorology, University of Cologne, Cologne, Germany"
	DS.attrs['Contact_person'] = "Andreas Walbroel (a.walbroel@uni-koeln.de)"
	DS.attrs['Source'] = "RPG HATPRO G5 and LHUMPRO-243-340 G5 microwave radiometer"
	DS.attrs['Dependencies'] = ("HATPRO and MiRAC-P brightness temperatures: " +
								"https://doi.org/10.1594/PANGAEA.941356, https://doi.org/10.1594/PANGAEA.941407")
	DS.attrs['Conventions'] = "CF-1.6"
	datetime_utc = dt.datetime.utcnow()
	DS.attrs['Processing_date'] = datetime_utc.strftime("%Y-%m-%d %H:%M:%S")
	DS.attrs['Author'] = "Andreas Walbroel (a.walbroel@uni-koeln.de)"
	DS.attrs['Comments'] = ""
	DS.attrs['License'] = "For non-commercial use only."
	DS.attrs['Measurement_site'] = "RV Polarstern"
	DS.attrs['Position_source'] = globat

	DS.attrs['retrieval_type'] = "Neural Network"
	DS.attrs['python_packages'] = (f"python version: {sys.version}, tensorflow: {tensorflow.__version__}, keras: {keras.__version__}, " +
									f"numpy: {np.__version__}, sklearn: {sklearn.__version__}, netCDF4: {nc.__version__}, " +
									f"matplotlib: {mpl.__version__}, xarray: {xr.__version__}, pandas: {pd.__version__}")

	DS.attrs['retrieval_net_architecture'] = f"n_hidden_layers: {str(aux_i['n_layers'])}; nodes_for_hidden_layers: {str(aux_i['n_nodes'])}"
	DS.attrs['retrieval_batch_size'] = f"{str(aux_i['batch_size'])}"
	DS.attrs['retrieval_epochs'] = f"{str(aux_i['epochs'])}"
	DS.attrs['retrieval_learning_rate'] = f"{str(aux_i['learning_rate'])}"
	DS.attrs['retrieval_activation_function'] = f"{aux_i['activation']} (from input to hidden layer and subsequent hidden layers)"
	DS.attrs['retrieval_feature_range'] = f"feature range of sklearn.preprocessing.MinMaxScaler: {str(aux_i['feature_range'])}"
	DS.attrs['retrieval_rng_seed'] = str(aux_i['seed'])
	DS.attrs['retrieval_kernel_initializer'] = f"{aux_i['kernel_init']}"
	DS.attrs['retrieval_optimizer'] = "keras.optimizers.Adam"
	DS.attrs['retrieval_callbacks'] = (f"EarlyStopping(monitor=val_loss, patience={str(aux_i['callback_patience'])}, " +
										f"min_delta={str(aux_i['min_delta'])}, restore_best_weights=True)")


	if aux_i['site'] == 'pol':
		DS.attrs['training_data'] = "ERA-Interim"
		tdy_str = ""
		for year_str in aux_i['yrs_training']: tdy_str += f"{str(year_str)}, "
		DS.attrs['training_data_years'] = tdy_str[:-2]

		if aux_i['nya_test_data']:
			DS.attrs['test_data'] = "Ny-Alesund radiosondes 2006-2017"
		else:
			DS.attrs['test_data'] = "ERA-Interim"
			tdy_str = ""
			for year_str in aux_i['yrs_testing']: tdy_str += f"{str(year_str)}, "
			DS.attrs['test_data_years'] = tdy_str[:-2]

	elif aux_i['site'] == 'nya':
		DS.attrs['training_data'] = "Ny-Alesund radiosondes 2006-2017"
		tdy_str = ""
		for year_str in aux_i['yrs_training']: tdy_str += f"{str(year_str)}, "
		DS.attrs['training_data_years'] = tdy_str[:-2]

		DS.attrs['test_data'] = "Ny-Alesund radiosondes 2006-2017"
		tdy_str = ""
		for year_str in aux_i['yrs_testing']: tdy_str += f"{str(year_str)}, "
		DS.attrs['test_data_years'] = tdy_str[:-2]

	elif aux_i['site'] == 'era5':
		DS.attrs['training_data'] = "ERA5"
		tdy_str = ""
		for year_str in aux_i['yrs_training']: tdy_str += f"{str(year_str)}, "
		DS.attrs['training_data_years'] = tdy_str[:-2]

		DS.attrs['test_data'] = "ERA5"
		tdy_str = ""
		for year_str in aux_i['yrs_testing']: tdy_str += f"{str(year_str)}, "
		DS.attrs['test_data_years'] = tdy_str[:-2]
		

	DS.attrs['n_training_samples'] = aux_i['n_training']
	DS.attrs['n_test_samples'] = aux_i['n_test']
	DS.attrs['training_test_TB_noise_std_dev'] = (f"22-60 GHz: {cat[test_id]['noise_kv']:.2g}, " +
													f"170-195 GHz: {cat[test_id]['noise_g']:.2g}, " +
													f"243 GHz: {cat[test_id]['noise_243']:.2g}, " +
													f"340 GHz: {cat[test_id]['noise_340']:.2g}")

	# input vector information: First, TBs, then remaining predictors
	DS.attrs['input_vector'] = "("
	for ff in prediction_ds.freq:
		DS.attrs['input_vector'] += f"TB_{ff.values:.2f}GHz, "
	DS.attrs['input_vector'] = DS.attrs['input_vector'][:-2]

	# add other input vector:
	predictor_transl_dict = {'pres_sfc': 'pres_sfc', 'DOY_1': 'cos(DayOfYear)',
								'DOY_2': 'sin(DayOfYear)', 'CF': 'liq_cloud_flag',
								'iwv': 'IWV', 't2m': "2m_air_temperature",
								'tb_bl': "boundary_layer_scan_TBs"}
	for pred_key in aux_i['predictors'][1:]:	# from index 1 because TBs have been listed already
		DS.attrs['input_vector'] = DS.input_vector + f", {predictor_transl_dict[pred_key]}"
	DS.attrs['input_vector'] = DS.input_vector + ")"
	DS.attrs['output_vector'] = f"{output_var}"


	# encode time:
	DS['time'] = prediction_ds.time.values.astype("datetime64[s]").astype(np.float64)
	DS['time'].attrs['units'] = "seconds since 1970-01-01 00:00:00"
	DS['time'].encoding['units'] = 'seconds since 1970-01-01 00:00:00'
	DS['time'].encoding['dtype'] = 'double'

	DS.to_netcdf(path_output_l2 + path_addition + nc_output_name + ".nc", mode='w', format="NETCDF4")
	DS.close()


def save_mosaic_test_obs(
	prediction, 
	predictand_id,
	aux_i,
	height=np.array([])):

	"""
	Saves the prediction of the Neural Network based on a small test set of real MOSAiC 
	observations.

	Parameters:
	-----------
	prediction : xarray DataArray of floats
		Variables predicted by the Neural Network.
	predictand_id : str
		String indicating which output variable is forwarded to the function.
	aux_i : dict
		Dictionary containing additional information.
	height : array of floats
		Height array for respective predictand or predictand profiles (of i.e., temperature or 
		humidity). Can be a 1D or 2D array (latter must be of shape (n_training,n_height)).
	"""

	# check if output path exists: if it doesn't, create it:
	path_output_dir = os.path.dirname(aux_i['path_output_pred_ref'])
	if not os.path.exists(path_output_dir):
		os.makedirs(path_output_dir)


	# create xarray Dataset:
	DS = xr.Dataset(coords={'time': 	(prediction.time),
							'height': 	(['height'], height[0,:],
										{'long_name': "Height", 'units': "m"})})

	# save data into it:
	if predictand_id in ['q']:
		DS[predictand_id] = prediction*0.001	# convert back to kg kg-1
	elif predictand_id in ['temp']:
		DS[predictand_id] = prediction
	elif predictand_id in ['iwv', 'lwp']:
		DS[predictand_id] = prediction
	DS[predictand_id].attrs = {'long_name': f"Predicted {predictand_id}", 'units': "SI units"}


	# GLOBAL ATTRIBUTES:
	DS.attrs['title'] = f"MOSAiC test observations; predicted {predictand_id}"
	DS.attrs['author'] = "Andreas Walbroel (a.walbroel@uni-koeln.de), Institute for Geophysics and Meteorology, University of Cologne, Cologne, Germany"
	DS.attrs['predictor_TBs'] = aux_i['predictor_TBs']
	DS.attrs['predictors'] = aux_i['predictors']
	DS.attrs['setup_id'] = aux_i['file_descr']
	datetime_utc = dt.datetime.utcnow()
	DS.attrs['processing_date'] = datetime_utc.strftime("%Y-%m-%d %H:%M:%S")
	DS.attrs['python_version'] = f"python version: {sys.version}"
	DS.attrs['python_packages'] = (f"numpy: {np.__version__}, matplotlib: {mpl.__version__}, " +
									f"xarray: {xr.__version__}, yaml: {yaml.__version__}, " +
									f"tensorflow: {tensorflow.__version__}, pandas: {pd.__version__}")

	# time encoding:
	DS['time'] = DS.time.values.astype("datetime64[s]").astype(np.float64)
	DS['time'].attrs['units'] = "seconds since 1970-01-01 00:00:00"
	DS['time'].encoding['units'] = 'seconds since 1970-01-01 00:00:00'
	DS['time'].encoding['dtype'] = 'double'


	# export to netCDF:
	save_filename = aux_i['path_output_pred_ref'] + f"MOSAiC_test_obs_synergetic_ret_prediction_{predictand_id}_{aux_i['file_descr']}.nc"
	DS.to_netcdf(save_filename, mode='w', format='NETCDF4')
	DS.close()
	print(f"Saved {save_filename}")


def simple_quality_control(predictand_training, predictand_test, aux_i):

	"""
	Quality control of the data: See RPG Software Manual 
	(RPG_MWR_STD_Software_Manual_G5_2021.pdf)

	Parameters:
	predictand_training : radiometer class
		Contains information about the training data predictand.
	predictand_test : radiometer class
		Contains information about the test data predictand.
	aux_i : dict
		Contains additional information.
	"""

	height_dif_training = np.diff(predictand_training.height, axis=1)
	height_dif_test = np.diff(predictand_test.height, axis=1)
	pres_dif_training = np.diff(predictand_training.pres, axis=1)
	pres_dif_test = np.diff(predictand_test.pres, axis=1)

	# check if height increases in subsequent levels, pressure decreases with height,
	# temp in [190, 330], pres_sfc > 50000, pres in [1, 107000], height in [-200, 70000],
	# temp and pres information at least up to 10 km; hum. information up to -30 deg C,
	# n_levels >= 10
	# (assert is split into many parts to more easily identify broken variables)
	# Broken temp, pres, height, or humidity values might cause the computed IWV to be
	# erroneous
	assert ((np.all(height_dif_training > 0)) and (np.all(height_dif_test > 0)) and 
			(np.all(pres_dif_training <= 0)) and (np.all(pres_dif_test <= 0)))
	assert ((np.all(predictand_training.temp <= 330)) and (np.all(predictand_training.temp >= 190)) 
			and (np.all(predictand_test.temp <= 330)) and (np.all(predictand_test.temp >= 190)))
	assert ((np.all(predictand_training.pres[:,0] > 50000)) and (np.all(predictand_test.pres[:,0] > 50000)) 
			and (np.all(predictand_training.pres > 1)) and (np.all(predictand_training.pres < 107000)) 
			and (np.all(predictand_test.pres > 1)) and (np.all(predictand_test.pres < 107000)))
	assert ((np.all(predictand_training.height[:,0] > -200)) and (np.all(predictand_training.height[:,-1] < 70000)) 
			and (np.all(predictand_test.height[:,0] > -200)) and (np.all(predictand_test.height[:,-1] < 70000)))
	assert (predictand_training.height.shape[1] >= 10) and (predictand_test.height.shape[1] >= 10)

	# on a regular grid, it's simple to check if temp and pres information exist up to 10 km height:
	idx_10km_train = np.where(predictand_training.height[0,:] >= 10000)[0]
	idx_10km_test = np.where(predictand_test.height[0,:] >= 10000)[0]

	for k in range(aux_i['n_training']): 
		assert ((np.any(~np.isnan(predictand_training.temp[k,idx_10km_train]))) and 
				(np.any(~np.isnan(predictand_training.pres[k,idx_10km_train]))))

		# check if hum. information available up to -30 deg C:
		idx_243K = np.where(predictand_training.temp[k,:] <= 243.15)[0]
		assert np.any(~np.isnan(predictand_training.rh[k,idx_243K]))

	for k in range(aux_i['n_test']): 
		assert ((np.any(~np.isnan(predictand_test.temp[k,idx_10km_test]))) and 
				(np.any(~np.isnan(predictand_test.pres[k,idx_10km_test]))))

		# check if hum. information available up to -30 deg C:
		idx_243K = np.where(predictand_test.temp[k,:] <= 243.15)[0]
		assert np.any(~np.isnan(predictand_test.rh[k,idx_243K]))


def NN_retrieval(predictor_training, predictand_training, predictor_test,
					predictand_test, aux_i, return_test_loss=True):

	print("(batch_size, epochs, seed)=", aux_i['batch_size'], aux_i['epochs'], aux_i['seed'])
	print("learning_rate=", aux_i['learning_rate'])

	# Initialize and define the NN model
	input_shape = predictor_training.input.shape
	output_shape = predictand_training.output.shape
	model = Sequential()

	model.add(Dense(aux_i['n_nodes'][0], input_dim=input_shape[1], activation=aux_i['activation'], kernel_initializer=aux_i['kernel_init']))
	if aux_i['batch_normalization']: model.add(BatchNormalization())
	if aux_i['dropout'] > 0.0: model.add(Dropout(aux_i['dropout']))

	# space for more layers:
	if (aux_i['n_layers'] > 1) and (aux_i['dropout'] > 0.0):
		for n_l in range(1,aux_i['n_layers']):
			model.add(Dense(aux_i['n_nodes'][n_l], activation=aux_i['activation'], kernel_initializer=aux_i['kernel_init']))
			if aux_i['batch_normalization']: model.add(BatchNormalization())
			model.add(Dropout(aux_i['dropout']))
	elif aux_i['n_layers'] > 1:
		for n_l in range(1,aux_i['n_layers']):
			model.add(Dense(aux_i['n_nodes'][n_l], activation=aux_i['activation'], kernel_initializer=aux_i['kernel_init']))
			if aux_i['batch_normalization']: model.add(BatchNormalization())

	model.add(Dense(output_shape[1], activation='linear'))		# output layer shape must be equal to retrieved variables

	# compile and train the NN model
	model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=aux_i['learning_rate']))
	history = model.fit(predictor_training.input_scaled, predictand_training.output, batch_size=aux_i['batch_size'],
				epochs=aux_i['epochs'], verbose=1,
				validation_data=(predictor_test.input_scaled, predictand_test.output),
				callbacks=[EarlyStopping(monitor='val_loss', patience=aux_i['callback_patience'], min_delta=aux_i['min_delta'],
				restore_best_weights=True)],
				)

	test_loss = np.asarray(history.history['val_loss']).min()			# test data MSE
	print("n_epochs executed: ", len(history.history['loss']))
	print("Test loss: ", test_loss)

	if return_test_loss:
		return model, test_loss
	else:
		return model


###################################################################################################
###################################################################################################


"""
	In this script, Tensorflow.Keras will be used to retrieve of at least IWV and humidity profiles, but
	possibly also LWP and temperature profiles from ground-based microwave radiometer (MWR) TB 
	measurements (MiRAC-P and HATPRO). The following steps are executed:
	- Importing training and test data; split into training and test data sets
	- quality control of the data;
	- define, rescale and build input vector (predictors)
	- define predictands
	- define and build Neural Network model
	- compile model: choose loss function and optimizer
	- fit model (training): try various subsets of the entire data as training; try
		different batch sizes and learning rates; validate with test data
	- evaluate model (with test data)
	- predict unknown output from new data (application on MiRAC-P obs during MOSAiC)
"""


# determine test_id
test_id = "000" # specify the intention of a test (used for the retrieval statistics output .nc file)
if len(sys.argv) == 2:
	test_id = sys.argv[1]
elif len(sys.argv) > 2:
	raise ValueError("Sorry, I didn't get that. Just type 'python3 NN_retrieval.py' or " +
					"'python3 NN_retrieval.py " + '"003"' + "' (as example for test run 003)....")

# exec_type determines if 20 random numbers shall be cycled through ("20_runs") or whether only one random
# number is to be used ("op_ret")
exec_type = 'op_ret'



# open test_purpose.YAML file to manage settings:
with open(wdir + "test_purpose.yaml", 'r') as f:
	cat = yaml.safe_load(f)


aux_i = dict()	# dictionary that collects additional information
rs_version = 'mwr_pro'			# radiosonde type: 'mwr_pro' means that the structure is built
								# so that mwr_pro retrieval can read it (for Ny-Alesund radiosondes and unconcatenated ERA-I)
aux_i['file_descr'] = test_id.replace(" ", "_").lower()	# file name addition (of some plots and netCDF output)
aux_i['site'] = 'era5'			# options of training and test data: 'nya' for Ny-Alesund radiosondes
								# 'pol': ERA-Interim grid points north of 84.5 deg N
								# 'era5': ERA5 training and test data
aux_i['predictors'] = cat[test_id]['predictors']	# specify input vector (predictors): options: TBs, DOY_1, DOY_2, pres_sfc, CF
													# TBs: up to all HATPRO and MiRAC-P channels
													# TB_BL: boundary layer scan TBs for V-band frequencies
													# DOY_1: cos(day_of_year)
													# DOY_2: sin(day_of_year)
													# pres_sfc: surface pressure (not recommended)
													# CF: binary cloud flag (either 0 or 1)
													# iwv: integrated water vapour (not possible if IWV in predictand list
													# t2m: 2 m air temperature in K
aux_i['predictor_TBs'] = cat[test_id]['predictor_TBs']	# string to identify which bands of TBs are used as predictors
														# syntax as in data_tools.select_MWR_channels

# NN settings:
aux_i['n_layers'] = cat[test_id]['n_layers']		# number of hidden layers (integer)
aux_i['n_nodes'] = cat[test_id]['n_nodes']			# number of nodes for each hidden layer as list: 
													# [n_node_layer0, n_node_layer1, n_node_layer2, ...]
aux_i['dropout'] = cat[test_id]['dropout']			# dropout chance in [0.0, 1.0]; if 0.0: no dropout layers
aux_i['batch_normalization'] = cat[test_id]['batch_normalization']	# bool if BatchNormalization layer is used in hidden layers
aux_i['activation'] = cat[test_id]['activ_f']		# default or best estimate for i.e., iwv: exponential
aux_i['feature_range'] = tuple(cat[test_id]['feature_range'])	# best est. with exponential (-3.0, 1.0)
aux_i['epochs'] = cat[test_id]['epochs']
aux_i['batch_size'] = cat[test_id]['batch_size']
aux_i['learning_rate'] = cat[test_id]['learning_rate']		# default: 0.001
aux_i['kernel_init'] = cat[test_id]['kernel_init']			# default: 'glorot_uniform'
aux_i['callback_patience'] = cat[test_id]['callback_patience']	# patience of the callback
aux_i['min_delta'] = cat[test_id]['min_delta']				# min val_loss improvement needed for earlystopping

aux_i['predictor_instrument'] = {	'pol': "syn_mwr_pro",	# argument to load predictor data
									'nya': "synthetic",
									'era5': "era5_pam"}
aux_i['predictand'] = cat[test_id]['predictand']			# output variable / predictand: options: 
															# list with elements in ["iwv", "lwp", "q", "temp"]


aux_i['yrs'] = {'pol': ["2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011",
				"2012", "2013", "2014", "2015", "2016", "2017"],
				'nya': ["2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", 
						"2015", "2016", "2017"],
				'era5': ["2002", "2003", "2004", "2005", "2007", "2008", "2009", "2010", 
				"2012", "2013", "2014", "2016", "2017", "2018"]}		# available years of data
aux_i['yrs'] = aux_i['yrs'][aux_i['site']]
n_yrs = len(aux_i['yrs'])
n_training = round(0.8*n_yrs)			# number of training years; default: 0.7
n_test = n_yrs - n_training


aux_i['add_TB_noise'] = True				# if True, random noise will be added to training and test data. 
											# Remember to define a noise dictionary if True
aux_i['vis_eval'] = True					# if True: visualize retrieval evaluation (test predictand vs prediction) (only true if aux_i['op_ret'] == False)
aux_i['save_figures'] = True				# if True: figures created will be saved to file
aux_i['lwp_offset_cor'] = False				# if True: LWP offset correction will be applied on the mosaic_test_subset
aux_i['tb_offset_cor'] = False				# if True: MOSAiC TB obs will be corrected for offsets
aux_i['op_ret'] = False						# if True: some NN output of one spec. random number will be generated
aux_i['save_obs_predictions'] = False	# if True, predictions made from MWR observations will be saved
										# to a netCDF file (i.e., for op_ret retrieval)
aux_i['mosaic_test_subset']	= True		# used to decide if also MOSAiC obs will be tested and saved for one RNG seed
aux_i['test_on_all_rngs'] = False		# if True, visualize_evaluation and mosaic_test_subset (if active) are exec. on all RNG seeds; should be False mostly
aux_i['1D_aligned'] = False					# indicator if training/test data is aligned on a 1D or 2D spatial grid
if aux_i['site'] == "era5":
	aux_i['1D_aligned'] = True

# decide if information content is to be computed:
aux_i['get_info_content'] = False


if exec_type == 'op_ret':
	aux_i['op_ret'] = True
	aux_i['vis_eval'] = False
	aux_i['save_obs_predictions'] = True
	aux_i['test_on_all_rngs'] = False

# remove IWV from predictors when it should be retrieved:
if ('iwv' in aux_i['predictand']) and ('iwv' in aux_i['predictors']): aux_i['predictors'].remove('iwv')


# paths:
if remote:
	aux_i['path_output'] = "/net/blanc/awalbroe/Data/synergetic_ret/tests_01/output/"				# path where output is saved to
	aux_i['path_output_info'] = "/net/blanc/awalbroe/Data/synergetic_ret/tests_01/info_content/"	# path where output is saved to
	aux_i['path_output_pred_ref'] = "/net/blanc/awalbroe/Data/synergetic_ret/tests_01/prediction_and_reference/"	# path where output is saved to
	aux_i['path_data'] = {'nya': "/net/blanc/awalbroe/Data/mir_fwd_sim/new_rt_nya/",
				'pol': "/net/blanc/awalbroe/Data/MOSAiC_radiometers/retrieval_training/mirac-p/",
				'era5': "/net/blanc/awalbroe/Data/synergetic_ret/training_data_01/merged/new_z_grid/"}		# path of training/test data
	aux_i['path_data'] = aux_i['path_data'][aux_i['site']]
	aux_i['path_ps_track'] = "/data/obs/campaigns/mosaic/polarstern_track/"							# path of Polarstern track data
	aux_i['path_tb_obs'] = {'hatpro': "/net/blanc/awalbroe/Data/MOSAiC_radiometers/HATPRO_l1_v01/",
							'mirac-p': "/net/blanc/awalbroe/Data/MOSAiC_radiometers/MiRAC-P_l1_v01/"} 	# path of published level 1 tb data
	aux_i['path_tb_offsets'] = "/net/blanc/awalbroe/Data/MOSAiC_radiometers/mwr_offsets/"			# path of the MOSAiC MWR TB offset correction
	aux_i['path_old_ret'] = "/net/blanc/awalbroe/Data/MOSAiC_radiometers/HATPRO_l2_v01/"				# path of old HATPRO retrievals
	aux_i['path_rs_obs'] = "/data/radiosondes/Polarstern/PS122_MOSAiC_upper_air_soundings/Level_2/"						############################################################################### LEVEL 3 !!!
	aux_i['path_plots'] = "/net/blanc/awalbroe/Plots/synergetic_ret/tests_01/"
	aux_i['path_plots_info'] = "/net/blanc/awalbroe/Plots/synergetic_ret/info_content/"

else:
	aux_i['path_output'] = "/mnt/f/heavy_data/synergetic_ret/synergetic_ret/tests_01/output/"			# path where output is saved to
	aux_i['path_output_info'] = "/mnt/f/heavy_data/synergetic_ret/synergetic_ret/tests_01/info_content/"	# path where output is saved to
	aux_i['path_output_pred_ref'] = "/mnt/f/heavy_data/synergetic_ret/synergetic_ret/tests_01/prediction_and_reference/"	# path where output is saved to
	aux_i['path_data'] = {'nya': "/mnt/f/heavy_data/synergetic_ret/mir_fwd_sim/new_rt_nya/",
				'pol': "/mnt/f/heavy_data/synergetic_ret/MOSAiC_radiometers/retrieval_training/mirac-p/",
				'era5': "/mnt/f/heavy_data/synergetic_ret/synergetic_ret/training_data_01/merged/new_z_grid/"}		# path of training/test data
	aux_i['path_data'] = aux_i['path_data'][aux_i['site']]
	aux_i['path_ps_track'] = "/mnt/f/heavy_data/polarstern_track/"										# path of Polarstern track data
	aux_i['path_tb_obs'] = {'hatpro': "/mnt/f/heavy_data/MOSAiC_radiometers/HATPRO_l1_v01/",
							'mirac-p': "/mnt/f/heavy_data/MOSAiC_radiometers/MiRAC-P_l1_v01/"} 	# path of published level 1 tb data
	aux_i['path_tb_offsets'] = "/mnt/f/heavy_data/MOSAiC_radiometers/mwr_offsets/"				# path of the MOSAiC MWR TB offset correction
	aux_i['path_old_ret'] = "/mnt/f/heavy_data/MOSAiC_radiometers/HATPRO_l2_v01/"				# path of old HATPRO retrievals
	aux_i['path_rs_obs'] = "/mnt/f/heavy_data/MOSAiC_radiosondes/"						######################################################################################################################## LEVEL 3 !!
	aux_i['path_plots'] = "/mnt/f/Studium_NIM/work/Plots/synergetic_ret/tests_01/"
	aux_i['path_plots_info'] = "/mnt/f/Studium_NIM/work/Plots/synergetic_ret/info_content/"


# time range of tb data to be imported (LATER PART OF THE RETRIEVAL DEV)
aux_i['mosaic_leg'] = 1
aux_i['considered_period'] = 'mosaic'	#f"leg{aux_i['mosaic_leg']}"	# specify which period shall be plotted or computed:
									# DEFAULT: 'mwr_range': 2019-09-30 - 2020-10-02
									# 'mosaic': entire mosaic period (2019-09-20 - 2020-10-12)
									# 'leg1': 2019-09-20 - 2019-12-13
									# 'leg2': 2019-12-13 - 2020-02-24
									# 'leg3': 2020-02-24 - 2020-06-04
									# 'leg4': 2020-06-04 - 2020-08-12
									# 'leg5': 2020-08-12 - 2020-10-12
									# ("leg%i"%(aux_i['mosaic_leg']))
									# 'user': user defined
daterange_options = {'mwr_range': 	["2019-09-30", "2020-10-02"],
					'mosaic': 		["2019-09-20", "2020-10-12"],
					'leg1':			["2019-09-20", "2019-12-12"],
					'leg2':			["2019-12-13", "2020-02-23"],
					'leg3':			["2020-02-24", "2020-06-03"],
					'leg4':			["2020-06-04", "2020-08-11"],
					'leg5':			["2020-08-12", "2020-10-12"],
					'user':			["2020-01-01", "2020-01-10"]}
aux_i['date_start'] = daterange_options[aux_i['considered_period']][0]
aux_i['date_end'] = daterange_options[aux_i['considered_period']][1]

# Training data months to be limited to certain season / months:
aux_i['training_data_months'] = []	# as list of integers; empty list if all months to be used


# eventually load more potential predictors (i.e., surface pressure data) and continue building input vector:
include_pres_sfc = False
include_CF = False
include_iwv = False
include_t2m = False
include_tb_bl = False
if 'pres_sfc' in aux_i['predictors']: include_pres_sfc = True
if 'CF' in aux_i['predictors']: include_CF = True
if 'iwv' in aux_i['predictors']: include_iwv = True
if 't2m' in aux_i['predictors']: include_t2m = True
if 'tb_bl' in aux_i['predictors']: include_tb_bl = True


# create output path if not existing:
outpath_dir = os.path.dirname(aux_i['path_output'])
if not os.path.exists(outpath_dir):
	os.makedirs(outpath_dir)
outpath_dir = os.path.dirname(aux_i['path_output_info'])
if not os.path.exists(outpath_dir):
	os.makedirs(outpath_dir)


# if desired, import some MOSAiC radiosondes and radiometer data (HATPRO and MiRAC-P) for
# a small test data set:
sonde_dict = dict()
MWR_DS = xr.Dataset()
if aux_i['mosaic_test_subset'] and not aux_i['op_ret']:

	# distinguish between LWP and other retrievals because radiosonde comparisons cannot be applied to LWP:
	if np.any(np.asarray(aux_i['predictand']) == np.array(['iwv', 'q', 'temp'])):
		date_0 = "2020-01-01"	# lower limit of dates
		date_1 = "2020-07-26"	# upper limit of dates (for first import of data)
		test_dates = ["2020-01-01", "2020-03-05", "2020-03-06", "2020-03-07", "2020-03-19", "2020-03-20",
						"2020-04-14", "2020-04-15", "2020-04-16", "2020-04-17", "2020-05-24", "2020-05-25",
						"2020-05-26", "2020-07-10", "2020-07-11", "2020-07-25", "2020-07-26"]
	else:
		date_0 = "2019-11-17"
		date_1 = "2020-09-13"
		test_dates = ["2019-11-17", "2019-12-08", "2019-12-21", "2020-03-05", "2020-03-07", "2020-04-14", 
						"2020-04-15", "2020-04-16", "2020-04-17", "2020-04-20", "2020-06-01", "2020-07-01",
						"2020-07-16", "2020-08-03", "2020-08-06", "2020-08-20", "2020-09-13"]
	test_dates_npdt = np.asarray([np.datetime64(td) for td in test_dates])


	# import radiometer data
	print("Importing HATPRO and MiRAC-P data....")
	hat_dict = import_hatpro_level1b_daterange_pangaea(aux_i['path_tb_obs']['hatpro'], test_dates)
	mir_dict = import_mirac_level1b_daterange_pangaea(aux_i['path_tb_obs']['mirac-p'], test_dates)


	# identify time duplicates (xarray dataset coords not permitted to have any):
	hat_dupno = np.where(~(np.diff(hat_dict['time']) == 0))[0]
	mir_dupno = np.where(~(np.diff(mir_dict['time']) == 0))[0]
	

	# Before merging, create xarray datasets:
	HAT_DS = xr.Dataset(coords={'time': (['time'], hat_dict['time'][hat_dupno].astype('datetime64[s]')),
								'freq': (['freq'], hat_dict['freq_sb'])})
	MIR_DS = xr.Dataset(coords={'time': (['time'], mir_dict['time'][mir_dupno].astype('datetime64[s]')),
								'freq': (['freq'], mir_dict['freq_sb'])})
	HAT_DS['flag'] = xr.DataArray(hat_dict['flag'][hat_dupno], dims=['time'])
	MIR_DS['flag'] = xr.DataArray(mir_dict['flag'][mir_dupno], dims=['time'])
	HAT_DS['tb'] = xr.DataArray(hat_dict['tb'][hat_dupno,:], dims=['time', 'freq'])
	MIR_DS['tb'] = xr.DataArray(mir_dict['tb'][mir_dupno,:], dims=['time', 'freq'])


	# eventually correct TB offsets:
	if aux_i['tb_offset_cor']:
		HAT_DS = mosaic_tb_offset_correction(HAT_DS, aux_i['path_tb_offsets'], 'hatpro')
		MIR_DS = mosaic_tb_offset_correction(MIR_DS, aux_i['path_tb_offsets'], 'mirac-p')

	# eventually load HATPRO temperature measurements:
	if 't2m' in aux_i['predictors']:
		# fill gaps, then reduce to non-time-duplicates and forward it to HAT_DS:
		HAT_T2m = xr.DataArray(hat_dict['ta'], dims=['time'], 
								coords={'time': (['time'], hat_dict['time'].astype('datetime64[s]'))})
		HAT_T2m = HAT_T2m.interpolate_na(dim='time', method='linear')
		HAT_T2m = HAT_T2m.ffill(dim='time')

		# apply smoothing to correct measurement errors: 60 min running mean:
		HAT_T2m_DF = HAT_T2m.to_dataframe(name='t2m')
		HAT_T2m = HAT_T2m_DF.rolling("60min", center=True).mean().to_xarray().t2m
		HAT_T2m += np.random.normal(0.0, 0.05, size=HAT_T2m.shape)
		HAT_DS['t2m'] = xr.DataArray(HAT_T2m.values[hat_dupno], dims=['time'])

		del HAT_T2m_DF, HAT_T2m


	# eventually load boundary layer scan TBs and interpolate to HAT_DS time grid:
	if 'tb_bl' in aux_i['predictors']:
		hat_bl_dict = import_hatpro_level1c_daterange_pangaea(aux_i['path_tb_obs']['hatpro'], test_dates)

		# select only V band frequencies because K band BL scan is not used because water vapour (or cloud
		# liquid) usually has a much stronger horizontal variability than oxygen:
		hat_bl_dict['tb'], hat_bl_dict['freq_sb'] = select_MWR_channels(hat_bl_dict['tb'], hat_bl_dict['freq_sb'], "V")

		# Create xarray dataset and interpolate to HAT_DS grid
		hat_bl_dupno = np.where(~(np.diff(hat_bl_dict['time']) == 0))[0]
		HAT_BL_DS = xr.Dataset(coords={'time_bl': (['time_bl'], hat_bl_dict['time'][hat_bl_dupno].astype('datetime64[s]')),
										'freq_bl': (['freq_bl'], hat_bl_dict['freq_sb']),
										'ang_bl': (['ang_bl'], hat_bl_dict['ele'])})
		HAT_BL_DS['tb_bl'] = xr.DataArray(hat_bl_dict['tb'][hat_bl_dupno,:,:], dims=['time_bl', 'ang_bl', 'freq_bl'])

		# only interpolate to HAT_DS time axis if those TBs are actually used. Else, avoid uncertainties induced by
		# interpolation:
		if ('TBs' in aux_i['predictors']):
			ele_angs = np.array([30.0,19.2,14.4,11.4,8.4,6.6,5.4])		# then, zenith is already included
			HAT_BL_DS = HAT_BL_DS.interp(time_bl=HAT_DS.time)
			HAT_BL_DS['tb_bl'] = HAT_BL_DS['tb_bl'].bfill(dim='time', limit=None)	# fill nans before time_bl[0]
			HAT_BL_DS['tb_bl'] = HAT_BL_DS['tb_bl'].ffill(dim='time', limit=None)	# fill nans after time_bl[-1]
			HAT_BL_DS = HAT_BL_DS.drop(['time_bl'])

		else:
			ele_angs = np.array([90.0,30.0,19.2,14.4,11.4,8.4,6.6,5.4])

		# delete unused (or redundant) angles because we already include them in HAT_DS:
		# In the MWR_PRO HATPRO BL scan temperature profiles, the lower two angles were excluded.
		HAT_BL_DS = HAT_BL_DS.sel(ang_bl=ele_angs)

		del hat_bl_dict


	del hat_dict, mir_dict

	
	# eventually flag bad values and then merge time axes of radiometer data into one dataset (intersection):
	# ok_idx = np.where((HAT_DS.flag == 0.0) | (HAT_DS.flag == 32.0))[0]
	# ok_idx_m = np.where(MIR_DS.flag == 0.0)[0]
	time_isct, t_i_hat, t_i_mir = np.intersect1d(HAT_DS.time.values, MIR_DS.time.values, return_indices=True)
	MWR_DS = xr.Dataset(coords={'time': (['time'], time_isct),
								'freq': (xr.concat([HAT_DS.freq, MIR_DS.freq], dim='freq'))})
	MWR_DS['tb'] = xr.concat([HAT_DS.tb[t_i_hat,:], MIR_DS.tb[t_i_mir,:]], dim='freq')
	MWR_DS['flag_h'] = HAT_DS.flag[t_i_hat]
	MWR_DS['flag_m'] = MIR_DS.flag[t_i_mir]
	if 't2m' in aux_i['predictors']: MWR_DS['t2m'] = HAT_DS.t2m[t_i_hat]

	if 'tb_bl' in aux_i['predictors']:
		# flatten the freq_bl and ang_bl dimensions:
		# tb_bl_r is sorted as follows: [(ele_ang=30,freq=50->58) -> (ele_ang=19.2,freq=50->58) -> ... ->
		# (ele_ang=5.4,freq=50->58)]
		if 'TBs' in aux_i['predictors']:
			HAT_BL_DS['tb_bl_r'] = xr.DataArray(np.reshape(HAT_BL_DS.tb_bl.values, 
												(len(HAT_BL_DS.time), len(HAT_BL_DS.freq_bl)*len(HAT_BL_DS.ang_bl))),
												dims=['time', 'n_bl'])
			MWR_DS['tb_bl'] = HAT_BL_DS.tb_bl_r[t_i_hat,:]

		else:
			# Two options: 1. Create tb_bl input vector as for the MWR_PRO retrieval: all frequencies, zenith 
			# followed by the 4 highest V band frequencies, each with the selected elevation angles.
			# 2. All frequencies and angles should be used. 
			# Uncomment the chosen option!
			# 1. MWR_PRO-like:
			tb_bl_r = HAT_BL_DS.tb_bl.sel(ang_bl=90.0).values
			mwr_pro_freq_bl = np.array([54.94, 56.66, 57.3, 58.0])
			for i_freq_bl, freq_bl in enumerate(mwr_pro_freq_bl):
				tb_bl_r = np.concatenate((tb_bl_r, HAT_BL_DS.tb_bl.sel(freq_bl=freq_bl,ang_bl=HAT_BL_DS.ang_bl[1:])), axis=-1)
			HAT_BL_DS['tb_bl_r'] = xr.DataArray(tb_bl_r, dims=['time_bl', 'n_bl'])

			# 2. all frequencies, all elevation angles: [(ele_ang=30,freq=50->58) -> ... -> (ele_ang=5.4,freq=50->58)]
			# HAT_BL_DS['tb_bl_r'] = xr.DataArray(np.reshape(HAT_BL_DS.tb_bl.values, 
												# (len(HAT_BL_DS.time_bl), len(HAT_BL_DS.freq_bl)*len(HAT_BL_DS.ang_bl))),
												# dims=['time_bl', 'n_bl'])
			MWR_DS['tb_bl'] = HAT_BL_DS.tb_bl_r
		

		HAT_BL_DS.close()
		del HAT_BL_DS, hat_bl_dupno


	# clear memory:
	HAT_DS.close()
	MIR_DS.close()
	del HAT_DS, MIR_DS, hat_dupno, mir_dupno


	# load radiosondes if needed:
	if np.any(np.asarray(aux_i['predictand']) == np.array(['iwv', 'q', 'temp'])):
		sonde_dict = import_radiosonde_daterange(aux_i['path_rs_obs'], date_0, date_1, s_version='level_2',
												with_wind=False, remove_failed=True)
		sonde_dict['launch_time_npdt'] = sonde_dict['launch_time'].astype('datetime64[s]')
		sonde_dict['lt_date'] = sonde_dict['launch_time_npdt'].astype('datetime64[D]')

		# reduce to test dates:
		sonde_idx = np.array([])
		for td_npdt in test_dates_npdt:
			idx_temp = np.where(sonde_dict['lt_date'] == td_npdt)[0]

			if len(idx_temp) > 0:
				sonde_idx = np.concatenate((sonde_idx, idx_temp))
		sonde_idx = sonde_idx.astype(np.int64)

		# loop over keys of sonde_dict to truncate the time dimension:
		time_keys = ['lat', 'lon', 'launch_time', 'launch_time_npdt', 'iwv']
		time_height_keys = ['pres', 'temp', 'rh', 'height', 'q', 'rho_v']
		for sk in sonde_dict.keys():
			if sk in time_keys:
				sonde_dict[sk] = sonde_dict[sk][sonde_idx]
			elif sk in time_height_keys:
				if sk == 'q': # convert to g kg-1
					sonde_dict[sk] = sonde_dict[sk][sonde_idx,:]*1000.
				else:
					sonde_dict[sk] = sonde_dict[sk][sonde_idx,:]


		# reduce radiometer data to times around radiosondes:
		if "TBs" in aux_i['predictors']:
			launch_window = 900		# duration (in sec) added to radiosonde launch time in which MWR data should be averaged

			# find overlap of radiosondes and radiometers:
			mwrson_idx = [np.argwhere((MWR_DS.time.values >= lt) &
							(MWR_DS.time.values < lt+np.timedelta64(launch_window, "s"))).flatten() for lt in sonde_dict['launch_time_npdt']]
			mwrson_idx_concat = np.array([])
			for mwrson in mwrson_idx:
				mwrson_idx_concat = np.concatenate((mwrson_idx_concat, mwrson))
			
			MWR_DS = MWR_DS.isel(time=mwrson_idx_concat.astype(np.int64))

		else:
			launch_window = 1800		# here, the window is rather: launch time +/- 30 minutes

			# find overlap of radiosondes and radiometers:
			mwrson_idx = [np.argwhere((MWR_DS.time_bl.values >= lt-np.timedelta64(launch_window, "s")) &
							(MWR_DS.time_bl.values <= lt+np.timedelta64(launch_window, "s"))).flatten() for lt in sonde_dict['launch_time_npdt']]
			mwrson_idx_concat = np.array([])
			for mwrson in mwrson_idx:
				mwrson_idx_concat = np.concatenate((mwrson_idx_concat, mwrson))

			MWR_DS = MWR_DS.isel(time_bl=mwrson_idx_concat.astype(np.int64))
			MWR_DS = MWR_DS.sel(time=MWR_DS.time_bl, method='nearest')

			# rename time dimension:
			MWR_DS = MWR_DS.drop('time').rename({'time_bl': 'time'})


		del mwrson_idx, mwrson_idx_concat, sonde_dict


	# load HATPRO LWP data if cloud flag is needed as predictor:
	if 'CF' in aux_i['predictors']:
		# import and convert to dataset:
		hatpro_dict = import_hatpro_level2a_daterange_pangaea(aux_i['path_old_ret'], test_dates, which_retrieval='lwp')
		HAT_DS = xr.Dataset(coords={'time': (['time'], hatpro_dict['time'].astype('datetime64[s]').astype('datetime64[ns]'))})
		HAT_DS['lwp'] = xr.DataArray(hatpro_dict['clwvi'], dims=['time'])

		# identify cloudy scenes (similar as in data_tools.py.offset_lwp()):
		hat_cloudy_idx = np.zeros_like(HAT_DS.lwp)
		LWP_DF = HAT_DS['lwp'].to_dataframe(name='LWP')	# PANDAS DF to be used to have rolling window width in time units
		LWP_std_2min = LWP_DF.rolling("2min", center=True, min_periods=30).std()
		LWP_std_max_20min = LWP_std_2min.rolling("20min", center=True).max().to_xarray().LWP

		idx_cloudy = np.where(LWP_std_max_20min >= 0.0015)[0]		# lwp std threshold is in kg m-2 (0.0015 has been used for MOSAiC)
		hat_cloudy_idx[idx_cloudy] = 1.0
		

		# interpolate to MWR_DS time grid:
		MWR_DS['CF'] = xr.DataArray(np.interp(MWR_DS.time.values.astype('datetime64[s]').astype(np.float64), 
									HAT_DS.time.values.astype('datetime64[s]').astype(np.float64), hat_cloudy_idx), dims=['time'])


		# clear memory:
		HAT_DS.close()
		del HAT_DS, LWP_DF, LWP_std_2min, LWP_std_max_20min, hatpro_dict, idx_cloudy, hat_cloudy_idx


	# load retrieved IWV if it is to be used as predictor:
	if 'iwv' in aux_i['predictors']:
		# import synergetic retrieval IWV and bring it on the MWR_DS time axis:
		SYN_DS = import_hatpro_mirac_level2a_daterange_pangaea(aux_i['path_output'] + "l2/", test_dates, 
														which_retrieval='iwv', data_version='v00')
		SYN_DS = SYN_DS.assign_coords(time=SYN_DS.time.astype('datetime64[s]').astype('datetime64[ns]'))
		SYN_DS = SYN_DS.sel(time=MWR_DS.time)

		# also apply some noise?
		SYN_DS['prw'] = SYN_DS.prw + np.random.normal(0.0, 0.25, size=SYN_DS['prw'].shape)
		SYN_DS['prw'][SYN_DS['prw'] < 0.] = 0.
		MWR_DS['iwv'] = SYN_DS.prw

		# clear memory:
		SYN_DS.close()
		del SYN_DS



# 20 random numbers generated with np.random.uniform(0, 1000, 20).astype(np.int32)
if exec_type == '20_runs':
	some_seeds = [773, 994, 815, 853, 939, 695, 472, 206, 159, 307, 
					612, 442, 405, 487, 549, 806, 45, 110, 35, 701]
elif exec_type == 'op_ret':
	if (aux_i['predictand'] == ["iwv"]) and (test_id == '126'): 
		some_seeds = [806]				# alternatively, [487]
	elif (aux_i['predictand'] == ['temp']) and (test_id == '417'):
		some_seeds = [487]
	elif (aux_i['predictand'] == ['temp']) and (test_id == '424'):
		some_seeds = [472]
	elif (aux_i['predictand'] == ['q']) and (test_id  == '472'):
		some_seeds = [442]
	else:
		some_seeds = [773]

# dict which will save information about each test
ret_metrics = ['rmse_tot', 'rmse_bot', 'rmse_mid', 'rmse_top', 'stddev', 'stddev_bot', 
				'stddev_mid', 'stddev_top', 'bias_tot', 'bias_bot', 'bias_mid', 'bias_top']
aux_i_stats = ['test_loss', 'training_loss', 'val_loss_array', 'loss_array', 'batch_size', 'epochs', 
				'elapsed_epochs', 'activation', 'seed', 'learning_rate', 'feature_range']


retrieval_stats_syn = dict()
for ais in aux_i_stats:
	if ais not in ['val_loss_array', 'loss_array']:
		retrieval_stats_syn[ais] = list()
	else:
		retrieval_stats_syn[ais] = np.full((len(some_seeds), aux_i['epochs']), np.nan)
for predictand in aux_i['predictand']:
	retrieval_stats_syn[predictand + "_metrics"] = dict()

	for ret_met in ret_metrics:
		retrieval_stats_syn[predictand + "_metrics"][ret_met] = list()

for k_s, aux_i['seed'] in enumerate(some_seeds):

	# set rng seeds
	np.random.seed(seed=aux_i['seed'])
	tensorflow.random.set_seed(aux_i['seed'])
	tensorflow.keras.utils.set_random_seed(aux_i['seed'])

	# randomly select training and test years
	yrs_idx_rng = np.random.permutation(np.arange(n_yrs))
	yrs_idx_training = sorted(yrs_idx_rng[:n_training])
	yrs_idx_test = sorted(yrs_idx_rng[n_training:])


	if aux_i['site'] in ['era5', 'pol']:
		aux_i['yrs_training'] = np.asarray(aux_i['yrs'])[yrs_idx_training]
		aux_i['yrs_testing'] = np.asarray(aux_i['yrs'])[yrs_idx_test]
	elif aux_i['site'] == 'nya':
		aux_i['yrs_training'] = np.asarray(aux_i['yrs'])
		aux_i['yrs_testing'] = np.asarray(aux_i['yrs'])

	print("Years Training: %s"%(aux_i['yrs_training']))
	print("Years Testing: %s"%(aux_i['yrs_testing']))


	# split training and test data:
	data_files_training = list()
	data_files_test = list()

	if aux_i['site'] == 'pol':
		data_files_training = sorted(glob.glob(aux_i['path_data'] + "MOSAiC_mirac-p_retrieval*.nc"))
		data_files_test = data_files_training
	elif aux_i['site'] == 'nya':
		data_files_training = sorted(glob.glob(aux_i['path_data'] + "rt_nya_vers01*.nc"))
		data_files_test = data_files_training
	elif aux_i['site'] == 'era5':
		data_files_training = sorted(glob.glob(aux_i['path_data'] + "*.nc"))
		data_files_test = data_files_training


	# Define noise strength dictionary for the function add_TB_noise in class radiometers:
	noise_dict = {	'22.24':	cat[test_id]['noise_kv'],
					'23.04':	cat[test_id]['noise_kv'],
					'23.84':	cat[test_id]['noise_kv'],
					'25.44':	cat[test_id]['noise_kv'],
					'26.24':	cat[test_id]['noise_kv'],
					'27.84':	cat[test_id]['noise_kv'],
					'31.40':	cat[test_id]['noise_kv'],
					'51.26':	cat[test_id]['noise_kv'],
					'52.28':	cat[test_id]['noise_kv'],
					'53.86':	cat[test_id]['noise_kv'],
					'54.94':	cat[test_id]['noise_kv'],
					'56.66':	cat[test_id]['noise_kv'],
					'57.30':	cat[test_id]['noise_kv'],
					'58.00':	cat[test_id]['noise_kv'],
					'183.91':	cat[test_id]['noise_g'],
					'184.81':	cat[test_id]['noise_g'],
					'185.81':	cat[test_id]['noise_g'],
					'186.81':	cat[test_id]['noise_g'],
					'188.31':	cat[test_id]['noise_g'],
					'190.81':	cat[test_id]['noise_g'],
					'243.00':	cat[test_id]['noise_243'],
					'340.00':	cat[test_id]['noise_340']}

	# Load radiometer TB data (independent predictor):
	predictor_training = radiometers(data_files_training, instrument=aux_i['predictor_instrument'][aux_i['site']], 
										include_pres_sfc=include_pres_sfc, include_CF=include_CF, include_iwv=include_iwv,
										include_t2m=include_t2m, include_tb_bl=include_tb_bl,
										add_TB_noise=aux_i['add_TB_noise'],
										noise_dict=noise_dict, 
										subset=aux_i['yrs_training'],
										subset_months=aux_i['training_data_months'],
										aligned_1D=aux_i['1D_aligned'],
										return_DS=True)

	predictor_test = radiometers(data_files_test, instrument=aux_i['predictor_instrument'][aux_i['site']], 
										include_pres_sfc=include_pres_sfc, include_CF=include_CF, include_iwv=include_iwv,
										include_t2m=include_t2m, include_tb_bl=include_tb_bl,
										add_TB_noise=aux_i['add_TB_noise'],
										noise_dict=noise_dict,
										subset=aux_i['yrs_testing'],
										subset_months=aux_i['training_data_months'],
										aligned_1D=aux_i['1D_aligned'],
										return_DS=True)


	# Load predictand data: (e.g., ERA5, Ny-Alesund radiosondes or ERA-I)
	if aux_i['site'] == 'pol':
		predictand_training = era_i(data_files_training, subset=aux_i['yrs_training'], subset_months=aux_i['training_data_months'])
		predictand_test = era_i(data_files_test, subset=aux_i['yrs_testing'], subset_months=aux_i['training_data_months'])
	elif aux_i['site'] == 'nya':
		predictand_training = radiosondes(data_files_test, s_version=rs_version)
		predictand_test = radiosondes(data_files_test, s_version=rs_version)
	elif aux_i['site'] == 'era5':
		processed_b = "new_z_grid" in aux_i['path_data']	# True if training data had been processed with training_data_new_height.py
		predictand_training = era5(data_files_training, subset=aux_i['yrs_training'], subset_months=aux_i['training_data_months'],
									return_DS=True, processed=processed_b)
		predictand_test = era5(data_files_test, subset=aux_i['yrs_testing'], subset_months=aux_i['training_data_months'],
									return_DS=True, processed=processed_b)


	# convert some units: first (second) element of list: must be added to the variable (the variable 
	# must be multiplied by) to get to the desired unit.
	# the multiplication is performed after adding the unit_conv_dict[key][0] value.
	unit_conv_dict = {'q': [0.0, 1000.]}	# from kg kg-1 to g kg-1
	for uc_key in unit_conv_dict.keys():
		if uc_key in predictand_training.__dict__.keys():
			predictand_training.__dict__[uc_key] = (predictand_training.__dict__[uc_key] + unit_conv_dict[uc_key][0])*unit_conv_dict[uc_key][1]
			predictand_test.__dict__[uc_key] = (predictand_test.__dict__[uc_key] + unit_conv_dict[uc_key][0])*unit_conv_dict[uc_key][1]


	# Need to convert the predictand and predictor data to a (n_training x n_input) (and respective
	# output): Before changing: time is FIRST dimension; height (or frequency) is LAST dimension
	check_dims_vars = {'temp_sfc': 1, 'height': 2, 'temp': 2, 'rh': 2, 'pres': 2, # int says how many dims it should have after reduction
						'sfc_slf': 1, 'iwv': 1, 'cwp': 1, 'rwp': 1, 'lwp': 1, 'swp': 1, 'iwp': 1, 'CF': 1, 't2m': 1, 'q': 2,
						'lat': 1, 'lon': 1, 'launch_time': 1, 'time': 1, 'freq': 1, 'flag': 1, 'TB': 2, 'TB_BL': 2, 'freq_bl': 1}

	predictand_training = reduce_dimensions(predictand_training, check_dims_vars)
	predictand_test = reduce_dimensions(predictand_test, check_dims_vars)

	predictor_training = reduce_dimensions(predictor_training, check_dims_vars)
	predictor_test = reduce_dimensions(predictor_test, check_dims_vars)


	# confine to sea grid cells only and create a new height grid (if era5):
	if aux_i['site'] == 'era5':
		predictand_training.sfc_mask = predictand_training.sfc_slf < 0.01
		predictand_test.sfc_mask = predictand_test.sfc_slf < 0.01

		predictand_training = apply_sea_mask(predictand_training, predictand_training.sfc_mask, check_dims_vars)
		predictand_test = apply_sea_mask(predictand_test, predictand_test.sfc_mask, check_dims_vars)
		predictor_training = apply_sea_mask(predictor_training, predictand_training.sfc_mask, check_dims_vars)
		predictor_test = apply_sea_mask(predictor_test, predictand_test.sfc_mask, check_dims_vars)


		# new height grid:
		height_vars = ['temp', 'rh', 'pres', 'q']
		new_height = np.array([0, 50, 100, 150, 200, 250, 325, 400, 475, 550, 625, 700, 800, 900,
								1000, 1150, 1300, 1450, 1600, 1800, 2000, 2250, 2500, 2750, 3000, 3250,
								3500, 3750, 4000, 4250, 4500, 4750, 5000, 5500, 6000, 6500, 7000, 7500,
								8000, 8500, 9000, 9500, 10000]).astype(np.float64)
		aux_i['n_height'] = len(new_height)

		# limit height to 0-8000 m for temperature profile retrievals because of missing signal above and to 
		# avoid the tropopause:
		if aux_i['predictand'] == ['temp']:
			idx_0_8000 = np.where(new_height <= 8000.)[0]
			for hgt_key in height_vars:
				predictand_training.__dict__[hgt_key] = predictand_training.__dict__[hgt_key][:,idx_0_8000]
				predictand_test.__dict__[hgt_key] = predictand_test.__dict__[hgt_key][:,idx_0_8000]
			predictand_training.height = predictand_training.height[:,idx_0_8000]
			predictand_test.height = predictand_test.height[:,idx_0_8000]


		if not processed_b:
			# interpolate data on new height grid:
			predictand_training = interp_to_new_hgt_grd(predictand_training, new_height, height_vars, aux_i)
			predictand_test = interp_to_new_hgt_grd(predictand_test, new_height, height_vars, aux_i)
			


	aux_i['n_training'] = len(predictand_training.launch_time)
	aux_i['n_test'] = len(predictand_test.launch_time)
	print(aux_i['n_training'], aux_i['n_test'])


	# Quality control (can be commented out if this part of the script has been performed successfully)
	# The quality control of the ERA-I and ERA5 data has already been performed on the files uploaded to ZENODO.
	# # simple_quality_control(predictand_training, predictand_test, aux_i)

	# further expand the quality control and check if IWV values are okay:
	# In the Ny Alesund radiosonde training data there are some questionable IWV values 
	# (< -80000 kg m^-2). These values need replacement:
	# also need to repair training TBs at the same spot:
	if aux_i['site'] == 'nya':
		iwv_broken_training = np.argwhere(predictand_training.iwv < 0).flatten()
		iwv_broken_test = np.argwhere(predictand_test.iwv < 0).flatten()
		if iwv_broken_training.size > 0:
			predictand_training.iwv[iwv_broken_training] = np.asarray([(predictand_training.iwv[ib-1] + 
															predictand_training.iwv[ib+1]) / 2 for ib in iwv_broken_training])
			predictor_training.TB[iwv_broken_training,:] = np.asarray([(predictor_training.TB[ib-1,:] + 
															predictor_training.TB[ib+1,:]) / 2 for ib in iwv_broken_training])

		if iwv_broken_test.size > 0:
			predictand_test.iwv[iwv_broken_test] = np.asarray([(predictand_test.iwv[ib-1] + 
															predictand_test.iwv[ib+1]) / 2 for ib in iwv_broken_test])
			predictor_test.TB[iwv_broken_test,:] = np.asarray([(predictor_test.TB[ib-1,:] + 
															predictor_test.TB[ib+1,:]) / 2 for ib in iwv_broken_test])



	# Start building input vector for training and test data: Eventually reduce TBs
	# to certain frequencies:
	predictor_training.TB, predictor_training.freq = select_MWR_channels(predictor_training.TB,
																		predictor_training.freq,
																		band=aux_i['predictor_TBs'],
																		return_idx=0)
	predictor_test.TB, predictor_test.freq = select_MWR_channels(predictor_test.TB,
																predictor_test.freq,
																band=aux_i['predictor_TBs'],
																return_idx=0)

	predictor_training = build_input_vector(predictor_training, 'training', aux_i)
	predictor_test = build_input_vector(predictor_test, 'test', aux_i)


	"""
		Define and build Neural Network model: Input_shape depends on whether or not
		DOY and surface pressure are included.
		Loss function: MSE, optimiser: adam (these options (among others) might also be changed during testing and
		build phase)
		Fit model, avoid overfitting by applying Early Stop: callbacks=[EarlyStopping(monitor='val_loss', patience=10)]
	"""

	print(aux_i['activation'], aux_i['feature_range'])

	# Rescale input: Use MinMaxScaler:
	scaler = MinMaxScaler(feature_range=aux_i['feature_range']).fit(predictor_training.input)
	predictor_training.input_scaled = scaler.transform(predictor_training.input)
	predictor_test.input_scaled = scaler.transform(predictor_test.input)

	
	# specify output:
	predictand_training.output = None
	predictand_test.output = None
	aux_i['n_ax1'] = dict()			# will contain information about the length of the 2nd dimension (i.e., height)
	for k, predictand in enumerate(aux_i['predictand']):

		aux_i['n_ax1'][predictand] = 0		# dimension of axis=1 of predictand (needed for concatenation)
		if predictand_training.__dict__[predictand].ndim == 1: 
			aux_i['n_ax1'][predictand] = 1
		elif predictand_training.__dict__[predictand].ndim == 2:
			aux_i['n_ax1'][predictand] = predictand_training.__dict__[predictand].shape[1]
		else:
			raise ValueError(f"Unexpected shape of {predictand} in predictand_training.")

		if k == 0:
			predictand_training.output = np.reshape(predictand_training.__dict__[predictand], (aux_i['n_training'],aux_i['n_ax1'][predictand]))
			predictand_test.output = np.reshape(predictand_test.__dict__[predictand], (aux_i['n_test'],aux_i['n_ax1'][predictand]))
		else:
			predictand_training.output = np.concatenate((predictand_training.output, 
														np.reshape(predictand_training.__dict__[predictand], (aux_i['n_training'],aux_i['n_ax1'][predictand]))),
														axis=1)
			predictand_test.output = np.concatenate((predictand_test.output, 
													np.reshape(predictand_test.__dict__[predictand], (aux_i['n_test'],aux_i['n_ax1'][predictand]))),
													axis=1)



	# Create the NN model and predict stuff from it:
	model, test_loss = NN_retrieval(predictor_training, predictand_training, predictor_test, 
									predictand_test, aux_i, return_test_loss=True)
	n_epochs_elapsed = len(model.history.epoch)		# number of elapsed epochs
	loss_array = np.asarray(model.history.history['loss'])
	val_loss_array = np.asarray(model.history.history['val_loss'])
	training_loss = loss_array[np.argmin(val_loss_array)]

	# make prediction:
	prediction_syn = model.predict(predictor_test.input_scaled)

	if exec_type == '20_runs':

		if (aux_i['mosaic_test_subset'] and (aux_i['seed'] == 773)) | (aux_i['mosaic_test_subset'] and aux_i['test_on_all_rngs']):
			# repeat what has been done to the predictor training data:
			MWR_TB, MWR_freq = select_MWR_channels(MWR_DS.tb.values, MWR_DS.freq.values,
													band=aux_i['predictor_TBs'],
													return_idx=0)

			# build input vector:
			MWR_input = MWR_TB
			if ("TBs" not in aux_i['predictors']) and ('tb_bl' in aux_i['predictors']):
				MWR_input = MWR_DS.tb_bl.values

			elif ("TBs" in aux_i['predictors']) and ('tb_bl' in aux_i['predictors']):
				MWR_input = np.concatenate((MWR_input,
											MWR_DS.tb_bl.values), axis=1)

			if "pres_sfc" in aux_i['predictors']:
				raise RuntimeError("It's not yet coded to have pres_sfc as predictor for MOSAiC observations.")

			if "CF" in aux_i['predictors']:
				MWR_input = np.concatenate((MWR_input, 
											np.reshape(MWR_DS.CF.values, (len(MWR_DS.CF),1))), axis=1)

			if 'iwv' in aux_i['predictors']:
				MWR_input = np.concatenate((MWR_input,
											np.reshape(MWR_DS.iwv.values, (len(MWR_DS.iwv),1))), axis=1)

			if 't2m' in aux_i['predictors']:
				MWR_input = np.concatenate((MWR_input,
											np.reshape(MWR_DS.t2m.values, (len(MWR_DS.t2m),1))), axis=1)

			if ("DOY_1" in aux_i['predictors']) and ("DOY_2" not in aux_i['predictors']):
				MWR_DOY_1, MWR_DOY_2 = compute_DOY(MWR_DS.time.values.astype('datetime64[s]').astype(np.float64), return_dt=False, reshape=True)
				MWR_input = np.concatenate((MWR_input,
											MWR_DOY_1), axis=1)
				
			elif ("DOY_2" in aux_i['predictors']) and ("DOY_1" not in aux_i['predictors']):
				MWR_DOY_1, MWR_DOY_2 = compute_DOY(MWR_DS.time.values.astype('datetime64[s]').astype(np.float64), return_dt=False, reshape=True)
				MWR_input = np.concatenate((MWR_input,
											MWR_DOY_2), axis=1)

			elif ("DOY_1" in aux_i['predictors']) and ("DOY_2" in aux_i['predictors']):
				MWR_DOY_1, MWR_DOY_2 = compute_DOY(MWR_DS.time.values.astype('datetime64[s]').astype(np.float64), return_dt=False, reshape=True)
				MWR_input = np.concatenate((MWR_input,
											MWR_DOY_1,
											MWR_DOY_2), axis=1)


			# Rescale input: Use MinMaxScaler:
			mosaic_input_scaled = scaler.transform(MWR_input)

			# retrieve:
			mosaic_output_pred = model.predict(mosaic_input_scaled)
			MWR_DS = MWR_DS.sel(freq=MWR_freq)


		# evaluate prediction of each predictand:
		error_dict_syn = dict()
		shape_pred_0 = 0
		shape_pred_1 = 0
		for id_i, predictand in enumerate(aux_i['predictand']):
			# inquire shape of current predictand and its position in the output vector or prediction:
			shape_pred_0 = shape_pred_1
			shape_pred_1 = shape_pred_1 + aux_i['n_ax1'][predictand]

			# compute error statistics:
			if predictand in ['iwv', 'lwp']:
				error_dict_syn = compute_error_stats(prediction_syn[:,shape_pred_0:shape_pred_1], 
													predictand_test.output[:,shape_pred_0:shape_pred_1], 
													predictand)

			elif predictand in ['temp', 'q']:
				error_dict_syn = compute_error_stats(prediction_syn[:,shape_pred_0:shape_pred_1], 
													predictand_test.output[:,shape_pred_0:shape_pred_1], 
													predictand, 
													predictand_test.height)

			else:
				raise ValueError("Unknown predictand.")

			# save error statistics in other dictionary:
			for ek in error_dict_syn.keys():
				retrieval_stats_syn[f"{predictand}_metrics"][ek].append(error_dict_syn[ek])


			# visualize evaluation if desired: (scatter plot for 1D, profiles for...profiles)
			# if aux_i['vis_eval']:
			if (aux_i['seed'] == 773) | aux_i['test_on_all_rngs']:
				visualize_evaluation(prediction_syn[:,shape_pred_0:shape_pred_1], 
									predictand_test.output[:,shape_pred_0:shape_pred_1],
									predictand, error_dict_syn, aux_i, predictand_test.height)

				# if predictand in ['iwv', 'q']: # to save test prediction
					# save_prediction_and_reference(prediction_syn[:,shape_pred_0:shape_pred_1], 
													# predictand_test.output[:,shape_pred_0:shape_pred_1],
													# predictand, aux_i, predictand_test.height)
					# pdb.set_trace()


				# save test MOSAiC obs data if desired:
				if aux_i['mosaic_test_subset']:
					if predictand in ['iwv', 'lwp']:
						MWR_DS['output'] = xr.DataArray(mosaic_output_pred[:,shape_pred_0:shape_pred_1].squeeze(), dims=['time'])

						if (predictand == 'lwp') and aux_i['lwp_offset_cor']: # then also apply clear sky LWP offset correction as in MWR_PRO
							lwp_cor = offset_lwp(MWR_DS.time.values.astype('datetime64[s]').astype(np.float64),
													MWR_DS['output'].values, lwp_std_thres=0.0015)
							MWR_DS['output'][:] = lwp_cor

					elif predictand in ['temp', 'q']:
						MWR_DS['output'] = xr.DataArray(mosaic_output_pred[:,shape_pred_0:shape_pred_1], dims=['time', 'height'],
														coords={'height': (['height'], predictand_test.height[0,:])})

					save_mosaic_test_obs(MWR_DS['output'], predictand, aux_i, predictand_test.height)


					# Execute script mosaic_test_obs_comp.py if desired:
					if aux_i['test_on_all_rngs']:
						subprocess.run(["python3", f"{wdir}mosaic_test_obs_comp.py", aux_i['file_descr'], str(aux_i['seed'])])
						print(f"Successfully executed {wdir}mosaic_test_obs_comp.py {aux_i['file_descr']}, {str(aux_i['seed'])} \n") 
						continue


	elif exec_type == 'op_ret':

		# evaluate prediction of each predictand:
		error_dict_syn = dict()
		shape_pred_0 = 0
		shape_pred_1 = 0
		for id_i, predictand in enumerate(aux_i['predictand']):
			# inquire shape of current predictand and its position in the output vector or prediction:
			shape_pred_0 = shape_pred_1
			shape_pred_1 = shape_pred_1 + aux_i['n_ax1'][predictand]

			# compute error statistics:
			if predictand in ['iwv', 'lwp']:
				error_dict_syn = compute_error_stats(prediction_syn[:,shape_pred_0:shape_pred_1], 
													predictand_test.output[:,shape_pred_0:shape_pred_1], 
													predictand)

			elif predictand in ['temp', 'q']:
				error_dict_syn = compute_error_stats(prediction_syn[:,shape_pred_0:shape_pred_1], 
													predictand_test.output[:,shape_pred_0:shape_pred_1], 
													predictand, 
													predictand_test.height)

			else:
				raise ValueError("Unknown predictand.")

			# save error statistics in other dictionary:
			for ek in error_dict_syn.keys():
				retrieval_stats_syn[f"{predictand}_metrics"][ek].append(error_dict_syn[ek])


			# visualize evaluation if desired: (scatter plot for 1D, profiles for...profiles)
			# if aux_i['vis_eval']:
			visualize_evaluation(prediction_syn[:,shape_pred_0:shape_pred_1], 
								predictand_test.output[:,shape_pred_0:shape_pred_1],
								predictand, error_dict_syn, aux_i, predictand_test.height)


		# To export predictions of real MOSAiC observations, Polarstern track data
		# needs to be loaded to include its information:
		ps_track_DS = load_geoinfo_MOSAiC_polarstern(aux_i)


		# import radiometer data and apply the retrieval for the entire MOSAiC period day by day to save memory:
		date_0_dt = dt.datetime.strptime(aux_i['date_start'], "%Y-%m-%d")
		date_1_dt = dt.datetime.strptime(aux_i['date_end'], "%Y-%m-%d")
		n_days = (date_1_dt - date_0_dt).days + 1
		for c_date in (date_0_dt + n*dt.timedelta(days=1) for n in range(n_days)): 
			c_date_str = c_date.strftime("%Y-%m-%d")
			ps_track_DS_c = ps_track_DS.sel(time=c_date_str)		# reduce to current date

			try:
				hat_dict = import_hatpro_level1b_daterange_pangaea(aux_i['path_tb_obs']['hatpro'], c_date_str, c_date_str)
				mir_dict = import_mirac_level1b_daterange_pangaea(aux_i['path_tb_obs']['mirac-p'], c_date_str, c_date_str)
				print(f"Processing HATPRO and MiRAC-P data for {c_date_str}....")
			except OSError:	# then, no files for HATPRO or MiRAC-P were found
				print(f"Skipping {c_date_str}....")
				continue


			# identify time duplicates (xarray dataset coords not permitted to have any):
			hat_dupno = np.where(~(np.diff(hat_dict['time']) == 0))[0]
			mir_dupno = np.where(~(np.diff(mir_dict['time']) == 0))[0]
			

			# Before merging, create xarray datasets:
			HAT_DS = xr.Dataset(coords={'time': (['time'], hat_dict['time'][hat_dupno].astype('datetime64[s]')),
										'freq': (['freq'], hat_dict['freq_sb'])})
			MIR_DS = xr.Dataset(coords={'time': (['time'], mir_dict['time'][mir_dupno].astype('datetime64[s]')),
										'freq': (['freq'], mir_dict['freq_sb'])})
			HAT_DS['flag'] = xr.DataArray(hat_dict['flag'][hat_dupno], dims=['time'])
			MIR_DS['flag'] = xr.DataArray(mir_dict['flag'][mir_dupno], dims=['time'])
			HAT_DS['tb'] = xr.DataArray(hat_dict['tb'][hat_dupno,:], dims=['time', 'freq'])
			MIR_DS['tb'] = xr.DataArray(mir_dict['tb'][mir_dupno,:], dims=['time', 'freq'])


			# eventually correct TB offsets:
			if aux_i['tb_offset_cor']:
				HAT_DS = mosaic_tb_offset_correction(HAT_DS, aux_i['path_tb_offsets'], 'hatpro')
				MIR_DS = mosaic_tb_offset_correction(MIR_DS, aux_i['path_tb_offsets'], 'mirac-p')

			# eventually load HATPRO temperature measurements:
			if 't2m' in aux_i['predictors']:
				# fill gaps, then reduce to non-time-duplicates and forward it to HAT_DS:
				HAT_T2m = xr.DataArray(hat_dict['ta'][hat_dupno], dims=['time'], 
										coords={'time': (['time'], hat_dict['time'][hat_dupno].astype('datetime64[s]'))})
				HAT_T2m = HAT_T2m.interpolate_na(dim='time', method='linear')
				HAT_T2m = HAT_T2m.ffill(dim='time')

				# apply smoothing to correct measurement errors: 60 min running mean, then apply some random noise
				# to avoid too smooth temperature data:
				HAT_T2m_DF = HAT_T2m.to_dataframe(name='t2m')
				HAT_T2m = HAT_T2m_DF.rolling("60min", center=True).mean().to_xarray().t2m
				HAT_T2m += np.random.normal(0.0, 0.05, size=HAT_T2m.shape)
				HAT_DS['t2m'] = xr.DataArray(HAT_T2m.values, dims=['time'])

				del HAT_T2m_DF, HAT_T2m


			# eventually load boundary layer scan TBs and interpolate to HAT_DS time grid:
			if 'tb_bl' in aux_i['predictors']:
				hat_bl_dict = import_hatpro_level1c_daterange_pangaea(aux_i['path_tb_obs']['hatpro'], c_date_str, c_date_str)

				# select only V band frequencies because K band BL scan is not used because water vapour (or cloud
				# liquid) usually has a much stronger horizontal variability than oxygen:
				hat_bl_dict['tb'], hat_bl_dict['freq_sb'] = select_MWR_channels(hat_bl_dict['tb'], hat_bl_dict['freq_sb'], "V")

				# Create xarray dataset and interpolate to HAT_DS grid
				hat_bl_dupno = np.where(~(np.diff(hat_bl_dict['time']) == 0))[0]
				HAT_BL_DS = xr.Dataset(coords={'time_bl': (['time_bl'], hat_bl_dict['time'][hat_bl_dupno].astype('datetime64[s]')),
												'freq_bl': (['freq_bl'], hat_bl_dict['freq_sb']),
												'ang_bl': (['ang_bl'], hat_bl_dict['ele'])})
				HAT_BL_DS['tb_bl'] = xr.DataArray(hat_bl_dict['tb'][hat_bl_dupno,:,:], dims=['time_bl', 'ang_bl', 'freq_bl'])

				# only interpolate to HAT_DS time axis if those TBs are actually used. Else, avoid uncertainties induced by
				# interpolation:
				if ('TBs' in aux_i['predictors']):
					ele_angs = np.array([30.0,19.2,14.4,11.4,8.4,6.6,5.4])		# then, zenith is already included
					HAT_BL_DS = HAT_BL_DS.interp(time_bl=HAT_DS.time)
					HAT_BL_DS['tb_bl'] = HAT_BL_DS['tb_bl'].bfill(dim='time', limit=None)	# fill nans before time_bl[0]
					HAT_BL_DS['tb_bl'] = HAT_BL_DS['tb_bl'].ffill(dim='time', limit=None)	# fill nans after time_bl[-1]
					HAT_BL_DS = HAT_BL_DS.drop(['time_bl'])

				else:
					ele_angs = np.array([90.0,30.0,19.2,14.4,11.4,8.4,6.6,5.4])

				# delete unused (or redundant) angles because we already include them in HAT_DS:
				# In the MWR_PRO HATPRO BL scan temperature profiles, the lower two angles were excluded.
				HAT_BL_DS = HAT_BL_DS.sel(ang_bl=ele_angs)

				del hat_bl_dict


			del hat_dict, mir_dict

			
			# eventually flag bad values and then merge time axes of radiometer data into one dataset (intersection):
			# ok_idx = np.where((HAT_DS.flag == 0.0) | (HAT_DS.flag == 32.0))[0]
			# ok_idx_m = np.where(MIR_DS.flag == 0.0)[0]
			time_isct, t_i_hat, t_i_mir = np.intersect1d(HAT_DS.time.values, MIR_DS.time.values, return_indices=True)
			MWR_DS = xr.Dataset(coords={'time': (['time'], time_isct),
										'freq': (xr.concat([HAT_DS.freq, MIR_DS.freq], dim='freq'))})
			MWR_DS['tb'] = xr.concat([HAT_DS.tb[t_i_hat,:], MIR_DS.tb[t_i_mir,:]], dim='freq')
			MWR_DS['flag_h'] = HAT_DS.flag[t_i_hat]
			MWR_DS['flag_m'] = MIR_DS.flag[t_i_mir]
			if 't2m' in aux_i['predictors']: MWR_DS['t2m'] = HAT_DS.t2m[t_i_hat]

			if 'tb_bl' in aux_i['predictors']:
				# flatten the freq_bl and ang_bl dimensions:
				# tb_bl_r is sorted as follows: [(ele_ang=30,freq=50->58) -> (ele_ang=19.2,freq=50->58) -> ... ->
				# (ele_ang=5.4,freq=50->58)]
				if 'TBs' in aux_i['predictors']:
					HAT_BL_DS['tb_bl_r'] = xr.DataArray(np.reshape(HAT_BL_DS.tb_bl.values, 
														(len(HAT_BL_DS.time), len(HAT_BL_DS.freq_bl)*len(HAT_BL_DS.ang_bl))),
														dims=['time', 'n_bl'])
					MWR_DS['tb_bl'] = HAT_BL_DS.tb_bl_r[t_i_hat,:]

				else:
					# Two options: 1. Create tb_bl input vector as for the MWR_PRO retrieval: all frequencies, zenith 
					# followed by the 4 highest V band frequencies, each with the selected elevation angles.
					# 2. All frequencies and angles should be used. 
					# Uncomment the chosen option!
					# 1. MWR_PRO-like:
					tb_bl_r = HAT_BL_DS.tb_bl.sel(ang_bl=90.0).values
					mwr_pro_freq_bl = np.array([54.94, 56.66, 57.3, 58.0])
					for i_freq_bl, freq_bl in enumerate(mwr_pro_freq_bl):
						tb_bl_r = np.concatenate((tb_bl_r, HAT_BL_DS.tb_bl.sel(freq_bl=freq_bl,ang_bl=HAT_BL_DS.ang_bl[1:])), axis=-1)
					HAT_BL_DS['tb_bl_r'] = xr.DataArray(tb_bl_r, dims=['time_bl', 'n_bl'])

					# 2. all frequencies, all elevation angles: [(ele_ang=30,freq=50->58) -> ... -> (ele_ang=5.4,freq=50->58)]
					# HAT_BL_DS['tb_bl_r'] = xr.DataArray(np.reshape(HAT_BL_DS.tb_bl.values, 
														# (len(HAT_BL_DS.time_bl), len(HAT_BL_DS.freq_bl)*len(HAT_BL_DS.ang_bl))),
														# dims=['time_bl', 'n_bl'])

					MWR_DS['tb_bl'] = HAT_BL_DS.tb_bl_r

					# adapt time dimension:
					if "TBs" not in aux_i['predictors']: # select time_bl only:
						MWR_DS = MWR_DS.sel(time=MWR_DS.time_bl, method='nearest')
						MWR_DS = MWR_DS.drop('time').rename({'time_bl': 'time'})
				

				HAT_BL_DS.close()
				del HAT_BL_DS, hat_bl_dupno

			# clear memory:
			HAT_DS.close()
			MIR_DS.close()
			del HAT_DS, MIR_DS, hat_dupno, mir_dupno


			# Compute a cloud flag if desired as input:
			# load HATPRO LWP data if cloud flag is needed as predictor:
			if "CF" in aux_i['predictors']:
				# import and convert to dataset:
				try:
					hatpro_dict = import_hatpro_level2a_daterange_pangaea(aux_i['path_old_ret'], c_date_str, c_date_str, which_retrieval='lwp')
				except OSError:
					print(f"Skipping {c_date_str}....")
					continue
				HAT_DS = xr.Dataset(coords={'time': (['time'], hatpro_dict['time'].astype('datetime64[s]').astype('datetime64[ns]'))})
				HAT_DS['lwp'] = xr.DataArray(hatpro_dict['clwvi'], dims=['time'])

				# identify cloudy scenes (similar as in data_tools.py.offset_lwp()):
				hat_cloudy_idx = np.zeros_like(HAT_DS.lwp)
				LWP_DF = HAT_DS['lwp'].to_dataframe(name='LWP')	# PANDAS DF to be used to have rolling window width in time units
				LWP_std_2min = LWP_DF.rolling("2min", center=True, min_periods=30).std()
				LWP_std_max_20min = LWP_std_2min.rolling("20min", center=True).max().to_xarray().LWP

				idx_cloudy = np.where(LWP_std_max_20min >= 0.0015)[0]		# lwp std threshold is in kg m-2 (0.0015 has been used for MOSAiC)
				hat_cloudy_idx[idx_cloudy] = 1.0
				

				# interpolate to MWR_DS time grid:
				MWR_DS['CF'] = xr.DataArray(np.interp(MWR_DS.time.values.astype('datetime64[s]').astype(np.float64), 
											HAT_DS.time.values.astype('datetime64[s]').astype(np.float64), hat_cloudy_idx,
											left=1.0, right=1.0), dims=['time'])


				# clear memory:
				HAT_DS.close()
				del HAT_DS, LWP_DF, LWP_std_2min, LWP_std_max_20min, hatpro_dict, idx_cloudy, hat_cloudy_idx


			# load retrieved IWV if it is to be used as predictor:
			if 'iwv' in aux_i['predictors']:
				# import synergetic retrieval IWV and bring it on the MWR_DS time axis:
				try:
					SYN_DS = import_hatpro_mirac_level2a_daterange_pangaea(aux_i['path_output'] + "l2/", c_date_str, c_date_str, 
																which_retrieval='iwv', data_version='v00')
				except OSError:
					print(f"Skipping {c_date_str}....")
					continue

				SYN_DS = SYN_DS.assign_coords(time=SYN_DS.time.astype('datetime64[s]').astype('datetime64[ns]'))
				SYN_DS = SYN_DS.sel(time=MWR_DS.time)

				# also apply some noise:
				SYN_DS['prw'] = SYN_DS.prw + np.random.normal(0.0, 0.25, size=SYN_DS['prw'].shape)
				SYN_DS['prw'][SYN_DS['prw'] < 0.] = 0.
				MWR_DS['iwv'] = SYN_DS.prw

				# clear memory:
				SYN_DS.close()
				del SYN_DS


			# repeat what has been done to the predictor training data:
			MWR_TB, MWR_freq = select_MWR_channels(MWR_DS.tb.values, MWR_DS.freq.values,
													band=aux_i['predictor_TBs'],
													return_idx=0)

			# build input vector:
			MWR_input = MWR_TB

			if ("TBs" not in aux_i['predictors']) and ('tb_bl' in aux_i['predictors']):
				MWR_input = MWR_DS.tb_bl.values

			elif ("TBs" in aux_i['predictors']) and ('tb_bl' in aux_i['predictors']):
				MWR_input = np.concatenate((MWR_input,
											MWR_DS.tb_bl.values), axis=1)

			if "pres_sfc" in aux_i['predictors']:
				raise RuntimeError("It's not yet coded to have pres_sfc as predictor for MOSAiC observations.")

			if "CF" in aux_i['predictors']:
				MWR_input = np.concatenate((MWR_input, 
											np.reshape(MWR_DS.CF.values, (len(MWR_DS.CF),1))), axis=1)

			if "iwv" in aux_i['predictors']:
				MWR_input = np.concatenate((MWR_input, 
											np.reshape(MWR_DS.iwv.values, (len(MWR_DS.iwv),1))), axis=1)

			if "t2m" in aux_i['predictors']:
				MWR_input = np.concatenate((MWR_input, 
											np.reshape(MWR_DS.t2m.values, (len(MWR_DS.t2m),1))), axis=1)

			if ("DOY_1" in aux_i['predictors']) and ("DOY_2" not in aux_i['predictors']):
				MWR_DOY_1, MWR_DOY_2 = compute_DOY(MWR_DS.time.values.astype("datetime64[s]").astype(np.float64), 
													return_dt=False, reshape=True)
				MWR_input = np.concatenate((MWR_input,
											MWR_DOY_1), axis=1)
				
			elif ("DOY_2" in aux_i['predictors']) and ("DOY_1" not in aux_i['predictors']):
				MWR_DOY_1, MWR_DOY_2 = compute_DOY(MWR_DS.time.values.astype("datetime64[s]").astype(np.float64), 
													return_dt=False, reshape=True)
				MWR_input = np.concatenate((MWR_input,
											MWR_DOY_2), axis=1)

			elif ("DOY_1" in aux_i['predictors']) and ("DOY_2" in aux_i['predictors']):
				MWR_DOY_1, MWR_DOY_2 = compute_DOY(MWR_DS.time.values.astype("datetime64[s]").astype(np.float64), 
													return_dt=False, reshape=True)
				MWR_input = np.concatenate((MWR_input,
											MWR_DOY_1,
											MWR_DOY_2), axis=1)


			# Rescale input: Use MinMaxScaler:
			mosaic_input_scaled = scaler.transform(MWR_input)

			# retrieve:
			mosaic_output_pred = model.predict(mosaic_input_scaled)
			MWR_DS = MWR_DS.sel(freq=MWR_freq)

			# separate predictands and save prediction on MOSAiC data:
			shape_pred_0 = 0
			shape_pred_1 = 0
			for id_i, predictand in enumerate(aux_i['predictand']):
				# inquire shape of current predictand and its position in the output vector or prediction:
				shape_pred_0 = shape_pred_1
				shape_pred_1 = shape_pred_1 + aux_i['n_ax1'][predictand]


				# save test MOSAiC obs data if desired:
				if predictand in ['iwv', 'lwp']:
					MWR_DS['output'] = xr.DataArray(mosaic_output_pred[:,shape_pred_0:shape_pred_1].squeeze(), dims=['time'])

				elif predictand in ['temp', 'q']:
					MWR_DS['output'] = xr.DataArray(mosaic_output_pred[:,shape_pred_0:shape_pred_1], dims=['time', 'height'],
													coords={'height': (['height'], predictand_test.height[0,:])})


				# Save to file for each day:
				if aux_i['save_obs_predictions']:
					save_obs_predictions(aux_i['path_output'], MWR_DS, predictand, c_date_str, aux_i, 
											ps_track_DS_c, predictand_test.height)


				# clear memory:
				del MWR_DS, mosaic_output_pred, MWR_TB, MWR_freq, mosaic_input_scaled, MWR_input



	# save other retrieval information (test loss and NN settings):
	retrieval_stats_syn['test_loss'].append(test_loss)		# likely equals np.nanmean((prediction_syn - predictand_test.output)**2)
	retrieval_stats_syn['training_loss'].append(training_loss)
	retrieval_stats_syn['elapsed_epochs'].append(n_epochs_elapsed)	# n elapsed epochs
	retrieval_stats_syn['val_loss_array'][k_s,:n_epochs_elapsed] = val_loss_array
	retrieval_stats_syn['loss_array'][k_s,:n_epochs_elapsed] = loss_array
	for ek in aux_i_stats:
		if ek in ['test_loss', 'elapsed_epochs', 'val_loss_array', 'loss_array', 'training_loss']:
			continue
		else:
			retrieval_stats_syn[ek].append(aux_i[ek])



	# info content:
	if aux_i['get_info_content']:
		i_cont = info_content(predictand_test.output, predictor_test.TB, prediction_syn, ax_samp=0, ax_comp=1,
								perturbation=1.01, perturb_type='multiply', aux_i=aux_i, 
								suppl_data={'lat': predictand_test.lat, 'lon': predictand_test.lon,
											'time': predictand_test.time, 'height': predictand_test.height,
											'rh': predictand_test.rh, 'temp': predictand_test.temp, 
											'pres': predictand_test.pres, 'temp_sfc': predictand_test.temp_sfc,
											'cwp': predictand_test.cwp, 'rwp': predictand_test.rwp,	# LWP == CWP
											'swp': predictand_test.swp, 'iwp': predictand_test.iwp,
											'q': predictand_test.q})


		# create new reference observation and state vector instead of the predictor_test.TB because that was created on
		# another height grid.
		print("Creating new reference observation vector....")
		i_cont.new_obs(False, what_data='samp')

		# Loop through test data set: perturb each state vector component, generate new obs 
		# via simulations, apply retrieval, compute AK:
		n_comp = i_cont.x.shape[i_cont.ax_c]
		for i_s in range(aux_i['n_test']):
			print(f"Computing info content for test case {i_s} of {aux_i['n_test']-1} (perturbation -> new obs vector -> Jacobian matrix -> AK -> DOF)....")
			i_cont.perturb('state', i_s, 'all')
			i_cont.new_obs(True, what_data='comp')
			i_cont.compute_jacobian()
			i_cont.compute_AK_i('matrix')
			i_cont.compute_DOF()
			# i_cont.visualise_AK_i()
		i_cont.visualise_mean_AK(aux_i['path_plots_info'])
		i_cont.save_info_content_data(aux_i['path_output_info'])
		pdb.set_trace()


if exec_type == '20_runs':

	# Save retrieval stats to xarray dataset, then to netcdf:
	nc_output_name = f"NN_syn_ret_retrieval_stat_test_{aux_i['file_descr']}"

	feature_range_0 = np.asarray([fr[0] for fr in retrieval_stats_syn['feature_range']])
	feature_range_1 = np.asarray([fr[1] for fr in retrieval_stats_syn['feature_range']])

	# start forming the data set, inserting retrieval setup information:
	RETRIEVAL_STAT_DS = xr.Dataset({'test_loss':	(['test_id'], np.asarray(retrieval_stats_syn['test_loss']),
													{'description': "Last epoch test data loss, mean square error",
													'units': "SI units"}),
									'training_loss':(['test_id'], np.asarray(retrieval_stats_syn['training_loss']),
													{'description': "Last epoch training data loss, mean square error",
													'units': "SI units"}),
									'val_loss':		(['test_id', 'n_epochs'], retrieval_stats_syn['val_loss_array'],
													{'description': "Test loss for each elapsed epoch, mean square error"}),
									'loss':			(['test_id', 'n_epochs'], retrieval_stats_syn['loss_array'],
													{'description': "Training loss for each elapsed epoch, mean square error"}),
									'batch_size':	(['test_id'], np.asarray(retrieval_stats_syn['batch_size']),
													{'description': "Neural Network training batch size"}),
									'epochs':		(['test_id'], np.asarray(retrieval_stats_syn['epochs']),
													{'description': "Neural Network training epoch number"}),
									'elapsed_epochs': (['test_id'], np.asarray(retrieval_stats_syn['elapsed_epochs']),
													{'description': "Number of epochs elapsed during training"}),
									'activation':	(['test_id'], np.asarray(retrieval_stats_syn['activation']),
													{'description': "Neural Network activation function from input to hidden layer"}),
									'seed':			(['test_id'], np.asarray(retrieval_stats_syn['seed']),
													{'description': "RNG seed for numpy.random.seed and tensorflow.random.set_seed"}),
									'learning_rate':(['test_id'], np.asarray(retrieval_stats_syn['learning_rate']),
													{'description': "Learning rate of NN optimizer"}),
									'feature_range0': (['test_id'], feature_range_0,
													{'description': "Lower end of feature range of tensorflow's MinMaxScaler"}),
									'feature_range1': (['test_id'], feature_range_1,
													{'description': "Upper end of feature range of tensorflow's MinMaxScaler"})},
									coords=			{'test_id': (['test_id'], np.arange(len(retrieval_stats_syn['test_loss'])),
													{'description': "Test number"}),
													'height': (['height'], predictand_test.height[0,:],
													{'description': "Test data height grid",
													'units': "m"})})

	# add the retrieval metrics of test data vs. prediction:
	ret_met_units = {'iwv': "mm", 'lwp': "kg m-2", 'temp': "K", 'q': "g kg-1"}
	ret_met_range = {	'iwv': {'bot': "[0,5) mm", 'mid': "[5,10) mm", 'top': "[10,100) mm"},
						'lwp': {'bot': "[0,0.025) kg m-2", 'mid': "[0.025,0.100) kg m-2", 'top': "[0.100, 1e+06) kg m-2"},
						'temp': {'bot': "[0,1500) m", 'mid': "[1500,5000) m", 'top': "[5000,15000) m"},
						'q': {'bot': "[0,1500) m", 'mid': "[1500,5000) m", 'top': "[5000,15000) m"}}

	for predictand in aux_i['predictand']:

		# description attributes:
		ret_met_descr = {'rmse_tot': f"Test data Root Mean Square Error (RMSE) of target and predicted {predictand}",
						'rmse_bot': f"Like rmse_tot but confined to {predictand} range {ret_met_range[predictand]['bot']}",
						'rmse_mid': f"Like rmse_tot but confined to {predictand} range {ret_met_range[predictand]['mid']}",
						'rmse_top': f"Like rmse_tot but confined to {predictand} range {ret_met_range[predictand]['top']}",
						'bias_tot': f"Bias of test data predicted - target {predictand}",
						'bias_bot': f"Like bias_tot but confined to {predictand} range {ret_met_range[predictand]['bot']}",
						'bias_mid': f"Like bias_tot but confined to {predictand} range {ret_met_range[predictand]['mid']}",
						'bias_top': f"Like bias_tot but confined to {predictand} range {ret_met_range[predictand]['top']}",
						'stddev': f"Test data standard deviation (bias corrected RMSE) of target and predicted {predictand}",
						'stddev_bot': f"Like stddev but confined to {predictand} range {ret_met_range[predictand]['bot']}",
						'stddev_mid': f"Like stddev but confined to {predictand} range {ret_met_range[predictand]['mid']}",
						'stddev_top': f"Like stddev but confined to {predictand} range {ret_met_range[predictand]['top']}"}

		# save retrieval metrics to dataset and forward the variable attributes:
		for ret_met in ret_metrics:
			if predictand in ['temp', 'q'] and ret_met in ['rmse_tot', 'bias_tot', 'stddev']:
				RETRIEVAL_STAT_DS[f"{predictand}_{ret_met}"] = xr.DataArray(np.asarray(retrieval_stats_syn[f"{predictand}_metrics"][ret_met]),
																			dims=['test_id', 'height'])

			else:
				RETRIEVAL_STAT_DS[f"{predictand}_{ret_met}"] = xr.DataArray(np.asarray(retrieval_stats_syn[f"{predictand}_metrics"][ret_met]),
																			dims=['test_id'])
			RETRIEVAL_STAT_DS[f"{predictand}_{ret_met}"].attrs['description'] = ret_met_descr[ret_met]
			RETRIEVAL_STAT_DS[f"{predictand}_{ret_met}"].attrs['units'] = ret_met_units[predictand]
			if "bot" in ret_met:
				RETRIEVAL_STAT_DS[f"{predictand}_{ret_met}"].attrs['range'] = ret_met_range[predictand]['bot']
			elif "mid" in ret_met:
				RETRIEVAL_STAT_DS[f"{predictand}_{ret_met}"].attrs['range'] = ret_met_range[predictand]['mid']
			elif "top" in ret_met:
				RETRIEVAL_STAT_DS[f"{predictand}_{ret_met}"].attrs['range'] = ret_met_range[predictand]['top']


	# Provide some global attributes
	RETRIEVAL_STAT_DS.attrs['test_purpose'] = test_id
	RETRIEVAL_STAT_DS.attrs['author'] = "Andreas Walbroel, a.walbroel@uni-koeln.de"
	RETRIEVAL_STAT_DS.attrs['predictands'] = ""
	for predictand in aux_i['predictand']: RETRIEVAL_STAT_DS.attrs['predictands'] += predictand + ", "
	RETRIEVAL_STAT_DS.attrs['predictands'] = RETRIEVAL_STAT_DS.attrs['predictands'][:-2]

	if aux_i['site'] == 'pol':
		RETRIEVAL_STAT_DS.attrs['training_data'] = "Subset of ERA-Interim 2001-2017, 8 virtual stations north of 84.5 deg N"
		if aux_i['nya_test_data']:
			RETRIEVAL_STAT_DS.attrs['test_data'] = "Ny-Alesund radiosondes 2006-2017"
		else:
			RETRIEVAL_STAT_DS.attrs['test_data'] = "Subset of ERA-Interim 2001-2017, 8 virtual stations north of 84.5 deg N"

	elif aux_i['site'] == 'nya':
		RETRIEVAL_STAT_DS.attrs['training_data'] = "Subset of Ny Alesund radiosondes 2006-2017"
		RETRIEVAL_STAT_DS.attrs['test_data'] = "Subset of Ny Alesund radiosondes 2006-2017"

	elif aux_i['site'] == 'era5':
		RETRIEVAL_STAT_DS.attrs['training_data'] = "ERA5, PAMTRA simulations"
		RETRIEVAL_STAT_DS.attrs['test_data'] = "ERA5, PAMTRA simulations"

	datetime_utc = dt.datetime.utcnow()
	RETRIEVAL_STAT_DS.attrs['datetime_of_creation'] = datetime_utc.strftime("%Y-%m-%d %H:%M:%S")


	# create output path if not existing:
	outpath_dir = os.path.dirname(aux_i['path_output'] + "ret_stat/")
	if not os.path.exists(outpath_dir):
		os.makedirs(outpath_dir)
	RETRIEVAL_STAT_DS.to_netcdf(aux_i['path_output'] + "ret_stat/" + nc_output_name + ".nc", mode='w', format="NETCDF4")
	RETRIEVAL_STAT_DS.close()


print(f"Test purpose: {test_id}")
print("Done....")
datetime_utc = dt.datetime.utcnow()
print(datetime_utc - ssstart)
