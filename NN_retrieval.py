import pdb
import glob
import copy
import datetime as dt
import numpy as np
import matplotlib as mpl
mpl.rcParams.update({'font.family': 'monospace'})

import matplotlib.pyplot as plt
import xarray as xr
import gc
import os
import sys

wdir = os.getcwd() + "/"

from matplotlib.ticker import PercentFormatter

sys.path.insert(0, os.path.dirname(wdir[:-1]) + "/")
from import_data import import_PS_mastertrack, import_mirac_level1b_daterange_pangaea, import_mirac_level1b_daterange
from my_classes import radiosondes, radiometers, era_i, era5
from info_content import info_content
from data_tools import *

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow

from sklearn.model_selection import KFold

ssstart = dt.datetime.utcnow()


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
			if check_dims_vars[dim_var] == 1 and dim_var not in ['freq']:
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

	# If chosen, add surface pressure to input vector:
	if "pres_sfc" in aux_i['predictors']:
		predictor.input = np.concatenate((predictor.input,
											np.reshape(predictor.pres, (aux_i[f'n_{specifier}'],1))),
											axis=1)

	# Compute Day of Year in radians if the sin and cos of it shall also be used in input vector:
	if ("DOY_1" in aux_i['predictors']) and ("DOY_2" in aux_i['predictors']):
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
		stats_dict = compute_retrieval_statistics(x_stuff, y_stuff, compute_stddev=True)

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

	# visualize:
	fs = 30
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
		a1.text(0.99, 0.01, f"N = {ret_stats_temp['N']}\nMean = {1000.0*np.mean(np.concatenate((x_fit, y_fit), axis=0)):.2f}\n" +
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

		y_lim = np.array([0.0, 15000.0])


		# bias profiles:
		ax_bias.plot(BIAS_pred, height, color=c_H, linewidth=1.5)
		ax_bias.plot(np.full_like(height, 0.0), height, color=(0,0,0), linewidth=1.0)


		# std dev profiles:
		ax_std.plot(STD_pred, height, color=c_H, linewidth=1.5)


		# add figure identifier of subplots: a), b), ...
		ax_bias.text(0.05, 0.98, "a)", fontsize=fs_small, fontweight='bold', ha='left', va='top', transform=ax_bias.transAxes)
		ax_std.text(0.05, 0.98, "b)", fontsize=fs_small, fontweight='bold', ha='left', va='top', transform=ax_std.transAxes)

		# legends:

		# axis lims:
		ax_bias.set_ylim(bottom=y_lim[0], top=y_lim[1])
		# ax_bias.set_xlim(left=-4, right=4)
		ax_std.set_ylim(bottom=y_lim[0], top=y_lim[1])
		# ax_std.set_xlim(left=0, right=3.5)

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
		ax_std.set_xlabel("$\sigma_{\mathrm{T}}$ (K)", fontsize=fs)
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
		STD_pred = ret_stats_dict['stddev']*1000.0		# in g kg-1
		BIAS_pred = ret_stats_dict['bias_tot']*1000.0		# in g kg-1


		f1 = plt.figure(figsize=(16,14))
		ax_bias = plt.subplot2grid((1,2), (0,0))			# bias profile
		ax_std = plt.subplot2grid((1,2), (0,1))				# std dev profile

		y_lim = np.array([0.0, height.max()])


		# bias profiles:
		ax_bias.plot(BIAS_pred, height, color=c_H, linewidth=1.5)
		ax_bias.plot(np.full_like(height, 0.0), height, color=(0,0,0), linewidth=1.0)


		# std dev profiles:
		ax_std.plot(STD_pred, height, color=c_H, linewidth=1.5)


		# add figure identifier of subplots: a), b), ...
		ax_bias.text(0.05, 0.98, "a)", fontsize=fs_small, fontweight='bold', ha='left', va='top', transform=ax_bias.transAxes)
		ax_std.text(0.05, 0.98, "b)", fontsize=fs_small, fontweight='bold', ha='left', va='top', transform=ax_std.transAxes)

		# legends:

		# axis lims:
		ax_bias.set_ylim(bottom=y_lim[0], top=y_lim[1])
		# # # # # # # # # # # # ax_bias.set_xlim(left=-0.6, right=0.6)
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
		ax_std.set_xlabel("$\sigma_{\mathrm{q}}$ ($\mathrm{g}\,\mathrm{kg}^{-1}$)", fontsize=fs)
		ax_bias.set_title(f"Bias, {aux_i['file_descr']}", fontsize=fs)
		ax_std.set_title(f"RMSE, {aux_i['file_descr']}", fontsize=fs)


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
		DS['prediction'] = xr.DataArray(prediction, dims=['n_s', 'n_height'], 
									attrs={	'long_name': f"Predicted {predictand_id}", 'units': "SI units"})
		DS['reference'] = xr.DataArray(predictand, dims=['n_s', 'n_height'],
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
	DS.attrs['predictor_TBs'] =aux_i['predictor_TBs']
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
	prediction,
	mwr_dict,
	aux_i):

	"""
	Save the Neural Network prediction to a netCDF file. Variables to be included:
	time, flag, output variable (prediction), standard error (std. dev. (bias corrected!),
	lat, lon, zsl (altitude above mean sea level),

	Parameters:
	-----------
	path_output : str
		Path where output is saved to.
	prediction : array of floats
		Array that contains the predicted output and is to be saved.
	mwr_dict : dict
		Dictionary containing data of the MiRAC-P.
	retrieval_stats_syn : dict
		Dictionary containing information about the errors (RMSE, bias).
	aux_i : dict
		Dictionary containing additional information about the NN.
	"""

	path_output_l1 = path_output + "l1/"
	path_output_l2 = path_output + "l2/"


	# Add geoinfo data: Load it, interpolate it on the MWR time axis, set the right attribute (source of
	# information), 
	# # # ps_track_dict = load_geoinfo_MOSAiC_polarstern()

	# # # # interpolate Polarstern track data on mwr data time axis:
	# # # for ps_key in ['lat', 'lon', 'time']:
		# # # ps_track_dict[ps_key + "_ip"] = np.interp(np.rint(mwr_dict['time']), ps_track_dict['time'], ps_track_dict[ps_key])

	# extract day, month and year from start date:
	date_start = dt.datetime.strptime(aux_i['date_start'], "%Y-%m-%d")
	date_end = dt.datetime.strptime(aux_i['date_end'], "%Y-%m-%d")

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
	l1_version = "i10"
	l2_version = "i10"
	if aux_i['predictand'] == 'iwv':
		output_var = 'prw'
		output_units = "kg m-2"

		prediction_thresh = [0, 100]		# kg m-2
		idx_beyond = np.where((prediction < prediction_thresh[0]) | (prediction > prediction_thresh[1]))[0]
		mwr_dict['flag'][idx_beyond] += 1024

	elif aux_i['predictand'] == 'lwp':
		output_var = 'clwvi'
		output_units = "kg m-2"

		prediction_thresh = [-0.2, 3.0]		# kg m-2
		idx_beyond = np.where((prediction < prediction_thresh[0]) | (prediction > prediction_thresh[1]))[0]
		mwr_dict['flag'][idx_beyond] += 1024

	now_date = date_start
	while now_date <= date_end:

		path_addition = f"{now_date.year:04}/{now_date.month:02}/{now_date.day:02}/"

		# check if path exists:
		path_output_dir = os.path.dirname(path_output_l1 + path_addition)
		if not os.path.exists(path_output_dir):
			os.makedirs(path_output_dir)
		path_output_dir = os.path.dirname(path_output_l2 + path_addition)
		if not os.path.exists(path_output_dir):
			os.makedirs(path_output_dir)

		print(now_date)

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

		# filter time:
		now_date_epoch = datetime_to_epochtime(now_date)
		now_date_epoch_plus = now_date_epoch + 86399		# plus one day minus one second
		time_idx = np.where((mwr_dict['time'] >= now_date_epoch) & (mwr_dict['time'] <= now_date_epoch_plus))[0]

		if len(time_idx) > 0:
			# Save predictions (level 2) to xarray dataset, then to netcdf:
			nc_output_name = f"MOSAiC_uoc_lhumpro-243-340_l2_{output_var}_{l2_version}_{dt.datetime.strftime(now_date, '%Y%m%d%H%M%S')}"

			# create Dataset:
			if aux_i['predictand'] == 'iwv':
				DS = xr.Dataset({'lat':			(['time'], ps_track_dict['lat_ip'][time_idx].astype(np.float32),
												{'units': "degree_north",
												'standard_name': "latitude",
												'long_name': "latitude of the RV Polarstern"}),
								'lon':			(['time'], ps_track_dict['lon_ip'][time_idx].astype(np.float32),
												{'units': "degree_east",
												'standard_name': "longitude",
												'long_name': "longitude of the RV Polarstern"}),
								'zsl':			(['time'], np.full_like(mwr_dict['time'][time_idx], 21.0).astype(np.float32),
												{'units': "m",
												'standard_name': "altitude",
												'long_name': "altitude above mean sea level"}),
								'azi':			(['time'], np.zeros_like(mwr_dict['time'][time_idx]).astype(np.float32),
												{'units': "degree",
												'standard_name': "sensor_azimuth_angle",
												'comment': "0=North, 90=East, 180=South, 270=West"}),
								'ele':			(['time'], np.full_like(mwr_dict['time'][time_idx], 89.97).astype(np.float32),
												{'units': "degree",
												'long_name': "sensor elevation angle"}),
								output_var:		(['time'], prediction.flatten()[time_idx].astype(np.float32),
												{'units': output_units,
												'standard_name': "atmosphere_mass_content_of_water_vapor",
												'comment': ("These values denote the vertically integrated amount of water vapor from the surface to TOA. " +
															"The (bias corrected) standard error of atmosphere mass content of water vapor is " +
															f"0.55 {output_units}. More specifically, the " +
															f"standard error of {output_var} in the ranges [0, 5), [5, 10), [10, 100) {output_units} is " +
															f"0.07, 0.36, 1.11 {output_units}.")}),
						output_var + "_offset": (['time'], np.full_like(mwr_dict['time'][time_idx], 0.0).astype(np.float32),
												{'units': output_units,
												'long_name': "atmosphere_mass_content_of_water_vapor offset correction based on brightness temperature offset",
												'comment': ("This value has been subtracted from the original prw value to account for instrument " +
															"calibration drifts. The information is designated for expert user use.")}),
								'flag':			(['time'], mwr_dict['flag'][time_idx].astype(np.short) - np.median(mwr_dict['flag']).astype(np.short),			# if version < v01: - 16 because MWR_PRO processing not made for MiRAC-P
												{'long_name': "quality control flags",
												'flag_masks': np.array([1,2,4,8,16,32,64,128,256,512,1024], dtype=np.short),
												'flag_meanings': ("visual_inspection_filter_band_1 visual_inspection_filter_band2 visual_inspection_filter_band3 " +
																	"rain_flag sanity_receiver_band1 sanity_receiver_band1 sun_in_beam unused " +
																	"unused tb_threshold_band1 iwv_lwp_threshold"),
												'comment': ("Flags indicate data that the user should only use with care. In cases of doubt, please refer " +
															"to the contact person. A Fillvalue of 0 means that data has not been flagged. " +
															"Bands refer to the measurement ranges (if applicable) of the microwave radiometer; " +
															"i.e band 1: all lhumpro frequencies (170-200, 243, and 340 GHz); tb valid range: " +
															"[  2.70, 330.00] in K; prw valid range: [0.,  100.] in kg m-2; clwvi (not considering " +
															"clwvi offset correction) valid range: [-0.2, 3.0] in k gm-2; ")})},
								coords=			{'time': (['time'], mwr_dict['time'][time_idx].astype(np.float64),
															{'units': "seconds since 1970-01-01 00:00:00 UTC",
															'standard_name': "time"})})

				# adapt fill values:
				# Make sure that _FillValue is not added to certain variables:
				exclude_vars_fill_value = ['time', 'lat', 'lon', 'zsl']
				for kk in exclude_vars_fill_value:
					DS[kk].encoding["_FillValue"] = None

				# add fill values to remaining variables:
				vars_fill_value = ['azi', 'ele', 'prw', 'prw_offset', 'flag']
				for kk in vars_fill_value:
					if kk != 'flag':
						DS[kk].encoding["_FillValue"] = float(-999.)
					else:
						DS[kk].encoding["_FillValue"] = np.array([0]).astype(np.short)[0]

				DS.attrs['Title'] = f"Microwave radiometer retrieved {output_var}"
				DS.attrs['Institution'] = "Institute for Geophysics and Meteorology, University of Cologne, Cologne, Germany"
				DS.attrs['Contact_person'] = "Andreas Walbroel (a.walbroel@uni-koeln.de)"
				DS.attrs['Source'] = "RPG LHUMPRO-243-340 G5 microwave radiometer"
				DS.attrs['Dependencies'] = f"MOSAiC_mirac-p_l1_tb"
				DS.attrs['Conventions'] = "CF-1.6"
				datetime_utc = dt.datetime.utcnow()
				DS.attrs['Processing_date'] = datetime_utc.strftime("%Y-%m-%d %H:%M:%S")
				DS.attrs['Author'] = "Andreas Walbroel (a.walbroel@uni-koeln.de)"
				DS.attrs['Comments'] = ""
				DS.attrs['License'] = "For non-commercial use only."
				DS.attrs['Measurement_site'] = "RV Polarstern"
				DS.attrs['Position_source'] = globat

				DS.attrs['retrieval_type'] = "Neural Network"
				DS.attrs['python_packages'] = (f"python version: 3.8.10, tensorflow: {tensorflow.__version__}, keras: {keras.__version__}, " +
												f"numpy: {np.__version__}, sklearn: {sklearn.__version__}, netCDF4: {nc.__version__}")
				DS.attrs['retrieval_batch_size'] = f"{str(aux_i['batch_size'])}"
				DS.attrs['retrieval_epochs'] = f"{str(aux_i['epochs'])}"
				DS.attrs['retrieval_learning_rate'] = f"{str(aux_i['learning_rate'])}"
				DS.attrs['retrieval_activation_function'] = f"{aux_i['activation']} (from input to hidden layer)"
				DS.attrs['retrieval_feature_range'] = f"feature range of sklearn.preprocessing.MinMaxScaler: {str(aux_i['feature_range'])}"
				DS.attrs['retrieval_rng_seed'] = str(aux_i['seed'])
				DS.attrs['retrieval_hidden_layers_nodes'] = f"1: 32 (kernel_initializer={aux_i['kernel_init']})"
				DS.attrs['retrieval_optimizer'] = "keras.optimizers.Adam"
				DS.attrs['retrieval_callbacks'] = "EarlyStopping(monitor=val_loss, patience=20, restore_best_weights=True)"


			if site == 'pol':
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

			elif site == 'nya':
				DS.attrs['training_data'] = "Ny-Alesund radiosondes 2006-2017"
				tdy_str = ""
				for year_str in aux_i['yrs_training']: tdy_str += f"{str(year_str)}, "
				DS.attrs['training_data_years'] = tdy_str[:-2]

				DS.attrs['test_data'] = "Ny-Alesund radiosondes 2006-2017"
				tdy_str = ""
				for year_str in aux_i['yrs_testing']: tdy_str += f"{str(year_str)}, "
				DS.attrs['test_data_years'] = tdy_str[:-2]

			DS.attrs['n_training_samples'] = aux_i['n_training']
			DS.attrs['n_test_samples'] = aux_i['n_test']


			DS.attrs['training_test_TB_noise_std_dev'] = ("TB_183.31+/-0.6GHz: 0.75, TB_183.31+/-1.5GHz: 0.75, TB_183.31+/-2.5GHz: 0.75, " +
															"TB_183.31+/-3.5GHz: 0.75, TB_183.31+/-5.0GHz: 0.75, " +
															"TB_183.31+/-7.5GHz: 0.75, TB_243.00GHz: 4.2, TB_340.00GHz: 4.5")

			DS.attrs['input_vector'] = ("(TB_183.31+/-0.6GHz, TB_183.31+/-1.5GHz, TB_183.31+/-2.5GHz, TB_183.31+/-3.5GHz, TB_183.31+/-5.0GHz, " +
													"TB_183.31+/-7.5GHz, TB_243.00GHz, TB_340.00GHz")
			if 'pres_sfc' in aux_i['predictors']:
				DS.attrs['input_vector'] = DS.input_vector + ", pres_sfc"

			if ("DOY_1" in aux_i['predictors']) and ("DOY_2" in aux_i['predictors']):
				DS.attrs['input_vector'] = DS.input_vector + ", cos(DayOfYear), sin(DayOfYear)"
			DS.attrs['input_vector'] = DS.input_vector + ")"
			DS.attrs['output_vector'] = f"({output_var})"


			# encode time:
			DS['time'].encoding['units'] = 'seconds since 1970-01-01 00:00:00'
			DS['time'].encoding['dtype'] = 'double'

			DS.to_netcdf(path_output_l2 + path_addition + nc_output_name + ".nc", mode='w', format="NETCDF4")
			DS.close()

		# update date:
		now_date += dt.timedelta(days=1)


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
	model.add(Dense(32, input_dim=input_shape[1], activation=aux_i['activation'], kernel_initializer=aux_i['kernel_init']))

	# space for more layers:
	model.add(Dense(32, activation=aux_i['activation'], kernel_initializer=aux_i['kernel_init']))

	model.add(Dense(output_shape[1], activation='linear'))		# output layer shape must be equal to retrieved variables

	# compile and train the NN model
	model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=aux_i['learning_rate']))
	history = model.fit(predictor_training.input_scaled, predictand_training.output, batch_size=aux_i['batch_size'],
				epochs=aux_i['epochs'], verbose=1,
				validation_data=(predictor_test.input_scaled, predictand_test.output),
				callbacks=[EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)],
				)

	test_loss = history.history['val_loss'][-1]			# test data MSE
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

# sys.argv[1] determines if 20 random numbers shall be cycled through ("20_runs") or whether only one random
# number is to be used ("op_ret")
if len(sys.argv) == 1:
	sys.argv.append('20_runs')
elif sys.argv[1] not in ['20_runs', 'op_ret']:
	raise ValueError("Script must be called with either 'python3 NN_retrieval.py " + '"20_runs"' +
						"' or 'python3 NN_retrieval.py " + '"op_ret"' + "'!")


aux_i = dict()	# dictionary that collects additional information
rs_version = 'mwr_pro'			# radiosonde type: 'mwr_pro' means that the structure is built
								# so that mwr_pro retrieval can read it (for Ny-Alesund radiosondes and unconcatenated ERA-I)
test_purpose = "044" # specify the intention of a test (used for the retrieval statistics output .nc file)
aux_i['file_descr'] = test_purpose.replace(" ", "_").lower()	# file name addition (of some plots and netCDF output)
aux_i['site'] = 'era5'			# options of training and test data: 'nya' for Ny-Alesund radiosondes
								# 'pol': ERA-Interim grid points north of 84.5 deg N
								# 'era5': ERA5 training and test data
aux_i['predictors'] = ["TBs"]		# specify input vector (predictors): options: TBs, DOY_1, DOY_2, pres_sfc
													# TBs: up to all HATPRO and MiRAC-P channels
													# DOY_1: cos(day_of_year)
													# DOY_2: sin(day_of_year)
													# pres_sfc: surface pressure (not recommended)
													# ....more?
aux_i['predictor_TBs'] = "K"				# string to identify which bands of TBs are used as predictors
													# syntax as in data_tools.select_MWR_channels
# NN settings:
aux_i['activation'] = "relu"					# default or best estimate for i.e., iwv: exponential
aux_i['feature_range'] = (0.0,1.0)				# best est. with exponential (-3.0, 1.0)
aux_i['epochs'] = 200
aux_i['batch_size'] = 64
aux_i['learning_rate'] = 0.001			# default: 0.001
aux_i['kernel_init'] = 'glorot_uniform'			# default: 'glorot_uniform'

aux_i['predictor_instrument'] = {	'pol': "syn_mwr_pro",	# argument to load predictor data
									'nya': "synthetic",
									'era5': "era5_pam"}
aux_i['predictand'] = ["q"]	# output variable / predictand: options: 
													# list with elements in ["iwv", "lwp", "q", "temp"]


aux_i['yrs'] = {'pol': ["2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011",
				"2012", "2013", "2014", "2015", "2016", "2017"],
				'nya': ["2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", 
						"2015", "2016", "2017"],
				'era5': ["2019", "2020"]}		# available years of data
aux_i['yrs'] = aux_i['yrs'][aux_i['site']]
n_yrs = len(aux_i['yrs'])
n_training = round(1.0*n_yrs)			# number of training years; default: 0.7																						################### MUST BE SET TO 0.7 AGAIN #######
n_test = n_yrs - n_training

aux_i['add_TB_noise'] = True				# if True, random noise will be added to training and test data. 
											# Remember to define a noise dictionary if True
aux_i['vis_eval'] = True					# if True: visualize retrieval evaluation (test predictand vs prediction) (only true if aux_i['op_ret'] == False)
aux_i['save_figures'] = True				# if True: figures created will be saved to file
aux_i['op_ret'] = False						# if True: some NN output of one spec. random number will be generated
aux_i['save_obs_predictions'] = False	# if True, predictions made from MiRAC-P observations will be saved
										# to a netCDF file (i.e., for op_ret retrieval)

# decide if information content is to be computed:
aux_i['get_info_content'] = False

if sys.argv[1] == 'op_ret':
	aux_i['op_ret'] = True
	aux_i['vis_eval'] = False
	# aux_i['save_obs_predictions'] = False


# paths:
remote = "/net/blanc/" in wdir		# identify if the code is executed on the blanc computer or at home
if remote:
	aux_i['path_output'] = "/net/blanc/awalbroe/Data/synergetic_ret/tests_00/output/"				# path where output is saved to
	aux_i['path_output_info'] = "/net/blanc/awalbroe/Data/synergetic_ret/tests_00/info_content/"	# path where output is saved to
	aux_i['path_output_pred_ref'] = "/net/blanc/awalbroe/Data/synergetic_ret/tests_00/prediction_and_reference/"	# path where output is saved to
	aux_i['path_data'] = {'nya': "/net/blanc/awalbroe/Data/mir_fwd_sim/new_rt_nya/",
				'pol': "/net/blanc/awalbroe/Data/MOSAiC_radiometers/retrieval_training/mirac-p/",
				'era5': "/net/blanc/awalbroe/Data/synergetic_ret/training_data_00/merged/"}		# path of training/test data
	aux_i['path_data'] = aux_i['path_data'][aux_i['site']]
	aux_i['path_tb_obs'] = {'hatpro': "/data/obs/campaigns/mosaic/hatpro/l1/",
							'mirac-p': "/data/obs/campaigns/mosaic/mirac-p/l1/"} # path of published level 1 tb data
	aux_i['path_plots'] = "/net/blanc/awalbroe/Plots/synergetic_ret/tests_00/"
	aux_i['path_plots_info'] = "/net/blanc/awalbroe/Plots/synergetic_ret/info_content/"

else:
	aux_i['path_output'] = "/mnt/f/heavy_data/synergetic_ret/synergetic_ret/tests_00/output/"			# path where output is saved to
	aux_i['path_output_info'] = "/mnt/f/heavy_data/synergetic_ret/synergetic_ret/tests_00/info_content/"	# path where output is saved to
	aux_i['path_output_pred_ref'] = "/mnt/f/heavy_data/synergetic_ret/synergetic_ret/tests_00/prediction_and_reference/"	# path where output is saved to
	aux_i['path_data'] = {'nya': "/mnt/f/heavy_data/synergetic_ret/mir_fwd_sim/new_rt_nya/",
				'pol': "/mnt/f/heavy_data/synergetic_ret/MOSAiC_radiometers/retrieval_training/mirac-p/",
				'era5': "/mnt/f/heavy_data/synergetic_ret/synergetic_ret/training_data_00/merged/"}		# path of training/test data
	aux_i['path_data'] = aux_i['path_data'][aux_i['site']]
	aux_i['path_tb_obs'] = {'hatpro': "/data/obs/campaigns/mosaic/hatpro/l1/",
							'mirac-p': "/data/obs/campaigns/mosaic/mirac-p/l1/"} # path of published level 1 tb data
	aux_i['path_plots'] = "/mnt/f/Studium_NIM/work/Plots/synergetic_ret/tests_00/"
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


# eventually load observed surface pressure data and continue building input vector:
include_pres_sfc = False
if 'pres_sfc' in aux_i['predictors']: include_pres_sfc = True



# dict which will save information about each test
ret_metrics = ['rmse_tot', 'rmse_bot', 'rmse_mid', 'rmse_top', 'stddev', 'stddev_bot', 
				'stddev_mid', 'stddev_top', 'bias_tot', 'bias_bot', 'bias_mid', 'bias_top']
aux_i_stats = ['test_loss', 'batch_size', 'epochs', 'activation', 
				'seed', 'learning_rate', 'feature_range']

retrieval_stats_syn = dict()
for ais in aux_i_stats:
	retrieval_stats_syn[ais] = list()
for predictand in aux_i['predictand']:
	retrieval_stats_syn[predictand + "_metrics"] = dict()

	for ret_met in ret_metrics:
		retrieval_stats_syn[predictand + "_metrics"][ret_met] = list()


# 20 random numbers generated with np.random.uniform(0, 1000, 20).astype(np.int32)
if sys.argv[1] == '20_runs':
	some_seeds = [773, 994, 815, 853, 939, 695, 472, 206, 159, 307, 
					612, 442, 405, 487, 549, 806, 45, 110, 35, 701]
	# some_seeds = [773, 994, 815]#, 853, 939, 695, 472, 206, 159, 307]
elif sys.argv[1] == 'op_ret':
	some_seeds = [612]
for aux_i['seed'] in some_seeds:

	# set rng seeds
	np.random.seed(seed=aux_i['seed'])
	tensorflow.random.set_seed(aux_i['seed'])
	# # tensorflow.keras.utils.set_random_seed(aux_i['seed'])

	# randomly select training and test years
	yrs_idx_rng = np.random.permutation(np.arange(n_yrs))
	yrs_idx_training = sorted(yrs_idx_rng[:n_training])
	# yrs_idx_test = sorted(yrs_idx_rng[n_training:])					########	######## LATER: USE SPLIT BETWEEN TRAINING/TEST DATA AGAIN
	yrs_idx_test = yrs_idx_training


	if aux_i['site'] == 'pol':
		aux_i['yrs_training'] = np.asarray(aux_i['yrs'])[yrs_idx_training]
		aux_i['yrs_testing'] = np.asarray(aux_i['yrs'])[yrs_idx_test]
	elif aux_i['site'] == 'nya':
		aux_i['yrs_training'] = np.asarray(aux_i['yrs'])
		aux_i['yrs_testing'] = np.asarray(aux_i['yrs'])
	elif aux_i['site'] == 'era5':
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
	noise_dict = {	'22.24':	0.5,
					'23.04':	0.5,
					'23.84':	0.5,
					'25.44':	0.5,
					'26.24':	0.5,
					'27.84':	0.5,
					'31.40':	0.5,
					'50.30':	0.5,
					'51.76':	0.5,
					'52.80':	0.5,
					'53.75':	0.5,
					'54.94':	0.5,
					'56.66':	0.5,
					'58.00':	0.5,
					'183.91':	0.75,
					'184.81':	0.75,
					'185.81':	0.75,
					'186.81':	0.75,
					'188.31':	0.75,
					'190.81':	0.75,
					'243.00':	4.20,
					'340.00':	4.50}

	# Load radiometer TB data (independent predictor):
	predictor_training = radiometers(data_files_training, instrument=aux_i['predictor_instrument'][aux_i['site']], 
										include_pres_sfc=include_pres_sfc, 
										add_TB_noise=aux_i['add_TB_noise'],
										noise_dict=noise_dict, 
										subset=aux_i['yrs_training'],
										subset_months=aux_i['training_data_months'],
										return_DS=True)

	predictor_test = radiometers(data_files_test, instrument=aux_i['predictor_instrument'][aux_i['site']], 
										include_pres_sfc=include_pres_sfc,
										add_TB_noise=aux_i['add_TB_noise'],
										noise_dict=noise_dict,
										subset=aux_i['yrs_testing'],
										subset_months=aux_i['training_data_months'],
										return_DS=True)


	# Load predictand data: (e.g., ERA5, Ny-Alesund radiosondes or ERA-I)
	if aux_i['site'] == 'pol':
		predictand_training = era_i(data_files_training, subset=aux_i['yrs_training'], subset_months=aux_i['training_data_months'])
		predictand_test = era_i(data_files_test, subset=aux_i['yrs_testing'], subset_months=aux_i['training_data_months'])
	elif aux_i['site'] == 'nya':
		predictand_training = radiosondes(data_files_test, s_version=rs_version)
		predictand_test = radiosondes(data_files_test, s_version=rs_version)
	elif aux_i['site'] == 'era5':
		predictand_training = era5(data_files_training, subset=aux_i['yrs_training'], subset_months=aux_i['training_data_months'],
									return_DS=True)
		predictand_test = era5(data_files_test, subset=aux_i['yrs_testing'], subset_months=aux_i['training_data_months'],
									return_DS=True)

	# Need to convert the predictand and predictor data to a (n_training x n_input) (and respective
	# output): Before changing: time is FIRST dimension; height is LAST dimension
	check_dims_vars = {'temp_sfc': 1, 'height': 2, 'temp': 2, 'rh': 2, 'pres': 2, # int says how many dims it should have after reduction
						'sfc_slf': 1, 'iwv': 1, 'cwp': 1, 'rwp': 1, 'lwp': 1, 'swp': 1, 'iwp': 1, 'q': 2,
						'lat': 1, 'lon': 1, 'launch_time': 1, 'time': 1, 'freq': 1, 'flag': 1, 'TB': 2}

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
		# new_height0 = np.arange(0.0, 1000.0, 25.0)
		# new_height1 = np.arange(1000.0, 2000.0, 50.0)
		# new_height2 = np.arange(2000.0, 5000.0, 100.0)
		# new_height3 = np.arange(5000.0, 10000.0, 250.0)
		# new_height4 = np.arange(10000.0, 15000.0001, 500.0)
		# # # new_height0 = np.arange(0.0, 2000.0, 200.0)
		# # # new_height1 = np.arange(2000.0, 10000.001, 500.0)
		new_height0 = np.arange(0.0, 2000.0, 250.0)
		new_height1 = np.arange(2000.0, 10000.001, 1000.0)
		new_height = np.concatenate((new_height0, new_height1))
		aux_i['n_height'] = len(new_height)

		# interpolate data on new height grid:
		height_vars = ['temp', 'rh', 'pres', 'q']
		predictand_training = interp_to_new_hgt_grd(predictand_training, new_height, height_vars, aux_i)
		predictand_test = interp_to_new_hgt_grd(predictand_test, new_height, height_vars, aux_i)


	aux_i['n_training'] = len(predictand_training.launch_time)
	aux_i['n_test'] = len(predictand_test.launch_time)
	print(aux_i['n_training'], aux_i['n_test'])


	# Quality control (can be commented out if this part of the script has been performed successfully)
	# The quality control of the ERA-I data has already been performed on the files uploaded to ZENODO.
	simple_quality_control(predictand_training, predictand_test, aux_i)

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
	if sys.argv[1] == '20_runs':
		model, test_loss = NN_retrieval(predictor_training, predictand_training, predictor_test, 
										predictand_test, aux_i, return_test_loss=True)

		# make prediction:
		prediction_syn = model.predict(predictor_test.input_scaled)


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
			if aux_i['seed'] == 773:
				visualize_evaluation(prediction_syn[:,shape_pred_0:shape_pred_1], 
									predictand_test.output[:,shape_pred_0:shape_pred_1],
									predictand, error_dict_syn, aux_i, predictand_test.height)

				# if predictand in ['iwv', 'q']:
					# save_prediction_and_reference(prediction_syn[:,shape_pred_0:shape_pred_1], 
													# predictand_test.output[:,shape_pred_0:shape_pred_1],
													# predictand, aux_i, predictand_test.height)
					# pdb.set_trace()


		# save other retrieval information (test loss and NN settings):
		retrieval_stats_syn['test_loss'].append(test_loss)		# likely equals np.nanmean((prediction_syn - predictand_test.output)**2)
		for ek in aux_i_stats:
			if ek == 'test_loss':
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


if sys.argv[1] == 'op_ret':

	# Predict actual observations: ###################### up to date??
	prediction_obs = model.predict(mwr_dict['input_scaled'])
	if aux_i['save_obs_predictions']:
		save_obs_predictions(aux_i['path_output'], prediction_obs, mwr_dict, aux_i)

elif sys.argv[1] == '20_runs':

	# Save retrieval stats to xarray dataset, then to netcdf:
	nc_output_name = f"NN_syn_ret_retrieval_stat_test_{aux_i['file_descr']}"

	feature_range_0 = np.asarray([fr[0] for fr in retrieval_stats_syn['feature_range']])
	feature_range_1 = np.asarray([fr[1] for fr in retrieval_stats_syn['feature_range']])

	# start forming the data set, inserting retrieval setup information:
	RETRIEVAL_STAT_DS = xr.Dataset({'test_loss':	(['test_id'], np.asarray(retrieval_stats_syn['test_loss']),
													{'description': "Test data loss, mean square error",
													'units': "SI units"}),
									'batch_size':	(['test_id'], np.asarray(retrieval_stats_syn['batch_size']),
													{'description': "Neural Network training batch size"}),
									'epochs':		(['test_id'], np.asarray(retrieval_stats_syn['epochs']),
													{'description': "Neural Network training epoch number"}),
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
	ret_met_units = {'iwv': "mm", 'lwp': "kg m-2", 'temp': "K", 'q': "kg kg-1"}
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
	RETRIEVAL_STAT_DS.attrs['test_purpose'] = test_purpose
	RETRIEVAL_STAT_DS.attrs['author'] = "Andreas Walbroel, a.walbroel@uni-koeln.de"
	RETRIEVAL_STAT_DS.attrs['predictands'] = ""
	for predictand in aux_i['predictand']: RETRIEVAL_STAT_DS.attrs['predictands'] += predictand + ", "
	RETRIEVAL_STAT_DS.attrs['predictands'] = RETRIEVAL_STAT_DS.attrs['predictands'][:-2]

	if aux_i['site'] == 'pol':
		RETRIEVAL_STAT_DS.attrs['training_data'] = "Subset of ERA-Interim 2001-2017, 8 virtual stations north of 84.5 deg N"
		if aux_i['nya_test_data']:
			RETRIEVAL_STAT_DS.attrs['test_data'] = "Ny Alesund radiosondes 2006-2017"
		else:
			RETRIEVAL_STAT_DS.attrs['test_data'] = "Subset of ERA-Interim 2001-2017, 8 virtual stations north of 84.5 deg N"

	elif aux_i['site'] == 'nya':
		RETRIEVAL_STAT_DS.attrs['training_data'] = "Subset of Ny Alesund radiosondes 2006-2017"
		RETRIEVAL_STAT_DS.attrs['test_data'] = "Subset of Ny Alesund radiosondes 2006-2017"

	elif aux_i['site'] == 'era5':
		RETRIEVAL_STAT_DS.attrs['training_data'] = "ERA5, PAMTRA simulations performed by Mario Mech"
		RETRIEVAL_STAT_DS.attrs['test_data'] = "ERA5, PAMTRA simulations performed by Mario Mech"

	datetime_utc = dt.datetime.utcnow()
	RETRIEVAL_STAT_DS.attrs['datetime_of_creation'] = datetime_utc.strftime("%Y-%m-%d %H:%M:%S")


	RETRIEVAL_STAT_DS.to_netcdf(aux_i['path_output'] + "ret_stat/" + nc_output_name + ".nc", mode='w', format="NETCDF4")
	RETRIEVAL_STAT_DS.close()


print(f"Test purpose: {test_purpose}")
print("Done....")
datetime_utc = dt.datetime.utcnow()
print(datetime_utc - ssstart)
