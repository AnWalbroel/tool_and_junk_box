import numpy as np
import xarray as xr
import pdb
import sys
import datetime as dt

import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/mnt/d/Studium_NIM/work/Codes/MOSAiC/")
from data_tools import (regression, select_MWR_channels, build_K_reg, compute_retrieval_statistics,
						compute_RMSE_profile)


def add_TB_noise(TB, freq, noise_dict):

	"""
	Adds random (un)correlated noise to the brightness temperatures, which must be
	in time (or samples) x freq shape.

	Parameters:
	-----------
	TB : array of floats
		2D array (samples x freq) of brightness temperatures in K. Noise will be added
		to them.
	freq : array of floats
		1D array containing frequencies (in GHz). 
	noise_dict : dict
		Dictionary that has the frequencies (with .2f floating point precision) as keys
		and the noise strength (in K) as value. Example: '190.71': 3.0
	"""

	n_time = TB.shape[0]

	# Loop through frequencies. Find which frequency is currently addressed and
	# create respective noise:
	for freq_sel in noise_dict.keys():
		frq_idx = np.where(np.isclose(freq, float(freq_sel), atol=0.01))[0]
		if len(frq_idx) > 0:
			frq_idx = frq_idx[0]
			TB[:,frq_idx] = TB[:,frq_idx] + np.random.normal(0.0, noise_dict[freq_sel], size=n_time)

	return TB


def add_TB_noise_BL(TB, freq, noise_dict):

	"""
	Adds random (un)correlated noise to the brightness temperatures, which must be
	in time (or samples) x angles x freq shape.

	Parameters:
	-----------
	TB : array of floats
		3D array (samples x angles x freq) of brightness temperatures in K. Noise will be added
		to them.
	freq : array of floats
		1D array containing frequencies (in GHz). 
	noise_dict : dict
		Dictionary that has the frequencies (with .2f floating point precision) as keys
		and the noise strength (in K) as value. Example: '190.71': 3.0
	"""

	tb_shape = TB.shape[:-1]

	# Loop through frequencies. Find which frequency is currently addressed and
	# create respective noise:
	for freq_sel in noise_dict.keys():
		frq_idx = np.where(np.isclose(freq, float(freq_sel), atol=0.01))[0]
		if len(frq_idx) > 0:
			frq_idx = frq_idx[0]
			TB[:,:,frq_idx] = TB[:,:,frq_idx] + np.random.normal(0.0, noise_dict[freq_sel], size=tb_shape)

	return TB


"""
	This script loads training data for a linear/quadratic regression of
	meteorological variables out of HATPRO brightness temperatures. The
	training coefficients will be saved to a file and can be directly used
	for retrievals.
	- load training data
	- train retrieval: generate retrieval coeffs with least squares approach
	- test retrieval (error statistics)
	- save trained coeffs
	- apply to real obs
"""


# paths:
path_data = {'training': "/mnt/d/Studium_NIM/work/Data/MOSAiC_radiometers/retrieval_training/",
			'obs': "",
			'output': "/mnt/d/Studium_NIM/work/Data/HATPRO_regression/"}


# settings: for plotting, saving data, for the regression
settings_dict = {	'ret_vars': ['prw', 'clwvi', 'ta', 'hua'],
					'zenith': False}		# if True: only zenith TBs


# load retrieval training data:
DS_ret = xr.open_dataset(path_data['training'] + "MOSAiC_hatpro_retrieval_nya_v00.nc")
DS_ret = DS_ret.isel(n_freq=np.arange(len(DS_ret.freq_sb)-1))	# remove the last frequency


# remove unwanted dimension (ele angles):
if settings_dict['zenith']:
	DS_ret['tb'] = DS_ret.tb[:,0,:]


	# add aritificial TB noise to training data:
	noise_dict = dict()
	for ff in DS_ret.freq_sb.values:
		noise_dict[f"{ff:.2f}"] = 0.5
	DS_ret['tb'] = add_TB_noise(DS_ret.tb, DS_ret.freq_sb, noise_dict)


	# take out outliers (quality control of test and training data):
	qc_idx = {'prw': np.where((DS_ret.prw.values < 100.0) & (DS_ret.prw.values > 0.0))[0],
						'clwvi': np.where((DS_ret.clwvi.values < 3.0) & (DS_ret.clwvi.values >= -0.2))[0],
						'ta': np.where(np.all(DS_ret.ta.values < 330.0, axis=1) & np.all(DS_ret.ta.values > 180.0, axis=1))[0],
						'hua': np.where(np.all(DS_ret.hua.values < 30.0, axis=1) & np.all(DS_ret.hua.values > -0.5, axis=1))[0]}


	# train retrieval: generate retrieval coeffs:
	K_band_idx = select_MWR_channels(DS_ret.tb.values, DS_ret.freq_sb.values, 'K', return_idx=2)
	V_band_idx = select_MWR_channels(DS_ret.tb.values, DS_ret.freq_sb.values, 'V', return_idx=2)
	n_height = len(DS_ret.height.values)
	n_obs_ta = len(V_band_idx)
	n_obs_hua = len(K_band_idx)
	n_samples = len(DS_ret.time.values)

	coeffs = dict()			# coeffs will be saved to this dict
	coeffs['prw'] = regression(DS_ret.prw.values[qc_idx['prw']], DS_ret.tb[qc_idx['prw'],K_band_idx].values, order=2)
	coeffs['clwvi'] = regression(DS_ret.clwvi.values[qc_idx['clwvi']], DS_ret.tb[qc_idx['clwvi'],K_band_idx].values, order=2)

	# for profiles, each altitude must be dealt with separately:
	# coefficient shape: (2*n_obs + 1)*n_height; (2*n_obs+1,n_height)
	coeffs['ta'] = np.full((2*n_obs_ta+1,n_height), np.nan)
	coeffs['hua'] = np.full((2*n_obs_hua+1,n_height), np.nan)
	for ii in range(n_height):
		coeffs['ta'][:,ii] = regression(DS_ret.ta[qc_idx['ta'],ii].values, DS_ret.tb[qc_idx['ta'],V_band_idx].values, order=2)
		coeffs['hua'][:,ii] = regression(DS_ret.hua[qc_idx['hua'],ii].values, DS_ret.tb[qc_idx['hua'],K_band_idx].values, order=2)


	# after completed training, test retrievals with test data (== training data here):
	# error statistics will be saved to stat_dict
	stat_dict = dict()
	test_data_modelled = dict()
	for key in settings_dict['ret_vars']:
		if key in ['prw', 'clwvi']:
			K_reg = build_K_reg(DS_ret.tb[qc_idx[key],K_band_idx].values, order=2)
			test_data_modelled[key] = K_reg.dot(coeffs[key])

		elif key == 'ta':
			K_reg = build_K_reg(DS_ret.tb[qc_idx[key],V_band_idx].values, order=2)
			test_data_modelled[key] = np.zeros((len(qc_idx[key]), n_height))
			for ii in range(n_height):
				test_data_modelled[key][:,ii] = K_reg.dot(coeffs[key][:,ii])

		else:
			K_reg = build_K_reg(DS_ret.tb[qc_idx[key],K_band_idx].values, order=2)
			test_data_modelled[key] = np.zeros((len(qc_idx[key]), n_height))
			for ii in range(n_height):
				test_data_modelled[key][:,ii] = K_reg.dot(coeffs[key][:,ii])


	# retrieval statistics:
	stat_dict['prw'] = compute_retrieval_statistics(DS_ret.prw.values[qc_idx['prw']], test_data_modelled['prw'],
													compute_stddev=True)
	stat_dict['clwvi'] = compute_retrieval_statistics(DS_ret.clwvi.values[qc_idx['clwvi']], test_data_modelled['clwvi'],
													compute_stddev=True)
	stat_dict['ta'] = {'RMSE': compute_RMSE_profile(test_data_modelled['ta'], DS_ret.ta.values[qc_idx['ta'],:], which_axis=0),
						'bias': np.nanmean(test_data_modelled['ta'] - DS_ret.ta.values[qc_idx['ta'],:], axis=0)}
	stat_dict['hua'] = {'RMSE': compute_RMSE_profile(test_data_modelled['hua'], DS_ret.hua.values[qc_idx['hua'],:], which_axis=0),
						'bias': np.nanmean(test_data_modelled['hua'] - DS_ret.hua.values[qc_idx['hua'],:], axis=0)}
	test_data_modelled['ta_biascorr'] = test_data_modelled['ta'] - stat_dict['ta']['bias']
	test_data_modelled['hua_biascorr'] = test_data_modelled['hua'] - stat_dict['hua']['bias']
	stat_dict['ta']['stddev'] = compute_RMSE_profile(test_data_modelled['ta_biascorr'], DS_ret.ta.values[qc_idx['ta'],:],
													which_axis=0)
	stat_dict['hua']['stddev'] = compute_RMSE_profile(test_data_modelled['hua_biascorr'], DS_ret.hua.values[qc_idx['hua'],:],
													which_axis=0)


	# save retrieval coeffs and error statistics into netCDF:
	RET_DS = xr.Dataset({'c_prw':			(['obs_prw'], coeffs['prw'].astype(np.float64),
											{'standard_name': 'retrieval coefficients for prw (IWV)',
											'comment': 'quadratic regression',
											'obs': DS_ret.freq_sb.values[K_band_idx]}),
						'c_clwvi':			(['obs_clwvi'], coeffs['clwvi'].astype(np.float64),
											{'standard_name': 'retrieval coefficients for clwvi (LWP)',
											'comment': 'quadratic regression',
											'obs': DS_ret.freq_sb.values[K_band_idx]}),
						'c_ta':				(['obs_ta', 'height'], coeffs['ta'].astype(np.float64),
											{'standard_name': 'retrieval coefficients for zenith temperature profile',
											'comment': 'quadratic regression',
											'obs': DS_ret.freq_sb.values[V_band_idx]}),
						'c_hua':			(['obs_hua', 'height'], coeffs['hua'].astype(np.float64),
											{'standard_name': 'retrieval coefficients for zenith humidity profile',
											'comment': 'quadratic regression',
											'obs': DS_ret.freq_sb.values[K_band_idx]}),
						'std_prw':			([], stat_dict['prw']['stddev'],
											{'standard_name': 'standard deviation of IWV',
											'units': 'kg m-2',
											'comment': 'of test (== training) data'}),
						'bias_prw':			([], stat_dict['prw']['bias'],
											{'standard_name': 'bias of IWV',
											'units': 'kg m-2',
											'comment': 'np.nanmean(modelled - obs)'}),
						'rmse_prw':			([], stat_dict['prw']['rmse'],
											{'standard_name': 'root mean squared error of IWV',
											'units': 'kg m-2',
											'comment': 'of test (== training) data'}),
						'std_clwvi':		([], stat_dict['clwvi']['stddev'],
											{'standard_name': 'standard deviation of LWP',
											'units': 'kg m-2',
											'comment': 'of test (== training) data'}),
						'bias_clwvi':		([], stat_dict['clwvi']['bias'],
											{'standard_name': 'bias of LWP',
											'units': 'kg m-2',
											'comment': 'np.nanmean(modelled - obs)'}),
						'rmse_clwvi':		([], stat_dict['clwvi']['rmse'],
											{'standard_name': 'root mean squared error of LWP',
											'units': 'kg m-2',
											'comment': 'of test (== training) data'}),
						'std_ta':			(['height'], stat_dict['ta']['stddev'],
											{'standard_name': 'standard deviation of temperature profile',
											'units': 'K',
											'comment': 'of test (== training) data; for each height'}),
						'bias_ta':			(['height'], stat_dict['ta']['bias'],
											{'standard_name': 'bias of temperature profile',
											'units': 'K',
											'comment': 'of test (== training) data; for each height'}),
						'rmse_ta':			(['height'], stat_dict['ta']['RMSE'],
											{'standard_name': 'root mean squared error of temperature profile',
											'units': 'K',
											'comment': 'of test (== training) data; for each height'}),
						'std_hua':			(['height'], stat_dict['hua']['stddev'],
											{'standard_name': 'standard deviation of humidity profile',
											'units': 'kg m-3',
											'comment': 'of test (== training) data; for each height'}),
						'bias_hua':			(['height'], stat_dict['hua']['bias'],
											{'standard_name': 'bias of humidity profile',
											'units': 'kg m-3',
											'comment': 'of test (== training) data; for each height'}),
						'rmse_hua':			(['height'], stat_dict['hua']['RMSE'],
											{'standard_name': 'root mean squared error of humidity profile',
											'units': 'kg m-3',
											'comment': 'of test (== training) data; for each height'})},
						coords=	{'height':	DS_ret.height})


	# set attributes:
	RET_DS.attrs['Title'] = "Regression coefficients and error statistics for HATPRO"
	RET_DS.attrs['Institution'] = "Institute for Geophysics and Meteorology, University of Cologne, Cologne, Germany"
	RET_DS.attrs['Contact_person'] = "Andreas Walbroel (a.walbroel@uni-koeln.de)"
	RET_DS.attrs['Processing_date'] = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
	RET_DS.attrs['Author'] = "Andreas Walbroel (a.walbroel@uni-koeln.de)"
	RET_DS.attrs['License'] = "For non-commercial use only."
	RET_DS.attrs['Measurement_site'] = "RV Polarstern"

	nc_output_name = "MOSAiC_HATPRO_regression_coeffs"
	RET_DS.to_netcdf(path_data['output'] + nc_output_name + ".nc", mode='w')
	RET_DS.close()


else:

	# update ret_vars:
	settings_dict['ret_vars'] = ['ta']

	# update elevation angles:
	DS_ret = DS_ret.isel(n_angle=[0,3,4,5,6,7])
	DS_ret['tb_ip'] = DS_ret.tb
	DS_ret['ele_ip'] = DS_ret.ele

	# # interpolate TBs to needed elevation angles:
	"""
	ele_obs = np.array([90, 14.4, 11.4, 8.4, 6.6, 5.4])[::-1]
	n_time_ret = len(DS_ret.time.values)
	n_ele_obs = len(ele_obs)
	n_freq_sb = len(DS_ret.freq_sb.values)
	tb_ip = np.zeros((n_time_ret, n_ele_obs, n_freq_sb))
	tb_t = DS_ret.tb.values[:,::-1,:]
	ele_t = DS_ret.ele.values[::-1]
	for k in range(n_time_ret):
		for l in range(n_freq_sb):
			tb_ip[k,:,l] = np.interp(ele_obs, ele_t, tb_t[k,:,l])
	tb_ip = tb_ip[:,::-1,:]
	DS_ret['tb_ip'] = xr.DataArray(tb_ip, dims=['time', 'n_angle_ip', 'n_freq'], 
									coords={'time': DS_ret.coords['time'], 
											'n_angle_ip': xr.DataArray(ele_obs[::-1], dims=['n_angle_ip']),
											'n_freq': DS_ret.freq_sb})
	DS_ret['ele_ip'] = xr.DataArray(xr.DataArray(ele_obs[::-1], dims=['n_angle_ip']))
	"""


	# add aritificial TB noise to training data:
	noise_dict = dict()
	for ff in DS_ret.freq_sb.values:
		noise_dict[f"{ff:.2f}"] = 1.0
	DS_ret['tb_ip'] = add_TB_noise_BL(DS_ret.tb_ip, DS_ret.freq_sb, noise_dict)

	# take out outliers (quality control of test and training data):
	qc_idx = {'ta': np.where(np.all(DS_ret.ta.values < 330.0, axis=1) & np.all(DS_ret.ta.values > 180.0, axis=1))[0]}


	# train retrieval: generate retrieval coeffs:
	V_band_idx = select_MWR_channels(DS_ret.tb_ip[:,0,:].values, DS_ret.freq_sb.values, 'V', return_idx=2)
	n_height = len(DS_ret.height.values)
	n_obs_ta = len(V_band_idx)
	n_angles = len(DS_ret.ele_ip.values)
	n_samples = len(DS_ret.time.values)

	coeffs = dict()			# coeffs will be saved to this dict

	# for profiles, each altitude must be dealt with separately:
	order = 2
	coeffs['ta_bl'] = np.full((order*n_obs_ta*n_angles+1,n_height), np.nan)
	tb_reshaped = np.reshape(DS_ret.tb_ip.values[:,:,V_band_idx], (n_samples, n_obs_ta*n_angles))
	for ii in range(n_height):
		coeffs['ta_bl'][:,ii] = regression(DS_ret.ta[qc_idx['ta'],ii].values, tb_reshaped[qc_idx['ta'],:], order=order)


	# after completed training, test retrievals with test data (== training data here):
	# error statistics will be saved to stat_dict
	stat_dict = dict()
	test_data_modelled = dict()
	K_reg = build_K_reg(np.reshape(DS_ret.tb_ip[qc_idx['ta'],:,V_band_idx].values, (len(qc_idx['ta']), n_obs_ta*n_angles)), order=order)
	test_data_modelled['ta_bl'] = np.zeros((len(qc_idx['ta']), n_height))
	for ii in range(n_height):
		test_data_modelled['ta_bl'][:,ii] = K_reg.dot(coeffs['ta_bl'][:,ii])


	# retrieval statistics:
	stat_dict['ta_bl'] = {'RMSE': compute_RMSE_profile(test_data_modelled['ta_bl'], DS_ret.ta.values[qc_idx['ta'],:], which_axis=0),
						'bias': np.nanmean(test_data_modelled['ta_bl'] - DS_ret.ta.values[qc_idx['ta'],:], axis=0)}
	test_data_modelled['ta_bl_biascorr'] = test_data_modelled['ta_bl'] - stat_dict['ta_bl']['bias']
	stat_dict['ta_bl']['stddev'] = compute_RMSE_profile(test_data_modelled['ta_bl_biascorr'], DS_ret.ta.values[qc_idx['ta'],:],
													which_axis=0)


	# save retrieval coeffs and error statistics into netCDF:
	RET_DS = xr.Dataset({'c_ta_bl':			(['obs_ta_bl', 'height'], coeffs['ta_bl'].astype(np.float64),
											{'standard_name': 'retrieval coefficients for BL temperature profile',
											'comment': 'quadratic regression',
											'obs_freq': DS_ret.freq_sb.values[V_band_idx],
											'obs_angles': DS_ret.ele_ip.values}),
						'std_ta_bl':		(['height'], stat_dict['ta_bl']['stddev'],
											{'standard_name': 'standard deviation of BL temperature profile',
											'units': 'K',
											'comment': 'of test (== training) data; for each height'}),
						'bias_ta_bl':		(['height'], stat_dict['ta_bl']['bias'],
											{'standard_name': 'bias of BL temperature profile',
											'units': 'K',
											'comment': 'of test (== training) data; for each height'}),
						'rmse_ta_bl':		(['height'], stat_dict['ta_bl']['RMSE'],
											{'standard_name': 'root mean squared error of BL temperature profile',
											'units': 'K',
											'comment': 'of test (== training) data; for each height'})},
						coords=	{'height':	DS_ret.height})


	# set attributes:
	RET_DS.attrs['Title'] = "Regression coefficients and error statistics for HATPRO BL profiles"
	RET_DS.attrs['Institution'] = "Institute for Geophysics and Meteorology, University of Cologne, Cologne, Germany"
	RET_DS.attrs['Contact_person'] = "Andreas Walbroel (a.walbroel@uni-koeln.de)"
	RET_DS.attrs['Processing_date'] = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
	RET_DS.attrs['Author'] = "Andreas Walbroel (a.walbroel@uni-koeln.de)"
	RET_DS.attrs['License'] = "For non-commercial use only."
	RET_DS.attrs['Measurement_site'] = "RV Polarstern"

	nc_output_name = "MOSAiC_HATPRO_regression_coeffs_BL"
	RET_DS.to_netcdf(path_data['output'] + nc_output_name + ".nc", mode='w')
	RET_DS.close()


print("Done....")