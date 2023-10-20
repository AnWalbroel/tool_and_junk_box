import numpy as np
from copy import deepcopy
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime as dt

import multiprocessing
import sys
import os

wdir = os.getcwd() + "/"
remote = "/net/blanc/" in wdir

sys.path.insert(0, os.path.dirname(wdir[:-1]) + "/")
from data_tools import *
from met_tools import *

import pyPamtra
import pdb

os.environ['OPENBLAS_NUM_THREADS'] = "1"


# some general settings for plots:
fs = 14
fs_small = fs - 2
fs_dwarf = fs - 4
marker_size = 15


class info_content:
	"""
		Compute information content based on optimal estimation theory. 

		Option 1 for degrees of freedom:
		To obtain the degrees of freedom (DOF) from the Averaging Kernel (AK), we firstly need to 
		perturb each component of the state vector (x -> x') of the (test) data sample step by 
		step to compute a new set of perturbed observations (y'). Then proceed to compute the 
		Jacobian matrix K (roughly dF(x)/dx ~ dy/dx). To obtain the AK, covariance matrix of the
		state vector (S_a) over the test data set and the observation error covariance matrix
		(S_eps) must be provided or computed. Then, the AK is computed as 
		(K.T*S_eps^(-1)*K + S_a^(-1))^(-1) * K.T*S_eps^(-1)*K .

		Option 2 for degrees of freedom: 
		To obtain the degrees of freedom (DOF) from the Averaging Kernel (AK), we firstly need to 
		perturb each component of the state vector (x -> x') of the (test) data sample step by 
		step to compute a new set of perturbed observations (y'). The new observations will be fed
		into the retrieval to generate a perturbed retrieved state vector (x_ret'). Differences of 
		x_ret' and x_ret divided by the difference of the (test) data state vectors x' and x yields
		the AK for one test data sample.

		Computing the gain matrix is simpler: Perturb the observation vector directly (y -> y')
		and have the retrieval generate a new x_ret'. The quotient of x_ret' - x_ret and y' - y
		yields a part of the gain matrix. This must be repeated for all obs. vector components
		(i.e., brightness temperature channels) to obtain the full gain matrix.
		
		Functions: perturb_state, perturb_obs, compute_AK (calls some of the prev functions?), 

		For initialisation, we need:
		x : array of floats
			State vector of the (test) data sample (not retrieved!). Has got either one or multiple
			components (i.e., height levels of a temperature profile). Currently, only the 
			following shape is supported: (data_samples, components).
		y : array of floats
			Observation vector, which must have the same number of data samples as x. Components of
			the observation vector could be, for example, brightness temperature channels. 
			Currently, only the following shape is supported: (data_samples, components).
		x_ret : array of floats
			Retrieved state vector. Must have the same number of components as x. Currently, only the
			following shape is supported: (data_samples, components).
		ax_samp : int
			Number indicating which axis of a numpy array represents the (test) data samples.
		ax_comp : int
			Number indicating which axis of a numpy array corresponds to the state vector 
			components.
		perturbation : float
			Float that will be added to or multiplied with the state vector or observation vector, 
			depending on whether "add" or "multiply" perturb_type is selected and whether gain 
			matrix or AK will be computed.
		perturb_type : str
			String indicating whether the perturbation is to be added or multiplied to the state or
			obs vector. Valid options: 'add', 'multiply'
		aux_i : dict
			Dictionary that can contain various information. It must contain information describing
			the state vector in the key "predictand".												#############################################################################################
		suppl_data : dict
			Dictionary containing supplemental data needed to run the PAMTRA simulations for the 
			function new_obs.

		**kwargs:
	"""

	def __init__(self, x, y, x_ret, ax_samp, ax_comp, perturbation, perturb_type, aux_i, suppl_data, **kwargs):
		
		# check for correct dimensions:
		assert x.shape == x_ret.shape
		assert y.shape[ax_samp] == x.shape[ax_samp]

		# set attributes:
		self.x = x
		self.y = y
		self.x_ret = x_ret
		self.perturb_type = perturb_type
		self.pert = perturbation
		self.ax_s = ax_samp
		self.ax_c = ax_comp
		self.aux_i = aux_i
		self.suppl_data = suppl_data
		self.i_s = 0		# (test) data sample index is set to 0 at the beginning (would also be done in function 'perturb')
		self.n_s = self.x.shape[self.ax_s]		# number of (test) data samples
		self.n_cx = self.x.shape[self.ax_c]		# number of state vector components
		self.n_cy = self.y.shape[self.ax_c]		# number of obs vector components
		self.x_i = self.x[self.i_s,:]			# i-th sample of state vector

		# compute some matrices:
		self.x_cov = np.cov(self.x, rowvar=False)		# apriori state vector error covariance matrix (statistical errors of prior state)
		self.y_cov = np.diag(np.ones((self.n_cy,))*0.5)	# observation error covariance matrix
		

		# initialize variables to be computed:
		self.K = np.zeros((self.n_cy, self.n_cx))		# jacobian matrix
		self.AK_i = np.zeros((self.n_cx, self.n_cx))	# i-th AK
		self.AK_diag = np.zeros((self.n_s, self.n_cx))	# main diagonal of AK matrix for all test cases and state vector components
		self.DOF = np.zeros((self.n_s,))				# saves Degrees of Freedom for each test case


	def perturb(self, wat, samp, comp):

		"""
		Perturbs a component of the state or observation vector with a given perturbation. Either
		additive or multiplicative perturbation will be performed.

		Parameters:
		-----------
		wat : str
			Specify if the observation or state vector is to be perturbed. Valid options: 'obs', 
			'state'
		samp : int
			Integer indicating which sample of the vector is processed. Must be within the range of
			the (test) data set.
		comp : int or str
			Integer that indicates which component of the vector will be perturbed. Must be within
			the range of the respective vector. OR: comp can also be a string: "all" which perturbs 
			all components of the state or obs vector step by step and saved the perturbed component
			of the vector in a new vector, which will contain all perturbed components.
		"""

		if comp == "all":

			self.i_s = samp		# current sample index being worked on
			self.x_i = self.x[self.i_s,:]		# i-th sample of state vector
			self.y_i = self.y[self.i_s,:]		# i-th sample of obs vector

			if wat == 'state':
				orig = self.x_i
				n_comp = self.n_cx
			else:
				orig = self.y_i
				n_comp = self.n_cy

			# perturb: save perturbed components into new vector full of perturbations:
			# i.e., orig_p_vec[0] will have the first perturbed component, ...[1] has the second perturbed component....
			if self.perturb_type == 'add':
				orig_p_vec = orig + self.pert

			else:
				orig_p_vec = orig * self.pert


			# save perturbed state or obs vector to a square matrix where i.e., entry [0,:]
			# contains the entire vector with the first component being perturbed. The matrix's
			# diagonal entries equal orig_p_vec.
			pert_vec_mat = np.broadcast_to(orig, (n_comp,n_comp))
			pert_vec_mat = pert_vec_mat - np.diag(orig) + np.diag(orig_p_vec)
			if wat == 'state':

				self.x_ip_mat = pert_vec_mat	# perturbed state vector for each perturbed component (i-th sample)
				self.dx_i = orig_p_vec - orig	# disturb. of all comp. of i-th sample of state vector

			else:
				self.y_ip_mat = pert_vec_mat	# perturbed obs vector for each perturbed component (i-th sample)
				self.dy_i = orig_p_vec - orig	# disturb. of all comp. of i-th sample of obs vector


		elif type(comp) == type(0):

			self.i_s = samp		# current sample index being worked on
			self.i_c = comp		# current component index being worked on
			self.x_i = self.x[self.i_s,:]		# i-th sample of state vector
			self.y_i = self.y[self.i_s,:]		# i-th sample of obs vector

			if wat == 'state':
				orig = self.x_i
			else:
				orig = self.y_i

			# perturb:
			orig_p = deepcopy(orig)				# will contain the perturbed vector
			if self.perturb_type == 'add':
				orig_p[self.i_c] = orig[self.i_c] + self.pert
			else:
				orig_p[self.i_c] = orig[self.i_c] * self.pert

			# save to attributes:
			if wat == 'state':
				self.x_ip = orig_p				# perturbed state vector (i-th sample)
				self.dx_ij = self.x_ip[self.i_c] - self.x_i[self.i_c]	# disturb. of j-th comp. of i-th sample of state vector

			else:
				self.y_ip = orig_p				# perturbed obs vector (i-th sample)
				self.dy_ij = self.y_ip[self.i_c] - self.y_i[self.i_c]	# disturb. of j-th comp. of i-th sample of obs vector


	def new_obs(self, perturbed, what_data='single'):
		
		"""
		Simulations of brightness temperatures (TB) with PAMTRA based on the perturbed atmospheric state.

		Parameters:
		-----------
		perturbed : bool
			Bool to specify whether new obs are generated for a disturbed or undisturbed state vector.
			If True, new TBs are generated for the perturbed state vector.
		what_data : str
			String indicating what will be forward simulated. Options: 'single': A single atmospheric profile 
			(state vector) is simulated, 'comp': Use this option if you want to simulate obs vector for all perturbed 
			state vectors of the i-th (test) data sample, 'samp': Simulate all state vectors in the (test) data set.
		"""

		def hyd_met_lin_dis(hgt_lev, temp_lev, rh_lev, pres_lev, cwp_i, iwp_i, rwp_i, swp_i):

			"""
			Distribute given integrated hydrometors (in kg m-2) uniformly on some cloudy layers that are
			crudely estimated based on temperature and relative humidity.

			Parameters:
			-----------
			hgt_lev : 1D array of floats
				Height levels in m in a 1D array.
			temp_lev : 1D array of floats
				Temperature on height levels in K.
			rh_lev : 1D array of floats
				Relative humidity on height levels in %.
			pres_lev : 1D array of floats
				Air pressure on height levels in Pa.
			cwp_i : float
				Cloud water path for the i-th test data sample in kg m-2.
			iwp_i : float
				Ice water path for the i-th test data sample in kg m-2.
			rwp_i : float
				Rain water path for the i-th test data sample in kg m-2.
			swp_i : float
				Snow water path for the i-th test data sample in kg m-2.
			"""

			# detect freezing level and levels where clouds could be present:
			# also -15 deg C level because of lwp
			hgt_lay = np.diff(hgt_lev)/2 + hgt_lev[:-1]	# height layers, not levels
			temp_lay = np.diff(temp_lev)/2 + temp_lev[:-1]
			rh_lay = np.diff(rh_lev)/2 + rh_lev[:-1]
			n_lay = len(hgt_lay)

			# indices indicating where the freezing level is located and where T < 0 deg C is present:
			below_0_idx = np.where(temp_lay < 273.15)[0]
			freezel_idx = below_0_idx[0]
			super_freezel_idx = np.where(temp_lay < 258.15)[0][0]

			# in case no temperatures above freezing are detected, just set a small height level
			# where cwp, rwp, lwp could be distributed to in case they are not 0.
			if freezel_idx == 0:
				freezel_idx = np.where(hgt_lay >= 400.0)[0][0]
			if super_freezel_idx == 0:
				super_freezel_idx = np.where(hgt_lay >= 1000.0)[0][0]
			# generate masks for potential liquid and ice clouds (sufficient as indices here):
			freezel_idx = np.arange(0, freezel_idx)
			super_freezel_idx = np.arange(0, super_freezel_idx)


			# check if cloud was detected: first, set thresholds relatively high,
			# but it may occur that cwp is > 0 despite relative humidity below 95 %. In that case,
			# relax the threshold(s).
			liq_rh_thres = 95.0
			ice_rh_thres = 85.0
			no_cloud_but_need_liq_cloud = False
			no_cloud_but_need_ice_cloud = False
			while (not no_cloud_but_need_liq_cloud) or (not no_cloud_but_need_ice_cloud):
				cloudy_idx = np.where(rh_lay >= liq_rh_thres)[0]
				ice_cloudy_idx = np.where(rh_lay >= ice_rh_thres)[0]

				liq_cloud_mask = np.intersect1d(super_freezel_idx, cloudy_idx)
				rain_cloud_mask = np.intersect1d(freezel_idx, cloudy_idx)
				ice_cloud_mask = np.intersect1d(below_0_idx, ice_cloudy_idx)
				len_liq_mask = len(liq_cloud_mask)
				len_rain_mask = len(rain_cloud_mask)
				len_ice_mask = len(ice_cloud_mask)

				# check if cloud is needed (when cwp > 0) but not detected (len_liq_mask==0):
				if cwp_i > 0.0 and len_liq_mask == 0:
					liq_rh_thres -= 2.5
					continue
				else:
					no_cloud_but_need_liq_cloud = True
				
				# repeat for other hyd. mets:
				if iwp_i > 0.0 and len_ice_mask == 0:
					ice_rh_thres -= 2.5
					continue
				else:
					no_cloud_but_need_ice_cloud = True
				if rwp_i > 0.0 and len_rain_mask == 0:
					liq_rh_thres -= 2.5
					continue
				else:
					no_cloud_but_need_liq_cloud = True
				if swp_i > 0.0 and len_ice_mask == 0:
					ice_rh_thres -= 2.5
					continue
				else:
					no_cloud_but_need_ice_cloud = True


			# distribute LWP, SWP, IWP, CWP, RWP linearly:
			cwc = np.zeros((n_lay,))		# cloud water content
			iwc = np.zeros((n_lay,))		# ice water content
			rwc = np.zeros((n_lay,))		# rain water content
			swc = np.zeros((n_lay,))		# snow water content

			# check if hydromet is actually > 0 and if clouds were detected (and to catch division by 0 errors)
			if cwp_i >= 0.0 and len_liq_mask > 0:
				# identify the heights of each cloudy layer, then compute how much CWC these layers
				# need to have to yield CWP = integral{CWC*dz}
				liq_lay_hgts = hgt_lev[liq_cloud_mask+1] - hgt_lev[liq_cloud_mask]
				cwc[liq_cloud_mask] = cwp_i / np.sum(liq_lay_hgts)

			if iwp_i >= 0.0 and len_ice_mask > 0:
				ice_lay_hgts = hgt_lev[ice_cloud_mask+1] - hgt_lev[ice_cloud_mask]
				iwc[ice_cloud_mask] = iwp_i / np.sum(ice_lay_hgts)

			if rwp_i >= 0.0 and len_rain_mask > 0:
				rain_lay_hgts = hgt_lev[rain_cloud_mask+1] - hgt_lev[rain_cloud_mask]
				rwc[rain_cloud_mask] = rwp_i / np.sum(rain_lay_hgts)

			if swp_i >= 0.0 and len_ice_mask > 0:
				ice_lay_hgts = hgt_lev[ice_cloud_mask+1] - hgt_lev[ice_cloud_mask]
				swc[ice_cloud_mask] = swp_i / np.sum(ice_lay_hgts)

			# convert cwc, iwc, ... to kg kg-1 instead of kg m-3:
			rho_v_lay = convert_rh_to_abshum(temp_lay, 0.01*rh_lay)
			pres_lay = np.diff(pres_lev)/2 + pres_lev[:-1]
			rho_lay = rho_air(pres_lay, temp_lay, rho_v_lay)
			cwc = cwc / rho_lay
			iwc = iwc / rho_lay
			rwc = rwc / rho_lay
			swc = swc / rho_lay

			return cwc, iwc, rwc, swc


		# identify what's the state vector:
		shape_pred_0 = 0
		shape_pred_1 = 0
		x_idx = dict()			# state vector x can consist of different meteorol. variables.
								# this identifies the indices where the single predictands are located
		for id_i, predictand in enumerate(self.aux_i['predictand']):
			# inquire shape of current predictand and its position in the output vector or prediction:
			shape_pred_0 = shape_pred_1
			shape_pred_1 = shape_pred_1 + self.aux_i['n_ax1'][predictand]
			x_idx[predictand] = [shape_pred_0, shape_pred_1]


		# create pam object:
		pam = pyPamtra.pyPamtra()


		# general settings:
		pam.nmlSet['passive'] = True						# passive simulation
		pam.nmlSet['active'] = False						# False: no radar
		if remote:
			pam.nmlSet['data_path'] = "/net/blanc/awalbroe/Codes/pamtra/"
		else:
			pam.nmlSet['data_path'] = "/home/tenweg/pamtra/"

		# define the pamtra profile: temp, relhum, pres, height, lat, lon, timestamp, lfrac, obs_height, ...
		pamData = dict()

		if what_data == 'samp':
			n_data = self.n_s
			shape2d = (n_data,1)

			# data needed for PAMTRA:
			lon = self.suppl_data['lon']
			lat = self.suppl_data['lat']
			timestamp = self.suppl_data['time']
			hgt_lev = self.suppl_data['height']
			pres_lev = self.suppl_data['pres']
			temp_sfc = self.suppl_data['temp_sfc']
			cwp = self.suppl_data['cwp']
			iwp = self.suppl_data['iwp']
			rwp = self.suppl_data['rwp']
			swp = self.suppl_data['swp']


			# use x_idx (who identified where which meteorological variable of the state vector is located)
			# to set the temp and rh data:
			if perturbed:
				pdb.set_trace()
				1/0			# not yet coded, and I don't think that this case will be planned

			else:
				if 'temp' in self.aux_i['predictand']:
					temp_lev = self.x[:, x_idx['temp'][0]:x_idx['temp'][1]]
				else:
					temp_lev = self.suppl_data['temp']

				# in case "unperturbed" is selected, we don't need to filter q out of the
				# state vector and convert it. We can just take the relative humidity instead 				# # or does it induce too many uncertainties to TBs
																											# # due to the conversion, and potentially overexpose
																											# # the effect by the perturbation lateron?
				# if 'q' in self.aux_i['predictand']:
					# # pdb.set_trace() # check for correct units
					# q_lev = self.x_i[x_idx['q'][0]:x_idx['q'][1]]

					# rho_v_lev = convert_spechum_to_abshum(temp_lev, 
														# self.suppl_data['pres'][self.i_s,:],
														# q_lev)
					# rh_lev = convert_abshum_to_relhum(temp_lev, rho_v_lev)

				# else:
				rh_lev = self.suppl_data['rh']


		elif what_data == 'single':
			shape2d = (1,1)

			# data needed for PAMTRA: select case
			lon = self.suppl_data['lon'][self.i_s]
			lat = self.suppl_data['lat'][self.i_s]
			timestamp = self.suppl_data['time'][self.i_s]
			hgt_lev = self.suppl_data['height'][self.i_s,:]
			pres_lev = self.suppl_data['pres'][self.i_s,:]
			temp_sfc = self.suppl_data['temp_sfc'][self.i_s]
			cwp = self.suppl_data['cwp'][self.i_s]
			iwp = self.suppl_data['iwp'][self.i_s]
			rwp = self.suppl_data['rwp'][self.i_s]
			swp = self.suppl_data['swp'][self.i_s]

			# use x_idx (who identified where which meteorological variable of the state vector is located)
			# to set the temp and rh data:
			if perturbed:
				if 'temp' in self.aux_i['predictand']:
					temp_lev = self.x_ip[x_idx['temp'][0]:x_idx['temp'][1]]
				else:
					temp_lev = self.suppl_data['temp'][self.i_s,:]

				# we need relative humidity: thus, if q is a predictand, it needs to be converted
				if 'q' in self.aux_i['predictand']:
					pdb.set_trace() # check units
					q_lev = self.x_ip[x_idx['q'][0]:x_idx['q'][1]]

					rho_v_lev = convert_spechum_to_abshum(temp_lev, pres_lev, q_lev)
					rh_lev = convert_abshum_to_relhum(temp_lev, rho_v_lev)

				else:
					rh_lev = self.suppl_data['rh'][self.i_s,:]

			else:
				if 'temp' in self.aux_i['predictand']:
					temp_lev = self.x_i[x_idx['temp'][0]:x_idx['temp'][1]]
				else:
					temp_lev = self.suppl_data['temp'][self.i_s,:]

				# in case "unperturbed" is selected, we don't need to filter q out of the
				# state vector and convert it. We can just take the relative humidity instead 				# # or does it induce too many uncertainties to TBs
																											# # due to the conversion, and potentially overexpose
																											# # the effect by the perturbation lateron?
				# if 'q' in self.aux_i['predictand']:
					# # pdb.set_trace() # check for correct units
					# q_lev = self.x_i[x_idx['q'][0]:x_idx['q'][1]]

					# rho_v_lev = convert_spechum_to_abshum(temp_lev, 
														# self.suppl_data['pres'][self.i_s,:],
														# q_lev)
					# rh_lev = convert_abshum_to_relhum(temp_lev, rho_v_lev)

				# else:
				rh_lev = self.suppl_data['rh'][self.i_s,:]


		elif what_data == 'comp':
			n_data = self.n_cx
			shape2d = (n_data,1)

			# data needed for PAMTRA: suppl_data must still be from i-th (test) data sample
			lon = np.broadcast_to(self.suppl_data['lon'][self.i_s], (n_data,))
			lat = np.broadcast_to(self.suppl_data['lat'][self.i_s], (n_data,))
			timestamp = np.broadcast_to(self.suppl_data['time'][self.i_s], (n_data,))
			hgt_lev = np.broadcast_to(self.suppl_data['height'][self.i_s,:], (n_data,n_data))
			pres_lev = np.broadcast_to(self.suppl_data['pres'][self.i_s,:], (n_data,n_data))
			temp_sfc = np.broadcast_to(self.suppl_data['temp_sfc'][self.i_s], (n_data,))
			cwp = np.broadcast_to(self.suppl_data['cwp'][self.i_s], (n_data,))
			iwp = np.broadcast_to(self.suppl_data['iwp'][self.i_s], (n_data,))
			rwp = np.broadcast_to(self.suppl_data['rwp'][self.i_s], (n_data,))
			swp = np.broadcast_to(self.suppl_data['swp'][self.i_s], (n_data,))


			# use x_idx (who identified where which meteorological variable of the state vector is located)
			# to set the temp and rh data:
			if perturbed:

				if 'temp' in self.aux_i['predictand']:
					temp_lev = self.x_ip_mat[:, x_idx['temp'][0]:x_idx['temp'][1]]
				else:
					temp_lev = np.broadcast_to(self.suppl_data['temp'][self.i_s,:], (n_data,n_data))

				# we need relative humidity: thus, if q is a predictand, it needs to be converted
				if 'q' in self.aux_i['predictand']:
					pdb.set_trace() # check units
					q_lev = self.x_ip_mat[:, x_idx['q'][0]:x_idx['q'][1]]

					rho_v_lev = convert_spechum_to_abshum(temp_lev, pres_lev, q_lev)
					rh_lev = convert_abshum_to_relhum(temp_lev, rho_v_lev)

				else:
					rh_lev = np.broadcast_to(self.suppl_data['rh'][self.i_s,:], (n_data,n_data))

			else:
				pdb.set_trace()
				1/0		# I also don't expect this case to happen


		# make sure relative humidity doesn't exceed sensible values:
		rh_lev = rh_lev*100.0		# convert to %
		rh_lev[rh_lev > 100.0] = 100.0
		rh_lev[rh_lev < 0.0] = 0.0


		# write data into pamData dict:
		pamData['lon'] = np.reshape(lon, shape2d)
		pamData['lat'] = np.reshape(lat, shape2d)
		pamData['timestamp'] = timestamp
		pamData['hgt_lev'] = hgt_lev


		# put meteo data into pamData:
		shape3d = shape2d + (pamData['hgt_lev'].shape[-1],)
		pamData['press_lev'] = np.reshape(pres_lev, shape3d)
		pamData['relhum_lev'] = np.reshape(rh_lev, shape3d)
		pamData['temp_lev'] = np.reshape(temp_lev, shape3d)

		# Surface data:
		# # pamData['wind10u'] = sonde_dict['u'][0]*0.0
		# # pamData['wind10v'] = sonde_dict['v'][0]*0.0
		pamData['groundtemp'] = temp_sfc

		# surface properties: either use lfrac or the other 4 lines
		pamData['sfc_type'] = np.ones(shape2d)
		pamData['sfc_model'] = np.zeros(shape2d)		# 0 = sea, 1 = land --> and we ve got sea conditions only
		pamData['sfc_refl'] = np.chararray(shape2d)
		pamData['sfc_refl'][:] = "L"


		pamData['obs_height'] = np.broadcast_to(np.array([0.0]), shape2d + (1,))
		
		# 4d variables: hydrometeors: distribute the given CWP, IWP, RWP, SWP linearly:
		shape3d_lay = shape2d + (pamData['hgt_lev'].shape[-1]-1,)
		shape4d = shape2d + (pamData['hgt_lev'].shape[-1]-1, 4)

		if what_data == 'samp':
			cwc = np.zeros(shape3d_lay)
			iwc = np.zeros(shape3d_lay)
			rwc = np.zeros(shape3d_lay)
			swc = np.zeros(shape3d_lay)
			for ii in range(n_data):
				cwc[ii], iwc[ii], rwc[ii], swc[ii] = hyd_met_lin_dis(pamData['hgt_lev'][0,:],
																	temp_lev[ii,:], rh_lev[ii,:], 
																	pres_lev[ii,:], cwp[ii], iwp[ii], rwp[ii], swp[ii])

		elif what_data == 'single':
			cwc, iwc, rwc, swc = hyd_met_lin_dis(pamData['hgt_lev'], temp_lev, rh_lev, pres_lev, cwp, iwp, rwp, swp)

		elif what_data == 'comp':
			cwc = np.zeros(shape3d_lay)
			iwc = np.zeros(shape3d_lay)
			rwc = np.zeros(shape3d_lay)
			swc = np.zeros(shape3d_lay)
			for ii in range(n_data):
				cwc[ii], iwc[ii], rwc[ii], swc[ii] = hyd_met_lin_dis(pamData['hgt_lev'][0,:],
																	temp_lev[ii,:], rh_lev[ii,:], 
																	pres_lev[ii,:], cwp[ii], iwp[ii], rwp[ii], swp[ii])

		pamData['hydro_q'] = np.zeros(shape4d)
		pamData["hydro_q"][:,:,:,0] = cwc
		pamData["hydro_q"][:,:,:,1] = iwc
		pamData["hydro_q"][:,:,:,2] = rwc
		pamData["hydro_q"][:,:,:,3] = swc

		descriptorFile = np.array([
			  #['hydro_name' 'as_ratio' 'liq_ice' 'rho_ms' 'a_ms' 'b_ms' 'alpha_as' 'beta_as' 'moment_in' 'nbin' 'dist_name' 'p_1' 'p_2' 'p_3' 'p_4' 'd_1' 'd_2' 'scat_name' 'vel_size_mod' 'canting']
			   ('cwc_q', -99.0, 1, -99.0, -99.0, -99.0, -99.0, -99.0, 3, 1, 'mono', -99.0, -99.0, -99.0, -99.0, 2e-05, -99.0, 'mie-sphere', 'khvorostyanov01_drops', -99.0),
			   ('iwc_q', 1.0, -1, 700.0, -99.0, -99.0, -99.0, -99.0, 3, 1, 'mono', -99.0, -99.0, -99.0, -99.0, 6e-05, -99.0, 'mie-sphere', 'heymsfield10_particles', -99.0),
			   ('rwc_q', -99.0, 1, -99.0, -99.0, -99.0, -99.0, -99.0, 3, 50, 'exp', 0.22, 2.2, -99.0, -99.0, 0.00012, 0.006, 'mie-sphere', 'khvorostyanov01_drops', -99.0),
			   ('swc_q', 1.0, -1, -99.0, 0.069, 2.0, -99.0, -99.0, 3, 50, 'exp', 2e-06, 0.0, -99.0, -99.0, 2e-04, 0.02, 'mie-sphere', 'heymsfield10_particles', -99.0)], 
			  dtype=[('hydro_name', 'S15'), ('as_ratio', '<f8'), ('liq_ice', '<i8'), ('rho_ms', '<f8'), ('a_ms', '<f8'), ('b_ms', '<f8'), ('alpha_as', '<f8'), ('beta_as', '<f8'), ('moment_in', '<i8'), ('nbin', '<i8'), ('dist_name', 'S15'), ('p_1', '<f8'), ('p_2', '<f8'), ('p_3', '<f8'), ('p_4', '<f8'), ('d_1', '<f8'), ('d_2', '<f8'), ('scat_name', 'S15'), ('vel_size_mod', 'S30'), ('canting', '<f8')]
		  )

		for hyd in descriptorFile: pam.df.addHydrometeor(hyd)


		# create pamtra profile from pamData and run pamtra at all specified frequencies:
		freqs = np.array([	22.240, 23.040, 23.840, 25.440, 26.240, 27.840, 31.400,
							51.260, 52.280, 53.860, 54.940, 56.660, 57.300, 58.000,
							175.810, 178.310, 179.810, 180.810, 181.810, 182.710,
							183.910, 184.810, 185.810, 186.810, 188.310, 190.810,
							243.000, 340.000])

		pam.createProfile(**pamData)

		if what_data == 'single':
			pam.runPamtra(freqs)
		else:
			n_cpus = int(multiprocessing.cpu_count()*0.5)
			pam.runParallelPamtra(freqs, pp_deltaX=1, pp_deltaY=0, pp_deltaF=0, pp_local_workers=n_cpus)
			# pam.runPamtra(freqs)


		# post process PAMTRA TBs:
		# TBs: angles: 0 = nadir; -1 = zenith <<->> angle[0] = 180.0; angle[-1] = 0.0
		TB = pam.r['tb'][:,0,0,-1,:,:].mean(axis=-1)
		TB, freqs = Gband_double_side_band_average(TB, freqs)

		# select the TBs used as obs vector:
		TB_obs, freq_obs = select_MWR_channels(TB, freqs, band=self.aux_i['predictor_TBs'], return_idx=0)

		# update obs vector
		if perturbed:
			if what_data == 'comp':
				self.y_ip_mat = TB_obs				# (self.n_cx, self.n_cy) matrix of obs vector based on perturbed
													# state vector: self.y_ip_max[0,:] yields the obs vector for the
													# 0-th-component-perturbed state vector
			elif what_data == 'single':
				self.y_ip = TB_obs[0,:]				# reduce to 1D array, removing the unnecessary dimension

		else:
			if what_data == 'samp':
				self.y = TB_obs
			elif what_data == 'single':
				self.y[self.i_s,:] = TB_obs[0,:]	# remove unnecessary dimension
				self.y_i = self.y[self.i_s,:]		# i-th sample of obs vector


	def compute_dx_ret_i(self, comp):

		"""
		Computes the difference of the perturbed retrieved state vector (generated from perturbed obs 
		vector) and the original retrieved state vector x_ret for the current i-th (test) data sample.

		Parameters:
		-----------
		comp : int or str
			Integer that indicates which component of the vector will be perturbed. Must be within
			the range of the respective vector. OR: comp can also be a string: "all" which perturbs 
			all components of the state or obs vector step by step and saved the perturbed component
			of the vector in a new vector, which will contain all perturbed components.
		"""

		if comp == 'all':
			self.dx_ret_i_mat = self.x_ret_ip_mat - self.x_ret[self.i_s,:]	# (n_c, n_c) array where the first row dx_ret_i_mat[0,:]
																	# indictes the ret state vector difference when the first 
																	# component was perturbed

		elif type(comp) == type(0):	############
			self.dx_ret_i = self.x_ret_ip - self.x_ret[self.i_s,:]


	def compute_jacobian_step(self):

		"""
		Computes the j-th (self.i_c) column of the Jacobian K with entries K_aj = dy_ia / dx_ij where dy_ia is the a-th
		component of the difference between the perturbed and reference obs vector of (test) data sample i. dx_ij is the 
		j-th component of the diff between the perturbed and reference state vector of (test) data sample i.
		"""

		jacobian_j = (self.y_ip - self.y_i) / self.dx_ij		# j-th column of the jacobian
		self.K[:,self.i_c] = jacobian_j


	def compute_jacobian(self):

		"""
		Computes the Jacobian K with entries K_aj = dy_ia / dx_ij where dy_ia is the a-th component of the 
		difference between the perturbed and reference obs vector of (test) data sample i. dx_ij is the 
		j-th component of the diff between the perturbed and reference state vector of (test) data sample i.
		"""

		jacobian = np.zeros((self.n_cy, self.n_cx))

		# loop through obs vector components:
		for a in range(self.n_cy):
			jacobian[a,:] = (self.y_ip_mat[:,a] - self.y_i[a]) / self.dx_i

		self.K = jacobian


	def compute_col_of_AK_i(self):

		"""
		Computes the j-th (i_j-th) column of the Averaging Kernel matrix of test case i.
		This function is needed when considering each component step by step, meaning that 'all' 
		has been used for comp in the other functions. Also, the main diagonal is set when all
		columns of the AK have been computed.
		"""

		self.AK_i[:,self.i_c] = self.dx_ret_i / self.dx_ij

		if self.i_c == self.n_cx - 1:
			self.AK_diag[self.i_s,:] = np.diag(self.AK_i)


	def compute_AK_i(self, how):

		"""
		Computes the the entire  Averaging Kernel matrix of test case i. Use this function when
		all components have been worked on in one batch (i.e., comp == 'all' in the functions above).
		Also, the main diagonal is set. The Averaging Kernel will either be computed via the 
		dx_ret / dx or via the matrix multiplication scheme.

		Parameters:
		-----------
		how : str
			String to choose between two ways of computing the AK matrix. Valid options: 'matrix': it
			requires to run perturb('state', i_s, 'all'), new_obs(True, what_data='comp') and 
			compute_jacobian(); 'ret': requires perturb('state', i_s, 'all'), new_obs(True, what_data='comp'),
			generation of new (perturbed) x_ret from perturbed obs vector, and compute_dx_ret_i('all').
		"""

		if how == 'ret':
			for jj in range(self.n_cx):
				self.AK_i[:,jj] = self.dx_ret_i_mat[jj,:] / self.dx_i[jj]

		elif how == 'matrix':
			x_cov_inv = np.linalg.inv(self.x_cov)
			y_cov_inv = np.linalg.inv(self.y_cov)
			KTSeK = self.K.T @ y_cov_inv @ self.K
			self.AK_i = np.linalg.inv(KTSeK + x_cov_inv) @ KTSeK


		else:
			raise ValueError("Argument 'how' of the function compute_AK_i must be either 'ret' or 'matrix'.")

		self.AK_diag[self.i_s,:] = np.diag(self.AK_i)


	def compute_DOF(self):

		"""
		Computes the degrees of freedom (DOF) from the trace of the AK of the i-th test case.
		"""

		self.DOF[self.i_s] = np.trace(self.AK_i)


	def visualise_AK_i(self):

		"""
		Visualises the main diagonal of the AK of the i-th (test) data sample.
		"""

		f1 = plt.figure(figsize=(7,11))
		a1 = plt.axes()

		# axis limits:
		ax_lims = {'y': [0.0, self.suppl_data['height'].max()]}

		# plot data:
		a1.plot(self.AK_diag[self.i_s,:], self.suppl_data['height'][self.i_s,:], color=(0,0,0), linewidth=1.25)

		# aux info:
		a1.text(0.98, 0.98, f"DOF = {self.DOF[self.i_s]}", 
				ha='right', va='top', color=(0,0,0), fontsize=fs_small, 
				bbox={'facecolor': (1.0, 1.0, 1.0, 0.8), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
				transform=a1.transAxes)

		# legend, colorbar

		# set axis limits:
		a1.set_ylim(ax_lims['y'][0], ax_lims['y'][1])

		# - set ticks and tick labels and parameters:
		a1.tick_params(axis='both', labelsize=fs_dwarf)

		# grid:
		a1.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

		# labels:
		a1.set_ylabel("Height (m)", fontsize=fs)
		a1.set_xlabel("Averaging Kernel diagonal (K/K)", fontsize=fs)

		plt.show()


	def visualise_mean_AK(self, path_output):

		"""
		Visualises the main diagonal of the mean of all test data sample AKs.

		Parameters:
		-----------
		path_output : str
			Path to save the plot to.
		"""

		mean_AK = np.mean(self.AK_diag, axis=0)
		mean_height = np.mean(self.suppl_data['height'], axis=0)
		
		f1 = plt.figure(figsize=(7,11))
		a1 = plt.axes()

		# axis limits:
		ax_lims = {'y': [0.0, self.suppl_data['height'].max()]}

		# plot data:
		a1.plot(mean_AK, mean_height, color=(0,0,0), linewidth=1.25)

		# aux info:
		a1.text(0.98, 0.98, f"DOF = {np.mean(self.DOF):.2f}", 
				ha='right', va='top', color=(0,0,0), fontsize=fs_small, 
				bbox={'facecolor': (1.0, 1.0, 1.0, 0.8), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
				transform=a1.transAxes)

		# legend, colorbar

		# set axis limits:
		a1.set_ylim(ax_lims['y'][0], ax_lims['y'][1])

		# - set ticks and tick labels and parameters:
		a1.tick_params(axis='both', labelsize=fs_dwarf)

		# grid:
		a1.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

		# labels:
		a1.set_ylabel("Height (m)", fontsize=fs)
		a1.set_xlabel("Averaging Kernel diagonal (K/K)", fontsize=fs)


		# check if output path exists:
		plotpath_dir = os.path.dirname(path_output)
		if not os.path.exists(plotpath_dir):
			os.makedirs(plotpath_dir)

		plot_file = path_output + f"MOSAiC_synergetic_ret_info_content_AK_diag_mean_DOF_{self.aux_i['file_descr']}.png"
		f1.savefig(plot_file, dpi=300, bbox_inches='tight')
		print(f" Saved {plot_file}")


	def save_info_content_data(self, path_output):

		"""
		Saves the AK_diag, DOF, and supplementary data of all (test) data samples to a netCDF file in the specified path.

		Parameters:
		-----------
		path_output : str
			Path to save the netCDF file to.
		"""

		# check if output path exists: if it doesn't, create it:
		path_output_dir = os.path.dirname(path_output)
		if not os.path.exists(path_output_dir):
			os.makedirs(path_output_dir)


		# create xarray Dataset:
		DS = xr.Dataset(coords={'n_s': 	(['n_s'], np.arange(self.n_s), {'long_name': "Number of data samples"}),
								'n_cx': (['n_cx'], np.arange(self.n_cx), {'long_name': "Number of state vector components"}),
								'n_cy': (['n_cy'], np.arange(self.n_cy), {'long_name': "Number of observation vector components"})})

		# save data into it:
		DS['height'] = xr.DataArray(self.suppl_data['height'], dims=['n_s', 'n_cx'], 
									attrs={	'long_name': "Height grid for state vector (profile)",
											'units': "m"})
		DS['pert_type'] = xr.DataArray(self.perturb_type, attrs={'long_name': "Perturbation type (either additive or multiplicative)"})
		DS['perturbation'] = xr.DataArray(self.pert, attrs={'long_name': "Perturbation factor or summand"})
		DS['AK_diag'] = xr.DataArray(self.AK_diag, dims=['n_s', 'n_cx'], 
									attrs={	'long_name': "Main diagonal of the Averaging Kernel matrix for each data sample",
											'units': "(unit_of_state_vector / unit_of_state_vector)"})
		DS['DOF'] = xr.DataArray(self.DOF, dims=['n_s'], attrs={'long_name': "Degrees Of Freedom, computed as trace of the Averaging Kernel matrix",
																'units': "(unit_of_state_vector / unit_of_state_vector)"})


		# GLOBAL ATTRIBUTES:
		DS.attrs['title'] = "Information content output"
		DS.attrs['author'] = "Andreas Walbroel (a.walbroel@uni-koeln.de), Institute for Geophysics and Meteorology, University of Cologne, Cologne, Germany"
		DS.attrs['predictor_TBs'] = self.aux_i['predictor_TBs']
		DS.attrs['setup_id'] = self.aux_i['file_descr']
		DS.attrs['python_version'] = f"python version: {sys.version}"
		DS.attrs['python_packages'] = f"numpy: {np.__version__}, xarray: {xr.__version__}, matplotlib: {mpl.__version__}, "

		datetime_utc = dt.datetime.utcnow()
		DS.attrs['processing_date'] = datetime_utc.strftime("%Y-%m-%d %H:%M:%S")

		# export to netCDF:
		DS.to_netcdf(path_output + f"MOSAiC_synergetic_ret_info_content_{self.aux_i['file_descr']}.nc", mode='w', format='NETCDF4')
		DS.close()




