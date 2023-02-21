import numpy as np
import copy
import datetime as dt
import xarray as xr
import os
import glob
import pdb
import warnings


def running_mean(x, N):

	"""
	Moving average of a 1D array x with a window width of N

	Parameters:
	-----------
	x : array of floats
		1D data vector of which the running mean is to be taken.
	N : int
		Running mean window width.
	"""
	x = x.astype(np.float64)
	x_m = copy.deepcopy(x)
	
	# run through the array:
	for k in range(len(x)):
		if k%400000 == 0: print(k/len(x))	# output required to avoid the ssh connection to
											# be automatically dropped

		# Identify which indices are addressed for the running
		# mean of the current index k:
		if N%2 == 0: 	# even:
			rm_range = np.arange(k - int(N/2), k + int(N/2), dtype = np.int32)
		else:			# odd:
			rm_range = np.arange(k - int(N/2), k + int(N/2) + 1, dtype=np.int32)

		# remove indices that exceed array bounds:
		rm_range = rm_range[(rm_range >= 0) & (rm_range < len(x))]

		# moving average:
		x_m[k] = np.mean(x[rm_range])
	
	return x_m


def running_mean_datetime(x, N, t):

	"""
	Moving average of a 1D array x with a window width of N in seconds.
	Here it is required to find out the actual window range. E.g. if
	the desired window width is 300 seconds but the measurement rate
	is one/minute, the actual window width is 5.

	Parameters:
	-----------
	x : array of floats
		1D data vector of which the running mean is to be taken.
	N : int
		Running mean window width in seconds.
	t : array of floats
		1D time vector (in seconds since a reference time) required to
		compute the actual running mean window width.
	"""

	x = x.astype(np.float64)
	x_m = copy.deepcopy(x)

	n_x = len(x)
	is_even = (N%2 == 0)		# if N is even: is_even is True

	# ii = np.arange(len(x))	# array of indices, required for the 'slow version'

	# inquire mean delta time to get an idea of how broad a window
	# must be (roughly) <-> used to speed up computation time:
	mdt = np.nanmean(t[1:] - t[:-1])
	look_range = int(np.ceil(N/mdt))
	
	# run through the array:
	look_save = 0
	for k in range(n_x):	# k, t_c in enumerate(t)?
		if k%400000 == 0: print(k/n_x)	# output required to avoid ssh connection to
										# be automatically dropped

		# Identify the correct running mean window width from the current
		# time t_c:
		t_c = t[k]
		if is_even:	# even:
			t_c_plus = t_c + int(N/2)
			t_c_minus = t_c - int(N/2)
			# t_range = t[(t >= t_c_minus) & (t <= t_c_plus)]	# not required
		else:			# odd:
			t_c_plus = t_c + int(N/2) + 1
			t_c_minus = t_c - int(N/2)
			# t_range = t[(t >= t_c_minus) & (t <= t_c_plus)]	# not required

		# rm_range_SAVE = ii[(t >= t_c_minus) & (t <= t_c_plus)]		# very slow for large time axis array but also works
		# faster:

		if (k > look_range) and (k < n_x - look_range):	# in between
			look_save = k-look_range
			rm_range = np.argwhere((t[k-look_range:k+look_range] >= t_c_minus) & (t[k-look_range:k+look_range] <= t_c_plus)).flatten() + look_save
		elif k <= look_range: 	# lower end of array
			look_save = 0
			rm_range = np.argwhere((t[:k+look_range] >= t_c_minus) & (t[:k+look_range] <= t_c_plus)).flatten()
		else:	# upper end of array
			look_save = k-look_range
			rm_range = np.argwhere((t[k-look_range:] >= t_c_minus) & (t[k-look_range:] <= t_c_plus)).flatten() + look_save
			

		# moving average:
		x_m[k] = np.mean(x[rm_range])
	
	return x_m


def running_mean_time_2D(x, N, t, axis=0):

	"""
	Moving average of a 2D+ array x with a window width of N in seconds.
	The moving average will be taken over the specifiec axis.
	Here it is required to find out the actual window range. E.g. if
	the desired window width is 300 seconds but the measurement rate
	is one/minute, the actual window width is 5.

	Parameters:
	-----------
	x : array of floats
		Data array (multi-dim) of which the running mean is to be taken for a
		certain axis.
	N : int
		Running mean window width in seconds.
	t : array of floats
		1D time vector (in seconds since a reference time) required to
		compute the actual running mean window width.
	axis : int
		Indicates, which axis represents the time axis, over which the moving
		average will be taken. Default: 0
	"""

	# check if shape of x is correct:
	n_x = x.shape[axis]
	assert n_x == len(t)

	x = x.astype(np.float64)
	x_m = copy.deepcopy(x)

	is_even = (N%2 == 0)		# if N is even: is_even is True

	# inquire mean delta time to get an idea of how broad a window
	# must be (roughly) <-> used to speed up computation time:
	mdt = np.nanmean(np.diff(t))
	look_range = int(np.ceil(N/mdt))
	
	# run through the array:
	look_save = 0
	for k in range(n_x):	# k, t_c in enumerate(t)?
		if k%400000 == 0: print(k/n_x)	# output required to avoid ssh connection to
										# be automatically dropped

		# Identify the correct running mean window width from the current
		# time t_c:
		t_c = t[k]
		if is_even:	# even:
			t_c_plus = t_c + int(N/2)
			t_c_minus = t_c - int(N/2)
		else:			# odd:
			t_c_plus = t_c + int(N/2) + 1
			t_c_minus = t_c - int(N/2)


		if (k > look_range) and (k < n_x - look_range):	# in between
			look_save = k-look_range
			rm_range = np.argwhere((t[k-look_range:k+look_range] >= t_c_minus) & (t[k-look_range:k+look_range] <= t_c_plus)).flatten() + look_save
		elif k <= look_range: 	# lower end of array
			look_save = 0
			rm_range = np.argwhere((t[:k+look_range] >= t_c_minus) & (t[:k+look_range] <= t_c_plus)).flatten()
		else:	# upper end of array
			look_save = k-look_range
			rm_range = np.argwhere((t[k-look_range:] >= t_c_minus) & (t[k-look_range:] <= t_c_plus)).flatten() + look_save
			

		# moving average:
		x_m[k] = np.mean(x[rm_range], axis=axis)
	
	return x_m


def datetime_to_epochtime(dt_array):
	
	"""
	This tool creates a 1D array (or of seconds since 1970-01-01 00:00:00 UTC
	(type: float) out of a datetime object or an array of datetime objects.

	Parameters:
	-----------
	dt_array : array of datetime objects or datetime object
		Array (1D) that includes datetime objects. Alternatively, dt_array is directly a
		datetime object.
	"""

	reftime = dt.datetime(1970,1,1)

	try:
		sec_epochtime = np.asarray([(dtt - reftime).total_seconds() for dtt in dt_array])
	except TypeError:	# then, dt_array is no array
		sec_epochtime = (dt_array - reftime).total_seconds()

	return sec_epochtime


def numpydatetime64_to_epochtime(npdt_array):

	"""
	Converts numpy datetime64 array to array in seconds since 1970-01-01 00:00:00 UTC (type:
	float).
	Alternatively, just use "some_array.astype(np.float64)".

	Parameters:
	-----------
	npdt_array : numpy array of type np.datetime64 or np.datetime64 type
		Array (1D) or directly a np.datetime64 type variable.
	"""

	sec_epochtime = npdt_array.astype(np.timedelta64) / np.timedelta64(1, 's')

	return sec_epochtime


def numpydatetime64_to_reftime(
	npdt_array, 
	reftime):

	"""
	Converts numpy datetime64 array to array in seconds since a reftime as type:
	float. Reftime could be for example: "2017-01-01 00:00:00" (in UTC)

	Parameters:
	-----------
	npdt_array : numpy array of type np.datetime64 or np.datetime64 type
		Array (1D) or directly a np.datetime64 type variable.
	reftime : str
		Specification of the reference time in "yyyy-mm-dd HH:MM:SS" (in UTC).
	"""

	time_dt = numpydatetime64_to_datetime(npdt_array)

	reftime = dt.datetime.strptime(reftime, "%Y-%m-%d %H:%M:%S")

	try:
		sec_epochtime = np.asarray([(dtt - reftime).total_seconds() for dtt in time_dt])
	except TypeError:	# then, time_dt is no array
		sec_epochtime = (time_dt - reftime).total_seconds()

	return sec_epochtime


def numpydatetime64_to_datetime(npdt_array):

	"""
	Converts numpy datetime64 array to a datetime object array.

	Parameters:
	-----------
	npdt_array : numpy array of type np.datetime64 or np.datetime64 type
		Array (1D) or directly a np.datetime64 type variable.
	"""

	sec_epochtime = npdt_array.astype(np.timedelta64) / np.timedelta64(1, 's')

	# sec_epochtime can be an array or just a float
	if sec_epochtime.ndim > 0:
		time_dt = np.asarray([dt.datetime.utcfromtimestamp(tt) for tt in sec_epochtime])

	else:
		time_dt = dt.datetime.utcfromtimestamp(sec_epochtime)

	return time_dt


def compute_DOY(
	time,
	return_dt=True,
	reshape=False):

	"""
	Compute the cos and sin of the day of the year for a given time.

	Parameters:
	-----------
	time : numpy array of floats or float
		Time data (must be in seconds since 1970-01-01 00:00:00 UTC) used to compute 
		the cos and sin of the day of the year.
	return_dt : bool
		If True the datetime object/array used for the computation is returned as well.
	reshape : bool
		If True an additional dimension of length 1 will be added to DOY_1 and DOY_2 via
		reshaping.
	"""

	time_dt = np.asarray([dt.datetime.utcfromtimestamp(ttt) for ttt in time])

	DOY = np.asarray([(ttt - dt.datetime(ttt.year,1,1)).days*2*np.pi/365 for ttt in time_dt])
	DOY_1 = np.cos(DOY)
	DOY_2 = np.sin(DOY)

	if reshape:
		n_data = len(time)
		DOY_1 = np.reshape(DOY_1, (n_data,1))
		DOY_2 = np.reshape(DOY_2, (n_data,1))

	if return_dt:
		return DOY_1, DOY_2, time_dt
	else:
		return DOY_1, DOY_2


def break_str_into_lines(
	le_string,
	n_max,
	split_at=' ',
	keep_split_char=False):

	"""
	Break a long strings into multiple lines if a certain number of chars may
	not be exceeded per line. String will be split into two lines if its length
	is > n_max but <= 2*n_max.

	Parameters:
	-----------
	le_string : str
		String that will be broken into several lines depending on n_max.
	n_max : int
		Max number of chars allowed in one line.
	split_at : str
		Character to look for where the string will be broken. Default: space ' '
	keep_split_char : bool
		If True, the split char indicated by split_at will not be removed (useful for "-" as split char).
		Default: False
	"""

	n_str = len(le_string)
	if n_str > n_max:
		# if string is > 2*n_max, then it has to be split into three lines, ...:
		n_lines = (n_str-1) // n_max		# // is flooring division

		# look though the string in backwards direction to find the first space before index n_max:
		le_string_bw = le_string[::-1]
		new_line_str = "\n"

		for k in range(n_lines):
			space_place = le_string_bw.find(split_at, n_str - (k+1)*n_max)
			if keep_split_char:
				le_string_bw = le_string_bw[:space_place].replace("\n","") + new_line_str + le_string_bw[space_place:]
			else:
				le_string_bw = le_string_bw[:space_place] + new_line_str + le_string_bw[space_place+1:]

		# reverse the string again
		le_string = le_string_bw[::-1]

	return le_string


def bin_to_dec(b_in):

	"""
	Converts a binary number given as string to normal decimal number (as integer).

	Parameters:
	-----------
	b_in : str
		String of a binary number that may either directly start with the
		binary number or start with "0b".
	"""

	d_out = 0		# output as decimal number (int or float)
	if "b" in b_in:
		b_in = b_in[b_in.find("b")+1:]	# actual bin number starts after "b"
	b_len = len(b_in)

	for ii, a in enumerate(b_in): d_out += int(a)*2**(b_len-ii-1)

	return d_out


def compute_retrieval_statistics(
	x_stuff,
	y_stuff,
	compute_stddev=False):

	"""
	Compute bias, RMSE and Pearson correlation coefficient (and optionally the standard deviation)
	from x and y data.

	Parameters:
	x_stuff : float or array of floats
		Data that is to be plotted on the x axis.
	y_stuff : float or array of floats
		Data that is to be plotted on the y axis.
	compute_stddev : bool
		If True, the standard deviation is computed (bias corrected RMSE).
	"""

	where_nonnan = np.argwhere(~np.isnan(y_stuff) & ~np.isnan(x_stuff)).flatten()
					# -> must be used to ignore nans in corrcoef
	stat_dict = {	'N': np.count_nonzero(~np.isnan(x_stuff) & ~np.isnan(y_stuff)),
					'bias': np.nanmean(y_stuff - x_stuff),
					'rmse': np.sqrt(np.nanmean((x_stuff - y_stuff)**2)),
					'R': np.corrcoef(x_stuff[where_nonnan], y_stuff[where_nonnan])[0,1]}

	if compute_stddev:
		stat_dict['stddev'] = np.sqrt(np.nanmean((x_stuff - (y_stuff - stat_dict['bias']))**2))

	return stat_dict


def compute_RMSE_profile(
	x,
	x_o,
	which_axis=0):

	"""
	Compute RMSE 'profile' of a i.e., (height x time)-matrix (e.g. temperature profile):
	RMSE(z_i) = sqrt(mean((x - x_o)^2, dims='time'))
	
	Parameters:
	-----------
	x : 2D array of numerical
		Data matrix whose deviation from a reference is desired.
	x_o : 2d array of numerical
		Data matrix of the reference.
	which_axis : int
		Indicator which axis is to be averaged over. For the RMSE profile, you would
		want to average over time!
	"""

	if which_axis not in [0, 1]:
		raise ValueError("'which_axis' must be either 0 or 1!")

	return np.sqrt(np.nanmean((x - x_o)**2, axis=which_axis))


def build_K_reg(
	y,
	order=1):

	"""
	Constructs the observation matrix typically used for regression retrievals where the
	rows indicate the samples (i.e., time series). The first column usually contains "1"
	only and the remaining columns contain observations in first and higher order.

	Parameters:
	-----------
	y : array of floats
		Observation vector. Must be a numpy array with M observations and N samples. The 
		shape must be N x M. (Also if M == 1, y must be a 2D array.)
	order : int
		Defines the order of the regression equation. Options: i.e., 1, 2, 3. Default:
		1
	"""

	n_obs = y.shape[1]		# == M
	n_samples = y.shape[0]	# == N

	assert y.shape == (n_samples,n_obs)

	# generate regression matrix K out of obs vector:
	K_reg = np.ones((n_samples, order*n_obs+1))
	K_reg[:,1:n_obs+1] = y

	if order > 1:
		for kk in range(order-1):
			jj = kk + 1
			K_reg[:,jj*n_obs+1:(jj+1)*n_obs+1] = y**(jj+1)

	return K_reg


def regression(
	x,
	y,
	order=1):
	
	"""
	Computes regression coefficients m_est to map observations y (i.e., brightness temperatures)
	to state variable x (i.e., temperature profile at one height level, or IWV). The regression
	order can also be specified.
	
	Parameters:
	-----------
	x : array of floats
		State variable vector. Must be a numpy array with N samples (N = training data size).
	y : array of floats
		Observation vector. Must be a numpy array with M observations (i.e., M frequencies) 
		and N samples. The shape must be N x M. (Also if M == 1, y must be a 2D array.)
	order : int
		Defines the order of the regression equation. Options: i.e., 1, 2, 3. Default:
		1
	"""

	# Generate matrix from observations:
	K_reg = build_K_reg(y, order)

	# compute m_est
	K_reg_T = K_reg.T
	m_est = np.linalg.inv(K_reg_T.dot(K_reg)).dot(K_reg_T).dot(x)

	return m_est


def Gband_double_side_band_average(
	TB,
	freqs,
	xarray_compatibility=False,
	freq_dim_name=""):

	"""
	Computes the double side band average of TBs that contain both
	sides of the G band absorption line. Returns either only the TBs
	or both the TBs and frequency labels with double side band avg.
	If xarray_compatibility is True, also more dimensional TB arrays
	can be included. Then, also the frequency dimension name must be
	supplied.

	Parameters:
	-----------
	TB : array of floats
		Brightness temperature array. Must have the following shape
		(time x frequency). More dimensions and other shapes are only
		allowed if xarray_compatibility=True.
	freqs : array of floats
		1D Array containing the frequencies of the TBs. The array must be
		sorted in ascending order.
	xarray_compatibility : bool
		If True, xarray utilities can be used, also allowing TBs of other
		shapes than (time x frequency). Then, also freq_dim_name must be
		provided.
	freq_dim_name : str
		Name of the xarray frequency dimension. Must be specified if 
		xarray_compatibility=True.
	"""

	if xarray_compatibility and not freq_dim_name:
		raise ValueError("Please specify 'freq_dim_name' when using the xarray compatible mode.")

	# Double side band average for G band if G band frequencies are available, which must first be clarified:
	# Determine, which frequencies are around the G band w.v. absorption line:
	g_upper_end = 183.31 + 15
	g_lower_end = 183.31 - 15
	g_freq = np.where((freqs > g_lower_end) & (freqs < g_upper_end))[0]
	non_g_freq = np.where(~((freqs > g_lower_end) & (freqs < g_upper_end)))[0]

	TB_dsba = copy.deepcopy(TB)

	if g_freq.size > 0: # G band within frequencies
		g_low = np.where((freqs <= 183.31) & (freqs >= g_lower_end))[0]
		g_high = np.where((freqs >= 183.31) & (freqs <= g_upper_end))[0]

		assert len(g_low) == len(g_high)
		if not xarray_compatibility:
			for jj in range(len(g_high)):
				TB_dsba[:,jj] = (TB[:,g_low[-1-jj]] + TB[:,g_high[jj]])/2.0

		else:
			for jj in range(len(g_high)):
				TB_dsba[{freq_dim_name: jj}] = (TB[{freq_dim_name: g_low[-1-jj]}] + TB[{freq_dim_name: g_high[jj]}])/2.0


	# Indices for sorting:
	idx_have = np.concatenate((g_high, non_g_freq), axis=0)
	idx_sorted = np.argsort(idx_have)

	# truncate and append the unedited frequencies (e.g. 243 and 340 GHz):
	if not xarray_compatibility:
		TB_dsba = TB_dsba[:,:len(g_low)]
		TB_dsba = np.concatenate((TB_dsba, TB[:,non_g_freq]), axis=1)

		# Now, the array just needs to be sorted correctly:
		TB_dsba = TB_dsba[:,idx_sorted]

		# define freq_dsba (usually, the upper side of the G band is then used as
		# frequency label:
		freq_dsba = np.concatenate((freqs[g_high], freqs[non_g_freq]))[idx_sorted]

	else:
		TB_dsba = TB_dsba[{freq_dim_name: slice(0,len(g_low))}]
		TB_dsba = xr.concat([TB_dsba, TB[{freq_dim_name: non_g_freq}]], dim=freq_dim_name)

		# Now, the array just needs to be sorted correctly:
		TB_dsba = TB_dsba[{freq_dim_name: idx_sorted}]

		# define freq_dsba (usually, the upper side of the G band is then used as
		# frequency label:
		freq_dsba = xr.concat([freqs[g_high], freqs[non_g_freq]], dim=freq_dim_name)[idx_sorted]


	return TB_dsba, freq_dsba


def Fband_double_side_band_average(
	TB,
	freqs,
	xarray_compatibility=False,
	freq_dim_name=""):

	"""
	Computes the double side band average of TBs that contain both
	sides of the F band absorption line. Returns either only the TBs
	or both the TBs and frequency labels with double side band avg.

	Parameters:
	-----------
	TB : array of floats
		Brightness temperature array. Must have the following shape
		(time x frequency).
	freqs : array of floats
		1D Array containing the frequencies of the TBs. The array must be
		sorted in ascending order.
	xarray_compatibility : bool
		If True, xarray utilities can be used, also allowing TBs of other
		shapes than (time x frequency). Then, also freq_dim_name must be
		provided.
	freq_dim_name : str
		Name of the xarray frequency dimension. Must be specified if 
		xarray_compatibility=True.
	"""

	if xarray_compatibility and not freq_dim_name:
		raise ValueError("Please specify 'freq_dim_name' when using the xarray compatible mode.")

	# Double side band average for F band if F band frequencies are available, which must first be clarified:
	# Determine, which frequencies are around the F band w.v. absorption line:
	upper_end = 118.75 + 10
	lower_end = 118.75 - 10
	f_freq = np.where((freqs > lower_end) & (freqs < upper_end))[0]
	non_f_freq = np.where(~((freqs > lower_end) & (freqs < upper_end)))[0]

	TB_dsba = copy.deepcopy(TB)
	
	if f_freq.size > 0: # F band within frequencies
		low = np.where((freqs <= 118.75) & (freqs >= lower_end))[0]
		high = np.where((freqs >= 118.75) & (freqs <= upper_end))[0]

		assert len(low) == len(high)
		if not xarray_compatibility:
			for jj in range(len(high)):
				TB_dsba[:,jj] = (TB[:,low[-1-jj]] + TB[:,high[jj]])/2.0

		else:
			for jj in range(len(high)):
				TB_dsba[{freq_dim_name: jj}] = (TB[{freq_dim_name: low[-1-jj]}] + TB[{freq_dim_name: high[jj]}])/2.0


	# Indices for sorting:
	idx_have = np.concatenate((high, non_f_freq), axis=0)
	idx_sorted = np.argsort(idx_have)

	# truncate and append the unedited frequencies (e.g. 243 and 340 GHz):
	if not xarray_compatibility:
		TB_dsba = TB_dsba[:,:len(low)]
		TB_dsba = np.concatenate((TB_dsba, TB[:,non_f_freq]), axis=1)

		# Now, the array just needs to be sorted correctly:
		TB_dsba = TB_dsba[:,idx_sorted]

		# define freq_dsba (usually, the upper side of the G band is then used as
		# frequency label:
		freq_dsba = np.concatenate((freqs[high], freqs[non_f_freq]))[idx_sorted]

	else:
		TB_dsba = TB_dsba[{freq_dim_name: slice(0,len(low))}]
		TB_dsba = xr.concat([TB_dsba, TB[{freq_dim_name: non_f_freq}]], dim=freq_dim_name)

		# Now, the array just needs to be sorted correctly:
		TB_dsba = TB_dsba[{freq_dim_name: idx_sorted}]

		# define freq_dsba (usually, the upper side of the G band is then used as
		# frequency label:
		freq_dsba = xr.concat([freqs[high], freqs[non_f_freq]], dim=freq_dim_name)[idx_sorted]

	return TB_dsba, freq_dsba


def select_MWR_channels(
	TB,
	freq,
	band,
	return_idx=0):

	"""
	This function selects certain frequencies (channels) of brightness temperatures (TBs)
	from a given set of TBs. The output will therefore be a subset of the input TBs. Single
	frequencies cannot be selected but only 'bands' (e.g. K band, V band, ...). Combinations
	are also possible.

	Parameters:
	-----------
	TB : array of floats
		2D array (i.e., time x freq; freq must be the second dimension) of TBs (in K).
	freq : array of floats
		1D array of frequencies (in GHz).
	band : str
		Specify the frequencies to be selected. Valid options:
		'K': 20-40 GHz, 'V': 50-60 GHz, 'W': 85-95 GHz, 'F': 110-130 GHz, 'G': 170-200 GHz,
		'243/340': 240-350 GHz
		Combinations are also possible: e.g. 'K+V+W' = 20-95 GHz
	return_idx : int
		If 0 the frq_idx list is not returned and merely TB and freq are returned.
		If 1 TB, freq, and frq_idx are returned. If 2 only frq_idx is returned.
	"""

	# define dict of band limits:
	band_lims = {	'K': [20, 40],
					'V': [50, 60],
					'W': [85, 95],
					'F': [110, 130],
					'G': [170, 200],
					'243/340': [240, 350]}

	# split band input:
	band_split = band.split('+')

	# cycle through all bands:
	frq_idx = list()
	for k, baba in enumerate(band_split):
		# find the right indices for the appropriate frequencies:
		frq_idx_temp = np.where((freq >= band_lims[baba][0]) & (freq <= band_lims[baba][1]))[0]
		for fit in frq_idx_temp: frq_idx.append(fit)

	# sort the list and select TBs:
	frq_idx = sorted(frq_idx)
	TB = TB[:, frq_idx]
	freq = freq[frq_idx]

	if return_idx == 0:
		return TB, freq

	elif return_idx == 1:
		return TB, freq, frq_idx

	elif return_idx == 2:
		return frq_idx

	else:
		raise ValueError("'return_idx' in function 'select_MWR_channels' must be an integer. Valid options: 0, 1, 2")


def filter_time(
	time_have,
	time_wanted,
	window=0,
	around=False):

	"""
	This function returns a mask (True, False) when the first argument (time_have) is in
	the range time_wanted:time_wanted+window (in seconds) (for around=False) or in the
	range time_wanted-window:time_wanted+window.

	It is important that time_have and time_wanted have the same units (e.g., seconds 
	since 1970-01-01 00:00:00 UTC). t_mask will be True when time_have and time_wanted
	overlap according to 'window' and 'around'. The overlap always includes the boundaries
	(e.g., time_have >= time_wanted & time_have <= time_wanted + window).

	Parameters:
	-----------
	time_have : 1D array of float or int
		Time array that should be masked so that you will know, when time_have overlaps
		with time_wanted.
	time_wanted : 1D array of float or int
		Time array around which 
	window : int or float
		Window in seconds around time_wanted (or merely from time_wanted until time_wanted
		+ window) that will be set True in the returned mask. If window = 0, the closest
		match will be used.
	around : bool
		If True, time_wanted - window : time_wanted + window is considered. If False,
		time_wanted : time_wanted + window is considered.
	"""

	if not isinstance(around, bool):
		return TypeError("Argument 'around' must be boolean.")

	# Initialise mask with False. Overlap with time_wanted will then be set True.
	have_shape = time_have.shape
	t_mask = np.full(have_shape, False)

	if window > 0:
		if around:	# search window is in both directions around time_wanted
			for tw in time_wanted:
				idx = np.where((time_have >= tw - window) & (time_have <= tw + window))[0]
				t_mask[idx] = True

		else:		# search window only in one direction
			for tw in time_wanted:
				idx = np.where((time_have >= tw) & (time_have <= tw + window))[0]
				t_mask[idx] = True

	else:	# window <= 0: use closest match; around = True or False doesn't matter here
		for tw in time_wanted:
			idx = np.argmin(np.abs(time_have - tw)).flatten()
			t_mask[idx] = True

	return t_mask


def find_files_daterange(
	all_files, 
	date_start_dt, 
	date_end_dt,
	idx):

	"""
	Filter from a given set of files the correct ones within the date range
	date_start_dt - date_end_dt (including start and end date).

	Parameters:
	-----------
	all_files : list of str
		List of str that includes all the files.
	date_start_dt : datetime object
		Start date as a datetime object.
	date_end_dt : datetime object
		End date as a datetime object.
	idx : list of int
		List of int where the one entry specifies the start and the second one
		the end of the date string in any all_files item (i.e., [-17,-9]).
	"""

	files = list()
	for pot_file in all_files:
		# check if file is within our date range:
		file_dt = dt.datetime.strptime(pot_file[idx[0]:idx[1]], "%Y%m%d")
		if (file_dt >= date_start_dt) & (file_dt <= date_end_dt):
			files.append(pot_file)

	return files


def vector_intersection_2d(
	A1,
	A2,
	B1,
	B2):

	"""
	Compute the intersection point between two 2D vectors (a: A1->A2 and b: B1->B2).
	a = A1 + nn*(A2 - A1)
	b = B1 + mm*(B2 - B1)
	Points A1 and A2 must not be identical. The same applies for B1 and B2.

	Parameters:
	-----------
	A1 : 2D array of float
		Origin of the first 2D vector a.
	A2 : 2D array of float
		Endpoint of the first 2D vector a.
	B1 : 2D array of float
		Origin of the second 2D vector b.
	B2 : 2D array of float
		Endpoint of the second 2D vector b.
	"""

	A1x = A1[0]
	A1y = A1[1]
	A2x = A2[0]
	A2y = A2[1]
	B1x = B1[0]
	B1y = B1[1]
	B2x = B2[0]
	B2y = B2[1]

	if A1x == A2x:
		aa = (B1x - A1x - (B1y - A1y)*(A2x - A1x) / (A2y - A1y))
		bb = ((B2y - B1y)*(A2x - A1x) / (A2y - A1y) - B2x + B1x)
		if bb == 0:
			mm = np.inf
		else:
			mm = aa / bb
		nn = (B1y - A1y + mm*(B2y - B1y)) / (A2y - A1y)

	else:
		aa = (B1y - A1y - (B1x - A1x)*(A2y - A1y) / (A2x - A1x))
		bb = ((B2x - B1x)*(A2y - A1y) / (A2x - A1x) - B2y + B1y)
		if bb == 0:
			mm = np.inf
		else:
			mm = aa / bb
		nn = (B1x - A1x + mm*(B2x - B1x)) / (A2x - A1x)

	return mm, nn


def filter_clear_sky_sondes_cloudnet(
	sonde_time,
	cn_time,
	cn_is_clear_sky,
	threshold=1.0,
	window=0):

	"""
	Find radiosonde launches that occur in clear sky scenes using cloudnet data. Before
	this function is run, the function cloudnet.clear_sky_only must have been executed
	already to provide a mask if time stamps are cloudy or not (cn_is_clear_sky).

	Only those sondes will be considered clear sky if all cloudnet time stamps in the
	time range 'sonde launch:sonde launch + window' are clear sky (cn_is_clear_sky = True).
	The function will return a mask (array of boolean type) indicating if a sonde fulfills
	the requirement.

	sonde_time and cn_time must have the same units!

	Parameters:
	-----------
	sonde_time : 1D array of int or float
		Launch time of radiosondes (preferably in sec since 1970-01-01 00:00:00 UTC).
	cn_time : 1D array of int or float
		Cloudnet time stamps (preferably in sec since 1970-01-01 00:00:00 UTC). Must have
		the same units as sonde_time!
	cn_is_clear_sky : 1D array of bool
		Cloudnet clear sky mask on the cloudnet time axis. Output of cloudnet.clear_sky_only.
	threshold : int
		Threshold of the fraction of cloudnet time stamps that must be 'clear sky' between
		sonde_time and sonde_time + window. Values between 0.95 and 1.0 are recommended.
		0.95 is more permitting (e.g. some more cloudnet noise can be neglected with that
		setting) while 1.0 requires exactly all cloudnet time stamps to be 'clear sky'.
		'clear sky' means that according to cloudnet target classification, it is clear sky,
		aerosols, insects or aerosols & insects. Additionally, some noisy pixels are filtered
		out (see my_classes.py -> class cloudnet -> clear_sky_only).
	window : int or float
		Time (in seconds) after sonde launch that is also considered for finding the clear
		sky sondes. Must be >= 0.
	"""

	if window < 0:
		raise ValueError("Argument 'window' must be >= 0 (int or float).")

	sonde_time_shape = sonde_time.shape
	t_mask = np.full(sonde_time_shape, False)	# initialise mask

	if np.any(cn_is_clear_sky):		# then we have at least got one clear sky scene
		for k, st in enumerate(sonde_time):
			c_idx = np.where((cn_time >= st) & (cn_time <= st + window))[0]
			n_c_idx = len(c_idx)
			if n_c_idx > 0:	# then it isn't empty: check, if all cn_time found are clear sky
				cn_all_clear_sky = np.count_nonzero(cn_is_clear_sky[c_idx])/n_c_idx >= threshold
				if cn_all_clear_sky:
					t_mask[k] = True

			else:
				cn_all_clear_sky = False

	return t_mask


def pam_out_drop_useless_dims(DS):

	"""
	Preprocessing the PAMTRA output file dataset before concatenation.
	Removing undesired dimensions (average over polarisation, use zenith only,
	remove y dimension). Additionally, it adds another variable (DataArray)
	containing the datatime in sec since epochtime.

	Parameters:
	-----------
	ds : xarray dataset
		Dataset of the PAMTRA output.
	"""

	# Zenith only ("-1"): PAMTRA angles is relative to ZENITH
	# Average over polarisation (".mean(axis=-1)")
	# Remove redundant dimensions: ("0,0")
	DS['tb'] = DS.tb[:,0,0,-1,:,:].mean(axis=-1)

	# And add a variable giving the datatime in seconds since 1970-01-01 00:00:00 UTC
	# along dimension grid_x:
	DS['time'] = xr.DataArray(numpydatetime64_to_epochtime(DS.datatime[:,0].values),
								dims=['grid_x'])

	return DS


def syn_MWR_cut_useless_variables_TB(DS):

	"""
	Preprocessing the synthetic TB data simulated typically with IDL STP
	to be applied to MWR_PRO retrievals before concatenation.
	Removing undesired dimensions and variables. 
	Also reduce to zenith observation only (elevation angle == 90) and
	remove n_cloud_model dimension.

	Parameters:
	-----------
	DS : xarray dataset
		Dataset of TB data.


	Remaining variables are:
		station_id
		elevation_angle
		date
		brightness_temperatures
		frequency

	Remaining dimensions:
		n_date
		string3
		n_angle				# will also be removed
		n_cloud_model		# will also be removed
		n_frequency
	"""

	# useless_vars = ['height_above_sea_level', 'data_path_to_rs', 'rh_thres', 'cap_height_above_sea_level', 
					# 'frequency_orig', 'cloud_model', 'cscl', 'gas_absorption_model', 
					# 'cloud_absorption_model', 'linewidth', 'cont_corr', 
					# 'air_mass_correction', 'number_of_levels_in_rs_ascent', 'highest_level_in_rs_ascent', 
					# 'k_index', 'ko_index', 'tt_index', 'li_index', 
					# 'si_index', 'cape_mu', 'wet_delay', 'liquid_water_path',
					# 'cloud_base', 'cloud_top', 'liquid_water_path_single_cloud',
					# 'optical_depth', 'optical_depth_wv', 'optical_depth_o2', 'optical_depth_liq_l91', 
					# 'aperture_correction', 'bandwidth_correction', 'brightness_temperatures_orig', 
					# 'brightness_temperatures_instrument', 'mean_radiating_temperature',
					# 'height_grid', 'atmosphere_temperature', 'atmosphere_humidity',
					# 'atmosphere_pressure', 'atmosphere_temperature_sfc', 'atmosphere_humidity_sfc',
					# 'atmosphere_pressure_sfc', 'integrated_water_vapor']

	useless_vars = ['height_above_sea_level', 'data_path_to_rs', 'rh_thres', 'cap_height_above_sea_level', 
					'frequency_orig', 'cloud_model', 'cscl', 'gas_absorption_model', 
					'cloud_absorption_model', 'linewidth', 'cont_corr', 
					'air_mass_correction', 'number_of_levels_in_rs_ascent', 'highest_level_in_rs_ascent', 
					'k_index', 'ko_index', 'tt_index', 'li_index', 
					'si_index', 'cape_mu', 'wet_delay', 'liquid_water_path',
					'cloud_base', 'cloud_top', 'liquid_water_path_single_cloud',
					'optical_depth', 'optical_depth_wv', 'optical_depth_o2', 'optical_depth_liq_l91', 
					'aperture_correction', 'bandwidth_correction', 'brightness_temperatures_orig', 
					'brightness_temperatures_instrument', 'mean_radiating_temperature',
					'atmosphere_temperature', 'atmosphere_humidity',
					'atmosphere_temperature_sfc', 'atmosphere_humidity_sfc',
					'integrated_water_vapor']

	# check if the variable names exist in DS:
	for idx, var in enumerate(useless_vars):
		if not var in DS:
			useless_vars.pop(idx)

	DS = DS.drop_vars(useless_vars)

	el_idx = np.where(DS.elevation_angle.values == 90.0)[0][0]
	DS['elevation_angle'] = DS.elevation_angle[el_idx]
	DS['brightness_temperatures'] = DS.brightness_temperatures[:,0,el_idx,:]

	return DS


def syn_MWR_cut_useless_variables_RS(DS):
	"""
	Preprocessing radiosonde data to be applied to MWR_PRO retrievals before concatenation.
	Removing undesired dimensions and variables. 

	Parameters:
	-----------
	DS : xarray dataset
		Dataset of TB data.


	Remaining variables are:
		station_id
		height_grid
		date
		atmosphere_temperature
		atmosphere_humidity
		atmosphere_pressure
		atmosphere_temperature_sfc
		atmosphere_humidity_sfc
		atmosphere_pressure_sfc
		integrated_water_vapor

	Remaining dimensions:
		n_date
		string3
		n_height
	"""

	useless_vars = ['height_above_sea_level', 'data_path_to_rs', 'rh_thres', 'cap_height_above_sea_level', 
					'frequency_orig', 'cloud_model', 'cscl', 'gas_absorption_model', 
					'cloud_absorption_model', 'linewidth', 'cont_corr', 
					'air_mass_correction', 'number_of_levels_in_rs_ascent', 'highest_level_in_rs_ascent', 
					'k_index', 'ko_index', 'tt_index', 'li_index', 
					'si_index', 'cape_mu', 'wet_delay', 
					'cloud_base', 'cloud_top', 'liquid_water_path_single_cloud',
					'optical_depth', 'optical_depth_wv', 'optical_depth_o2', 'optical_depth_liq_l91', 
					'aperture_correction', 'bandwidth_correction', 'brightness_temperatures_orig', 
					'brightness_temperatures_instrument', 'mean_radiating_temperature',
					'elevation_angle', 'brightness_temperatures', 'frequency']

	# check if the variable names exist in DS:
	for idx, var in enumerate(useless_vars):
		if not var in DS:
			useless_vars.pop(idx)

	DS = DS.drop_vars(useless_vars)

	return DS


def sigmoid(x):
	
	"""
	Compute sigmoid of x.

	Parameters:
	-----------
	x : float or array of floats
		Input vector, array or number.
	"""
	return 1 / (1 + np.exp(-x))


def save_concat_IWV(
	file_save,
	mwr_dict,
	master_time,
	instrument):

	"""
	Saves concatenated IWV (over a date range) of a 
	microwave radiometer (which one to be specified in 'instrument') to a netcdf file.

	Parameters:
	-----------
	file_save : str
		Path and filename where the concatenated IWV is to be saved to.
	mwr_dict : dictionary
		Dictionary of a microwave radiometer containing time and IWV arrays.
	master_time : array of float
		Array that contains the time from 2019-09-30 through 2020-10-02 in 
		seconds since 1970-01-01 00:00:00 UTC.
	instrument : str
		Specifies which instrument is mwr_dict. Can only be 'hatpro', 'mirac' or 'arm'.
	"""

	if instrument not in ['hatpro', 'mirac', 'arm']:
		raise ValueError("'instrument' must be either 'hatpro', 'mirac' or 'arm'.")

	# create Dataset:
	MWR_DS = xr.Dataset({
		'IWV': 		(['time'], mwr_dict['IWV_master'],
					{'description': "Integrated water vapor retrieved from microwave radiometer",
					'units': "kg m^-2"})},
		coords =	{'time': (['time'], master_time,
					{'description': "Time stamp",
					'units': "seconds since 1970-01-01 00:00:00 UTC"})})

	# Set global attributes:
	MWR_DS.attrs['description'] = ("Concatenated IWV over the period " + 
		dt.datetime.utcfromtimestamp(master_time[0]).strftime("%Y-%m-%d") + " through " + 
		dt.datetime.utcfromtimestamp(master_time[-1]).strftime("%Y-%m-%d"))
	MWR_DS.attrs['instrument'] = instrument
	MWR_DS.attrs['author'] = "Andreas Walbröl, a.walbroel@uni-koeln.de"
	MWR_DS.attrs['history'] = "Created on: " + dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

	MWR_DS.to_netcdf(file_save, mode='w', format="NETCDF4")
	MWR_DS.close()


def save_concat_LWP(
	file_save,
	mwr_dict,
	master_time,
	instrument):

	"""
	Saves concatenated LWP (over a date range) of a 
	microwave radiometer (which one to be specified in 'instrument') to a netcdf file.

	Parameters:
	-----------
	file_save : str
		Path and filename where the concatenated LWP is to be saved to.
	mwr_dict : dictionary
		Dictionary of a microwave radiometer containing time and LWP arrays.
	master_time : array of float
		Array that contains the time from e.g. 2019-09-30 through 2020-10-02 in 
		seconds since 1970-01-01 00:00:00 UTC.
	instrument : str
		Specifies which instrument is mwr_dict. Can only be 'hatpro', 'mirac' or 'arm'.
	"""

	if instrument not in ['hatpro', 'mirac', 'arm']:
		raise ValueError("'instrument' must be either 'hatpro', 'mirac' or 'arm'.")

	# create Dataset:
	MWR_DS = xr.Dataset({
		'LWP': 		(['time'], mwr_dict['LWP_master'],
					{'description': "Liquid water path retrieved from microwave radiometer",
					'units': "kg m^-2"})},
		coords =	{'time': (['time'], master_time,
					{'description': "Time stamp",
					'units': "seconds since 1970-01-01 00:00:00 UTC"})})

	# Set global attributes:
	MWR_DS.attrs['description'] = ("Concatenated LWP over the period " + 
		dt.datetime.utcfromtimestamp(master_time[0]).strftime("%Y-%m-%d") + " through " + 
		dt.datetime.utcfromtimestamp(master_time[-1]).strftime("%Y-%m-%d"))
	MWR_DS.attrs['instrument'] = instrument
	MWR_DS.attrs['author'] = "Andreas Walbröl, a.walbroel@uni-koeln.de"
	MWR_DS.attrs['history'] = "Created on: " + dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

	MWR_DS.to_netcdf(file_save, mode='w', format="NETCDF4")
	MWR_DS.close()


def save_IWV_running_mean(
	file_save,
	mwr_dict,
	rm_window,
	instrument):

	"""
	Saves concatenated IWV, to which a running mean has been applied, on the instrument
	specific time axis (over the date range 2019-10-22 through 2020-10-02) of a 
	microwave radiometer (specified in 'instrument') to a netcdf file.

	Parameters:
	-----------
	file_save : str
		Path and filename where the concatenated IWV is to be saved to.
	mwr_dict : dictionary
		Dictionary of a microwave radiometer containing time and IWV arrays.
	rm_window : int
		Range of the moving average / running mean in seconds.
	instrument : str
		Specifies which instrument is mwr_dict. Can only be 'hatpro', 'mirac' or 'arm'.
	"""

	if instrument not in ['hatpro', 'mirac', 'arm']:
		raise ValueError("'instrument' must be either 'hatpro', 'mirac' or 'arm'.")

	elif instrument in ['hatpro', 'mirac']:
		mwr_dict['IWV'] = mwr_dict['prw']
	elif instrument == 'arm':
		mwr_dict['IWV'] = mwr_dict['prw']
		mwr_dict['flag'] = mwr_dict['prw_flag']

	if type(rm_window) != type(1):
		raise TypeError("'rm_window' must be an integer representing the number of seconds, " +
			"over which the running average is applied.")

	# create Dataset:
	MWR_DS = xr.Dataset({
		'IWV':		(['time'], mwr_dict['IWV'],
					{'description': "Integrated water vapor retrieved from microwave radiometer",
					'units': "kg m-2"}),
		'flag':		(['time'], mwr_dict['flag'],
					{'description': "Flag values that are greater than 0 for faulty measurements"}),
		'rm_window':([], rm_window,
					{'description': ("Range of the moving average (running mean): time-rm_window " +
									"until time+rm_window considered for averaging"),
					'units': "seconds"})},
		coords = 	{'time': (['time'], mwr_dict['time'],
					{'description': "Instrument specific time stamp (may have jumps in the time series)",
					'units': "seconds since 1970-01-01 00:00:00 UTC"})})

	# Set global attributes:
	MWR_DS.attrs['description'] = ("Concatenated IWV with running mean over the period " + 
		dt.datetime.utcfromtimestamp(mwr_dict['time'][0]).strftime("%Y-%m-%d") + " through " + 
		dt.datetime.utcfromtimestamp(mwr_dict['time'][-1]).strftime("%Y-%m-%d") +
		", with instrument specific time axis.")
	MWR_DS.attrs['instrument'] = instrument
	MWR_DS.attrs['running_mean_window'] = rm_window
	MWR_DS.attrs['author'] = "Andreas Walbröl, a.walbroel@uni-koeln.de"
	MWR_DS.attrs['history'] = "Created on: " + dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

	MWR_DS.to_netcdf(file_save, mode='w', format="NETCDF4")
	MWR_DS.close()


def save_LWP_running_mean(
	file_save,
	mwr_dict,
	rm_window,
	instrument):

	"""
	Saves concatenated LWP, to which a running mean has been applied, on the instrument
	specific time axis (over the date range 2019-10-22 through 2020-10-02) of a 
	microwave radiometer (specified in 'instrument') to a netcdf file.

	Parameters:
	-----------
	file_save : str
		Path and filename where the concatenated LWP is to be saved to.
	mwr_dict : dictionary
		Dictionary of a microwave radiometer containing time and LWP arrays.
	rm_window : int
		Range of the moving average / running mean in seconds.
	instrument : str
		Specifies which instrument is mwr_dict. Can only be 'hatpro', 'mirac' or 'arm'.
	"""

	if instrument not in ['hatpro', 'mirac', 'arm']:
		raise ValueError("'instrument' must be either 'hatpro', 'mirac' or 'arm'.")

	elif instrument in ['hatpro', 'mirac']:
		mwr_dict['LWP'] = mwr_dict['clwvi']
	elif instrument == 'arm':
		mwr_dict['LWP'] = mwr_dict['lwp']
		mwr_dict['flag'] = mwr_dict['lwp_flag']

	if type(rm_window) != type(1):
		raise TypeError("'rm_window' must be an integer representing the number of seconds, " +
			"over which the running average is applied.")

	# create Dataset:
	MWR_DS = xr.Dataset({
		'LWP':		(['time'], mwr_dict['LWP'],
					{'description': "Liquid water path retrieved from microwave radiometer",
					'units': "kg m-2"}),
		'flag':		(['time'], mwr_dict['flag'],
					{'description': "Flag values that are greater than 0 for faulty measurements"}),
		'rm_window':([], rm_window,
					{'description': ("Range of the moving average (running mean): time-rm_window " +
									"until time+rm_window considered for averaging"),
					'units': "seconds"})},
		coords = 	{'time': (['time'], mwr_dict['time'],
					{'description': "Instrument specific time stamp (may have jumps in the time series)",
					'units': "seconds since 1970-01-01 00:00:00 UTC"})})

	# Set global attributes:
	MWR_DS.attrs['description'] = ("Concatenated LWP with running mean over the period " + 
		dt.datetime.utcfromtimestamp(mwr_dict['time'][0]).strftime("%Y-%m-%d") + " through " + 
		dt.datetime.utcfromtimestamp(mwr_dict['time'][-1]).strftime("%Y-%m-%d") +
		", with instrument specific time axis.")
	MWR_DS.attrs['instrument'] = instrument
	MWR_DS.attrs['running_mean_window'] = rm_window
	MWR_DS.attrs['author'] = "Andreas Walbröl, a.walbroel@uni-koeln.de"
	MWR_DS.attrs['history'] = "Created on: " + dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

	MWR_DS.to_netcdf(file_save, mode='w', format="NETCDF4")
	MWR_DS.close()


def create_IWV_concat_master_time(
	date_start,
	date_end,
	mwr_dict,
	instrument,
	path_save,
	IWV_concat_mwr_filename):

	"""
	Create a master time axis that spans the MOSAiC measurement period (date_start - date_end)
	of HATPRO, MiRAC-P and ARM MWRs onboard Polarstern with 1 second spacing. At time stamps when the
	respective instrument has recorded an observation, the value (IWV) is saved if the flag
	value is 0 (=measurement okay). At the end, the IWV values of an instrument on the new
	time axis will be saved to a netcdf file for each instrument.

	Parameters:
	-----------
	date_start : str
		Marks the first day of the desired period. To be specified in yyyy-mm-dd (e.g. 2019-09-30)!
	date_end : str
		Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2020-10-02)!
	mwr : dictionary
		Dictionary of microwave radiometer containing 'time', IWV ('prw') and 'flag' arrays.
	instrument : str
		Specifies which instrument is mwr_dict. Can only be 'hatpro', 'mirac', 'arm'.
	path_save : str
		Path where the concatenated IWV is to be saved to.
	IWV_concat_mwr_filename : str
		Filename of the concatenated IWV data for one of the MWRs.
	"""

	# equalise variable names:
	if instrument == 'hatpro':
		mwr_dict['IWV'] = mwr_dict['prw']
	elif instrument == 'arm':
		mwr_dict['IWV'] = mwr_dict['prw']
		mwr_dict['flag'] = mwr_dict['prw_flag']
	elif instrument == 'mirac':
		mwr_dict['IWV'] = mwr_dict['prw']
	else:
		raise ValueError("'instrument' must be either 'hatpro', 'mirac' or 'arm'.")

	master_time = np.arange((dt.datetime.strptime(date_start, "%Y-%m-%d") - dt.datetime(1970,1,1)).total_seconds(),
						(dt.datetime.strptime(date_end, "%Y-%m-%d") - dt.datetime(1969,12,31)).total_seconds())
					# --> It is intended to have datetime(1969,12,31,0,0,0) because the master time axis
					# is supposed to end at the end of the day specified in date_end, not right before that
					# day begins!
	mwr_dict['IWV_master'] = np.full_like(master_time, np.nan)	# iwv values on master time axis
	mwrtime_save = 0		# variable to save where mwr time equalled master time (speeds up the code)
	look_range = 8	# how many indices around mwrtime_save will be looked
	n_time_master = len(master_time)
	for idx, m_time in enumerate(master_time):

		if idx % 500000 == 0: print(idx/n_time_master)

		# check if mwr time equals m_time: mwr_check will contain the index where mwr time equals m_time
		if (mwrtime_save > look_range) & (mwrtime_save < n_time_master - look_range):
			mwr_check = np.argwhere(mwr_dict['time'][mwrtime_save-look_range:mwrtime_save+look_range] == m_time).flatten() + (mwrtime_save-look_range)
		elif mwrtime_save <= look_range:	# lower end of range
			mwr_check = np.argwhere(mwr_dict['time'][:mwrtime_save+look_range] == m_time).flatten()
		else:	# case: mwrtime_save >= n_time_master - look_range: upper end of range
			mwr_check = np.argwhere(mwr_dict['time'][mwrtime_save-look_range:] == m_time).flatten() + (mwrtime_save-look_range)

		if mwr_check.size > 0:
			mwrtime_save = mwr_check[0]
			if np.all(mwr_dict['flag'][mwr_check] == 0):
				mwr_dict['IWV_master'][idx] = np.nanmean(mwr_dict['IWV'][mwr_check])


	save_concat_IWV(path_save + IWV_concat_mwr_filename, mwr_dict, master_time, instrument)


def create_LWP_concat_master_time(
	date_start,
	date_end,
	mwr_dict,
	instrument,
	path_save,
	LWP_concat_mwr_filename):

	"""
	Create a master time axis that spans the MOSAiC measurement period (date_start - date_end)
	of HATPRO, MiRAC-P and ARM MWRs onboard Polarstern with 1 second spacing. At time stamps when the
	respective instrument has recorded an observation, the value (LWP) is saved if the flag
	value is 0 (=measurement okay). At the end, the LWP values of an instrument on the new
	time axis will be saved to a netcdf file for each instrument.

	Parameters:
	-----------
	date_start : str
		Marks the first day of the desired period. To be specified in yyyy-mm-dd (e.g. 2019-09-30)!
	date_end : str
		Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2020-10-02)!
	mwr : dictionary
		Dictionary of microwave radiometer containing 'time', LWP ('clwvi') and 'flag' arrays.
	instrument : str
		Specifies which instrument is mwr_dict. Can only be 'hatpro', 'mirac', 'arm'.
	path_save : str
		Path where the concatenated LWP is to be saved to.
	LWP_concat_mwr_filename : str
		Filename of the concatenated LWP data for one of the MWRs.
	"""

	# equalise variable names:
	if instrument == 'hatpro':
		mwr_dict['LWP'] = mwr_dict['clwvi']
	elif instrument == 'arm':
		mwr_dict['LWP'] = mwr_dict['lwp']
		mwr_dict['flag'] = mwr_dict['lwp_flag']
	elif instrument == 'mirac':
		mwr_dict['flag'] = mwr_dict['RF']
	else:
		raise ValueError("'instrument' must be either 'hatpro', 'mirac' or 'arm'.")

	master_time = np.arange((dt.datetime.strptime(date_start, "%Y-%m-%d") - dt.datetime(1970,1,1)).total_seconds(),
						(dt.datetime.strptime(date_end, "%Y-%m-%d") - dt.datetime(1969,12,31)).total_seconds())
					# --> It is intended to have datetime(1969,12,31,0,0,0) because the master time axis
					# is supposed to end at the end of the day specified in date_end, not right before that
					# day begins!
	mwr_dict['LWP_master'] = np.full_like(master_time, np.nan)	# LWP values on master time axis
	mwrtime_save = 0		# variable to save where mwr time equalled master time (speeds up the code)
	look_range = 8	# how many indices around mwrtime_save will be looked
	n_time_master = len(master_time)
	for idx, m_time in enumerate(master_time):

		if idx % 500000 == 0: print(idx/n_time_master)

		# check if mwr time equals m_time: mwr_check will contain the index where mwr time equals m_time
		if (mwrtime_save > look_range) & (mwrtime_save < n_time_master - look_range):
			mwr_check = np.argwhere(mwr_dict['time'][mwrtime_save-look_range:mwrtime_save+look_range] == m_time).flatten() + (mwrtime_save-look_range)
		elif mwrtime_save <= look_range:	# lower end of range
			mwr_check = np.argwhere(mwr_dict['time'][:mwrtime_save+look_range] == m_time).flatten()
		else:	# case: mwrtime_save >= n_time_master - look_range: upper end of range
			mwr_check = np.argwhere(mwr_dict['time'][mwrtime_save-look_range:] == m_time).flatten() + (mwrtime_save-look_range)

		if mwr_check.size > 0:
			mwrtime_save = mwr_check[0]
			if np.all(mwr_dict['flag'][mwr_check] == 0):
				mwr_dict['LWP_master'][idx] = np.nanmean(mwr_dict['LWP'][mwr_check])


	save_concat_LWP(path_save + LWP_concat_mwr_filename, mwr_dict, master_time, instrument)


def save_PS_mastertrack_as_nc(
	export_file,
	pstrack_dict,
	attribute_info):

	"""
	Saves Polarstern master track during MOSAiC to a netCDF4 file.

	Parameters:
	-----------
	export_file : str
		Path where the file is to be saved to and filename.
	pstrack_dict : dict
		Dictionary that contains the Polarstern track information.
	attribute_info : dict
		Dictionary that contains global attributes found in the .tab header.
	"""

	PS_DS = xr.Dataset({'Latitude': 	(['time'], pstrack_dict['Latitude'],
										{'units': "deg N"}),
						'Longitude':	(['time'], pstrack_dict['Longitude'],
										{'units': "deg E"}),
						'Speed':		(['time'], pstrack_dict['Speed'],
										{'description': "Cruise speed",
										'units': "knots"}),
						'Course':		(['time'], pstrack_dict['Course'],
										{'description': "Cruise heading",
										'units': "deg"})},
						coords = 		{'time': (['time'], pstrack_dict['time'],
										{'description': "Time stamp or seconds since 1970-01-01 00:00:00 UTC"})})

	# Set global attributes:
	for attt in attribute_info:
		if (":" in attt[0]) & (len(attt) > 1):
			PS_DS.attrs[attt[0].replace(":","")] = attt[1]
	PS_DS.attrs['Author_of_netCDF'] = "Andreas Walbroel, a.walbroel@uni-koeln.de"

	# encode time:
	encoding = {'time': dict()}
	encoding['time']['dtype'] = 'int64'
	encoding['time']['units'] = 'seconds since 1970-01-01 00:00:00'

	PS_DS.to_netcdf(export_file, mode='w', format="NETCDF4", encoding=encoding)
	PS_DS.close()


def save_MOSAiC_Radiosondes_PS122_Level2_as_nc(
	export_file,
	rs_dict,
	attribute_info):

	"""
	Saves single MOSAiC Polarstern Level 2 Radiosonde to a netCDF4 file.

	Parameters:
	-----------
	export_file : str
		Path and filename to which the file is to be saved to.
	rs_dict : dict
		Dictionary that contains the radiosonde information.
	attribute_info : dict
		Dictionary that contains global attributes found in the .tab header.
	"""

	RS_DS = xr.Dataset({'Latitude': 	(['time'], rs_dict['Latitude'],
										{'units': "deg N"}),
						'Longitude':	(['time'], rs_dict['Longitude'],
										{'units': "deg E"}),
						'Altitude':		(['time'], rs_dict['Altitude'],
										{'description': "Altitude",
										'units': "m"}),
						'h_geom':		(['time'], rs_dict['h_geom'],
										{'description': "Geometric Height",
										'units': "m"}),
						'ETIM':			(['time'], rs_dict['ETIM'],
										{'description': "Elapsed time since sonde start"}),
						'P':			(['time'], rs_dict['P'],
										{'description': "hPa",
										'units': "deg"}),
						'T':			(['time'], rs_dict['T'],
										{'description': "Temperature",
										'units': "deg C"}),
						'RH':			(['time'], rs_dict['RH'],
										{'description': "Relative humidity",
										'units': "percent"}),
						'wdir':			(['time'], rs_dict['wdir'],
										{'description': "Wind direction",
										'units': "deg"}),
						'wspeed':		(['time'], rs_dict['wspeed'],
										{'description': "Wind speed",
										'units': "m s^-1"}),
						'q':			(['time'], rs_dict['q'],
										{'description': "Specific humidity",
										'conversion': "Saturation water vapour pressure based on Hyland and Wexler, 1983.",
										'units': "kg kg^-1"}),
						'rho_v':		(['time'], rs_dict['rho_v'],
										{'description': "Absolute humidity",
										'conversion': "Saturation water vapour pressure based on Hyland and Wexler, 1983.",
										'units': "kg m^-3"}),
						'IWV':			([], rs_dict['IWV'],
										{'description': "Integrated Water Vapour",
										'calculation': ("Integration of (specific humidity x pressure). " +
														"Humidity conversion based on Hyland and Wexler, 1983. "),
										'further_comment': ("IWV computation function checks if pressure truely " +
															"decreases with increasing time since sonde start."),
										'units': "kg m^-2"})},
						coords = 		{'time': (['time'], rs_dict['time_sec'],
										{'description': "Time stamp or seconds since 1970-01-01 00:00:00 UTC",
										'units': "seconds since 1970-01-01 00:00:00 UTC"})})

	# Set global attributes:
	for attt in attribute_info:
		if (":" in attt[0]) & (len(attt) > 1):
			RS_DS.attrs[attt[0].replace(":","")] = attt[1]
	RS_DS.attrs['Author_of_netCDF'] = "Andreas Walbroel, a.walbroel@uni-koeln.de"

	# encode time:
	RS_DS['time'].encoding['units'] = 'seconds since 1970-01-01 00:00:00'
	RS_DS['time'].encoding['dtype'] = 'double'

	RS_DS.to_netcdf(export_file, mode='w', format="NETCDF4")
	RS_DS.close()


def write_polarstern_track_data_into_mwr_pro_output(
	path_mwr_pro_output,
	ps_track_dict,
	date_start,
	date_end,
	instrument,
	level,
	verbose=0):

	"""
	Imports output of the mwr_pro program (level 1 and level 2) on a daily basis. When a day with data
	is found, the data is loaded into an xarray. Then, the Polarstern track information is interpolated
	on the time grid of that day and inserted into the respective lat and lon variables of the xarray
	dataset. Finally, the xarray is saved as netCDF without touching anything else!

	Parameters:
	-----------
	path_mwr_pro_output : str
		Base path of level 1 or 2 data. This directory contains subfolders representing the year, which,
		in turn, contain months, which contain day subfolders. Example path_mwr_pro_output =
		"/data/obs/campaigns/mosaic/hatpro/l2/"
	ps_track_dict : dict
		This dictionary contains latitude, longitude and time information of the Polarstern track during
		MOSAiC as numpy arrays.
	date_start : str
		Marks the first day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
	date_end : str
		Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
	instrument : str
		Specifies which instrument is considered. Can only be 'hatpro', 'mirac', 'arm'.
	level : str
		Specifies which mwr_pro output level is considered. Can only be 'level_1' or 'level_2'.
	verbose : bool
	If 0, output is suppressed. If 1, basic output is printed. If 2, more output (more warnings,...)
		is printed.
	"""

	# The following variables are allowed to have the _FillValue attribute:
	exclude_variables_fill_value = ["azi", "ele", "ele_ret", "prw", "prw_offset", "prw_off_zenith", 
									"prw_off_zenith_offset", "prw_err", "flag", "clwvi", "clwvi_off_zenith",
									"clwvi_err", "clwvi_offset_zeroing", "clwvi_offset", "clwvi_off_zenith_offset",
									"height", "hua", "hua_offset", "hua_err", "ta", "ta_offset", "ta_err",
									"tb", "tb_bias_estimate", "freq_shift", "tb_absolute_accuracy", "tb_cov", 
									"tb_irp", "ele_irp", "pa", "hur"]

	# check for correct input parameters:
	if instrument == 'mirac':
		raise ValueError("'write_polarstern_track_data_into_mwr_pro_output' has not yet been coded for MiRAC-P data!")
	elif instrument == 'arm':
		raise ValueError("'write_polarstern_track_data_into_mwr_pro_output' has not yet been coded for ARM MWR data!")
	elif instrument not in ['hatpro', 'mirac', 'arm']:
		raise ValueError("'instrument' must be either 'hatpro', 'mirac' or 'arm'.")

	if level not in ['level_1', 'level_2']: raise ValueError("'level' must be either 'level_1' or 'level_2'.")

	if not isinstance(instrument, str): raise TypeError("'instrument' must be a string (either 'hatpro', 'mirac' or 'arm').")
	if not isinstance(level, str): raise TypeError("'level' must be a string (either 'level_1' or 'level_2').")


	# extract day, month and year from start date:
	date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
	date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")

	# number of days:
	n_days = (date_end - date_start).days + 1

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
	

	# distinguish between different levels and instruments:
	if instrument == 'hatpro':

		# identify the correct level 1 files (v00)
		# cycle through all years, all months and days:
		for now_date in (date_start + dt.timedelta(days=n) for n in range(n_days)):

			if verbose >= 1: print("Working on %s '%s', "%(instrument, level), now_date)

			yyyy = now_date.year
			mm = now_date.month
			dd = now_date.day

			day_path = path_mwr_pro_output + "%04i/%02i/%02i/"%(yyyy,mm,dd)

			if not os.path.exists(os.path.dirname(day_path)):
				continue

			# list of files:
			mwr_pro_output_files_nc = sorted(glob.glob(day_path + "ioppol_tro*_v00_*.nc"))					# different for different levels and instruments

			# day_path directory might be empty:
			if len(mwr_pro_output_files_nc) == 0:
				if verbose >= 2:
					warnings.warn("No netcdf files found for date %04i-%02i-%02i."%(yyyy,mm,dd))
				continue

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


			# load one mwr_pro output file after the other on the current day:
			for output_file in mwr_pro_output_files_nc:
				# _test.nc and _v01_ should already have lat, lon from Polarstern
				if ("_test.nc" in output_file) or ("_v01_" in output_file): continue

				MWR_DS = xr.open_dataset(output_file, decode_times=False)

				# interpolate Polarstern track data on mwr data time axis:
				for ps_key in ['lat', 'lon', 'time']:
					ps_track_dict[ps_key + "_ip"] = np.interp(np.rint(MWR_DS.time.values), ps_track_dict['time'], ps_track_dict[ps_key])

				MWR_DS['zsl'] = xr.DataArray(np.full_like(MWR_DS.time.values, 15.0).astype(np.float32), coords=MWR_DS.time.coords, dims=['time'])
				MWR_DS['zsl'].attrs['units'] = "m"
				MWR_DS['zsl'].attrs['standard_name'] = "altitude"
				MWR_DS['zsl'].attrs['long_name'] = "altitude above mean sea level"

				# lat, lon as dtype=dtype('float32') into MWR_DS while preserving and updating encoding and attributes:
				for geo_key in ['lat', 'lon']:
					original_encoding = MWR_DS[geo_key].encoding
					original_attrs = MWR_DS[geo_key].attrs
					MWR_DS[geo_key] = xr.DataArray(np.float32(ps_track_dict[geo_key + '_ip']),
													coords = MWR_DS.time.coords)

					# update encoding
					for enc_key in original_encoding.keys():
						if enc_key not in ['original_shape']:
							MWR_DS[geo_key].encoding[enc_key] = original_encoding[enc_key]
						else:
							MWR_DS[geo_key].encoding[enc_key] = MWR_DS.time.encoding[enc_key]
					
					# update / preserve attributes:
					for att_key in original_attrs.keys():
						MWR_DS[geo_key].attrs[att_key] = original_attrs[att_key]
				

				# Make sure that _FillValue is not added to more variables than before:
				for varr in MWR_DS.variables:
					if varr not in exclude_variables_fill_value:
						MWR_DS[varr].encoding["_FillValue"] = None

				# save to netcdf again:
				outfile = output_file.replace('_v00_', '_v01_')

				# add another global attribute:
				MWR_DS.attrs['Position_source'] = globat
				MWR_DS.to_netcdf(outfile, mode='w', format='NETCDF3_CLASSIC')

				MWR_DS.close()


	elif instrument == 'mirac':
		if level == 'level_1':
			print("It's pretty empty here.")


		elif level == 'level_2':
			print("It's pretty empty here.")

	elif instrument == 'arm':
		if level == 'level_1':
			print("It's pretty empty here.")


		elif level == 'level_2':
			print("It's pretty empty here.")
