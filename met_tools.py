import numpy as np
import pdb


# constants:
R_d = 287.04  	# gas constant of dry air, in J kg^-1 K^-1
R_v = 461.5  	# gas constant of water vapour, in J kg^-1 K^-1
M_dv = R_d / R_v # molar mass ratio , in ()
e_0 = 611		# saturation water vapour pressure at freezing point (273.15 K), in Pa
T0 = 273.15		# freezing temperature, in K
g = 9.80665 	# gravitation acceleration, in m s^-2 (from https://doi.org/10.6028/NIST.SP.330-2019 )


def compute_IWV(
	rho_v,
	z,
	nan_threshold=0.0,
	scheme='balanced'):

	"""
	Compute Integrated Water Vapour (also known as precipitable water content)
	out of absolute humidity (in kg m^-3) and height (in m).
	The moisture data may contain certain number gaps (up to nan_threshold*n_levels) but
	the height variable must be free of gaps.

	Parameters:
	-----------
	rho_v : array of floats
		One dimensional array of absolute humidity in kg m^-3.
	z : array of floats
		One dimensional array of sorted height axis (ascending order) in m.
	nan_threshold : float, optional
		Threshold describing the fraction of nan values of the total height level
		number that is still permitted for computation.
	scheme : str, optional
		Chose the scheme 'balanced' or 'top_weighted'. They differ in the way the altitude
		levels are used to compute IWV. Recommendation and default: 'balanced'
	"""

	# Check if the height axis is sorted in ascending order:
	if np.any((z[1:] - z[:-1]) < 0):
		raise ValueError("Height axis must be in ascending order to compute the integrated" +
			" water vapour.")

	n_height = len(z)
	# Check if rho_v has got any gaps:
	n_nans_rho_v = np.count_nonzero(np.isnan(rho_v))


	# If no nans exist, the computation is simpler. If some nans exist IWV will still be
	# computed but needs to look for the next non-nan value. If too many nans exist IWV
	# won't be computed.
	if scheme == 'balanced':
		if (n_nans_rho_v == 0):

			IWV = 0.0
			for k in range(n_height):
				if k == 0:		# bottom of grid
					dz = 0.5*(z[k+1] - z[k])		# just a half of a level difference
					IWV = IWV + rho_v[k]*dz

				elif k == n_height-1:	# top of grid
					dz = 0.5*(z[k] - z[k-1])		# the other half level difference
					IWV = IWV + rho_v[k]*dz

				else:			# mid of grid
					dz = 0.5*(z[k+1] - z[k-1])
					IWV = IWV + rho_v[k]*dz

		elif n_nans_rho_v / n_height < nan_threshold:

			# Loop through height grid:
			IWV = 0.0
			k = 0
			prev_nonnan_idx = -1
			while k < n_height:
				
				# check if hum on current level is nan:
				# if so search for the next non-nan level:
				if np.isnan(rho_v[k]):
					next_nonnan_idx = np.where(~np.isnan(rho_v[k:]))[0]

					if (len(next_nonnan_idx) > 0) and (prev_nonnan_idx >= 0):	# mid or near top of height grid
						next_nonnan_idx = next_nonnan_idx[0] + k	# plus k because searched over part of rho_v
						IWV += 0.25*(rho_v[next_nonnan_idx] + rho_v[prev_nonnan_idx])*(z[k+1] - z[k-1])
					
					elif (len(next_nonnan_idx) > 0) and (prev_nonnan_idx < 0):	# bottom of height grid
						next_nonnan_idx = next_nonnan_idx[0] + k	# plus k because searched over part of rho_v
						IWV += 0.5*rho_v[next_nonnan_idx]*(z[k+1] - z[k])
						

					else: # reached top of grid
						IWV += 0.0

				else:
					prev_nonnan_idx = k

					if k == 0:			# bottom of grid
						IWV += 0.5*rho_v[k]*(z[k+1] - z[k])
					elif (k > 0) and (k < n_height-1):	# mid of grid
						IWV += 0.5*rho_v[k]*(z[k+1] - z[k-1])
					else:				# top of grid
						IWV += 0.5*rho_v[k]*(z[-1] - z[-2])

				k += 1		

		else:
			IWV = np.nan


	elif scheme == 'top_weighted':
		if (n_nans_rho_v == 0):

			IWV = 0.0
			for k in range(n_height):
				if k < n_height-2:		# bottom or mid of grid
					dz = z[k+1] - z[k]
					IWV = IWV + rho_v[k]*dz

				else:	# top and next to top of grid
					dz = 0.5*(z[-1] - z[-2])		# half the height for top two levels
					IWV = IWV + rho_v[k]*dz

		elif n_nans_rho_v / n_height < nan_threshold:

			# Loop through height grid:
			IWV = 0.0
			k = 0
			prev_nonnan_idx = -1
			while k < n_height:
				
				# check if hum on current level is nan:
				# if so search for the next non-nan level:
				if np.isnan(rho_v[k]):
					next_nonnan_idx = np.where(~np.isnan(rho_v[k:]))[0]

					if (len(next_nonnan_idx) > 0) and (prev_nonnan_idx >= 0):	# mid of height grid
						next_nonnan_idx = next_nonnan_idx[0] + k	# plus k because searched over part of rho_v

						if k+1 != n_height-1:
							IWV += 0.5*(rho_v[next_nonnan_idx] + rho_v[prev_nonnan_idx])*(z[k+1] - z[k])
						else:	# near top of grid
							IWV += 0.25*(rho_v[next_nonnan_idx] + rho_v[prev_nonnan_idx])*(z[k+1] - z[k])
					
					elif (len(next_nonnan_idx) > 0) and (prev_nonnan_idx < 0):	# bottom of height grid
						next_nonnan_idx = next_nonnan_idx[0] + k	# plus k because searched over part of rho_v
						IWV += rho_v[next_nonnan_idx]*(z[k+1] - z[k])
						

					else: # reached top of grid
						IWV += 0.0

				else:
					prev_nonnan_idx = k

					if k < n_height-2:	# bottom or mid of grid
						IWV += rho_v[k]*(z[k+1] - z[k])
					else:				# top of grid
						IWV += 0.5*rho_v[k]*(z[-1] - z[-2])

				k += 1		

		else:
			IWV = np.nan
		
	return IWV


def compute_IWV_q(
	q,
	press,
	nan_threshold=0.0,
	scheme='balanced'):

	"""
	Compute Integrated Water Vapour (also known as precipitable water content)
	out of specific humidity (in kg kg^-1), gravitational constant and air pressure (in Pa).
	The moisture data may contain certain number gaps (up to nan_threshold*n_levels) but
	the height variable must be free of gaps.

	Parameters:
	-----------
	q : array of floats
		One dimensional array of specific humidity in kg kg^-1.
	press : array of floats
		One dimensional array of pressure in Pa.
	nan_threshold : float, optional
		Threshold describing the fraction of nan values of the total height level
		number that is still permitted for computation.
	scheme : str, optional
		Chose the scheme 'balanced' or 'top_weighted'. They differ in the way the altitude
		levels are used to compute IWV. Recommendation and default: 'balanced'
	"""

	# Check if the Pressure axis is sorted in descending order:
	if np.any((press[1:] - press[:-1]) > 0):
		pdb.set_trace()
		raise ValueError("Height axis must be in ascending order to compute the integrated" +
			" water vapour.")

	n_height = len(press)
	# Check if q has got any gaps:
	n_nans = np.count_nonzero(np.isnan(q))


	# If no nans exist, the computation is simpler. If some nans exist IWV will still be
	# computed but needs to look for the next non-nan value. If too many nans exist IWV
	# won't be computed.
	if scheme == 'balanced':
		if (n_nans == 0):

			IWV = 0.0
			for k in range(n_height):
				if k == 0:		# bottom of grid
					dp = 0.5*(press[k+1] - press[k])		# just a half of a level difference
					IWV = IWV - q[k]*dp

				elif k == n_height-1:	# top of grid
					dp = 0.5*(press[k] - press[k-1])		# the other half level difference
					IWV = IWV - q[k]*dp

				else:			# mid of grid
					dp = 0.5*(press[k+1] - press[k-1])
					IWV = IWV - q[k]*dp


		elif n_nans / n_height < nan_threshold:

			# Loop through height grid:
			IWV = 0.0
			k = 0
			prev_nonnan_idx = -1
			while k < n_height:
				
				# check if hum on current level is nan:
				# if so search for the next non-nan level:
				if np.isnan(q[k]):
					next_nonnan_idx = np.where(~np.isnan(q[k:]))[0]

					if (len(next_nonnan_idx) > 0) and (prev_nonnan_idx >= 0):	# mid or near top of height grid
						next_nonnan_idx = next_nonnan_idx[0] + k	# plus k because searched over part of rho_v
						IWV -= 0.25*(q[next_nonnan_idx] + q[prev_nonnan_idx])*(press[k+1] - press[k-1])
					
					elif (len(next_nonnan_idx) > 0) and (prev_nonnan_idx < 0):	# bottom of height grid
						next_nonnan_idx = next_nonnan_idx[0] + k	# plus k because searched over part of q
						IWV -= 0.5*q[next_nonnan_idx]*(press[k+1] - press[k])
						

					else: # reached top of grid
						IWV += 0.0

				else:
					prev_nonnan_idx = k

					if k == 0:			# bottom of grid
						IWV -= 0.5*q[k]*(press[k+1] - press[k])
					elif (k > 0) and (k < n_height-1):	# mid of grid
						IWV -= 0.5*q[k]*(press[k+1] - press[k-1])
					else:				# top of grid
						IWV -= 0.5*q[k]*(press[-1] - press[-2])

				k += 1

		else:
			IWV = np.nan


	elif scheme == 'top_weighted':
		if (n_nans == 0):

			IWV = 0.0
			for k in range(n_height):
				if k < n_height-2:		# bottom or mid of grid
					dp = press[k+1] - press[k]
					IWV = IWV - q[k]*dp

				else:	# top and next to top of grid
					dp = 0.5*(press[-1] - press[-2])		# half the height for top two levels
					IWV = IWV - q[k]*dp

		elif n_nans / n_height < nan_threshold:

			# Loop through height grid:
			IWV = 0.0
			k = 0
			prev_nonnan_idx = -1
			while k < n_height:
				
				# check if hum on current level is nan:
				# if so search for the next non-nan level:
				if np.isnan(q[k]):
					next_nonnan_idx = np.where(~np.isnan(q[k:]))[0]

					if (len(next_nonnan_idx) > 0) and (prev_nonnan_idx >= 0):	# mid of height grid
						next_nonnan_idx = next_nonnan_idx[0] + k	# plus k because searched over part of q

						if k+1 != n_height-1:
							IWV -= 0.5*(q[next_nonnan_idx] + q[prev_nonnan_idx])*(press[k+1] - press[k])
						else:	# near top of grid
							IWV -= 0.25*(q[next_nonnan_idx] + q[prev_nonnan_idx])*(press[k+1] - press[k])
					
					elif (len(next_nonnan_idx) > 0) and (prev_nonnan_idx < 0):	# bottom of height grid
						next_nonnan_idx = next_nonnan_idx[0] + k	# plus k because searched over part of q
						IWV -= q[next_nonnan_idx]*(press[k+1] - press[k])
						

					else: # reached top of grid
						IWV += 0.0

				else:
					prev_nonnan_idx = k

					if k < n_height-2:	# bottom or mid of grid
						IWV -= q[k]*(press[k+1] - press[k])
					else:				# top of grid
						IWV -= 0.5*q[k]*(press[-1] - press[-2])

				k += 1

		else:
			IWV = np.nan


	IWV = IWV / g		# yet had to be divided by gravitational acceleration

	return IWV


def wspeed_wdir_to_u_v(
	wspeed,
	wdir,
	convention='towards'):

	"""
	This will compute u and v wind components from wind speed and wind direction
	(in deg from northward facing wind). u and v will have the same units as
	wspeed. The default convention is that wdir indicates where the wind will flow
	to.

	Parameters:
	-----------
	wspeed : array of float or int
		Wind speed array.
	wdir : array of float or int
		Wind direction in deg from northward facing (or northerly) wind (for convention
		= towards, the wind flows northwards, wdir is 0; for convention = from, the wind
		comes from the north for wdir = 0).
	convention : str
		Convention of how wdir is to be interpreted. Options: 'towards' means that
		wdir indicates where the wind points to (where parcels will move to); 'from'
		means that wdir indicates where the wind comes from.
	"""

	if convention == 'towards':
		wdir_rad = wdir*2*np.pi/360
		u = np.sin(wdir_rad)*wspeed
		v = np.cos(wdir_rad)*wspeed

	elif convention == 'from':
		wdir_rad = (wdir+180)*2*np.pi/360
		wdir_rad[wdir_rad > 2*np.pi] -= 2*np.pi

		u = np.sin(wdir_rad)*wspeed
		v = np.cos(wdir_rad)*wspeed

	return u, v


def e_sat(
	temp,
	which_algo='hyland_and_wexler'):

	"""
	Calculates the saturation pressure over water after Goff and Gratch (1946)
	or Hyland and Wexler (1983).
	Source: Smithsonian Tables 1984, after Goff and Gratch 1946
	http://cires.colorado.edu/~voemel/vp.html
	http://hurri.kean.edu/~yoh/calculations/satvap/satvap.html

	e_sat_gg_water in Pa.

	Parameters:
	-----------
	temp : array of floats
		Array of temperature (in K).
	which_algo : str
		Specify which algorithm is chosen to compute e_sat (in Pa). Options:
		'hyland_and_wexler' (default), 'goff_and_gratch'
	"""

	if which_algo == 'hyland_and_wexler':
		e_sat_gg_water = temp**(0.65459673e+01) * np.exp(-0.58002206e+04 / temp + 0.13914993e+01 - 0.48640239e-01*temp + 
								0.41764768e-04*(temp**2) - 0.14452093e-07*(temp**3))

	elif which_algo == 'goff_and_gratch':
		e_sat_gg_water = 100 * 1013.246 * 10**(-7.90298*(373.16/temp-1) + 5.02808*np.log10(
				373.16/temp) - 1.3816e-7*(10**(11.344*(1-temp/373.16))-1) + 8.1328e-3 * (10**(-3.49149*(373.16/temp-1))-1))

	return e_sat_gg_water


def convert_rh_to_abshum(
	temp,
	relhum):

	"""
	Convert array of relative humidity (between 0 and 1) to absolute humidity
	in kg m^-3. 

	Saturation water vapour pressure computation is based on: see e_sat(temp).

	Parameters:
	-----------
	temp : array of floats
		Array of temperature (in K).
	relhum : array of floats
		Array of relative humidity (between 0 and 1).
	"""

	e_sat_water = e_sat(temp)

	rho_v = relhum * e_sat_water / (R_v * temp)

	return rho_v


def convert_rh_to_spechum(
	temp,
	pres,
	relhum):

	"""
	Convert array of relative humidity (between 0 and 1) to specific humidity
	in kg kg^-1.

	Saturation water vapour pressure computation is based on: see e_sat(temp).

	Parameters:
	-----------
	temp : array of floats
		Array of temperature (in K).
	pres : array of floats
		Array of pressure (in Pa).
	relhum : array of floats
		Array of relative humidity (between 0 and 1).
	"""

	e_sat_water = e_sat(temp)

	e = e_sat_water * relhum
	q = M_dv * e / (e*(M_dv - 1) + pres)

	return q
	
	
def convert_abshum_to_spechum(
	temp,
	pres,
	abshum):

	"""
	Convert array of absolute humidity (kg m^-3) to specific humidity
	in kg kg^-1.


	Parameters:
	-----------
	temp : array of floats
		Array of temperature (in K).
	pres : array of floats
		Array of pressure (in Pa).
	abshum : array of floats
		Array of absolute humidity (in kg m^-3).
	"""

	q = abshum / (abshum*(1 - M_dv) + (pres/(R_d*temp)))

	return q


def rho_air(
	pres,
	temp,
	abshum):

	"""
	Compute the density of air (in kg m-3) with a certain moisture load.

	Parameters:
	-----------
	pres : array of floats
		Array of pressure (in Pa).
	temp : array of floats
		Array of temperature (in K).
	abshum : array of floats
		Array of absolute humidity (in kg m^-3).
	"""

	rho = (pres - abshum*R_v*temp) / (R_d*temp) + abshum

	return rho


def convert_spechum_to_abshum(
	temp,
	pres,
	q):

	"""
	Convert array of specific humidity (kg kg^-1) to absolute humidity
	in kg m^-3.


	Parameters:
	-----------
	temp : array of floats
		Array of temperature (in K).
	pres : array of floats
		Array of pressure (in Pa).
	q : array of floats
		Array of specific humidity (in kg kg^-1).
	"""

	abshum = pres / (R_d*temp*(1/q + M_dv - 1))

	return abshum


def convert_abshum_to_relhum(
	temp,
	abshum):

	"""
	Convert array of absolute humidity (in kg m^-3) to relative humidity (in [0...1]).

	Parameters:
	-----------
	temp : array of floats
		Array of temperature (in K).
	abshum : array of floats
		Array of absolute humidity (in kg m^-3).
	"""

	e = abshum*R_v*temp
	e_sat_water = e_sat(temp)
	relhum = e/e_sat_water

	return relhum


def Z_from_GP(
	gp):

	"""
	Computes geopotential height (in m) from geopotential.

	Parameters:
	gp : float or array of float
		Geopotential in m^2 s^-2.
	"""

	return gp / g


def ZR_rain_rate(
	Z,
	dsd='mp'):

	"""
	Compute rain rate from ZR relation: Z = a*R^b
	a and b vary.

	Parameters:
	-----------
	Z : float or array of floats
		Radar reflectivity factor (in mm^6 m^-3).
	dsd : str
		Identifier of the drop size distribution. Valid options: 'mp'
		'mp': Marshall and Palmer 1948
	"""

	if dsd == 'mp':
		a = 296
		b = 1.47
		R = (Z/a)**(1/b)

	else:
		raise ValueError("Function 'ZR_rain_rate' currently only implemented dsd='mp'.")

	return R


def compute_LWC_from_Z(
	Z,
	cloud_type,
	**kwargs):

	"""
	Compute Liquid Water Content (LWC, in g m^-3) of a cloud from radar reflectivity factor Z
	(in mm^6 m^-3) using the Z-LWC relation Z = a*LWC**b

	Parameters:
	-----------
	Z : float or array of floats
		Radar reflectivity factor (in mm^6 m^-3).
	cloud_type : str
		Specification of the cloud type. Valid options: 'no_drizzle', 'light_drizzle', 
		'heavy_drizzle'

	**kwargs:
	algorithm : str
		Specfiy the algorithm. This argument is ignored if cloud_type is not 'no_drizzle'.
		Valid options: 'Fox_and_Illingworth_1997_i', 'Fox_and_Illingworth_1997_ii',
		'Sauvageot_and_Omar_1987', 'Liao_and_Sassen_1994'
	"""

	algorithms = {	'Fox_and_Illingworth_1997_i':		{'a': 0.012, 'b': 1.16},		# no drizzle
					'Fox_and_Illingworth_1997_ii':		{'a': 0.031, 'b': 1.56},		# no drizzle
					'Sauvageot_and_Omar_1987':			{'a': 0.030, 'b': 1.31},		# no drizzle
					'Liao_and_Sassen_1994':				{'a': 0.036, 'b': 1.80},		# no drizzle
					'Baedi_et_al_2000': 				{'a': 57.54, 'b': 5.17},		# light drizzle
					'Krasnov_and_Russchenberg_2002':	{'a': 323.59, 'b': 1.58}}		# heavy drizzle

	# select algorithm:
	if cloud_type == 'no_drizzle':
		algorithm = 'Fox_and_Illingworth_1997_i'

		if 'algorithm' in kwargs.keys():
			algorithm = kwargs['algorithm']

	elif cloud_type == 'light_drizzle':
		algorithm = 'Baedi_et_al_2000'

	elif cloud_type == 'heavy_drizzle':
		algorithm = 'Krasnov_and_Russchenberg_2002'

	else:
		raise ValueError("'cloud_type' for compute_LWC_from_Z must be 'no_drizzle, " +
							"'light_drizzle', or 'heavy_drizzle'.")

	LWC = (Z / algorithms[algorithm]['a'])**(1/algorithms[algorithm]['b'])

	return LWC
