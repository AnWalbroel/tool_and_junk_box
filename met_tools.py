import numpy as np
import xarray as xr
import pdb


# constants:
R_d = 287.04  	# gas constant of dry air, in J kg-1 K-1
R_v = 461.5  	# gas constant of water vapour, in J kg-1 K-1
M_dv = R_d / R_v # molar mass ratio , in ()
e_0 = 611		# saturation water vapour pressure at freezing point (273.15 K), in Pa
T0 = 273.15		# freezing temperature, in K
g = 9.80665 	# gravitation acceleration, in m s^-2 (from https://doi.org/10.6028/NIST.SP.330-2019 )
c_pd = 1005.7	# specific heat capacity of dry air at constant pressure, in J kg-1 K-1
c_vd = 719.0	# specific heat capacity of dry air at constant volume, in J kg-1 K-1
c_h2o = 4187.0	# specific heat capacity of water at 15 deg C; in J kg-1 K-1
L_v = 2.501e+06	# latent heat of vaporization, in J kg-1
omega_earth = 2*np.pi / 86164.09	# earth's angular velocity: World Book Encyclopedia Vol 6. Illinois: World Book Inc.: 1984: 12.


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
	to. Note, that meteorological wind direction is defined as from where the wind is coming
	from.

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
		wdir_rad = np.radians(wdir)
		u = np.sin(wdir_rad)*wspeed
		v = np.cos(wdir_rad)*wspeed

	elif convention == 'from':
		wdir_rad = np.radians(wdir+180)
		wdir_rad[wdir_rad > 2*np.pi] -= 2*np.pi

		u = np.sin(wdir_rad)*wspeed
		v = np.cos(wdir_rad)*wspeed

	return u, v


def u_v_to_wspeed_wdir(
	u,
	v,
	convention='towards'):

	"""
	This will compute wind speed (in units of u and v) and wind direction (in deg from 
	northward facing or from north coming wind (depends on convention)) from u and v wind 
	components.The default convention is that wdir indicates where the wind will flow
	to. Note, that meteorological wind direction is defined as from where the wind is coming
	from.

	Parameters:
	-----------
	u : array of float or int
		Zonal component of wind (eastwards > 0).
	v : array of float or int
		Meridional component of wind (northwards > 0).
	convention : str
		Convention of how wdir is to be interpreted. Options: 'towards' means that
		wdir indicates where the wind points to (where parcels will move to); 'from'
		means that wdir indicates where the wind comes from.
	"""

	assert u.shape == v.shape	# check if both have the same dimension

	# flatten array and put it back into shape later.
	u_shape = u.shape
	u = u.flatten()
	v = v.flatten()
	wspeed = (u**2.0 + v**2.0)**0.5

	if convention == 'from':
		u *= (-1.0)
		v *= (-1.0)

	# distinguish the two semi circles to compute the correct wind direction:
	u_greater_0 = np.where(u >= 0)[0]
	u_smaller_0 = np.where(u < 0)[0]

	# compute wind direction based on the semi circle:
	wdir = np.zeros(u.shape)
	wdir[u_greater_0] = np.arccos(v[u_greater_0] / wspeed[u_greater_0])
	wdir[u_smaller_0] = 2.0*np.pi - np.arccos(v[u_smaller_0] / wspeed[u_smaller_0])

	# convert wdir to deg:
	wdir = np.degrees(wdir)

	# back to old shape:
	wspeed = np.reshape(wspeed, u_shape)
	wdir = np.reshape(wdir, u_shape)

	return wspeed, wdir


def compute_divergence(
	u,
	v,
	lon,
	lat):

	"""
	Computes convergence of a wind field (u,v) on a coordinate grid
	(lon, lat) on a height layer and for a certain time (u,v are both 2D arrays).
	The formula behind it is convergence = du/dx + dv/dy.

	Parameters:
	u : array of floats
		Wind vector in zonal direction (in m s-1).
	v : array of floats
		Wind vector in meridional direction (in m s-1).
	lon : array of floats
		Longitude grid points (in decimal degrees East). 1D array.
	lat : array of floats
		Latitude grid points (in decimal degrees North). 1D array.
	"""

	from geopy import distance		# needed to compute dx and dy

	nx = len(lon)
	ny = len(lat)

	divergence = np.full((ny, nx), np.nan)
	for ii in range(ny):		# loop through row indices (latitudes)
		for jj in range(nx):	# loop through column indices (longitudes)

			# compute du, dv, dx, dy:
			if ii == 0:		# first row (highest latitude)
				dy = distance.distance((lat[ii+1], lon[jj]), (lat[ii], lon[jj])).km*1000.0
				dv = v[ii,jj] - v[ii+1,jj]

			elif ii < ny-1:
				# 2 steps for centered difference:
				dy = distance.distance((lat[ii-1], lon[jj]), (lat[ii+1], lon[jj])).km*1000.0
				dv = v[ii-1,jj] - v[ii+1,jj]

			else:	# last row (lowest latitude)
				dy = distance.distance((lat[ii-1], lon[jj]), (lat[ii], lon[jj])).km*1000.0
				dv = v[ii-1,jj] - v[ii,jj]

			if jj == 0: # western border
				du = u[ii,jj+1] - u[ii,jj]
				dx = distance.distance((lat[ii], lon[jj+1]), (lat[ii], lon[jj])).km*1000.0		# distance in meters

			elif jj < nx-1:	# between western and eastern border
				du = u[ii,jj+1] - u[ii,jj-1]
				dx = distance.distance((lat[ii], lon[jj+1]), (lat[ii], lon[jj-1])).km*1000.0		# distance in meters

			else:	# eastern border
				dx = distance.distance((lat[ii], lon[jj-1]), (lat[ii], lon[jj])).km*1000.0
				du = u[ii,jj] - u[ii,jj-1]

			# compute convergece: du/dx + dv/dy
			if dx == 0 and dy != 0:
				divergence[ii,jj] = dv/dy
			elif dy == 0 and dx != 0:
				divergence[ii,jj] = du/dx
			elif dx == 0 and dy == 0:
				divergence[ii,jj] = 0.0
			else:
				divergence[ii,jj] = du/dx + dv/dy

	return divergence


def relative_vorticity_advection(
	u,
	v,
	lon,
	lat):

	"""
	Computes relative vorticity advection according to -v*grad(rel_vorticity_z) where
	v is the 2D wind vector of (u, v) = (zonal, meridional) wind, rel_vorticity_z is 
	rot(v) z component on a (lon,lat) grid. The wind components are 2D arrays of the 
	shape (len(lat), len(lon)).

	Parameters:
	u : array of floats
		Wind vector in zonal direction (in m s-1).
	v : array of floats
		Wind vector in meridional direction (in m s-1).
	lon : array of floats
		Longitude grid points (in decimal degrees East). 1D array.
	lat : array of floats
		Latitude grid points (in decimal degrees North). 1D array.
	"""

	from geopy import distance		# needed to compute dx and dy

	nx = len(lon)
	ny = len(lat)
	dx = np.zeros((ny,nx))
	dy = np.zeros((ny,nx))
	du = np.zeros((ny,nx))
	dv = np.zeros((ny,nx))
	du_dy = np.zeros((ny,nx))
	dv_dx = np.zeros((ny,nx))

	rva = np.full((ny, nx), np.nan)
	for ii in range(ny):		# loop through row indices (latitudes)
		for jj in range(nx):	# loop through column indices (longitudes)

			# compute du, dv, dx, dy:
			if ii == 0:		# first row (highest latitude)
				dy[ii,jj] = distance.distance((lat[ii+1], lon[jj]), (lat[ii], lon[jj])).km*1000.0
				dv[ii,jj] = v[ii,jj] - v[ii+1,jj]

			elif ii < ny-1:
				# 2 steps for centered difference:
				dy[ii,jj] = distance.distance((lat[ii-1], lon[jj]), (lat[ii+1], lon[jj])).km*1000.0
				dv[ii,jj] = v[ii-1,jj] - v[ii+1,jj]

			else:	# last row (lowest latitude)
				dy[ii,jj] = distance.distance((lat[ii-1], lon[jj]), (lat[ii], lon[jj])).km*1000.0
				dv[ii,jj] = v[ii-1,jj] - v[ii,jj]

			if jj == 0: # western border
				du[ii,jj] = u[ii,jj+1] - u[ii,jj]
				dx[ii,jj] = distance.distance((lat[ii], lon[jj+1]), (lat[ii], lon[jj])).km*1000.0		# distance in meters

			elif jj < nx-1:	# between western and eastern border
				du[ii,jj] = u[ii,jj+1] - u[ii,jj-1]
				dx[ii,jj] = distance.distance((lat[ii], lon[jj+1]), (lat[ii], lon[jj-1])).km*1000.0		# distance in meters

			else:	# eastern border
				du[ii,jj] = u[ii,jj] - u[ii,jj-1]
				dx[ii,jj] = distance.distance((lat[ii], lon[jj-1]), (lat[ii], lon[jj])).km*1000.0


			# first, compute for all grid points: du/dy and dv/dx
			if dy[ii,jj] != 0:
				du_dy[ii,jj] = du[ii,jj]/dy[ii,jj]
			else:
				du_dy[ii,jj] = 0.0
			if dx[ii,jj] != 0:
				dv_dx[ii,jj] = dv[ii,jj]/dx[ii,jj]


	# now we need to know how du/dy and dv/dx change in x and y directions:
	for ii in range(ny):		# loop through row indices (latitudes)
		for jj in range(nx):	# loop through column indices (longitudes)

			# compute d(du_dy)/dx, d(du_dy)/dy, d(dv_dx)/dx, d(dv_dx)/dy
			if ii == 0:		# first row (highest latitude)
				if dy[ii,jj] == 0:		# capture division by zero before errors occur
					ddu_dydy = 0.0
					ddv_dxdy = 0.0
				else:
					ddu_dydy = (du_dy[ii,jj] - du_dy[ii+1,jj])/dy[ii,jj]
					ddv_dxdy = (dv_dx[ii,jj] - dv_dx[ii+1,jj])/dy[ii,jj]

			elif ii < ny-1:
				# 2 steps for centered difference:
				if dy[ii,jj] == 0:
					ddu_dydy = 0.0
					ddv_dxdy = 0.0
				else:
					ddu_dydy = (du_dy[ii-1,jj] - du_dy[ii+1,jj])/dy[ii,jj]
					ddv_dxdy = (dv_dx[ii-1,jj] - dv_dx[ii+1,jj])/dy[ii,jj]

			else:	# last row (lowest latitude)
				if dy[ii,jj] == 0:
					ddu_dydy = 0.0
					ddv_dxdy = 0.0
				else:
					ddu_dydy = (du_dy[ii-1,jj] - du_dy[ii,jj])/dy[ii,jj]
					ddv_dxdy = (dv_dx[ii-1,jj] - dv_dx[ii,jj])/dy[ii,jj]

			if jj == 0: # western border
				if dx[ii,jj] == 0:
					ddu_dydx = 0.0
					ddv_dxdx = 0.0
				else:
					ddu_dydx = (du_dy[ii,jj+1] - du_dy[ii,jj])/dx[ii,jj]
					ddv_dxdx = (dv_dx[ii,jj+1] - dv_dx[ii,jj])/dx[ii,jj]

			elif jj < nx-1:	# between western and eastern border
				if dx[ii,jj] == 0:
					ddu_dydx = 0.0
					ddv_dxdx = 0.0
				else:
					ddu_dydx = (du_dy[ii,jj+1] - du_dy[ii,jj-1])/dx[ii,jj]
					ddv_dxdx = (dv_dx[ii,jj+1] - dv_dx[ii,jj-1])/dx[ii,jj]				

			else:	# eastern border
				if dx[ii,jj] == 0:
					ddu_dydx = 0.0
					ddv_dxdx = 0.0
				else:
					ddu_dydx = (du_dy[ii,jj] - du_dy[ii,jj-1])/dx[ii,jj]
					ddv_dxdx = (dv_dx[ii,jj] - dv_dx[ii,jj-1])/dx[ii,jj]

			# compute relative vorticity advection (rva):
			rva[ii,jj] = u[ii,jj]*(ddu_dydx - ddv_dxdx) + v[ii,jj]*(ddu_dydy - ddv_dxdy)

	return rva


def absolute_vorticity_advection(
	u,
	v,
	lon,
	lat):

	"""
	Computes absolute vorticity advection (relative + planetary) according to 
	-v*grad(rel_vorticity_z + f) where v is the 2D wind vector of (u, v) = (zonal, meridional) wind, 
	rel_vorticity_z is rot(v) z component, f is Coriolis parameter on a (lon,lat) grid. The wind 
	components are 2D arrays of the shape (len(lat), len(lon)).

	Parameters:
	u : array of floats
		Wind vector in zonal direction (in m s-1).
	v : array of floats
		Wind vector in meridional direction (in m s-1).
	lon : array of floats
		Longitude grid points (in decimal degrees East). 1D array.
	lat : array of floats
		Latitude grid points (in decimal degrees North). 1D array.
	"""

	from geopy import distance		# needed to compute dx and dy

	nx = len(lon)
	ny = len(lat)
	dx = np.zeros((ny,nx))
	dy = np.zeros((ny,nx))
	du = np.zeros((ny,nx))
	dv = np.zeros((ny,nx))
	du_dy = np.zeros((ny,nx))
	dv_dx = np.zeros((ny,nx))

	# compute coriolis parameter for each grid point:
	f = np.repeat(np.reshape(2*omega_earth*np.sin(np.radians(lat)), (ny,1)), nx, axis=1)

	ava = np.full((ny, nx), np.nan)
	for ii in range(ny):		# loop through row indices (latitudes)
		for jj in range(nx):	# loop through column indices (longitudes)

			# compute du, dv, dx, dy:
			if ii == 0:		# first row (highest latitude)
				dy[ii,jj] = distance.distance((lat[ii+1], lon[jj]), (lat[ii], lon[jj])).km*1000.0
				dv[ii,jj] = v[ii,jj] - v[ii+1,jj]

			elif ii < ny-1:
				# 2 steps for centered difference:
				dy[ii,jj] = distance.distance((lat[ii-1], lon[jj]), (lat[ii+1], lon[jj])).km*1000.0
				dv[ii,jj] = v[ii-1,jj] - v[ii+1,jj]

			else:	# last row (lowest latitude)
				dy[ii,jj] = distance.distance((lat[ii-1], lon[jj]), (lat[ii], lon[jj])).km*1000.0
				dv[ii,jj] = v[ii-1,jj] - v[ii,jj]


			if jj == 0: # western border
				du[ii,jj] = u[ii,jj+1] - u[ii,jj]
				dx[ii,jj] = distance.distance((lat[ii], lon[jj+1]), (lat[ii], lon[jj])).km*1000.0		# distance in meters

			elif jj < nx-1:	# between western and eastern border
				du[ii,jj] = u[ii,jj+1] - u[ii,jj-1]
				dx[ii,jj] = distance.distance((lat[ii], lon[jj+1]), (lat[ii], lon[jj-1])).km*1000.0		# distance in meters

			else:	# eastern border
				du[ii,jj] = u[ii,jj] - u[ii,jj-1]
				dx[ii,jj] = distance.distance((lat[ii], lon[jj-1]), (lat[ii], lon[jj])).km*1000.0


			# first, compute for all grid points: du/dy and dv/dx
			if dy[ii,jj] != 0:
				du_dy[ii,jj] = du[ii,jj]/dy[ii,jj]
			else:
				du_dy[ii,jj] = 0.0
			if dx[ii,jj] != 0:
				dv_dx[ii,jj] = dv[ii,jj]/dx[ii,jj]


	# now we need to know how du/dy and dv/dx change in x and y directions:
	for ii in range(ny):		# loop through row indices (latitudes)
		for jj in range(nx):	# loop through column indices (longitudes)

			# compute d(du_dy)/dx, d(du_dy)/dy, d(dv_dx)/dx, d(dv_dx)/dy
			if ii == 0:		# first row (highest latitude)
				if dy[ii,jj] == 0:		# capture division by zero before errors occur
					ddu_dydy = 0.0
					ddv_dxdy = 0.0
					df_dy = 0.0
				else:
					ddu_dydy = (du_dy[ii,jj] - du_dy[ii+1,jj])/dy[ii,jj]
					ddv_dxdy = (dv_dx[ii,jj] - dv_dx[ii+1,jj])/dy[ii,jj]
					df_dy = (f[ii,jj] - f[ii+1,jj])/dy[ii,jj]

			elif ii < ny-1:
				# 2 steps for centered difference:
				if dy[ii,jj] == 0:
					ddu_dydy = 0.0
					ddv_dxdy = 0.0
					df_dy = 0.0
				else:
					ddu_dydy = (du_dy[ii-1,jj] - du_dy[ii+1,jj])/dy[ii,jj]
					ddv_dxdy = (dv_dx[ii-1,jj] - dv_dx[ii+1,jj])/dy[ii,jj]
					df_dy = (f[ii-1,jj] - f[ii+1,jj])/dy[ii,jj]

			else:	# last row (lowest latitude)
				if dy[ii,jj] == 0:
					ddu_dydy = 0.0
					ddv_dxdy = 0.0
					df_dy = 0.0
				else:
					ddu_dydy = (du_dy[ii-1,jj] - du_dy[ii,jj])/dy[ii,jj]
					ddv_dxdy = (dv_dx[ii-1,jj] - dv_dx[ii,jj])/dy[ii,jj]
					df_dy = (f[ii-1,jj] - f[ii,jj])/dy[ii,jj]


			if jj == 0: # western border
				if dx[ii,jj] == 0:
					ddu_dydx = 0.0
					ddv_dxdx = 0.0
					df_dx = 0.0
				else:
					ddu_dydx = (du_dy[ii,jj+1] - du_dy[ii,jj])/dx[ii,jj]
					ddv_dxdx = (dv_dx[ii,jj+1] - dv_dx[ii,jj])/dx[ii,jj]
					df_dx = (f[ii,jj+1] - f[ii,jj])/dx[ii,jj]

			elif jj < nx-1:	# between western and eastern border
				if dx[ii,jj] == 0:
					ddu_dydx = 0.0
					ddv_dxdx = 0.0
					df_dx = 0.0
				else:
					ddu_dydx = (du_dy[ii,jj+1] - du_dy[ii,jj-1])/dx[ii,jj]
					ddv_dxdx = (dv_dx[ii,jj+1] - dv_dx[ii,jj-1])/dx[ii,jj]
					df_dx = (f[ii,jj+1] - f[ii,jj-1])/dx[ii,jj]

			else:	# eastern border
				if dx[ii,jj] == 0:
					ddu_dydx = 0.0
					ddv_dxdx = 0.0
					df_dx = 0.0
				else:
					ddu_dydx = (du_dy[ii,jj] - du_dy[ii,jj-1])/dx[ii,jj]
					ddv_dxdx = (dv_dx[ii,jj] - dv_dx[ii,jj-1])/dx[ii,jj]
					df_dx = (f[ii,jj] - f[ii,jj-1])/dx[ii,jj]

			# compute absolute vorticity advection (ava):
			ava[ii,jj] = -1.0*(u[ii,jj]*(ddv_dxdx - ddu_dydx + df_dx) + v[ii,jj]*(ddv_dxdy - ddu_dydy + df_dy))

	return ava


def potential_temperature(
	press,
	temp,
	press_sfc=100000.0,
	height_axis=None):

	"""
	Computes potential temperature theta from pressure and temperature of a certain level, and
	surface pressure according to theta = T*(p_s/p)**(R/c_p).

	Parameters:
	temp : array of floats
		Temperature at a certain height level in K.
	press : array of floats
		Pressure (best in Pa, else: same units as press_sfc) at a certain height level. Shape can
		be equal to temp.shape but can also be a 1D array.
	press_sfc : float
		Surface or reference pressure (in same units as press, preferably in Pa or hPa). Usually
		100000 Pa. 
	height_axis : int or None
		Identifier to locate the height axis of temp (i.e., 0, 1 or 2).
	"""

	if press.ndim == 1:	# expand press to shape of temp
		n_press = len(press)
		
		if height_axis == None:
			raise ValueError("Please specify which is the height axis of the temperature data as integer.")

		else:
			# build new shape list
			press_shape_new = list()
			for k in range(temp.ndim): press_shape_new.append(1)
			press_shape_new[height_axis] = temp.shape[height_axis]
			press = np.reshape(press, press_shape_new)

			# repeat pressure values:
			for k, tt in enumerate(temp.shape):
				if k != height_axis:
					press = np.repeat(press, tt, axis=k)

			# compute pot. temperature:
			theta = temp*(press_sfc/press)**(R_d/c_pd)

	elif press.shape == temp.shape:
		theta = temp*(press_sfc/press)**(R_d/c_pd)

	return theta


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


def convert_spechum_to_mix_rat(
	q,
	q_add=np.nan):

	"""
	Convert array (of float) of specific humidity (kg kg-1) to water vapour 
	mixing ratio (in kg kg-1). Also other hydrometeors (cloud liquid, 
	cloud rain water, snow, ice) can be respected.

	Parameters:
	-----------
	q : float or array of floats
		Specific humidity in kg kg-1.
	q_add : float or array of floats
		Sum of other hydrometeors (i.e., cloud liquid, cloud ice, snow) as
		'specific' contents (in kg kg-1). 
	"""

	if ((type(q_add) == type(np.array([]))) and q_add.size == 0) or ((type(q_add) == float) and (np.isnan(q_add))):
		r_v = q / (1 - q)
	else:
		r_v = q / (1 - q - q_add)

	return r_v


def convert_relhum_to_mix_rat(
	relhum,
	temp,
	pres):

	"""
	Convert relative humidity (in [0,1]) to water vapour mixing ratio (in kg kg-1).

	Parameters:
	-----------
	relhum : array of floats or float
		Array of relative humidity (between 0 and 1).
	temp : array of floats or float
		Array of temperature (in K).
	pres : array of floats or float
		Array of air pressure (in Pa).
	"""

	# convert relhum to abshum:
	abshum = convert_rh_to_abshum(temp, relhum)
	r_v = abshum / ((pres - e_sat(temp)*relhum) / (R_d * temp))

	return r_v


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


def detect_hum_inversions(
	q,
	z,
	return_inv_strength_height=False):

	"""
	Detect humidity inversions defined as increasing specific or absolute humidity
	with altitude, resulting in local maxima and minima of the humidity. q and z must
	have the same shape (even if z doesn't vary in the other dimensions).

	Parameters:
	-----------
	q : array of floats
		Array of specific (in kg kg-1) or absolute (kg m-3) humidity with the shape
		(height,) or (time or space,height). Also 3D array with shape of i.e., 
		(lat,lon,height) is permitted. Bottom of the profile must be at [0], top at [-1].
	z : array of floats
		Array of height (in m) with shape (height,) or (time or space,height). Also 3D array 
		with shape of i.e., (lat,lon,height) is permitted. Bottom must be at [0], 
		monotonically increases to top [-1]. Shape must be identical to q.shape.
	return_inv_strength_height : bool
		If True, inversion height (in m) and inversion strength (in units of q) for each 
		detected inversion is returned. If False, they aren't.
	"""

	if q.shape != z.shape:
		raise ValueError("z and q must have the same shape for humidity inversion detection.")

	def hum_inv_det_algo(extreme_points_loc, ddq_dzdz_loc, q_loc, z_loc):
		"""
		The detection algorithm reduced to 1D arrays (height axis only).
		"""

		# initialize variables where inversion info will be saved to and returned:
		z_inv_loc = {'bot': np.full(extreme_points_loc.shape, np.nan),
					'top': np.full(extreme_points_loc.shape, np.nan)}
		q_inv_loc = {'bot': np.full(extreme_points_loc.shape, np.nan),
					'top': np.full(extreme_points_loc.shape, np.nan)}

		# do extreme points exist?
		if len(extreme_points_loc) > 0:
			ddq_dzdz_inv = ddq_dzdz_loc[extreme_points_loc]

			# bottom and top to be identified via second derivative:
			z_x = z_loc[1:-1][extreme_points_loc]		# heights of extreme points
			q_x = q_loc[1:-1][extreme_points_loc]
			z_inv_bot = z_x[ddq_dzdz_inv > 0]		# inv bottom
			z_inv_top = z_x[ddq_dzdz_inv < 0]		# inv top
			q_inv_bot = q_x[ddq_dzdz_inv > 0]		# q at inv bottom
			q_inv_top = q_x[ddq_dzdz_inv < 0]		# q at inv top


			# check cases:
			if (len(z_inv_bot) == len(z_inv_top)) and (np.all(z_inv_top - z_inv_bot > 0)):
				# then all is good and inversions can be stored:
				z_inv_loc['bot'][:len(z_inv_bot)] = z_inv_bot
				z_inv_loc['top'][:len(z_inv_top)] = z_inv_top
				q_inv_loc['bot'][:len(q_inv_bot)] = q_inv_bot
				q_inv_loc['top'][:len(q_inv_top)] = q_inv_top

			elif len(z_inv_bot) > 0:	# make sure that an inversion bottom exists
				# repair bottom: tops existing before the first inv bottom:
				i = 0
				while ((i < len(z_inv_top)) and (z_inv_top[i] < z_inv_bot[0])):
					z_inv_top = z_inv_top[i+1:]		# now removed
					q_inv_top = q_inv_top[i+1:]

				# then check if everything looks good and save:
				if (len(z_inv_bot) == len(z_inv_top)) and (np.all(z_inv_top - z_inv_bot > 0)):
					z_inv_loc['bot'][:len(z_inv_bot)] = z_inv_bot
					z_inv_loc['top'][:len(z_inv_top)] = z_inv_top
					q_inv_loc['bot'][:len(q_inv_bot)] = q_inv_bot
					q_inv_loc['top'][:len(q_inv_top)] = q_inv_top

				elif z_inv_top[-1] - z_inv_bot[-1] < 0:	# an inversion bottom at top of height prof might be unpaired:
					i = 1
					while ((i <= len(z_inv_bot)) and (z_inv_top[-1] - z_inv_bot[-1*i] < 0)):
						z_inv_bot = z_inv_bot[:-1*i]
						q_inv_bot = q_inv_bot[:-1*i]

					# then check if everything looks good and save:
					if (len(z_inv_bot) == len(z_inv_top)) and (np.all(z_inv_top - z_inv_bot > 0)):
						z_inv_loc['bot'][:len(z_inv_bot)] = z_inv_bot
						z_inv_loc['top'][:len(z_inv_top)] = z_inv_top
						q_inv_loc['bot'][:len(q_inv_bot)] = q_inv_bot
						q_inv_loc['top'][:len(q_inv_top)] = q_inv_top

				else: # ???????
					raise RuntimeError("???????")
					
					
			elif (len(z_inv_bot) == 0) and (len(z_inv_top) == 1):	# only a top exists (prolly close to sfc); 
				# and only one top can exist ... otherwise something's wrong with the data (?)
				z_inv_bot = z_loc[0]					# set the surface as inversion bottom
				q_inv_bot = q_loc[0]

				# then check if everything looks good and save:
				if (len(z_inv_bot) == len(z_inv_top)) and (np.all(z_inv_top - z_inv_bot >= 0)):
					z_inv_loc['bot'][:len(z_inv_bot)] = z_inv_bot
					z_inv_loc['top'][:len(z_inv_top)] = z_inv_top
					q_inv_loc['bot'][:len(q_inv_bot)] = q_inv_bot
					q_inv_loc['top'][:len(q_inv_top)] = q_inv_top


			else: # something else might be wrong... happy debugging
				raise RuntimeError("An unexpected height inversion case occurred... oh dear")

		return q_inv_loc, z_inv_loc


	# # # # # # # # # # # # # # # # # # # # # # #


	# identify humidity inversions; then find respective bottom and top:
	# first: use first derivative: dq/dz = 0: find where dq/dz changes sign with increasing z:
	# multiplication of (dq/dz)(z_i) * (dq/dz)(z_i+1) < 0  <<-- change of sign
	dq = np.diff(q, axis=-1)
	dz = np.diff(z, axis=-1)
	dq_dz = dq/dz			# respective height axis: z[...,1:]


	# Identify change of sign:
	extreme_points = dq_dz[...,1:] * dq_dz[...,:-1] < 0		# q[...,1:-1][extreme_points] would be the q at extreme points

	# Theoretically, extreme points should come almost always in pairs, but to make sure 
	# to correctly identify bottom and top, we use second derivative:
	ddq_dzdz = np.diff(dq_dz, axis=-1) / dz[...,1:]


	# Slow solution of not losing information which inversion bottoms and tops belong 
	# spatio-temporally together: Loops
	z_inv = {'bot': np.full(extreme_points.shape, np.nan),
			'top': np.full(extreme_points.shape, np.nan)}
	q_inv = {'bot': np.full(extreme_points.shape, np.nan),
			'top': np.full(extreme_points.shape, np.nan)}

	if q.ndim == 2 and q.shape[0] > 1:	# loop over time (or space) required
		len_x = extreme_points.shape[0]

		for m in range(len_x):

			# reduce variable to local time or space coordinate; but leave full height dimension
			ddq_dzdz_loc = ddq_dzdz[m,:]
			extreme_points_loc = extreme_points[m,:]
			q_loc = q[m,:]
			z_loc = z[m,:]

			# use algorithm to find inversions and save the info:
			q_inv_loc, z_inv_loc = hum_inv_det_algo(extreme_points_loc, ddq_dzdz_loc, q_loc, z_loc)
			z_inv['bot'][m,:len(z_inv_loc['bot'])] = z_inv_loc['bot']
			z_inv['top'][m,:len(z_inv_loc['top'])] = z_inv_loc['top']
			q_inv['bot'][m,:len(q_inv_loc['bot'])] = q_inv_loc['bot']
			q_inv['top'][m,:len(q_inv_loc['top'])] = q_inv_loc['top']


	elif q.ndim == 3:					# 2 loops over time (or space)
		len_x = extreme_points.shape[0]
		len_y = extreme_points.shape[1]

		for m in range(len_x):
			for n in range(len_y):

				# reduce variable to local time or space coordinate; but leave full height dimension
				ddq_dzdz_loc = ddq_dzdz[m,n,:]
				extreme_points_loc = extreme_points[m,n,:]
				q_loc = q[m,n,:]
				z_loc = z[m,n,:]

				# use algorithm to find inversions and save the info:
				q_inv_loc, z_inv_loc = hum_inv_det_algo(extreme_points_loc, ddq_dzdz_loc, q_loc, z_loc)
				z_inv['bot'][m,n,:len(z_inv_loc['bot'])] = z_inv_loc['bot']
				z_inv['top'][m,n,:len(z_inv_loc['top'])] = z_inv_loc['top']
				q_inv['bot'][m,n,:len(q_inv_loc['bot'])] = q_inv_loc['bot']
				q_inv['top'][m,n,:len(q_inv_loc['top'])] = q_inv_loc['top']


	elif q.ndim == 1:					# no loops required
		q_inv, z_inv = hum_inv_det_algo(extreme_points, ddq_dzdz, q, z)
		

	# compute inversion heights and strengths if desired:
	if return_inv_strength_height:
		inv_height = z_inv['top'] - z_inv['bot']		
		inv_strength = q_inv['top'] - q_inv['bot']

		return q_inv, z_inv, inv_strength, inv_height

	else:
		return q_inv, z_inv


def equiv_pot_temperature(
	temp,
	pres,
	relhum=np.array([]),
	q=np.array([]),
	q_hyd=np.array([]),
	neglect_rtc=True):

	"""
	Computes the equivalent potential temperature following 
	https://glossary.ametsoc.org/wiki/Equivalent_potential_temperature .
	The given air pressure must be reduced to partial pressure of dry air.
	temp, pres, relhum, q and q_hyd must have the same shape. Either relhum
	or q must be provided.

	Parameters:
	-----------
	temp : array of floats
		Temperature in K.
	pres : rray of floats
		Air pressure in Pa.
	relhum : array of floats
		Relative humidity in [0,1].
	q : array of floats
		Specific humidity in kg kg-1.
	q_hyd : array of floats
		Specific content of several hydrometeors (i.e., cloud liquid, ice, snow, rain)
		in kg kg-1. Can be neglected
	neglect_rtc : bool
		Option whether to neglect the terms r_t*c_h2o (setting r_t = 0) or not.
		According to https://glossary.ametsoc.org/wiki/Equivalent_potential_temperature
		both can be used with good accuracy.
	"""

	if (relhum.size == 0) and (q.size == 0):
		raise ValueError("Specific or relative humidity must be provided.")
	elif q.size == 0:
		r_v = convert_relhum_to_mix_rat(relhum, temp, pres)
		e = e_sat(temp) * relhum
	else:
		r_v = convert_spechum_to_mix_rat(q, q_hyd)
		e = pres / (1 + M_dv*(1/q - 1))		# partial pressure of water vapour in Pa

	if q_hyd.size == 0:
		neglect_rtc = True

	pres_dry = pres - e					# partial pressure of dry air in Pa

	# compute total water mixing ratio (vapour, liquid, ice, snow, rain) in kg kg-1
	if neglect_rtc:
		r_t = np.zeros(temp.shape)		# total water mixing ratio (vapour, liquid, ice, snow, rain)
	else:
		# convert q_hyd + q to r_t
		r_t = convert_spechum_to_mix_rat(q_hyd + q)

	cpd_rtc = c_pd + r_t*c_h2o
	theta_e = temp * (100000.0 / pres_dry)**(R_d / cpd_rtc) * relhum**(-r_v*R_v / cpd_rtc) * np.exp(L_v*r_v / (cpd_rtc*temp))

	return theta_e


def Z_from_GP(
	gp):

	"""
	Computes geopotential height (in m) from geopotential.

	Parameters:
	gp : float or array of float
		Geopotential in m^2 s^-2.
	"""

	return gp / g


def mean_scale_height(
	pres,
	temp):

	"""
	Computes the mean scale height in m. H = R_d <T> / g with 
	<T> = integral(p2, p1){T(p) d lnp} / integral(p2, p1){d lnp}

	Parameters:
	-----------
	pres : array of floats
		Air pressure in Pa.
	temp : array of floats
		Temperature in K.
	"""

	# need to compute the layer averaged mean vertical temperature:
	pdb.set_trace()

	temp_m = np.sum(temp[...,:-1] * np.diff(np.log(pres))) / (np.cumsum(np.diff(np.log(pres))))
	MSH = R_d * temp_m / g

	return MSH


def Z_from_pres(
	pres,
	rho,
	pres_sfc):

	"""
	Computes the geopotential height in m based on the hydrostatic equation. Pressure must be sorted 
	from high pressure at index 0 to low pressure at the last index. Height axis must be the last axis.
	The returned height array will be sorted from low values at the first to high values at the last index.
	Surface pressure can be added to identify whether the pressure axis contains values below the actual 
	surface. 3D or higher dimension arrays are not (yet) supported.

	Parameters:
	-----------
	pres : array of floats
		Air pressure in Pa.
	rho : array of floats
		Air density (with moisture content) in kg m-3.
	pres_sfc : array of floats or float
		Surface air pressure in Pa.
	"""

	# check if pressure array is sorted correctly. If it isn't, flip pressure
	# and density arrays:
	if np.any(np.diff(pres, axis=-1) > 0.0):
		pres = pres[...,::-1]
		rho = rho[...,::-1]

	# check if pres has got values below the surface pressure:
	# pres and pres_sfc might have the same numer of dimensions. 
	# then pres is probably just a "height axis", while pres_sfc contains several data samples
	if pres.ndim == pres_sfc.ndim and len(pres_sfc) > 1:
		# expand pres dimensions:
		pres = np.repeat(np.reshape(pres, (1,len(pres))), len(pres_sfc), axis=0)

		# identify locations below surface
		idx_not_sub = [np.where(pres[k,:] <= pres_sfc[k])[0] for k in range(len(pres_sfc))]

	# compute Z:
	Z = np.full_like(rho, 0.0)

	if rho.ndim == 2:
		n_s = rho.shape[0]
		for k in range(n_s):
			pdb.set_trace()
			Z[k,idx_not_sub[k][:-1]] = -(1.0/g) * np.cumsum((1/rho[k,idx_not_sub[k][:-1]]) * np.diff(pres[k,idx_not_sub[k]], axis=-1), axis=-1)
			Z[k,idx_not_sub[k][-1]] = (Z[k,idx_not_sub[k][-2]] - (1.0/g) * (1/rho[k,idx_not_sub[k][-1]]) * (pres[k,idx_not_sub[k][-1]] - pres[k,idx_not_sub[k][-2]]))

	1/0	# costruction site

	return Z


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
