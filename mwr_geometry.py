import numpy as np
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt
import pdb


BLEN = 10.0			# dummy length of beam before hitting the mirror (must be greater than
					# max distance between mirror boundaries and sensor_centre
o_angle = 4.0		# MWR sensor opening angle (to each side)


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


def compute_intersection(
	ele_rad,
	M1x,
	M1y,
	M2x,
	M2y,
	PMx,
	PMy):

	"""
	Compute intersection of MWR beam and mirror with given MWR elevation angle,
	and positions of the mirror and MWR sensor.

	Parameters:
	ele_rad : float
		MWR elevation angle in rad.
	M1x : float
		X position of the 'left' mirror end.
	M1y : float
		Y position of the 'left' mirror end.
	M2x : float
		X position of the 'right' mirror end.
	M2y : float
		Y position of the 'right' mirror end.
	PMx : float
		X position of the MWR's parabolic mirror.
	PMy : float
		Y position of the MWR's parabolic mirror.
	"""

	sinacosa = np.sin(ele_rad) / np.cos(ele_rad)
	mm = (M1y - PMy - M1x*sinacosa + PMx*sinacosa) / (sinacosa*(M2x - M1x) + M1y - M2y)
	nn = (M1x + mm *(M2x - M1x) - PMx) / (BLEN * np.cos(ele_rad))

	return mm, nn


"""
	Script to visualize and compute the MWR geometry (depending on MWR elevation angle,
	mirror angle wrt. the horizon. Respect opening
	- declare some basic (sort-of fix geometries)
	- compute angles
	- check for boundary conditions
	- visualize the whole thing
"""


# changables:
mwr_ele = 40.0						# elevation angle of MWR (in 2nd quadrant); relative to horizon, in deg
mir_angle = 1.5						# mirror angle in deg; counter-clockwise is > 0
delta = 2*mir_angle - mwr_ele		# angle to horizon after reflection off mirror

mwr_ele_rad = np.radians(mwr_ele)
mir_angle_rad = np.radians(mir_angle)
delta_rad = np.radians(delta)


# left and right part of the beam:
mwr_ele_1 = mwr_ele + o_angle		# 'left' part of the beam
mwr_ele_2 = mwr_ele - o_angle		# 'right' part of the beam
delta_1 = 2*mir_angle - mwr_ele_1	# angle to horizon after reflection off mirror (left beam part)
delta_2 = 2*mir_angle - mwr_ele_2	# angle to horizon after reflection off mirror (right beam part)

mwr_ele_rad_1 = np.radians(mwr_ele_1)
mwr_ele_rad_2 = np.radians(mwr_ele_2)
delta_rad_1 = np.radians(delta_1)
delta_rad_2 = np.radians(delta_2)


# start with basic geometry: ([x,y]), all in m
mwr_stand = np.array([0.0, 0.950])
hgt_stand_sensor = np.array([0.0, 0.3935])
sensor_centre = mwr_stand + hgt_stand_sensor		# centre of parabolic mirror
sensor_len = 0.250		# diameter of parabolic mirror
half_sensor_len = sensor_len*0.5

mirror_centre = sensor_centre + np.array([0.755, 0.610])
mirror_len = 0.740
half_mirror_len = mirror_len*0.5

guard_rail_dist = 0.500		# distance between MWR centre and guard rail centre in x dir
guard_rail_hgt = 1.110		# guard rail height in m
guard_rail_width = 0.080	# width of guard rail in m
guard_rail_centre = np.array([mwr_stand[0] + guard_rail_dist, 0.0])


# respect and compute that the sensor parabolic mirror is there, too:
par_mir_bounds_1 = (sensor_centre + half_sensor_len*np.array([np.cos(np.radians(90 + mwr_ele)), 
					np.sin(np.radians(90 + mwr_ele))]))		# left / top boundary of parabolic mirror
par_mir_bounds_2 = (sensor_centre + half_sensor_len*np.array([np.cos(np.radians(270 + mwr_ele)), 
					np.sin(np.radians(270 + mwr_ele))]))		# right / bottom boundary of parabolic mirror

# find mirror boundaries: 1 and 2 are bottom left and right boundaries
axis_plane_offset_vector = 0.017*np.array([np.cos(np.radians(mir_angle - 90.0)), np.sin(np.radians(mir_angle - 90.0))])
mir_bounds_1 = (mirror_centre + half_mirror_len*np.array([np.cos(np.radians(180.0 + mir_angle)), 
				np.sin(np.radians(180.0 + mir_angle))]) + axis_plane_offset_vector)
mir_bounds_2 = (mirror_centre + half_mirror_len*np.array([np.cos(mir_angle_rad),
				np.sin(mir_angle_rad)]) + axis_plane_offset_vector)
mir_bounds_3 = (mirror_centre + half_mirror_len*np.array([np.cos(mir_angle_rad),
				np.sin(mir_angle_rad)]) - axis_plane_offset_vector)		# top right boundary
mir_bounds_4 = (mirror_centre + half_mirror_len*np.array([np.cos(np.radians(180.0 + mir_angle)), 
				np.sin(np.radians(180.0 + mir_angle))]) - axis_plane_offset_vector)	# top left boundary


# compute intersection of centre of mwr beam with mirror:
# mm, nn = compute_intersection(mwr_ele_rad, mir_bounds_1[0], mir_bounds_1[1], mir_bounds_2[0], mir_bounds_2[1],
								# sensor_centre[0], sensor_centre[1])
mm, nn = vector_intersection_2d(A1=sensor_centre, A2=sensor_centre+BLEN*np.array([np.cos(mwr_ele_rad), np.sin(mwr_ele_rad)]),
								B1=mir_bounds_1, B2=mir_bounds_2)

# repeat for left and right parts of mwr beam:
mm1, nn1 = vector_intersection_2d(A1=par_mir_bounds_1, A2=par_mir_bounds_1+BLEN*np.array([np.cos(mwr_ele_rad_1), np.sin(mwr_ele_rad_1)]),
								B1=mir_bounds_1, B2=mir_bounds_2)
mm2, nn2 = vector_intersection_2d(A1=par_mir_bounds_2, A2=par_mir_bounds_2+BLEN*np.array([np.cos(mwr_ele_rad_2), np.sin(mwr_ele_rad_2)]),
								B1=mir_bounds_1, B2=mir_bounds_2)

all_okay = True
if np.any(np.array([mm, mm1, mm2]) > 1.0) or np.any(np.array([mm,mm1,mm2]) < 0.0): # then, mirror not hit
	print("CAUTION: One part of the radiometer 'beam' doesn't hit the mirror.")
	all_okay = False

# remove reflection along mirror line when not hitting the mirror:
# for this, I use a dummy factor which will be set to 0 if no reflection occurs.
refl_factor = 1.0
refl_factor1 = 1.0
refl_factor2 = 1.0
if mm > 1.0 or mm < 0.0:
	nn = 1.0
	refl_factor = 0.0
if mm1 > 1.0 or mm1 < 0.0:
	nn1 = 1.0
	refl_factor1 = 0.0
if mm2 > 1.0 or mm2 < 0.0:
	nn2 = 1.0
	refl_factor2 = 0.0
	

# intersection with mirror:
MBC = sensor_centre + nn*BLEN*np.array([np.cos(mwr_ele_rad), np.sin(mwr_ele_rad)])			# mirror-beam_centre intersec.
MB1 = par_mir_bounds_1 + nn1*BLEN*np.array([np.cos(mwr_ele_rad_1), np.sin(mwr_ele_rad_1)])	# mirror-beam_left intersec.
MB2 = par_mir_bounds_2 + nn2*BLEN*np.array([np.cos(mwr_ele_rad_2), np.sin(mwr_ele_rad_2)])	# mirror-beam_left intersec.


# virtual point some meters after interaction with the mirror:
FBC = MBC + refl_factor*BLEN*np.array([np.cos(delta_rad), np.sin(delta_rad)])				# final beam_centre point
FB1 = MB1 + refl_factor1*BLEN*np.array([np.cos(delta_rad_1), np.sin(delta_rad_1)])			# final beam_left point
FB2 = MB2 + refl_factor2*BLEN*np.array([np.cos(delta_rad_2), np.sin(delta_rad_2)])			# final beam_right point



# visualize:
fs = 16
fs_small = fs - 2
fs_dwarf = fs - 4
axis_lims = [-0.5, 2.5]


f1 = plt.figure(figsize=(10,10))
a1 = plt.axes()

# visualize MWR and mirror position
a1.plot(np.array([0, mwr_stand[0]]), np.array([0, mwr_stand[1]]), 
			linewidth=1.0, color=(0,0,0))
a1.plot(np.array([mwr_stand[0], mwr_stand[0] + hgt_stand_sensor[0]]), np.array([mwr_stand[1], mwr_stand[1] + hgt_stand_sensor[1]]), 
			linewidth=1.0, color=(0,0,0))
a1.plot(np.array([mirror_centre[0], mirror_centre[0]]), np.array([0.0, mirror_centre[1]]),
			linewidth=1.0, color=(0,0,0))

# visualize Polarstern guard rail:
guard_rail_dist = 0.500		# distance between MWR and guard rail in x dir
guard_rail_hgt = 1.110		# guard rail height in m
guard_rail_width = 0.080	# width of guard rail in m
guard_rail_centre = np.array([mwr_stand[0] + guard_rail_dist, 0.0])
a1.plot(np.array([guard_rail_centre[0] - 0.5*guard_rail_width, guard_rail_centre[0] - 0.5*guard_rail_width, 
		guard_rail_centre[0] + 0.5*guard_rail_width, guard_rail_centre[0] + 0.5*guard_rail_width]), 
		np.array([guard_rail_centre[1], guard_rail_centre[1] + guard_rail_hgt, guard_rail_centre[1] + guard_rail_hgt, guard_rail_centre[1]]), 
		linewidth=1.0, color=(0,0,0))

# plot mirror and parabolic mirror bounds:
a1.plot(np.array([par_mir_bounds_1[0], par_mir_bounds_2[0]]), np.array([par_mir_bounds_1[1], par_mir_bounds_2[1]]), 
			linewidth=2.0, color=(0,0,0))
a1.plot(np.array([mir_bounds_1[0], mir_bounds_2[0]]), np.array([mir_bounds_1[1], mir_bounds_2[1]]), 
			linewidth=2.0, color=(0,0,0))
a1.plot(np.array([mir_bounds_2[0], mir_bounds_3[0]]), np.array([mir_bounds_2[1], mir_bounds_3[1]]),
			linewidth=1.0, color=(0,0,0))
a1.plot(np.array([mir_bounds_3[0], mir_bounds_4[0]]), np.array([mir_bounds_3[1], mir_bounds_4[1]]),
			linewidth=1.0, color=(0,0,0))
a1.plot(np.array([mir_bounds_4[0], mir_bounds_1[0]]), np.array([mir_bounds_4[1], mir_bounds_1[1]]),
			linewidth=1.0, color=(0,0,0))

# mark centre positions:
a1.plot(np.array([sensor_centre[0], sensor_centre[0]]), np.array([sensor_centre[1], sensor_centre[1]]),
			linestyle='none', marker='.', markersize=9.0, color=(0,0,0))
a1.plot(np.array([mirror_centre[0], mirror_centre[0]]), np.array([mirror_centre[1], mirror_centre[1]]),
			linestyle='none', marker='.', markersize=9.0, color=(0,0,0))


# plot mwr line of sight (referred to as beam): for beam centre, left and right edges
a1.plot(np.array([sensor_centre[0], MBC[0]]), np.array([sensor_centre[1], MBC[1]]),
			linewidth=1.5, color=(0.8,0,0))
a1.plot(np.array([MBC[0], FBC[0]]), np.array([MBC[1], FBC[1]]),
			linewidth=1.5, color=(0.8,0,0))

a1.plot(np.array([par_mir_bounds_1[0], MB1[0]]), np.array([par_mir_bounds_1[1], MB1[1]]),
			linewidth=1.5, color=(0.8,0,0), linestyle='dotted')
a1.plot(np.array([MB1[0], FB1[0]]), np.array([MB1[1], FB1[1]]),
			linewidth=1.5, color=(0.8,0,0), linestyle='dotted')

a1.plot(np.array([par_mir_bounds_2[0], MB2[0]]), np.array([par_mir_bounds_2[1], MB2[1]]),
			linewidth=1.5, color=(0.8,0,0), linestyle='dotted')
a1.plot(np.array([MB2[0], FB2[0]]), np.array([MB2[1], FB2[1]]),
			linewidth=1.5, color=(0.8,0,0), linestyle='dotted')


# some dummy lines:
a1.plot([-20,20], [0,0], linewidth=1.0, color=(0,0,0))

# text mentioning the resulting zenith angle (and more text labels):
a1.text(0.02, 0.02, "ELE$_{\mathrm{MWR}}$: " + f"{mwr_ele:.2f}" + "$^{\circ}$\nmirror angle: " + 
		f"{mir_angle:.2f}" + "$^{\circ}$\n" + "$\mathbf{resulting}$ $\mathbf{zenith}$ $\mathbf{angle}$: " + 
		f"{90.0 + delta:.2f}" + "$^{\circ}$",
		color=(0,0,0), fontsize=fs, ha='left', va='bottom', transform=a1.transAxes)
a1.text(mirror_centre[0] - axis_plane_offset_vector[0], mirror_centre[1] - axis_plane_offset_vector[1], "mirror", 
		rotation=mir_angle, rotation_mode='anchor', ha='center', va='bottom', fontsize=fs_small)
a1.text(guard_rail_centre[0], guard_rail_centre[1] + 0.5*guard_rail_hgt, "guard rail", ha='center',
		va='center', rotation=90.0, rotation_mode='anchor', fontsize=fs_dwarf)
a1.text(axis_lims[0], 0.0, "Polarstern deck", ha='left', va='top', fontsize=fs_dwarf)

if not all_okay:
	a1.text(0.5, 1.01, "Radiometer misses mirror!", ha='center', va='bottom', 
			fontweight='bold', fontsize=fs, transform=a1.transAxes)

# axis properties:
a1.axis('equal')
a1.set_xlim(axis_lims[0], axis_lims[1])
a1.set_ylim(axis_lims[0], axis_lims[1])
a1.minorticks_on()
a1.grid(which='major', axis='both', color=(0.5,0.5,0.5), alpha=0.5)
a1.grid(which='minor', axis='both', color=(0.5,0.5,0.5), alpha=0.2)

# labels
a1.set_xlabel("x (m)", fontsize=fs)
a1.set_ylabel("y (m)", fontsize=fs)


plt.show()
