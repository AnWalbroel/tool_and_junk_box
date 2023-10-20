import numpy as np
import matplotlib as mpl
mpl.use("WebAgg")
import matplotlib.pyplot as plt
import pdb
import sys

if len(sys.argv) == 1:
	sys.argv.append("hatpro")

BLEN = 100.0			# dummy length of beam before hitting the mirror (must be greater than
					# max distance between mirror boundaries and sensor_centre
if sys.argv[1] == 'hatpro':
	o_angle = 3.7*0.5		# MWR sensor opening angle (to each side) for HATPRO, not both sides together
elif sys.argv[1] == 'mirac-p':
	o_angle = 1.3*0.5
o_angle_mirac = 1.3*0.5
o_angle_rad = np.radians(o_angle)


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


# paths:
path_plots = "/mnt/d/Studium_NIM/work/Plots/WALSEMA/mwr_geometry/"


# visualization options:
save_figures = True				# if True, saves figures to path_plots
mwr_mir_zoomed = True			# plot showing if mirror and mwr construction zoomed in
mwr_mir_sea_sfc = False			# plot showing mirror and mwr, but also the sea surface
include_IR = False				# if true, IR camera geometry is added to mwr_mir_sea_sfc plot
correct_footprints = True		# if True, the sensor length is reduced to a small value instead of
								# using the full parabolic mirror diameter (0.25 m). Thus, True represents
								# the true footprint size more accurately.
view_3D = True					# if true, a 3d view of the geometry will be generated
view = 'footprint'					# defines axis limits of view_3D plot: 'full' shows entire scenery
								# 'instrument' focuses on the MWRs and IR camera, 'footprint' zooms
								# into the footprints

if not mwr_mir_sea_sfc and not view_3D:
	include_IR = False
if view_3D:
	mwr_mir_sea_sfc = True


# changables: angles: 0.0 = horizontal; 90.0 = zenith; -90.0 = nadir
mwr_ele = 40.0						# elevation angle of MWR (in 2nd quadrant); relative to horizon, in deg
mir_angle = 1.5						# mirror angle in deg; counter-clockwise is > 0
delta = 2*mir_angle - mwr_ele		# angle to horizon after reflection off mirror
if include_IR: 
	ir_o_angle = 22.0	# opening angle
	ir_ele = -40.0		# elevation angle of IR camera
	ir_ele_1 = ir_ele + ir_o_angle		# 'left' part of the beam
	ir_ele_2 = ir_ele - ir_o_angle		# 'right' part of the beam 
	ir_ele_rad = np.radians(ir_ele)
	ir_ele_rad_1 = np.radians(ir_ele_1)
	ir_ele_rad_2 = np.radians(ir_ele_2)

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
if sys.argv[1] == 'hatpro': 
	mwr_stand = np.array([0.0, 0.992])
elif sys.argv[1] == 'mirac-p':
	mwr_stand = np.array([0.0, 0.997])
hgt_stand_sensor = np.array([0.0, 0.3935])
sensor_centre = mwr_stand + hgt_stand_sensor		# centre of parabolic mirror
if correct_footprints: 
	sensor_len = 0.0			# assumed sensor size: point-like
else:
	sensor_len = 0.250			# diameter of parabolic mirror
half_sensor_len = sensor_len*0.5

mirror_centre = sensor_centre + np.array([0.755, 0.610])
mirror_len = 0.740
half_mirror_len = mirror_len*0.5

guard_rail_dist = 0.500		# distance between MWR centre and guard rail centre in x dir
guard_rail_hgt = 1.110		# guard rail height in m
guard_rail_width = 0.081	# width of guard rail in m
guard_rail_centre_width = 0.060		# width of guard rail without small horizontal bars
guard_rail_centre = np.array([mwr_stand[0] + guard_rail_dist, 0.0])

dist_gre_dec = 0.060		# distance between guard rail and deck end construction
deck_end_1 = np.array([guard_rail_centre[0] + 0.5*guard_rail_centre_width + dist_gre_dec,
						0.0])
deck_end_2 = np.array([deck_end_1[0], 0.192])
deck_end_3 = np.array([deck_end_1[0] + 0.006, deck_end_2[1]])
deck_end_4 = np.array([deck_end_1[0] + 0.006, 0.0])


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


# include IR camera geometry if desired:
if include_IR:
	lens_position = np.array([guard_rail_centre[0] - 0.125, 1.488])


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
MB2 = par_mir_bounds_2 + nn2*BLEN*np.array([np.cos(mwr_ele_rad_2), np.sin(mwr_ele_rad_2)])	# mirror-beam_right intersec.


# virtual point some meters after interaction with the mirror:
FBC = MBC + refl_factor*BLEN*np.array([np.cos(delta_rad), np.sin(delta_rad)])				# final beam_centre point
FB1 = MB1 + refl_factor1*BLEN*np.array([np.cos(delta_rad_1), np.sin(delta_rad_1)])			# final beam_left point
FB2 = MB2 + refl_factor2*BLEN*np.array([np.cos(delta_rad_2), np.sin(delta_rad_2)])			# final beam_right point


if mwr_mir_sea_sfc:		# then also compute intersection of MBC -> FBC with sea surface:
	sea_sfc_ship = np.array([deck_end_4[0], -20.000])
	sea_sfc_dist = np.array([deck_end_4[0] + 2*BLEN, -20.000])

	# intersection of MWR beam with sea surface:
	kk, ll = vector_intersection_2d(A1=MBC, A2=FBC, B1=sea_sfc_ship, B2=sea_sfc_dist)	# kk for B1->B2, ll for A1->A2
	kk1, ll1 = vector_intersection_2d(A1=MB1, A2=FB1, B1=sea_sfc_ship, B2=sea_sfc_dist)
	kk2, ll2 = vector_intersection_2d(A1=MB2, A2=FB2, B1=sea_sfc_ship, B2=sea_sfc_dist)

	# check if sea surface is hit:
	if kk < 0: ll = 1.0
	if kk1 < 0: ll1 = 1.0
	if kk2 < 0: ll2 = 1.0

	# sea surface hit point centre, left (1) and right (2) beam
	SHPC = MBC + ll*(FBC - MBC)
	SHP1 = MB1 + ll1*(FB1 - MB1)
	SHP2 = MB2 + ll2*(FB2 - MB2)

	if include_IR:
		# intersection of IR camera beam with sea surface:
		irno, iryes = vector_intersection_2d(A1=lens_position, 
											A2=lens_position + BLEN*np.array([np.cos(ir_ele_rad), np.sin(ir_ele_rad)]),
											B1=sea_sfc_ship, B2 = sea_sfc_dist)
		irno1, iryes1 = vector_intersection_2d(A1=lens_position, 
											A2=lens_position + BLEN*np.array([np.cos(ir_ele_rad_1), np.sin(ir_ele_rad_1)]),
											B1=sea_sfc_ship, B2 = sea_sfc_dist)
		irno2, iryes2 = vector_intersection_2d(A1=lens_position, 
											A2=lens_position + BLEN*np.array([np.cos(ir_ele_rad_2), np.sin(ir_ele_rad_2)]),
											B1=sea_sfc_ship, B2 = sea_sfc_dist)

		# check if sea surface is hit at correct positions:
		if irno < 0: iryes = 1.0
		if irno1 < 0: iryes1 = 1.0
		if irno2 < 0: iryes2 = 1.0

		IR_SHPC = lens_position + iryes*BLEN*np.array([np.cos(ir_ele_rad), np.sin(ir_ele_rad)])
		IR_SHP1 = lens_position + iryes1*BLEN*np.array([np.cos(ir_ele_rad_1), np.sin(ir_ele_rad_1)])
		IR_SHP2 = lens_position + iryes2*BLEN*np.array([np.cos(ir_ele_rad_2), np.sin(ir_ele_rad_2)])


# visualize:
fs = 16
fs_small = fs - 2
fs_dwarf = fs - 4


if mwr_mir_zoomed:
	axis_lims = [-0.5, 2.5]

	f1 = plt.figure(figsize=(10,10))
	a1 = plt.axes()

	# visualize MWR and mirror position (just as unprecise sketches)
	a1.plot(np.array([0, mwr_stand[0]]), np.array([0, mwr_stand[1]]), 
				linewidth=1.0, color=(0,0,0))
	a1.plot(np.array([mwr_stand[0], mwr_stand[0] + hgt_stand_sensor[0]]), np.array([mwr_stand[1], mwr_stand[1] + hgt_stand_sensor[1]]), 
				linewidth=1.0, color=(0,0,0))
	# a1.plot(np.array([mirror_centre[0], mirror_centre[0]]), np.array([0.0, mirror_centre[1]]),
				# linewidth=1.0, color=(0,0,0))
	a1.plot(np.array([mwr_stand[0], mirror_centre[0], mirror_centre[0]]), np.array([0.80, 0.80, mirror_centre[1]]),
				linewidth=1.0, color=(0,0,0))

	# visualize Polarstern guard rail and deck:
	a1.plot(np.array([guard_rail_centre[0] - 0.5*guard_rail_width, guard_rail_centre[0] - 0.5*guard_rail_width, 
			guard_rail_centre[0] + 0.5*guard_rail_width, guard_rail_centre[0] + 0.5*guard_rail_width]), 
			np.array([guard_rail_centre[1], guard_rail_centre[1] + guard_rail_hgt, guard_rail_centre[1] + guard_rail_hgt, guard_rail_centre[1]]), 
			linewidth=1.0, color=(0,0,0))
	a1.plot(np.array([axis_lims[0], deck_end_1[0], deck_end_2[0], deck_end_3[0], deck_end_4[0]]),
			np.array([0.0, deck_end_1[1], deck_end_2[1], deck_end_3[1], deck_end_4[1]]),
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
	a1.plot([-20,deck_end_4[0]], [0,0], linewidth=1.0, color=(0,0,0))

	# text mentioning the resulting zenith angle (and more text labels):
	a1.text(mirror_centre[0] - axis_plane_offset_vector[0], mirror_centre[1] - axis_plane_offset_vector[1], "mirror", 
			rotation=mir_angle, rotation_mode='anchor', ha='center', va='bottom', fontsize=fs_small)
	a1.text(guard_rail_centre[0], guard_rail_centre[1] + 0.5*guard_rail_hgt, "guard rail", ha='center',
			va='center', rotation=90.0, rotation_mode='anchor', fontsize=fs_dwarf)
	a1.text(axis_lims[0], 0.0, "Polarstern deck", ha='left', va='top', fontsize=fs_dwarf)

	a1.text(0.02, 1.01, sys.argv[1], ha='left', va='bottom',
			fontsize=fs, transform=a1.transAxes)

	if not all_okay:
		a1.text(0.5, 1.01, "Radiometer misses mirror!", ha='center', va='bottom', 
				fontweight='bold', fontsize=fs, transform=a1.transAxes)

		if mm > 1.0 or mm < 0.0:
			res_zen_angle_str = ""
		else:
			res_zen_angle_str = f"{90.0 + delta:.2f}" + "$^{\circ}$"
	else:
		res_zen_angle_str = f"{90.0 + delta:.2f}" + "$^{\circ}$"

	a1.text(0.02, 0.02, "ELE$_{\mathrm{MWR}}$: " + f"{mwr_ele:.2f}" + "$^{\circ}$\nmirror angle: " + 
			f"{mir_angle:.2f}" + "$^{\circ}$\n" + "$\mathbf{resulting}$ $\mathbf{zenith}$ $\mathbf{angle}$: " + 
			res_zen_angle_str,
			color=(0,0,0), fontsize=fs, ha='left', va='bottom', transform=a1.transAxes)


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

	plot_file = path_plots + f"MWR_geometry_mwr_ele{int(mwr_ele)}_{sys.argv[1]}_zoomed_2D.png"
	if save_figures: 
		f1.savefig(plot_file, dpi=300, bbox_inches='tight')
	else:
		plt.show()
	plt.close()


"""
if mwr_mir_sea_sfc:
	axis_lims_x = [-0.5, 40.0]
	axis_lims_y = [-22.5, 2.5]

	f1 = plt.figure(figsize=(15,7.5))
	a1 = plt.axes()

	# visualize MWR and mirror position (just as unprecise sketches)
	a1.plot(np.array([0, mwr_stand[0]]), np.array([0, mwr_stand[1]]), 
				linewidth=1.0, color=(0,0,0))
	a1.plot(np.array([mwr_stand[0], mwr_stand[0] + hgt_stand_sensor[0]]), np.array([mwr_stand[1], mwr_stand[1] + hgt_stand_sensor[1]]), 
				linewidth=1.0, color=(0,0,0))
	a1.plot(np.array([mwr_stand[0], mirror_centre[0], mirror_centre[0]]), np.array([0.80, 0.80, mirror_centre[1]]),
				linewidth=1.0, color=(0,0,0))

	# visualize Polarstern guard rail and deck:
	a1.plot(np.array([guard_rail_centre[0] - 0.5*guard_rail_width, guard_rail_centre[0] - 0.5*guard_rail_width, 
			guard_rail_centre[0] + 0.5*guard_rail_width, guard_rail_centre[0] + 0.5*guard_rail_width]), 
			np.array([guard_rail_centre[1], guard_rail_centre[1] + guard_rail_hgt, guard_rail_centre[1] + guard_rail_hgt, guard_rail_centre[1]]), 
			linewidth=1.0, color=(0,0,0))
	a1.plot(np.array([axis_lims_x[0], deck_end_1[0], deck_end_2[0], deck_end_3[0], deck_end_4[0]]),
			np.array([0.0, deck_end_1[1], deck_end_2[1], deck_end_3[1], deck_end_4[1]]),
			linewidth=1.2, color=(0,0,0))

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

	# plot sea surface and rough sketch of Polarstern boundary
	a1.plot(np.array([sea_sfc_ship[0], sea_sfc_dist[0]]), np.array([sea_sfc_ship[1], sea_sfc_dist[1]]),
				linewidth=1.2, color=(0,0,0.75))
	a1.plot(np.array([deck_end_4[0], sea_sfc_ship[0]]), np.array([deck_end_4[1], sea_sfc_ship[1]]),
				linewidth=1.2, color=(0,0,0))

	# mark centre positions:
	a1.plot(np.array([sensor_centre[0], sensor_centre[0]]), np.array([sensor_centre[1], sensor_centre[1]]),
				linestyle='none', marker='.', markersize=9.0, color=(0,0,0))
	a1.plot(np.array([mirror_centre[0], mirror_centre[0]]), np.array([mirror_centre[1], mirror_centre[1]]),
				linestyle='none', marker='.', markersize=9.0, color=(0,0,0))


	# plot mwr line of sight (referred to as beam): for beam centre, left and right edges
	a1.plot(np.array([sensor_centre[0], MBC[0]]), np.array([sensor_centre[1], MBC[1]]),
				linewidth=1.5, color=(0.8,0,0))
	a1.plot(np.array([MBC[0], SHPC[0]]), np.array([MBC[1], SHPC[1]]),
				linewidth=1.5, color=(0.8,0,0))

	a1.plot(np.array([par_mir_bounds_1[0], MB1[0]]), np.array([par_mir_bounds_1[1], MB1[1]]),
				linewidth=1.5, color=(0.8,0,0), linestyle='dotted')
	a1.plot(np.array([MB1[0], SHP1[0]]), np.array([MB1[1], SHP1[1]]),
				linewidth=1.5, color=(0.8,0,0), linestyle='dotted')

	a1.plot(np.array([par_mir_bounds_2[0], MB2[0]]), np.array([par_mir_bounds_2[1], MB2[1]]),
				linewidth=1.5, color=(0.8,0,0), linestyle='dotted')
	a1.plot(np.array([MB2[0], SHP2[0]]), np.array([MB2[1], SHP2[1]]),
				linewidth=1.5, color=(0.8,0,0), linestyle='dotted')

	# visualize and label footprint:
	a1.plot(np.array([SHP1[0], SHP2[0]]), np.array([SHP1[1], SHP2[1]]), 
				linewidth=1.5, color=(0.8,0,0), label='MWR')
	a1.scatter(np.array([SHP1[0]]), np.array([SHP1[1]]), s=50.0, marker='<', color=(0.8,0,0))
	a1.scatter(np.array([SHP2[0]]), np.array([SHP2[1]]), s=50.0, marker='>', color=(0.8,0,0))
	a1.text(SHPC[0], SHPC[1], f"{SHP2[0]-SHP1[0]:.1f}" + "$\,$m", color=(0.8,0,0), 
			fontsize=fs_small, ha='center', va='top')


	# add IR camera if desired:
	if include_IR:
		a1.plot(np.array([lens_position[0], IR_SHPC[0]]), np.array([lens_position[1], IR_SHPC[1]]),
				linewidth=1.5, color=(0.25, 1.0, 0.25), label='IR')
		a1.plot(np.array([lens_position[0], IR_SHP1[0]]), np.array([lens_position[1], IR_SHP1[1]]),
				linewidth=1.5, color=(0.25, 1.0, 0.25), linestyle='dotted')
		a1.plot(np.array([lens_position[0], IR_SHP2[0]]), np.array([lens_position[1], IR_SHP2[1]]),
				linewidth=1.5, color=(0.25, 1.0, 0.25), linestyle='dotted')

		lh, ll = a1.get_legend_handles_labels()
		a1.legend(handles=lh, labels=ll, loc="center left", framealpha=1.0, fontsize=fs_small, markerscale=1.5)


	# some dummy lines:
	a1.plot([-20,deck_end_4[0]], [0,0], linewidth=1.2, color=(0,0,0))

	# text mentioning the resulting zenith angle (and more text labels):
	a1.text(sea_sfc_ship[0], sea_sfc_ship[1], "Sea surface", ha='left', va='top', 
			color=(0,0,0.75), fontsize=fs_dwarf)
	a1.text(0.02, 1.01, sys.argv[1] + "; (0,0): (parab mirror, Peil deck)", ha='left', va='bottom',
			fontsize=fs, transform=a1.transAxes)

	if not all_okay:
		a1.text(0.5, 1.01, "Radiometer misses mirror!", ha='center', va='bottom', 
				fontweight='bold', fontsize=fs, transform=a1.transAxes)

		if mm > 1.0 or mm < 0.0:
			res_zen_angle_str = ""
		else:
			res_zen_angle_str = f"{90.0 + delta:.2f}" + "$^{\circ}$"
	else:
		res_zen_angle_str = f"{90.0 + delta:.2f}" + "$^{\circ}$"

	a1.text(0.98, 0.98, "$\mathbf{resulting}$ $\mathbf{zenith}$ $\mathbf{angle}$: " + 
			res_zen_angle_str + f"\nsfc hit point: {SHPC[0]:.1f}" + "$\,$m\nlength footprint: " +
			f"{SHP2[0]-SHP1[0]:.1f}" + "$\,$m",
			color=(0,0,0), fontsize=fs, ha='right', va='top', transform=a1.transAxes)


	# axis properties:
	a1.axis('equal')
	a1.set_xlim(axis_lims_x[0], axis_lims_x[1])
	a1.set_ylim(axis_lims_y[0], axis_lims_y[1])
	a1.minorticks_on()
	a1.grid(which='major', axis='both', color=(0.5,0.5,0.5), alpha=0.5)
	a1.grid(which='minor', axis='both', color=(0.5,0.5,0.5), alpha=0.2)

	# labels
	a1.set_xlabel("x (m)", fontsize=fs)
	a1.set_ylabel("y (m)", fontsize=fs)


	plt.show()
"""

if view_3D:

	# define MiRAC-P geometries:
	o_angle_mirac_rad = np.radians(o_angle_mirac)
	mwr_ele_mirac_1 = mwr_ele + o_angle_mirac
	mwr_ele_mirac_2 = mwr_ele - o_angle_mirac
	delta_mirac_1 = 2*mir_angle - mwr_ele_mirac_1
	delta_mirac_2 = 2*mir_angle - mwr_ele_mirac_2

	mwr_ele_rad_mirac_1 = np.radians(mwr_ele_mirac_1)
	mwr_ele_rad_mirac_2 = np.radians(mwr_ele_mirac_2)
	delta_rad_mirac_1 = np.radians(delta_mirac_1)
	delta_rad_mirac_2 = np.radians(delta_mirac_2)

	mwr_stand_mirac = np.array([0.0, 0.997])
	hgt_stand_sensor_mirac = np.array([0.0, 0.3935])
	sensor_centre_mirac = mwr_stand_mirac + hgt_stand_sensor_mirac
	mirror_centre_mirac = sensor_centre_mirac + np.array([0.755, 0.610])
	par_mir_bounds_mirac_1 = (sensor_centre_mirac + half_sensor_len*np.array([np.cos(np.radians(90 + mwr_ele)), 
								np.sin(np.radians(90 + mwr_ele))]))
	par_mir_bounds_mirac_2 = (sensor_centre_mirac + half_sensor_len*np.array([np.cos(np.radians(270 + mwr_ele)), 
								np.sin(np.radians(270 + mwr_ele))]))
	mir_bounds_mirac_1 = (mirror_centre_mirac + half_mirror_len*np.array([np.cos(np.radians(180.0 + mir_angle)), 
							np.sin(np.radians(180.0 + mir_angle))]) + axis_plane_offset_vector)
	mir_bounds_mirac_2 = (mirror_centre_mirac + half_mirror_len*np.array([np.cos(mir_angle_rad),
							np.sin(mir_angle_rad)]) + axis_plane_offset_vector)
	mir_bounds_mirac_3 = (mirror_centre_mirac + half_mirror_len*np.array([np.cos(mir_angle_rad),
							np.sin(mir_angle_rad)]) - axis_plane_offset_vector)
	mir_bounds_mirac_4 = (mirror_centre_mirac + half_mirror_len*np.array([np.cos(np.radians(180.0 + mir_angle)), 
							np.sin(np.radians(180.0 + mir_angle))]) - axis_plane_offset_vector)


	# compute intersection of centre of mwr beam with mirror:
	mm, nn = vector_intersection_2d(A1=sensor_centre_mirac, A2=sensor_centre_mirac+BLEN*np.array([np.cos(mwr_ele_rad), np.sin(mwr_ele_rad)]),
									B1=mir_bounds_mirac_1, B2=mir_bounds_mirac_2)

	# repeat for left and right parts of mwr beam:
	mm1, nn1 = vector_intersection_2d(A1=par_mir_bounds_mirac_1, A2=par_mir_bounds_mirac_1+BLEN*np.array([np.cos(mwr_ele_rad_mirac_1), np.sin(mwr_ele_rad_mirac_1)]),
									B1=mir_bounds_mirac_1, B2=mir_bounds_mirac_2)
	mm2, nn2 = vector_intersection_2d(A1=par_mir_bounds_mirac_2, A2=par_mir_bounds_mirac_2+BLEN*np.array([np.cos(mwr_ele_rad_mirac_2), np.sin(mwr_ele_rad_mirac_2)]),
									B1=mir_bounds_mirac_1, B2=mir_bounds_mirac_2)

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
	MBC_mirac = sensor_centre_mirac + nn*BLEN*np.array([np.cos(mwr_ele_rad), np.sin(mwr_ele_rad)])			# mirror-beam_centre intersec.
	MB1_mirac = par_mir_bounds_mirac_1 + nn1*BLEN*np.array([np.cos(mwr_ele_rad_mirac_1), np.sin(mwr_ele_rad_mirac_1)])	# mirror-beam_left intersec.
	MB2_mirac = par_mir_bounds_mirac_2 + nn2*BLEN*np.array([np.cos(mwr_ele_rad_mirac_2), np.sin(mwr_ele_rad_mirac_2)])	# mirror-beam_left intersec.

	# virtual point some meters after interaction with the mirror:
	FBC_mirac = MBC_mirac + refl_factor*BLEN*np.array([np.cos(delta_rad), np.sin(delta_rad)])				# final beam_centre point
	FB1_mirac = MB1_mirac + refl_factor1*BLEN*np.array([np.cos(delta_rad_mirac_1), np.sin(delta_rad_mirac_1)])			# final beam_left point
	FB2_mirac = MB2_mirac + refl_factor2*BLEN*np.array([np.cos(delta_rad_mirac_2), np.sin(delta_rad_mirac_2)])			# final beam_right point

	# intersection of MWR beam with sea surface:
	kk, ll = vector_intersection_2d(A1=MBC_mirac, A2=FBC_mirac, B1=sea_sfc_ship, B2=sea_sfc_dist)	# kk for B1->B2, ll for A1->A2
	kk1, ll1 = vector_intersection_2d(A1=MB1_mirac, A2=FB1_mirac, B1=sea_sfc_ship, B2=sea_sfc_dist)
	kk2, ll2 = vector_intersection_2d(A1=MB2_mirac, A2=FB2_mirac, B1=sea_sfc_ship, B2=sea_sfc_dist)

	# check if sea surface is hit:
	if kk < 0: ll = 1.0
	if kk1 < 0: ll1 = 1.0
	if kk2 < 0: ll2 = 1.0

	# sea surface hit point centre, left (1) and right (2) beam
	SHPC_mirac = MBC_mirac + ll*(FBC_mirac - MBC_mirac)
	SHP1_mirac = MB1_mirac + ll1*(FB1_mirac - MB1_mirac)
	SHP2_mirac = MB2_mirac + ll2*(FB2_mirac - MB2_mirac)


	# translate relevant vectors to 3D:
	mir_y = 1.780		# distace between HATPRO and MiRAC-P
	mwr_stand = np.array([mwr_stand[0], 0.0, mwr_stand[1]])
	hgt_stand_sensor = np.array([hgt_stand_sensor[0], 0.0, hgt_stand_sensor[1]])
	mirror_centre = np.array([mirror_centre[0], 0.0, mirror_centre[1]])
	mwr_stand_mirac = np.array([mwr_stand_mirac[0], mir_y, mwr_stand_mirac[1]])
	hgt_stand_sensor_mirac = np.array([hgt_stand_sensor_mirac[0], mir_y, hgt_stand_sensor_mirac[1]])
	mirror_centre_mirac = np.array([mirror_centre_mirac[0], mir_y, mirror_centre_mirac[1]])
	guard_rail_centre = np.array([guard_rail_centre[0], 0.0, guard_rail_centre[1]])
	deck_end_1 = np.array([deck_end_1[0], 0.0, deck_end_1[1]])
	deck_end_2 = np.array([deck_end_2[0], 0.0, deck_end_2[1]])
	deck_end_3 = np.array([deck_end_3[0], 0.0, deck_end_3[1]])
	deck_end_4 = np.array([deck_end_4[0], 0.0, deck_end_4[1]])
	par_mir_bounds_1 = np.array([par_mir_bounds_1[0], 0.0, par_mir_bounds_1[1]])
	par_mir_bounds_2 = np.array([par_mir_bounds_2[0], 0.0, par_mir_bounds_2[1]])
	par_mir_bounds_mirac_1 = np.array([par_mir_bounds_mirac_1[0], mir_y, par_mir_bounds_mirac_1[1]])
	par_mir_bounds_mirac_2 = np.array([par_mir_bounds_mirac_2[0], mir_y, par_mir_bounds_mirac_2[1]])
	mir_bounds_1 = np.array([mir_bounds_1[0], 0.0, mir_bounds_1[1]])
	mir_bounds_2 = np.array([mir_bounds_2[0], 0.0, mir_bounds_2[1]])
	mir_bounds_3 = np.array([mir_bounds_3[0], 0.0, mir_bounds_3[1]])
	mir_bounds_4 = np.array([mir_bounds_4[0], 0.0, mir_bounds_4[1]])
	mir_bounds_mirac_1 = np.array([mir_bounds_mirac_1[0], mir_y, mir_bounds_mirac_1[1]])
	mir_bounds_mirac_2 = np.array([mir_bounds_mirac_2[0], mir_y, mir_bounds_mirac_2[1]])
	mir_bounds_mirac_3 = np.array([mir_bounds_mirac_3[0], mir_y, mir_bounds_mirac_3[1]])
	mir_bounds_mirac_4 = np.array([mir_bounds_mirac_4[0], mir_y, mir_bounds_mirac_4[1]])
	sea_sfc_ship = np.array([sea_sfc_ship[0], 0.0, sea_sfc_ship[1]])
	sea_sfc_dist = np.array([sea_sfc_dist[0], 0.0, sea_sfc_dist[1]])
	sensor_centre = np.array([sensor_centre[0], 0.0, sensor_centre[1]])
	sensor_centre_mirac = np.array([sensor_centre_mirac[0], mir_y, sensor_centre_mirac[1]])
	MBC = np.array([MBC[0], 0.0, MBC[1]])
	MB1 = np.array([MB1[0], 0.0, MB1[1]])
	MB2 = np.array([MB2[0], 0.0, MB2[1]])
	SHPC = np.array([SHPC[0], 0.0, SHPC[1]])
	SHP1 = np.array([SHP1[0], 0.0, SHP1[1]])
	SHP2 = np.array([SHP2[0], 0.0, SHP2[1]])
	MBC_mirac = np.array([MBC_mirac[0], mir_y, MBC_mirac[1]])
	MB1_mirac = np.array([MB1_mirac[0], mir_y, MB1_mirac[1]])
	MB2_mirac = np.array([MB2_mirac[0], mir_y, MB2_mirac[1]])
	SHPC_mirac = np.array([SHPC_mirac[0], mir_y, SHPC_mirac[1]])
	SHP1_mirac = np.array([SHP1_mirac[0], mir_y, SHP1_mirac[1]])
	SHP2_mirac = np.array([SHP2_mirac[0], mir_y, SHP2_mirac[1]])
	IR_y = 3.345		# distance between HATPRO (y=0) and IR camera
	if include_IR: 
		lens_position = np.array([lens_position[0], IR_y, lens_position[1]])
		IR_SHPC = np.array([IR_SHPC[0], IR_y, IR_SHPC[1]])
		IR_SHP1 = np.array([IR_SHP1[0], IR_y, IR_SHP1[1]])
		IR_SHP2 = np.array([IR_SHP2[0], IR_y, IR_SHP2[1]])
	

	# actual mirror vertices
	mir_bounds_5 = np.array([mir_bounds_1[0], mir_bounds_1[1] - half_mirror_len, mir_bounds_1[2]])
	mir_bounds_6 = np.array([mir_bounds_2[0], mir_bounds_2[1] - half_mirror_len, mir_bounds_2[2]])
	mir_bounds_7 = np.array([mir_bounds_2[0], mir_bounds_2[1] + half_mirror_len, mir_bounds_2[2]])
	mir_bounds_8 = np.array([mir_bounds_1[0], mir_bounds_1[1] + half_mirror_len, mir_bounds_1[2]])
	mir_bounds_9 = np.array([mir_bounds_4[0], mir_bounds_4[1] - half_mirror_len, mir_bounds_4[2]])
	mir_bounds_10 = np.array([mir_bounds_3[0], mir_bounds_3[1] - half_mirror_len, mir_bounds_3[2]])
	mir_bounds_11 = np.array([mir_bounds_3[0], mir_bounds_3[1] + half_mirror_len, mir_bounds_3[2]])
	mir_bounds_12 = np.array([mir_bounds_4[0], mir_bounds_4[1] + half_mirror_len, mir_bounds_4[2]])

	mir_bounds_mirac_5 = np.array([mir_bounds_mirac_1[0], mir_bounds_mirac_1[1] - half_mirror_len, mir_bounds_mirac_1[2]])
	mir_bounds_mirac_6 = np.array([mir_bounds_mirac_2[0], mir_bounds_mirac_2[1] - half_mirror_len, mir_bounds_mirac_2[2]])
	mir_bounds_mirac_7 = np.array([mir_bounds_mirac_2[0], mir_bounds_mirac_2[1] + half_mirror_len, mir_bounds_mirac_2[2]])
	mir_bounds_mirac_8 = np.array([mir_bounds_mirac_1[0], mir_bounds_mirac_1[1] + half_mirror_len, mir_bounds_mirac_1[2]])
	mir_bounds_mirac_9 = np.array([mir_bounds_mirac_4[0], mir_bounds_mirac_4[1] - half_mirror_len, mir_bounds_mirac_4[2]])
	mir_bounds_mirac_10 = np.array([mir_bounds_mirac_3[0], mir_bounds_mirac_3[1] - half_mirror_len, mir_bounds_mirac_3[2]])
	mir_bounds_mirac_11 = np.array([mir_bounds_mirac_3[0], mir_bounds_mirac_3[1] + half_mirror_len, mir_bounds_mirac_3[2]])
	mir_bounds_mirac_12 = np.array([mir_bounds_mirac_4[0], mir_bounds_mirac_4[1] + half_mirror_len, mir_bounds_mirac_4[2]])

	# parabolic mirror vertices:
	par_mir_bounds_3 = np.array([par_mir_bounds_2[0], par_mir_bounds_2[1] - half_sensor_len, par_mir_bounds_2[2]])
	par_mir_bounds_4 = np.array([par_mir_bounds_2[0], par_mir_bounds_2[1] + half_sensor_len, par_mir_bounds_2[2]])
	par_mir_bounds_5 = np.array([par_mir_bounds_1[0], par_mir_bounds_1[1] + half_sensor_len, par_mir_bounds_1[2]])
	par_mir_bounds_6 = np.array([par_mir_bounds_1[0], par_mir_bounds_1[1] - half_sensor_len, par_mir_bounds_1[2]])

	par_mir_bounds_mirac_3 = np.array([par_mir_bounds_mirac_2[0], par_mir_bounds_mirac_2[1] - half_sensor_len, par_mir_bounds_mirac_2[2]])
	par_mir_bounds_mirac_4 = np.array([par_mir_bounds_mirac_2[0], par_mir_bounds_mirac_2[1] + half_sensor_len, par_mir_bounds_mirac_2[2]])
	par_mir_bounds_mirac_5 = np.array([par_mir_bounds_mirac_1[0], par_mir_bounds_mirac_1[1] + half_sensor_len, par_mir_bounds_mirac_1[2]])
	par_mir_bounds_mirac_6 = np.array([par_mir_bounds_mirac_1[0], par_mir_bounds_mirac_1[1] - half_sensor_len, par_mir_bounds_mirac_1[2]])


	# y direction parts of MWR beam: '3': -y; '4': +y:
	beam_3_start = np.array([sensor_centre[0], sensor_centre[1] - half_sensor_len, sensor_centre[2]])
	beam_4_start = np.array([sensor_centre[0], sensor_centre[1] + half_sensor_len, sensor_centre[2]])

	all_okay = True
	if np.any(np.array([mm,mm1,mm2]) > 1.0) or np.any(np.array([mm,mm1,mm2]) < 0.0): # then, mirror not hit
		print("CAUTION: One part of the radiometer 'beam' doesn't hit the mirror.")
		all_okay = False


	# y axis widening from sensor to mirror, and from mirror to sea surface: First, compute the
	# distance: sensor - mirror and mirror - surface in x-z plane (y not relevant here):
	dist_sens_mir_3d = ((MBC[0] - beam_4_start[0])**2 + (MBC[2] - beam_4_start[2])**2)**0.5
	dist_mir_sfc_3d = ((SHPC[0] - MBC[0])**2 + (SHPC[2] - MBC[2])**2)**0.5
	beam_widen_y_mir = np.tan(o_angle_rad) * dist_sens_mir_3d
	beam_widen_y_sfc = np.tan(o_angle_rad) * dist_mir_sfc_3d

	# compute the coordinates of beams 3 and 4 at the mirror and sea surface:
	MB3 = np.array([MBC[0], beam_3_start[1] - beam_widen_y_mir, MBC[2]])
	MB4 = np.array([MBC[0], beam_4_start[1] + beam_widen_y_mir, MBC[2]])

	SHP3 = np.array([SHPC[0], MB3[1] - beam_widen_y_sfc, SHPC[2]])
	SHP4 = np.array([SHPC[0], MB4[1] + beam_widen_y_sfc, SHPC[2]])

	# Repeat for the other radiometer:
	beam_3_start_mirac = np.array([sensor_centre_mirac[0], sensor_centre_mirac[1] - half_sensor_len, sensor_centre_mirac[2]])
	beam_4_start_mirac = np.array([sensor_centre_mirac[0], sensor_centre_mirac[1] + half_sensor_len, sensor_centre_mirac[2]])

	dist_sens_mir_3d_mirac = ((MBC_mirac[0] - beam_4_start_mirac[0])**2 + (MBC_mirac[2] - beam_4_start_mirac[2])**2)**0.5
	dist_mir_sfc_3d_mirac = ((SHPC_mirac[0] - MBC_mirac[0])**2 + (SHPC_mirac[2] - MBC_mirac[2])**2)**0.5
	beam_widen_y_mir_mirac = np.tan(o_angle_mirac_rad) * dist_sens_mir_3d_mirac
	beam_widen_y_sfc_mirac = np.tan(o_angle_mirac_rad) * dist_mir_sfc_3d_mirac

	MB3_mirac = np.array([MBC_mirac[0], beam_3_start_mirac[1] - beam_widen_y_mir_mirac, MBC_mirac[2]])
	MB4_mirac = np.array([MBC_mirac[0], beam_4_start_mirac[1] + beam_widen_y_mir_mirac, MBC_mirac[2]])

	SHP3_mirac = np.array([SHPC_mirac[0], MB3_mirac[1] - beam_widen_y_sfc_mirac, SHPC_mirac[2]])
	SHP4_mirac = np.array([SHPC_mirac[0], MB4_mirac[1] + beam_widen_y_sfc_mirac, SHPC_mirac[2]])

	"""
	### alternative computation of MB3, MB4, SHP3, SHP4, MB3_mirac, MB4_mirac, SHP3_mirac, SHP4_mirac ###
	mm3, nn3 = vector_intersection_2d(A1=beam_3_start, 
										A2=beam_3_start + BLEN*np.array([np.cos(np.radians(360-o_angle))*np.cos(mwr_ele_rad), 
										np.sin(np.radians(360-o_angle))*np.cos(mwr_ele_rad), np.sin(mwr_ele_rad)]),
										B1=mir_bounds_5 + 0.5*(mir_bounds_6 - mir_bounds_5),	# -y end of mirror
										B2=mir_bounds_5 + 0.5*(mir_bounds_6 - mir_bounds_5) + BLEN*np.array([0,1,0])) ########################################
	mm4, nn4 = vector_intersection_2d(A1=beam_4_start, 
										A2=beam_4_start + BLEN*np.array([np.cos(o_angle_rad)*np.cos(mwr_ele_rad), 
										np.sin(o_angle_rad)*np.cos(mwr_ele_rad), np.sin(mwr_ele_rad)]),
										B1=mir_bounds_5 + 0.5*(mir_bounds_6 - mir_bounds_5),	# -y end of mirror
										B2=mir_bounds_5 + 0.5*(mir_bounds_6 - mir_bounds_5) + BLEN*np.array([0,1,0])) ########################################

	all_okay = True
	if np.any(np.array([mm,mm1,mm2,mm3,mm4]) > 1.0) or np.any(np.array([mm,mm1,mm2,mm3,mm4]) < 0.0): # then, mirror not hit
		print("CAUTION: One part of the radiometer 'beam' doesn't hit the mirror.")
		all_okay = False

	refl_factor3 = 1.0
	refl_factor4 = 1.0
	if mm3 > 1.0 or mm3 < 0.0:
		nn3 = 1.0
		refl_factor3 = 0.0
	if mm4 > 1.0 or mm4 < 0.0:
		nn4 = 1.0
		refl_factor4 = 0.0

	# intersection with mirror:
	MB3 = (beam_3_start + nn3*BLEN*np.array([np.cos(np.radians(360-o_angle))*np.cos(mwr_ele_rad), 
			np.sin(np.radians(360-o_angle))*np.cos(mwr_ele_rad), np.sin(mwr_ele_rad)]))
	MB4 = (beam_4_start + nn4*BLEN*np.array([np.cos(o_angle_rad)*np.cos(mwr_ele_rad), 
			np.sin(o_angle_rad)*np.cos(mwr_ele_rad), np.sin(mwr_ele_rad)]))

	# final points:
	FB3 = (MB3 + refl_factor3*BLEN*np.array([np.cos(np.radians(360-o_angle))*np.cos(delta_rad), 
											np.sin(np.radians(360-o_angle))*np.cos(delta_rad), np.sin(delta_rad)]))
	FB4 = (MB4 + refl_factor4*BLEN*np.array([np.cos(o_angle_rad)*np.cos(delta_rad), np.sin(o_angle_rad)*np.cos(delta_rad),
											np.sin(delta_rad)]))

	# intersection of MWR beam with sea surface:
	kk3, ll3 = vector_intersection_2d(A1=MB3, A2=FB3, B1=SHPC, B2=SHPC + BLEN*np.array([0,1,0]))
	kk4, ll4 = vector_intersection_2d(A1=MB4, A2=FB4, B1=SHPC, B2=SHPC + BLEN*np.array([0,1,0]))

	# sea surface hit point:
	SHP3 = MB3 + ll3*(FB3 - MB3)
	SHP4 = MB4 + ll4*(FB4 - MB4)


	# and all that for the second radiometer:
	beam_3_start_mirac = np.array([sensor_centre_mirac[0], sensor_centre_mirac[1] - half_sensor_len, sensor_centre_mirac[2]])
	beam_4_start_mirac = np.array([sensor_centre_mirac[0], sensor_centre_mirac[1] + half_sensor_len, sensor_centre_mirac[2]])
	mm3, nn3 = vector_intersection_2d(A1=beam_3_start_mirac, 
										A2=beam_3_start_mirac + BLEN*np.array([np.cos(np.radians(360-o_angle_mirac))*np.cos(mwr_ele_rad), 
										np.sin(np.radians(360-o_angle_mirac))*np.cos(mwr_ele_rad), np.sin(mwr_ele_rad)]),
										B1=mir_bounds_mirac_5 + 0.5*(mir_bounds_mirac_6 - mir_bounds_mirac_5),	# -y end of mirror
										B2=mir_bounds_mirac_5 + 0.5*(mir_bounds_mirac_6 - mir_bounds_mirac_5) + BLEN*np.array([0,1,0])) #############################
	mm4, nn4 = vector_intersection_2d(A1=beam_4_start_mirac, 
										A2=beam_4_start_mirac + BLEN*np.array([np.cos(o_angle_mirac_rad)*np.cos(mwr_ele_rad), 
										np.sin(o_angle_mirac_rad)*np.cos(mwr_ele_rad), np.sin(mwr_ele_rad)]),
										B1=mir_bounds_mirac_5 + 0.5*(mir_bounds_mirac_6 - mir_bounds_mirac_5),	# -y end of mirror
										B2=mir_bounds_mirac_5 + 0.5*(mir_bounds_mirac_6 - mir_bounds_mirac_5) + BLEN*np.array([0,1,0])) #############################

	all_okay = True
	if np.any(np.array([mm,mm1,mm2,mm3,mm4]) > 1.0) or np.any(np.array([mm,mm1,mm2,mm3,mm4]) < 0.0): # then, mirror not hit
		print("CAUTION: One part of the radiometer 'beam' doesn't hit the mirror.")
		all_okay = False

	refl_factor3 = 1.0
	refl_factor4 = 1.0
	if mm3 > 1.0 or mm3 < 0.0:
		nn3 = 1.0
		refl_factor3 = 0.0
	if mm4 > 1.0 or mm4 < 0.0:
		nn4 = 1.0
		refl_factor4 = 0.0

	# intersection with mirror:
	MB3_mirac = (beam_3_start_mirac + nn3*BLEN*np.array([np.cos(np.radians(360-o_angle_mirac))*np.cos(mwr_ele_rad), 
			np.sin(np.radians(360-o_angle_mirac))*np.cos(mwr_ele_rad), np.sin(mwr_ele_rad)]))
	MB4_mirac = (beam_4_start_mirac + nn4*BLEN*np.array([np.cos(o_angle_mirac_rad)*np.cos(mwr_ele_rad), 
			np.sin(o_angle_mirac_rad)*np.cos(mwr_ele_rad), np.sin(mwr_ele_rad)]))

	# final points:
	FB3_mirac = (MB3_mirac + refl_factor3*BLEN*np.array([np.cos(np.radians(360-o_angle_mirac))*np.cos(delta_rad), 
											np.sin(np.radians(360-o_angle_mirac))*np.cos(delta_rad), np.sin(delta_rad)]))
	FB4_mirac = (MB4_mirac + refl_factor4*BLEN*np.array([np.cos(o_angle_mirac_rad)*np.cos(delta_rad), np.sin(o_angle_mirac_rad)*np.cos(delta_rad),
											np.sin(delta_rad)]))

	# intersection of MWR beam with sea surface:
	kk3, ll3 = vector_intersection_2d(A1=MB3_mirac, A2=FB3_mirac, B1=SHPC_mirac, B2=SHPC_mirac + BLEN*np.array([0,1,0]))
	kk4, ll4 = vector_intersection_2d(A1=MB4_mirac, A2=FB4_mirac, B1=SHPC_mirac, B2=SHPC_mirac + BLEN*np.array([0,1,0]))

	# sea surface hit point:
	SHP3_mirac = MB3_mirac + ll3*(FB3_mirac - MB3_mirac)
	SHP4_mirac = MB4_mirac + ll4*(FB4_mirac - MB4_mirac)
	"""


	if view == 'full':
		axis_lims_x = [-2.0, 38.0]
		axis_lims_y = [-20.0, 20.0]
		axis_lims_z = [-20.0, 2.5]
	elif view == 'instrument':
		axis_lims_x = [-1.0, 3.5]
		axis_lims_y = [-0.5, 4.0]
		axis_lims_z = [-2.0, 2.5]
	elif view == 'footprint':
		# axis_lims_x = [np.floor(np.min([SHP1[0], SHP1_mirac[0]]))-0.25, np.ceil(np.max([SHP2[0], SHP2_mirac[0]]))+0.25]
		axis_lims_x = [np.min([SHP1[0], SHP1_mirac[0]])-0.3, np.max([SHP2[0], SHP2_mirac[0]])+0.3]
		axis_lims_y = [-0.5*(axis_lims_x[1]-axis_lims_x[0])+0.4, 0.5*(axis_lims_x[1]-axis_lims_x[0])+0.4]
		axis_lims_z = [-20.0, -17.0]


	f1 = plt.figure(figsize=(15,7.5))
	a1 = f1.add_subplot(111, projection='3d')	# 111: one row, one column, at position "1"

	# visualize MWR and mirror position (just as unprecise sketches)
	a1.plot(np.array([0, mwr_stand[0]]), np.array([0, mwr_stand[1]]), np.array([0, mwr_stand[2]]),
				linewidth=1.0, color=(0,0,0))
	a1.plot(np.array([mwr_stand[0], mwr_stand[0] + hgt_stand_sensor[0]]), np.array([mwr_stand[1], mwr_stand[1] + hgt_stand_sensor[1]]), 
				np.array([mwr_stand[2], mwr_stand[2] + hgt_stand_sensor[2]]),
				linewidth=1.0, color=(0,0,0))
	a1.plot(np.array([mwr_stand[0], mirror_centre[0], mirror_centre[0]]), np.array([mwr_stand[1], mirror_centre[1], mirror_centre[1]]),
				np.array([0.80, 0.80, mirror_centre[2]]),
				linewidth=1.0, color=(0,0,0))

	# repeat for other radiometer:
	a1.plot(np.array([mwr_stand_mirac[0], mwr_stand_mirac[0]]), np.array([mwr_stand_mirac[1], mwr_stand_mirac[1]]), 
				np.array([0, mwr_stand_mirac[2]]),
				linewidth=1.0, color=(0,0,0))
	a1.plot(np.array([mwr_stand_mirac[0], hgt_stand_sensor_mirac[0]]), 
				np.array([mwr_stand_mirac[1], hgt_stand_sensor_mirac[1]]), 
				np.array([mwr_stand_mirac[2], mwr_stand_mirac[2] + hgt_stand_sensor_mirac[2]]),
				linewidth=1.0, color=(0,0,0))
	a1.plot(np.array([mwr_stand_mirac[0], mirror_centre_mirac[0], mirror_centre_mirac[0]]), 
				np.array([mwr_stand_mirac[1], mirror_centre_mirac[1], mirror_centre_mirac[1]]),
				np.array([0.80, 0.80, mirror_centre_mirac[2]]),
				linewidth=1.0, color=(0,0,0))

	# visualize Polarstern guard rail and deck:
	grc_plus = guard_rail_centre[0] + 0.5*guard_rail_width
	grc_minus = guard_rail_centre[0] - 0.5*guard_rail_width
	a1.plot(np.array([grc_minus, grc_minus, grc_plus, grc_plus]), 
			np.array([axis_lims_y[0], axis_lims_y[0], axis_lims_y[0], axis_lims_y[0]]),
			np.array([guard_rail_centre[2], guard_rail_centre[2] + guard_rail_hgt, guard_rail_centre[2] + guard_rail_hgt, 
			guard_rail_centre[2]]),
			linewidth=1.0, color=(0,0,0))
	a1.plot(np.array([grc_minus, grc_minus, grc_plus, grc_plus]), 
			np.array([axis_lims_y[1], axis_lims_y[1], axis_lims_y[1], axis_lims_y[1]]),
			np.array([guard_rail_centre[2], guard_rail_centre[2] + guard_rail_hgt, guard_rail_centre[2] + guard_rail_hgt, 
			guard_rail_centre[2]]),
			linewidth=1.0, color=(0,0,0))
	a1.plot(np.array([grc_plus, grc_plus, grc_plus, grc_plus, grc_minus, grc_minus, grc_minus, grc_minus]),
			np.array([axis_lims_y[0], axis_lims_y[1], axis_lims_y[1], axis_lims_y[0], axis_lims_y[0], axis_lims_y[1],
			axis_lims_y[1], axis_lims_y[0]]),
			np.array([guard_rail_centre[2], guard_rail_centre[2], guard_rail_centre[2] + guard_rail_hgt, 
			guard_rail_centre[2] + guard_rail_hgt, guard_rail_centre[2] + guard_rail_hgt, guard_rail_centre[2] + guard_rail_hgt,
			guard_rail_centre[2], guard_rail_centre[2]]), linewidth=1.2, color=(0,0,0))

	a1.plot(np.array([axis_lims_x[0], deck_end_1[0], deck_end_2[0], deck_end_3[0], deck_end_4[0]]),
			np.array([axis_lims_y[0], axis_lims_y[0], axis_lims_y[0], axis_lims_y[0], axis_lims_y[0]]),
			np.array([0.0, deck_end_1[2], deck_end_2[2], deck_end_3[2], deck_end_4[2]]),
			linewidth=1.2, color=(0,0,0))
	a1.plot(np.array([axis_lims_x[0], deck_end_1[0], deck_end_2[0], deck_end_3[0], deck_end_4[0]]),
			np.array([axis_lims_y[1], axis_lims_y[1], axis_lims_y[1], axis_lims_y[1], axis_lims_y[1]]),
			np.array([0.0, deck_end_1[2], deck_end_2[2], deck_end_3[2], deck_end_4[2]]),
			linewidth=1.2, color=(0,0,0))
	a1.plot(np.array([deck_end_4[0], deck_end_4[0], deck_end_3[0], deck_end_3[0], deck_end_2[0], deck_end_2[0],
			deck_end_1[0], deck_end_1[0]]),
			np.array([axis_lims_y[0], axis_lims_y[1], axis_lims_y[1], axis_lims_y[0], axis_lims_y[0], axis_lims_y[1],
					axis_lims_y[1], axis_lims_y[0]]),
			np.array([deck_end_4[2], deck_end_4[2], deck_end_3[2], deck_end_3[2], deck_end_2[2], deck_end_2[2],
					deck_end_1[2], deck_end_1[2]]),
			linewidth=1.2, color=(0,0,0))

	# plot mirror and parabolic mirror bounds:
	a1.plot(np.array([par_mir_bounds_3[0], par_mir_bounds_4[0], par_mir_bounds_5[0], par_mir_bounds_6[0], par_mir_bounds_3[0]]),
				np.array([par_mir_bounds_3[1], par_mir_bounds_4[1], par_mir_bounds_5[1], par_mir_bounds_6[1], par_mir_bounds_3[1]]),
				np.array([par_mir_bounds_3[2], par_mir_bounds_4[2], par_mir_bounds_5[2], par_mir_bounds_6[2], par_mir_bounds_3[2]]),
				linewidth=1.2, color=(0,0,0))
	a1.plot(np.array([mir_bounds_5[0], mir_bounds_6[0], mir_bounds_7[0], mir_bounds_8[0], mir_bounds_5[0]]),
				np.array([mir_bounds_5[1], mir_bounds_6[1], mir_bounds_7[1], mir_bounds_8[1], mir_bounds_5[1]]),
				np.array([mir_bounds_5[2], mir_bounds_6[2], mir_bounds_7[2], mir_bounds_8[2], mir_bounds_5[2]]),
				linewidth=1.2, color=(0,0,0))
	a1.plot(np.array([mir_bounds_9[0], mir_bounds_10[0], mir_bounds_11[0], mir_bounds_12[0], mir_bounds_9[0]]),
				np.array([mir_bounds_9[1], mir_bounds_10[1], mir_bounds_11[1], mir_bounds_12[1], mir_bounds_9[1]]),
				np.array([mir_bounds_9[2], mir_bounds_10[2], mir_bounds_11[2], mir_bounds_12[2], mir_bounds_9[2]]),
				linewidth=1.2, color=(0,0,0))

	# also for the other radiometer:
	a1.plot(np.array([par_mir_bounds_mirac_3[0], par_mir_bounds_mirac_4[0], par_mir_bounds_mirac_5[0], par_mir_bounds_mirac_6[0], par_mir_bounds_mirac_3[0]]),
				np.array([par_mir_bounds_mirac_3[1], par_mir_bounds_mirac_4[1], par_mir_bounds_mirac_5[1], par_mir_bounds_mirac_6[1], par_mir_bounds_mirac_3[1]]),
				np.array([par_mir_bounds_mirac_3[2], par_mir_bounds_mirac_4[2], par_mir_bounds_mirac_5[2], par_mir_bounds_mirac_6[2], par_mir_bounds_mirac_3[2]]),
				linewidth=1.2, color=(0,0,0))
	a1.plot(np.array([mir_bounds_mirac_5[0], mir_bounds_mirac_6[0], mir_bounds_mirac_7[0], mir_bounds_mirac_8[0], mir_bounds_mirac_5[0]]),
				np.array([mir_bounds_mirac_5[1], mir_bounds_mirac_6[1], mir_bounds_mirac_7[1], mir_bounds_mirac_8[1], mir_bounds_mirac_5[1]]),
				np.array([mir_bounds_mirac_5[2], mir_bounds_mirac_6[2], mir_bounds_mirac_7[2], mir_bounds_mirac_8[2], mir_bounds_mirac_5[2]]),
				linewidth=1.2, color=(0,0,0))
	a1.plot(np.array([mir_bounds_mirac_9[0], mir_bounds_mirac_10[0], mir_bounds_mirac_11[0], mir_bounds_mirac_12[0], mir_bounds_mirac_9[0]]),
				np.array([mir_bounds_mirac_9[1], mir_bounds_mirac_10[1], mir_bounds_mirac_11[1], mir_bounds_mirac_12[1], mir_bounds_mirac_9[1]]),
				np.array([mir_bounds_mirac_9[2], mir_bounds_mirac_10[2], mir_bounds_mirac_11[2], mir_bounds_mirac_12[2], mir_bounds_mirac_9[2]]),
				linewidth=1.2, color=(0,0,0))

	# plot sea surface and rough sketch of Polarstern boundary
	a1.plot(np.array([sea_sfc_ship[0], sea_sfc_dist[0], sea_sfc_dist[0], sea_sfc_ship[0], sea_sfc_ship[0]]), 
				np.array([axis_lims_y[0], axis_lims_y[0], axis_lims_y[1], axis_lims_y[1], axis_lims_y[0]]),
				np.array([sea_sfc_ship[2], sea_sfc_dist[2], sea_sfc_dist[2], sea_sfc_ship[2], sea_sfc_ship[2]]),
				linewidth=1.2, color=(0,0,0.75))
	a1.plot(np.array([sea_sfc_ship[0], sea_sfc_dist[0], sea_sfc_dist[0], sea_sfc_ship[0], sea_sfc_ship[0]]), 
				np.array([sea_sfc_ship[1], sea_sfc_dist[1], sea_sfc_dist[1], sea_sfc_ship[1], sea_sfc_ship[1]]),
				np.array([sea_sfc_ship[2], sea_sfc_dist[2], sea_sfc_dist[2], sea_sfc_ship[2], sea_sfc_ship[2]]),
				linewidth=1.0, linestyle='dotted', color=(0,0,0))
	a1.plot(np.array([deck_end_4[0], sea_sfc_ship[0]]), np.array([axis_lims_y[0], axis_lims_y[0]]),
				np.array([deck_end_4[2], sea_sfc_ship[2]]),
				linewidth=1.2, color=(0,0,0))
	a1.plot(np.array([deck_end_4[0], sea_sfc_ship[0]]), np.array([deck_end_4[1], sea_sfc_ship[1]]),
				np.array([deck_end_4[2], sea_sfc_ship[2]]),
				linewidth=1.0, linestyle='dotted', color=(0,0,0))
	a1.plot(np.array([deck_end_4[0], sea_sfc_ship[0]]), np.array([axis_lims_y[1], axis_lims_y[1]]),
				np.array([deck_end_4[2], sea_sfc_ship[2]]),
				linewidth=1.2, color=(0,0,0))


	# plot mwr line of sight (referred to as beam): for beam centre, left and right edges
	a1.plot(np.array([sensor_centre[0], MBC[0]]), np.array([sensor_centre[1], MBC[1]]),
				np.array([sensor_centre[2], MBC[2]]),
				linewidth=1.5, color=(0.8,0,0))
	a1.plot(np.array([MBC[0], SHPC[0]]), np.array([MBC[1], SHPC[1]]), np.array([MBC[2], SHPC[2]]),
				linewidth=1.5, color=(0.8,0,0))

	a1.plot(np.array([par_mir_bounds_1[0], MB1[0]]), np.array([par_mir_bounds_1[1], MB1[1]]),
				np.array([par_mir_bounds_1[2], MB1[2]]),
				linewidth=1.5, color=(0.8,0,0), linestyle='dotted')
	a1.plot(np.array([MB1[0], SHP1[0]]), np.array([MB1[1], SHP1[1]]), np.array([MB1[2], SHP1[2]]),
				linewidth=1.5, color=(0.8,0,0), linestyle='dotted')

	a1.plot(np.array([par_mir_bounds_2[0], MB2[0]]), np.array([par_mir_bounds_2[1], MB2[1]]),
				np.array([par_mir_bounds_2[2], MB2[2]]),
				linewidth=1.5, color=(0.8,0,0), linestyle='dotted')
	a1.plot(np.array([MB2[0], SHP2[0]]), np.array([MB2[1], SHP2[1]]), np.array([MB2[2], SHP2[2]]),
				linewidth=1.5, color=(0.8,0,0), linestyle='dotted')

	# repeat for other radiometer:
	a1.plot(np.array([sensor_centre_mirac[0], MBC_mirac[0]]), np.array([sensor_centre_mirac[1], MBC_mirac[1]]),
				np.array([sensor_centre_mirac[2], MBC_mirac[2]]),
				linewidth=1.5, color=(0.8,0,0))
	a1.plot(np.array([MBC_mirac[0], SHPC_mirac[0]]), np.array([MBC_mirac[1], SHPC_mirac[1]]), np.array([MBC_mirac[2], SHPC_mirac[2]]),
				linewidth=1.5, color=(0.8,0,0))

	a1.plot(np.array([par_mir_bounds_mirac_1[0], MB1_mirac[0]]), np.array([par_mir_bounds_mirac_1[1], MB1_mirac[1]]),
				np.array([par_mir_bounds_mirac_1[2], MB1_mirac[2]]),
				linewidth=1.5, color=(0.8,0,0), linestyle='dotted')
	a1.plot(np.array([MB1_mirac[0], SHP1_mirac[0]]), np.array([MB1_mirac[1], SHP1_mirac[1]]), np.array([MB1_mirac[2], SHP1_mirac[2]]),
				linewidth=1.5, color=(0.8,0,0), linestyle='dotted')

	a1.plot(np.array([par_mir_bounds_mirac_2[0], MB2_mirac[0]]), np.array([par_mir_bounds_mirac_2[1], MB2_mirac[1]]),
				np.array([par_mir_bounds_mirac_2[2], MB2_mirac[2]]),
				linewidth=1.5, color=(0.8,0,0), linestyle='dotted')
	a1.plot(np.array([MB2_mirac[0], SHP2_mirac[0]]), np.array([MB2_mirac[1], SHP2_mirac[1]]), np.array([MB2_mirac[2], SHP2_mirac[2]]),
				linewidth=1.5, color=(0.8,0,0), linestyle='dotted')


	# mwr line of sight, y beam direction:
	a1.plot(np.array([beam_3_start[0], MB3[0], SHP3[0]]),
			np.array([beam_3_start[1], MB3[1], SHP3[1]]),
			np.array([beam_3_start[2], MB3[2], SHP3[2]]),
			linewidth=1.5, color=(0.8,0,0), linestyle='dotted')
	a1.plot(np.array([beam_4_start[0], MB4[0], SHP4[0]]),
			np.array([beam_4_start[1], MB4[1], SHP4[1]]),
			np.array([beam_4_start[2], MB4[2], SHP4[2]]),
			linewidth=1.5, color=(0.8,0,0), linestyle='dotted')

	# y beam direction for the other radiometer:
	a1.plot(np.array([beam_3_start_mirac[0], MB3_mirac[0], SHP3_mirac[0]]),
			np.array([beam_3_start_mirac[1], MB3_mirac[1], SHP3_mirac[1]]),
			np.array([beam_3_start_mirac[2], MB3_mirac[2], SHP3_mirac[2]]),
			linewidth=1.5, color=(0.8,0,0), linestyle='dotted')
	a1.plot(np.array([beam_4_start_mirac[0], MB4_mirac[0], SHP4_mirac[0]]),
			np.array([beam_4_start_mirac[1], MB4_mirac[1], SHP4_mirac[1]]),
			np.array([beam_4_start_mirac[2], MB4_mirac[2], SHP4_mirac[2]]),
			linewidth=1.5, color=(0.8,0,0), linestyle='dotted')

	# visualize and label footprint:
	a1.plot(np.array([SHP1[0], SHP2[0]]), np.array([SHP1[1], SHP2[1]]), np.array([SHP1[2], SHP2[2]]),
				linewidth=1.5, color=(0.8,0,0), label='MWR')


	# ellipsis of footprint:
	# computation of the ellipsis's extents cannot 
	e_c_hatpro_x = (SHP2[0]-SHP1[0])*0.5 + SHP1[0]		# centre of the ellipsis (x dir)
	e_c_hatpro_y = (SHP4[1]-SHP3[1])*0.5 + SHP3[1]		# centre of the ellipsis (y dir)
	e_c_hatpro_z = sea_sfc_dist[2]						# centre of the ellipsis (z dir)
	hatpro_f_a = np.abs(SHP2[0]-SHP1[0])*0.5
	hatpro_f_b = np.abs(SHP4[1]-SHP3[1])*0.5
	hatpro_ell_x = np.arange(-hatpro_f_a, hatpro_f_a + 0.000001, 0.0001)
	hatpro_ell_y1 = SHPC[1] + np.sqrt(hatpro_f_b**2 - (hatpro_f_b**2 / hatpro_f_a**2)*hatpro_ell_x**2)
	hatpro_ell_y2 = SHPC[1] - np.sqrt(hatpro_f_b**2 - (hatpro_f_b**2 / hatpro_f_a**2)*hatpro_ell_x**2)
	hatpro_ellipsis = np.array([e_c_hatpro_x + hatpro_ell_x, hatpro_ell_y1, -20.0*np.ones(hatpro_ell_x.shape)])
	hatpro_ellipsis2 = np.array([e_c_hatpro_x + hatpro_ell_x, hatpro_ell_y2, -20.0*np.ones(hatpro_ell_x.shape)])
	a1.plot(hatpro_ellipsis[0], hatpro_ellipsis[1], hatpro_ellipsis[2], linewidth=1.5, color=(0.8,0,0))
	a1.plot(hatpro_ellipsis2[0], hatpro_ellipsis2[1], hatpro_ellipsis2[2], linewidth=1.5, color=(0.8,0,0))
	a1.scatter(np.array([hatpro_ellipsis[0][0]]), np.array([hatpro_ellipsis[1][0]]), np.array([hatpro_ellipsis[2][0]]), s=50.0, marker='o', color=(0.8,0,0))
	a1.scatter(np.array([hatpro_ellipsis[0][-1]]), np.array([hatpro_ellipsis[1][-1]]), np.array([hatpro_ellipsis[2][-1]]), s=50.0, marker='o', color=(0.8,0,0))
	a1.text(e_c_hatpro_x, e_c_hatpro_y, e_c_hatpro_z, f"{SHP2[0]-SHP1[0]:.1f}" + "$\,$m", color=(0.8,0,0), 
			fontsize=fs_small, ha='center', va='top')


	# also for the other radiometer
	a1.plot(np.array([SHP1_mirac[0], SHP2_mirac[0]]), np.array([SHP1_mirac[1], SHP2_mirac[1]]), np.array([SHP1_mirac[2], SHP2_mirac[2]]),
				linewidth=1.5, color=(0.8,0,0), label='MWR')

	# ellipsis of footprint for the other radiometer:
	e_c_mirac_x = (SHP2_mirac[0]-SHP1_mirac[0])*0.5 + SHP1_mirac[0]		# centre of the ellipsis (x dir)
	e_c_mirac_y = (SHP4_mirac[1]-SHP3_mirac[1])*0.5 + SHP3_mirac[1]		# centre of the ellipsis (y dir)
	e_c_mirac_z = sea_sfc_dist[2]						# centre of the ellipsis (z dir)
	mirac_f_a = np.abs(SHP2_mirac[0] - SHP1_mirac[0])*0.5
	mirac_f_b = np.abs(SHP4_mirac[1] - SHP3_mirac[1])*0.5
	mirac_ell_x = np.arange(-mirac_f_a, mirac_f_a + 0.000001, 0.0001)
	mirac_ell_y1 = SHPC_mirac[1] + np.sqrt(mirac_f_b**2 - (mirac_f_b**2 / mirac_f_a**2)*mirac_ell_x**2)
	mirac_ell_y2 = SHPC_mirac[1] - np.sqrt(mirac_f_b**2 - (mirac_f_b**2 / mirac_f_a**2)*mirac_ell_x**2)
	mirac_ellipsis = np.array([e_c_mirac_x + mirac_ell_x, mirac_ell_y1, -20.0*np.ones(mirac_ell_x.shape)])
	mirac_ellipsis2 = np.array([e_c_mirac_x + mirac_ell_x, mirac_ell_y2, -20.0*np.ones(mirac_ell_x.shape)])
	a1.plot(mirac_ellipsis[0], mirac_ellipsis[1], mirac_ellipsis[2], linewidth=1.5, color=(0.8,0,0))
	a1.plot(mirac_ellipsis2[0], mirac_ellipsis2[1], mirac_ellipsis2[2], linewidth=1.5, color=(0.8,0,0))
	a1.scatter(np.array([mirac_ellipsis[0][0]]), np.array([mirac_ellipsis[1][0]]), np.array([mirac_ellipsis[2][0]]), s=50.0, marker='o', color=(0.8,0,0))
	a1.scatter(np.array([mirac_ellipsis[0][-1]]), np.array([mirac_ellipsis[1][-1]]), np.array([mirac_ellipsis[2][-1]]), s=50.0, marker='o', color=(0.8,0,0))
	a1.text(e_c_mirac_x, e_c_mirac_y, e_c_mirac_z, f"{SHP2_mirac[0]-SHP1_mirac[0]:.1f}" + "$\,$m", color=(0.8,0,0), 
			fontsize=fs_small, ha='center', va='top')


	# add IR camera if desired:
	if include_IR:
		# visualize tripod:
		a1.plot(np.array([lens_position[0], lens_position[0]]),
				np.array([lens_position[1], lens_position[1]]),
				np.array([deck_end_1[2], lens_position[2]]),
				linewidth=1.0, color=(0,0,0))

		# plot line of sight:
		a1.plot(np.array([lens_position[0], IR_SHPC[0]]), np.array([lens_position[1], IR_SHPC[1]]),
				np.array([lens_position[2], IR_SHPC[2]]),
				linewidth=1.5, color=(0.25, 1.0, 0.25), label='IR')
		a1.plot(np.array([lens_position[0], IR_SHP1[0]]), np.array([lens_position[1], IR_SHP1[1]]),
				np.array([lens_position[2], IR_SHP1[2]]),
				linewidth=1.5, color=(0.25, 1.0, 0.25), linestyle='dotted')
		a1.plot(np.array([lens_position[0], IR_SHP2[0]]), np.array([lens_position[1], IR_SHP2[1]]),
				np.array([lens_position[2], IR_SHP2[2]]),
				linewidth=1.5, color=(0.25, 1.0, 0.25), linestyle='dotted')


	# some dummy lines:
	a1.plot([axis_lims_x[0], deck_end_4[0]], [0,0], [0,0], linewidth=1.0, linestyle='dotted', color=(0,0,0))

	# text labels:
	# a1.text(sea_sfc_ship[0], axis_lims_y[0], sea_sfc_ship[2], "Sea surface", ha='left', va='top', 
			# color=(0,0,0.75), fontsize=fs_dwarf)
	a1.text(30.8, 1.4, sea_sfc_ship[2], "Sea surface", ha='left', va='top', 
			color=(0,0,0.75), fontsize=fs_dwarf)


	# axis properties:
	a1.axis('auto')
	a1.set_xlim(axis_lims_x[0], axis_lims_x[1])
	a1.set_ylim(axis_lims_y[0], axis_lims_y[1])
	a1.set_zlim(axis_lims_z[0], axis_lims_z[1])
	a1.minorticks_on()
	a1.grid(which='major', axis='both', color=(0.5,0.5,0.5), alpha=0.5)
	a1.grid(which='minor', axis='both', color=(0.5,0.5,0.5), alpha=0.2)

	# labels
	a1.set_xlabel("x (m)", fontsize=fs)
	a1.set_ylabel("y (m)", fontsize=fs)
	a1.set_zlabel("z (m)", fontsize=fs)

	# set box aspect ratio: for undistorted view, consider the axis limits to compute the
	# aspect ratios
	a1.set_box_aspect([np.ptp(axis_lims_x),np.ptp(axis_lims_y),np.ptp(axis_lims_z)])


	plot_file = path_plots + f"WALSEMA_mwr_geometry_hatpro_mirac_ir_{view}_view_3D.png"
	a1.view_init(elev=35, azim=-29)
	if save_figures:
		f1.savefig(plot_file, dpi=300, bbox_inches='tight')
	else:
		plt.show()

print(f"HATPRO footprint extent: {SHP2[0]-SHP1[0]:.1f} m x {SHP4[1] - SHP3[1]:.1f} m")
print(f"MiRAC-P footprint extent: {SHP2_mirac[0]-SHP1_mirac[0]:.1f} m x {SHP4_mirac[1] - SHP3_mirac[1]:.1f} m")
pdb.set_trace()
