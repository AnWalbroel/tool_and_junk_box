import os
import sys
import glob
import pdb
import datetime as dt

wdir = os.getcwd() + "/"

import numpy as np
import xarray as xr
import matplotlib as mpl
mpl.use("WebAgg")
import matplotlib.pyplot as plt


def import_eq_data_txt(file):

	"""
	Imports earthquake data copied from https://www.volcanoesandearthquakes.com/map/Italy
	into a text (.txt) file and returns an xarray dataset.

	Parameters:
	-----------
	file : str
		String with the full path and filename of the data.
	"""

	# open file:
	with open(file, 'r', encoding='utf-8', errors='ignore') as f_handler:

		n_data = 0
		for k, line in enumerate(f_handler):
			line_elems = line.split("\t")
			if len(line_elems) > 2: n_data += 1

		# initialize arrays:
		data_dict = {'time': np.full((n_data,), np.datetime64("1970-01-01T00:00:00")),
					'lat': np.full((n_data,), np.nan),
					'lon': np.full((n_data,), np.nan),
					'depth': np.full((n_data,), np.nan),
					'mag': np.full((n_data,), np.nan)}

		f_handler.seek(0,0)		# reset handler to beginning
		n = 0
		year = 0
		for k, line in enumerate(f_handler):
			line_elems = line.split("\t")
			n_elems = len(line_elems)

			if n_elems == 5:

				# extract data:
				time_str_elems = line_elems[0].strip().split(" ")
				hhmmss = time_str_elems[3]
				mon = int(dt.datetime.strptime(time_str_elems[2], "%b").month)
				day = int(time_str_elems[1])
				data_dict['time'][n] = np.datetime64(f"{year:04}-{mon:02}-{day:02}T{hhmmss}")

				mag_dep_elems = line_elems[1].strip().split(" ")
				data_dict['mag'][n] = float(mag_dep_elems[1])
				data_dict['depth'][n] = float(mag_dep_elems[3].replace("km",""))*(-1000.)

				loc_elems = line_elems[3].strip().split(" / ")
				data_dict['lat'][n] = float(loc_elems[0])
				data_dict['lon'][n] = float(loc_elems[1])

				n += 1

			elif n_elems == 1:	# to be used to extract the year
				year = int(line_elems[0].strip()[-4:])

			else:
				print("unexpected number of entries....")
				pdb.set_trace()


	# put data into xarray dataset:
	DS = xr.Dataset(coords={'time': (['time'], data_dict['time'].astype('datetime64[ns]'))})

	attr_dict = {'mag': {'long_name': "magnitude", 'units': "Richter Scale"},
				'depth': {'long_name': "depth below sea level / surface?", 'units': 'm'},
				'lat': {'long_name': "latitude", 'units': 'deg north'},
				'lon': {'long_name': "longitude", 'units': 'deg east'}}
	for key in data_dict.keys():
		if key != 'time':
			DS[key] = xr.DataArray(data_dict[key], dims=['time'], attrs=attr_dict[key])

	# sort the dataset chronologically:
	DS = DS.sortby('time')

	return DS


def plot_time_coloured(DS, set_dict):
	fs = 14			# fontsize

	f1 = plt.figure(figsize=(16,9))
	a1 = plt.axes(projection='3d')


	# build colormap:
	n_data = len(DS.time)
	n_cmap_sec = 5
	n_levels_p = int(n_data / n_cmap_sec)

	# build cmap:
	cmap_dict = {'0': mpl.cm.get_cmap('Purples', n_levels_p*2),
				'1': mpl.cm.get_cmap('Blues', n_levels_p*2),	# *2 because I only take one half of the cmap
				'2': mpl.cm.get_cmap('Greens', n_levels_p*2),
				'3': mpl.cm.get_cmap('YlOrBr', n_levels_p*2),
				'4': mpl.cm.get_cmap('Reds', n_levels_p*2)}
	for key in cmap_dict.keys():
		# pdb.set_trace()
		cmap_dict[key] = cmap_dict[key](range(int(n_levels_p*2)))
		cmap_dict[key] = cmap_dict[key][int(0.25*n_levels_p*2):,:]

		# stack cmaps:
		if key == '0': 
			cmap = cmap_dict[key]

		else:
			cmap = np.concatenate((cmap, cmap_dict[key]), axis=0)

	# convert back to colormap:
	cmap = mpl.colors.ListedColormap(cmap)

	# cmap = plt.cm.get_cmap('turbo', n_data)

	sc = a1.scatter(DS.lon.values, DS.lat.values, DS.depth.values, s=40.0*DS.mag.values,
				c=DS.time.values, marker='o', cmap=cmap)

	a1.set_xlim(set_dict['lon_lims'])
	a1.set_ylim(set_dict['lat_lims'])
	a1.set_zlim(0., -5000.)


	# colorbar(s) and legends:
	lh, ll = sc.legend_elements(prop='sizes', alpha=0.6, num=5, fmt="{x:.1f}", func=lambda x: x/40.0)
	a1.legend(lh, ll, loc='upper left', title='Magnitude')

	cb_var = f1.colorbar(mappable=sc, ax=a1, orientation='vertical', 
							fraction=0.04, pad=0.05, shrink=0.65,
							ticks=np.arange(set_dict['min_date'], 
											set_dict['max_date'] + np.timedelta64(1, "D"), 
											np.timedelta64(1, "D")).astype("datetime64[ns]").astype(np.float64))
	cb_var.ax.set_yticklabels(cb_var.get_ticks().astype("datetime64[ns]").astype("datetime64[h]").astype("str"))
	cb_var.ax.tick_params(labelsize=fs-6)

	a1.set_xlabel("Longitude ($^{\circ}\mathrm{E}$)", fontsize=fs)
	a1.set_ylabel("Latitude ($^{\circ}\mathrm{N}$)", fontsize=fs)
	a1.set_zlabel("Depth (m)", fontsize=fs)
	dt_range_str = (str(DS.time.values[0].astype('datetime64[s]')) + " - " +
						str(DS.time.values[-1].astype('datetime64[s]')))
	dt_range_str_file = (str(DS.time.values[0].astype('datetime64[s]')).replace("-","").replace("T","").replace(":","") + "-" +
						str(DS.time.values[-1].astype('datetime64[s]')).replace("-","").replace("T","").replace(":",""))
	a1.set_title("Earthquake depths 3D, time coloured; \n" + dt_range_str,
					fontsize=fs)
	a1.invert_zaxis()

	if set_dict['save_figures']: 
		plotfile = set_dict['path_plots'] + f"Italy_EQ_depth_{dt_range_str_file}.jpg"
		f1.savefig(plotfile, dpi=400)
		print(f"Saved {plotfile}....")
		
	else: 
		plt.show()


"""
	This script visualizes earthquakes in a 3D space (x,y,z = longitude,latitude,height/depth) 
	with colours and markersize indicating the time and magnitude of the earthquakes, respectively.
"""


# paths:
path_data = wdir		# files are in the same folder as the script
path_plots = wdir + "Plots/"


# settings:
set_dict = {"save_figures": False,
			"path_plots": path_plots,
			'lat_lims': np.array([40.65, 40.95]),
			'lon_lims': np.array([13.9, 14.6]),
			}

# create non-existing plot path:
plot_dir = os.path.dirname(path_plots)
if not os.path.exists(plot_dir):
	os.makedirs(plot_dir)


# import earthquake data:
file_data = path_data + "eq_20231001-20231012.txt"
DS = import_eq_data_txt(file_data)


# cut to the desired area:
idx_loc = np.where((DS.lon >= set_dict['lon_lims'][0]) & (DS.lon <= set_dict['lon_lims'][1]) & 
					(DS.lat >= set_dict['lat_lims'][0]) & (DS.lat <= set_dict['lat_lims'][1]))[0]
DS = DS.isel(time=idx_loc)
set_dict['min_date'] = DS.time.values[0].astype('datetime64[D]')
set_dict['max_date'] = DS.time.values[-1].astype('datetime64[D]')
plot_time_coloured(DS, set_dict)
