import fs
import sys
import datetime as dt
import os
import pdb
import time
import tkinter as tk
import tkinter.messagebox as tkmb
import glob

sys.path.insert(0, "/net/blanc/awalbroe/Codes/MOSAiC/")


"""
	This script checks if new files have been created OR existing files grow.
	The directory to be checked must be handed over as system argument when
	calling the script: e.g.: python3 data_growth_sentinel.py "/net/blanc/awalbroe/Plots/spammer/"
	- record time when monitoring starts
	- increment a time variable every time a check (new files or file growth) is performed
	- dictionary that stores filenames as keys, size as data
	- length of dictionary key list equals number of files in directory, also must be conserved
	- check every 15 or 30 seconds
	- if no new file has been created or no file has grown, print huge warning
"""

time_sleep = 10		# interval of checks in seconds

if len(sys.argv) == 1:
	raise IndexError("This script must be called by including the path to the directory to be " +
						"monitored. Example: python3 data_growth_sentinel.py " +
						"'/net/blanc/awalbroe/Plots/spammer/'")
elif len(sys.argv) == 2:
	path_output = sys.argv[1]
else:
	raise RuntimeError("Too many arguments when calling this script.")


# establish connection to remote file system: remote_fs.listdir can be used instead of os.listdir()
# or glob.glob(). File size can be inquired via remote_fs.filterdir() and then .size
username = ""
host = ""
pw = "---------"
remote_fs = fs.open_fs(f"ssh://{username}:{pw}@{host}{path_output}")


# record time when monitoring starts:
monitoring_start_time = dt.datetime.utcnow()
last_update_time = dt.datetime.utcnow()
time_dif = dt.timedelta(seconds=0)
time_thres = dt.timedelta(minutes=1)	# threshold: if exceeded -> give warning

# open loop that will only be closed when the script is quit manually or when no new data comes in:
# dictionaries will save file sizes and filenames:
all_okay = True		# False when either no new file has been created or existing files didn't grow
dict_files_old = dict()
dict_files_new = dict()
while all_okay:
	# update dict_files_old:
	for key in dict_files_new.keys():
		dict_files_old[key] = dict_files_new[key]

	# safe some stats:
	n_files_old = len(dict_files_old.keys())

	# investigate which files exist; split filename from path:
	remote_dir = remote_fs.filterdir("./", files=["*"], namespaces=['details'])
	n_files_new = len(remote_fs.listdir("./"))

	# update dict_files_new:
	for file in remote_dir:
		dict_files_new[file.name] = file.size

	# assert len(dict_files_new.keys()) == n_files_new

	# comparing stats:
	new_file = n_files_old < n_files_new
	file_grows = list()
	for key in dict_files_new.keys():
		if key in dict_files_old.keys():
			file_grows.append(dict_files_old[key] < dict_files_new[key])

	# if no new file was found and if all files have the same size (file_grows == False):
	# increment time_dif
	if (not new_file) and (not any(file_grows)):
		time_dif = dt.datetime.utcnow() - last_update_time
	else:	# then there was an update: set time_dif to 0
		last_update_time = dt.datetime.utcnow()
		time_dif = dt.timedelta(seconds=0)

	print(f"{dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')}: Last update was {time_dif.seconds} " +
			"seconds ago.")

	if time_dif >= time_thres: # print warning
		info_message = "HEY, WAKE UP! SOMETHING'S WRONG WITH THE DATA. I CAN'T FIND ANY NEW STUFF :("
		print(info_message)

		master = tk.Tk()
		master.title("Oops!")

		btn = tk.Button(master, text="Got it! (Click X on top right)", command=tk.Label(master, text=info_message).grid(row=2, 
		column=1))
		btn.grid(row=3, column=1, pady=4)

		master.mainloop()
		all_okay = False

	# if time_dif > dt.timedelta(seconds=120): pdb.set_trace()

	# pause script for certain time:
	time.sleep(time_sleep)


print("Check instruments.")