# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, pi
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

import os
import pickle

import thompson_pulsed as tp # v0.7.0
import three_level as e3l

"""
SETTINGS: Define all datasets to preprocess
"""
folder = os.path.join(os.getcwd(), '..')
sd48551 = e3l.SetDef(folder, "s48551", seq0=48551, nseq=20, chiN_nom=1040, deltaZ=0, theta0=0.5, heating=3)

# Sets to preprocess
sds = [sd48551]

"""
SETTINGS: Define experimental parameters and options
"""
params = e3l.Parameters()
params.t_run = 50 * 1e-6        # 50 us
params.t_drive = 0.3 * 1e-6     # 0.3 us
params.t_cav_pulse = 5 * 1e-6   # 5 us
params.f0_cav = 10 * 1e6        # 10 MHz
params.f0_atom = 8.75 * 1e6       # 10 MHz
params.dt = 1/60 * 1e-6

option_reload_all = False

"""
Iterate over sets
"""
for j in range(len(sds)):
	sd = sds[j]
	pkl_file = os.path.join(sd.folder, sd.dset_name + '.pickle')
	
	if not os.path.exists(sd.folder):
		raise FileNotFoundError("""
			Data not found. Make sure you have the right folder!
			""")
			
	# OPTION: Skip sets which have already been preprocessed, unless told otherwise
	if os.path.exists(pkl_file) and (not option_reload_all):
		print(f'Skipping set {sd.dset_name} ({j+1} of {len(sds)}).')
		continue
	
	# Create experiment object for set and preprocess
	expt = e3l.Experiment(params)
	for i in range(sd.seq0, sd.seq0 + sd.nseq):
		file_num = str(i).zfill(sd.zpad)
		file_name = sd.expt_title + file_num + '.txt'
		file = os.path.join(folder, file_name)
	
		expt.load_sequence(file)
	
		print(f'Sequence {i-sd.seq0+1} of {sd.nseq} loaded.')
	
	data, premeasure, postmeasure = expt.preprocess(
		premeasure_interleaved = True, postmeasure = 5, n_warmups = 1)
	
	# Bundle up data and pickle
	pkl_dict = {
		"data": data,
		"premeasure": premeasure,
		"postmeasure": postmeasure,
		"sd": sd
	}
	pickle.dump(pkl_dict, open(pkl_file, 'wb'))
	
	print(f'Set {sd.dset_name} ({j+1} of {len(sds)}) preprocessed and pickled.')