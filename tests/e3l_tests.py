# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 11:13:54 2022

@author: dylan

This script shows an example processing script for the three_level experiment.
"""

# TESTING ONLY: Adds the local package temporarily to the path for testing
import sys
sys.path.insert(0, '../src')

import thompson_pulsed as tp
import thompson_pulsed.expts.three_level as e3l
import numpy as np
import matplotlib.pyplot as plt

import os

# TESTING ONLY: Remove package from path when done
sys.path.remove('../src')

"""
Define files to import as sequences of the experiment
"""
zpad = 0
n_seq_0, n_seqs = 0, 10

file_title = "3L_TEST_NS"
folder = 'example_data/e3l'
if not os.path.exists(folder):
	raise FileNotFoundError("""
		Example data not found. Make sure you have copied the 'example_data'
		folder from OneDrive into your thompson_pulsed distribution's tests/
		directory and try again.
		""")

"""
Define experimental parameters
"""
params = e3l.Parameters()
params.t_run = 50 * 1e-6        # 50 us
params.t_bin = 0.5 * 1e-6       # 0.5 us
params.t_drive = 0.5 * 1e-6     # 0.5 us
params.t_fft_pad = 100 * 1e-6   # 100 us
params.t_cav_pulse = 5 * 1e-6   # 5 us
params.f0_cav = 10 * 1e6        # 10 MHz
params.f0_atom = 10 * 1e6       # 10 MHz
params.fft_fit = e3l.sinc_symm_fitter(params.t_bin)
dt = 1/60 * 1e-6
params.demod_smoother = lambda V: \
    e3l.moving_average(V, round(1/params.f0_atom/dt/2))

"""
Create experiment object and load with sequences
"""
expt = e3l.Experiment(params)
for i in range(n_seq_0, n_seq_0 + n_seqs):
    file_num = str(i).zfill(zpad)
    file_name = file_title + file_num + '.txt'
    file = os.path.join(folder, file_name)

    expt.load_sequence(file)

    print(f'Sequence {i+1} of {n_seqs} loaded.')

expt.preprocess()

"""
Cavity resonance frequency probe
"""
# cav_freqs = expt.data.track_cav_frequency_iq()
# cav_freqs_mean = np.mean(cav_freqs.V, axis=0)
# cav_freqs_stdmean = np.std(cav_freqs.V, axis=0) / np.sqrt(cav_freqs.V.shape[0])

# plt.figure()
# plt.errorbar(cav_freqs.t, cav_freqs_mean, cav_freqs_stdmean)

plt.figure()
for atom_run in expt.data.atom_runs.V[0, :10, ...]:
    plt.plot(expt.data.atom_runs.t, atom_run)
    
"""
Atomic self-radiated field probe
"""
atom_demod = expt.data.demod_atom_trace()

plt.figure()
for run in atom_demod.V[:10]:
    plt.plot(atom_demod.t, run)

atom_demod_fft = atom_demod.fft()
plt.figure()
for i in range(10):
    plt.plot(atom_demod_fft.f, np.abs(atom_demod_fft.V[i,...])**2)

plt.figure()
plt.plot(atom_demod.t, np.mean(atom_demod.V[:25,...], axis=0))