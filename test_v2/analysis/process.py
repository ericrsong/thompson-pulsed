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
from three_level.constants import *

"""
SETTINGS: Define datasets to analyze
"""
folder = os.path.join(os.getcwd(), '..')
# dset_name = "s48551"
dset_names = [
    "s48551"   # χN = 1040 kHz, δz = 0 kHz, Θ0 = 0.5π, heating 3
    ]     

"""
SETTINGS: Processing options
"""

# Analysis values
t_bin = 1/3 * 1e-6       # 0.33 us

"""
Import dsets to analyze
"""

dsets = [e3l.Dataset(folder, dset_name, \
                     t_bin=t_bin) for dset_name in dset_names]
e3l.plot_splus(dsets, yscale="linear", show_sp_decay=True)
# e3l.plot_splus(dsets, yscale="log", show_sp_decay=True)
e3l.plot_cav_probe(dsets, yscale="log")
e3l.plot_splus_fft(dsets, yscale="linear")