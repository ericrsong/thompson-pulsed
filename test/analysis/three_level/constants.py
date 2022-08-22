# -*- coding: utf-8 -*-

from scipy.constants import c, pi
import numpy as np

cav_coupling = 0.54
eta_m = np.sqrt(cav_coupling)
kappa689 = 2*pi * 153. * 1e3
omega689 = 2*pi * c/(689*1e-9)
Delta = 2*pi * 51.7 * 1e6
g0689 = 2*pi * 10.6 * 1e3
chi = g0689**2/Delta
gamma689 = 2*pi * 7.5 * 1e3
C689 = 4*g0689**2 / (kappa689 * gamma689)

alpha_c = 0.873 * 1e6 # V^-2
alpha_a = 0.382 * 1e6 # V^-2

# Calibration data and physics values
JPerp_per_V = Delta/g0689 * np.sqrt(alpha_a) # V^-1
