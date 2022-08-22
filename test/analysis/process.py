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
# Analysis options
option_use_mag2_average = True
option_include_pulse_in_plots = False
option_plot_mag2 = True
option_show_photons_lost = False
option_fit_decay = True
option_show_sp_decay = False
option_shots_used_per_seq = 8

# Main plots
plot_sr_field = True
plot_cav_probe = False
plot_sr_mag_fft = False

# Diagnostics
plot_cav_log = False
plot_higgs_finder = False
plot_cav_photons = False
plot_fi_shot = True
plot_intensity_vs_shot = False

# Analysis values
t_bin = 1/3 * 1e-6       # 0.33 us

"""
Import dsets to analyze
"""

dsets = [e3l.Dataset(folder, dset_name, \
                     t_bin=t_bin ) for dset_name in dset_names]
e3l.plot_splus(dsets, yscale="linear", show_sp_decay=True)
# e3l.plot_splus(dsets, yscale="log", show_sp_decay=True)
e3l.plot_cav_probe(dsets, yscale="log")



# """
# BODY: Initialize multiset figures and data containers
# """
# if plot_sr_field:
#     fig_time, ax_time = plt.subplots(1, 1, figsize=(6, 4.5))
#     if option_show_photons_lost:
#         ax_time2 = ax_time.twinx()
# if plot_sr_mag_fft:
#     fig_fft, ax_fft = plt.subplots(1, 1, figsize=(6, 4.5))
# if plot_cav_probe:
#     fig_cav, ax_cav = plt.subplots(1, 1, figsize=(5, 4.5))
# if plot_cav_photons:
#     fig_mc, ax_mc = plt.subplots(1,1)
# if plot_fi_shot:
#     fig_fishot, ax_fishot = plt.subplots(1, 1, figsize=(6, 4.5))
# if plot_intensity_vs_shot:
#     fig_intvshot, ax_intvshot = plt.subplots(1, 1, figsize=(6,4.5))
# if plot_higgs_finder:
#     fig_higgs, ax_higgs = plt.subplots(1, 1, figsize=(6,4.5))




# # # Sort sets by fi and plot them to show where the fi are
# # fig, ax = plt.subplots(1,1)
# # ax.plot(np.sort(np.average(self.data.fi,axis=1)),'.')
# # ax.grid()




# """
# PLOTTING: Plot data for set
# """
# if plot_cav_probe:        
#     ax_cav.errorbar(t_us_cav, cav_freq_shift * 1e-3, yerr=cav_freqs.dV * 1e-3,
#                         fmt='.-', color='darkmagenta', alpha=0.5, label=set_string)
    
#     if plot_cav_log:
#         cav_freq_shift = cav_freqs.V - f0
        
#         # ifit = np.argwhere( (t_us_cav > 10) & (t_us_cav < 30) ).flatten()
#         # pOpt, pCov = curve_fit(fit_exp, t_us_cav[ifit], cav_freq_shift[ifit],
#         #                        sigma=cav_freqs.dV[ifit],
#         #                        p0=(20, 500), bounds=([0,1],[100,np.inf]))
        
#         ax_cav.errorbar(t_us_cav, cav_freq_shift * 1e-3, yerr=cav_freqs.dV * 1e-3,
#                             fmt='.-', color='darkmagenta', alpha=0.5, label=set_string)
        
#         ax_cav.set_yscale('log')
#         ax_cav.set_ylim((1e1,1e3))
#         # ax_cav.plot(t_us_cav, fit_exp(t_us_cav, *pOpt) * 1e-3, 'k', label=f'tau = {round(pOpt[0],1)} us')
#     else:
#         ax_cav.errorbar(t_us_cav, cav_freq_shift * 1e-3, yerr=cav_freqs.dV * 1e-3,
#                             fmt='.-', color='darkmagenta', alpha=0.5, label=set_string)
#         # ax_cav.set_ylim((-set_vals*1.1)) # For plots vs. χN
#         ax_cav.set_ylim((-900, -100))
    
#     ax_cav.set_xlim((-1,40))        
#     ax_cav.set_ylabel(r'$f_{cav}$ (kHz)')
#     ax_cav.grid()
#     ax_cav.legend()

# if plot_cav_photons:
#     mag = self.data._seq_cav_probe_mag()
#     mc = alpha_c * np.average(mag.V, axis=0)**2
#     ax_mc.plot(mag.t * 1e6, mc * 1e-3, label=set_string)

# if plot_sr_field:
#     t_us_atom = (atom_demod.t-t0)*1e6
#     ax_time.plot(t_us_atom, coherence_normalized, 'k', label=set_string)
#     ax_time.fill_between(t_us_atom,
#                             coherence_normalized + err_coherence,
#                             coherence_normalized - err_coherence,
#                             color='k', alpha=0.25)
#     ax_time.grid()
#     ax_time.set_xlim((0,20))
#     ax_time.set_ylim((0, 0.09))
#     # ax_time.set_yscale('log')
#     # ax_time.set_ylim((1e-3,1e-1))
#     if option_plot_mag2:
#         ax_time.set_ylabel(r'$|S^{+}|^2/N^2$')
#     else:
#         ax_time.set_ylabel(r'$|S^{+}|/N$')
        
#     if option_fit_decay:
#         ax_time.plot(t_us_atom, fit_decay(t_us_atom, *pOpt_decay), '--k', alpha=0.5,
#                         label=fr'tau = {round(pOpt_decay[0],2)} us')
    
#     if option_show_sp_decay:
#         ax_time.plot(t_us_atom, coherence_decay_n2, '--', color='darkcyan', alpha=0.5,
#                         label='Decay from radial temp')
#         ax_time.plot(t_us_atom, coherence_decay_n3, '--', color='indigo', alpha=0.5,
#                         label='Decay from radial and axial temp')
    
#     if option_show_photons_lost:
#         ax_time2.semilogy((atom_demod.t - t0) * 1e6, M_emitted, '--', color='chocolate')
#         ax_time2.set_ylim((10**1, 10**5))
#         ax_time2.set_ylabel('Photons lost', color='chocolate')
#     ax_time.legend()
    
#     xs.append(t_us_atom)
#     ys.append(coherence_normalized)

# if plot_sr_mag_fft:
#     i0 = int(mag2_fft.V.shape/2)
#     PSD = np.abs(mag2_fft.V)**2
#     # ax_fft.plot(mag2_fft.f * 1e-6, PSD/np.max(PSD), color='forestgreen', label=set_string)
#     ax_fft.plot(mag2_fft.f * 1e-6, PSD, color='forestgreen', label=set_string)
#     # ax_fft.set_yscale('log')
#     ax_fft.set_xlim((0,6))
#     # ax_fft.set_ylim((1e-5,1e-1))
#     ax_fft.set_ylim((0,0.5))
#     ax_fft.set_ylabel('PSD (arb.)')
#     ax_fft.grid()
#     ax_fft.legend()
    
#     if plot_higgs_finder:
#         ax_higgs.plot(higgs_finder.t * 1e6, higgs_finder.V, color='darkgray', label=set_string)
#         ax_higgs.set_xlim((0,20))
#         ax_higgs.set_ylim((-0.01,0.01))
#         ax_higgs.grid()
#         ax_higgs.set_ylabel(r'$|S^{+}|^2$ minus exponential fit')
#         ax_higgs.legend()
    
# if plot_fi_shot:
#     fi_seq_avg = np.average(self.data.fi, axis=0)
#     runs = np.arange(self.data.fi.shape[1])
#     ax_fishot.plot(runs, fi_seq_avg*1e-3, '-ok')
#     ax_fishot.set_ylabel(r'$f_{cav}$ (kHz)')
#     ax_fishot.grid()

# if plot_intensity_vs_shot:
#     # Use detected peak position to align phasors
#     post_phasors = self.postmeasure.demod_atom_trace()
#     t_all, Mc_shots = post_phasors.t, post_phasors.mag().V**2 * alpha_a
#     ti_pulse = 7.3e-6
#     tf_pulse = 7.45e-6
#     idx = np.where((t_all >= ti_pulse) & (t_all < tf_pulse))
    
#     phase_shots = np.sum(self.params.dt * 2*g0689 * np.sqrt(Mc_shots[:,idx]), axis=-1)
    
#     ax_intvshot.plot(phase_shots, 'ok')
#     ax_intvshot.set_ylim((0,None))
#     ax_intvshot.grid()
#     ax_intvshot.set_ylabel(r'Est. $\theta_{Rabi}$')  
    
# """
# Finish setting up figures
# """
# if plot_cav_probe:
#     ax_cav.set_xlabel(r'Time ($\mu s$)')
#     ax_cav.set_title(r'$f_{cav}$ vs. time')
#     fig_cav.tight_layout()

# if plot_cav_photons:
#     ax_mc.set_xlabel(r'Time ($\mu s$)')
#     ax_mc.set_ylabel(r'$M_c (\times 10^3)$')
#     ax_mc.set_title(r'$M_c$ vs. time')
#     ax_mc.grid()
#     ax_mc.legend()
#     fig_mc.tight_layout()

# if plot_sr_field:
#     ax_time.set_xlabel(r'Time ($\mu s$)')
#     if option_plot_mag2:
#         ax_time.set_title(r'$|S^{+}|^2$ vs. time')
#     else:
#         ax_time.set_title(r'$|S^{+}|$ vs. time')
#     fig_time.tight_layout()

# if plot_sr_mag_fft:
#     ax_fft.set_xlabel('Frequency (MHz)')
#     ax_fft.set_title('Fourier transform: t_pad = 250 us')
#     fig_fft.tight_layout()
    
#     if plot_higgs_finder:
#         ax_higgs.set_xlabel(r'Time ($\mu s$)')
    
# if plot_fi_shot:
#     ax_fishot.set_title('Cavity frequency vs. shot within a sequence')
#     ax_fishot.set_xlabel('Shot count')
#     fig_fishot.tight_layout()

# if plot_intensity_vs_shot:
#     ax_intvshot.set_title('Intensity fluctuations in Rabi drive')
#     ax_intvshot.set_xlabel('Shot count')
#     fig_intvshot.tight_layout()