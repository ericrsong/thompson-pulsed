# -*- coding: utf-8 -*-
import numpy as np
from scipy.constants import c, pi
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

import os
import pickle

import thompson_pulsed as tp # v0.7.0
from .constants import chi, g0689, kappa689, \
    alpha_a, JPerp_per_V, Tax_dict, Trad_dict

__all__ = ["SetDef", "Dataset"]


def fit_exp(x, x0, A):
    return( A*np.exp(-x/x0) )

class SetDef():
    """
    Object containing metadata about a particular dataset.
    """
    def __init__(self, folder, dset_name, seq0, nseq, chiN_nom, deltaZ, theta0, heating,
                 misc="", expt_title="e3L_NS", zpad=0):
        self.folder = folder
        self.dset_name = dset_name
        self.seq0 = seq0
        self.nseq = nseq
        
        self.chiN = chiN_nom
        self.deltaZ = deltaZ
        self.theta0 = theta0
        self.heating = heating
        
        # Optional args
        self.misc = misc
        self.expt_title = expt_title
        self.zpad = zpad

class Dataset():
    """
    Object to contain processing methods and processed data
    """
    def __init__(self, folder, dset_name,
                 t_bin = 0.5 * 1e-6, **kwargs):
        pkl_file = os.path.join(folder, dset_name + '.pickle')
        if not os.path.exists(pkl_file):
            raise FileNotFoundError(f"""
            		Data not found for dataset {dset_name}. Have you preprocessed all files?
            		""")
        else:
            pkl_dict = pickle.load(open(pkl_file, 'rb'))
            
            self.data = pkl_dict["data"]
            self.premeasure = pkl_dict["premeasure"]
            self.postmeasure = pkl_dict["postmeasure"]
            self.sd = pkl_dict["sd"]
            self.params = self.data.params
        
        # Quantities filled by process_data()
        self.splus = None                   # Time_Multitrace
        self.splus_sp = None                # Time_Multitrace
        self.splus_sp3 = None               # Time_Multitrace
        self.splus_longdecay = None         # Time_Multitrace
        self.splus_longdecay_full = None    # Time Multitrace
        self.splus_fft = None               # Frequency_Multitrace
        
        self.M_emitted = None               # Time_Multitrace
        self.Omega0 = None                  # Time Multitrace
        
        self.cav_probe_shift = None         # Time_Multitrace
        self.cav_probe_N = None             # Time Multitrace
        self.cav_f0 = None                  # float

        
        self.process_data(t_bin, **kwargs)
            
    def process_data(self, t_bin,
                     shots_used_per_seq = None, 
                     include_pulse_in_traces = False,
                     use_mag2_average = True,
                     plot_mag2 = True,
                     assert_chiN = None):
        
        """
        From postmeasure, calculate Rabi frequency, total drive phase, and pulse length
        """
        ####
        # Use self.postmeasure drive pulse to calibrate drive angle θ0
        ####
        
        # Detect sides of drive using a threshold
        post_avg_mag2 = self.postmeasure.demod_atom_trace().mag2().average_over(axis=0)
        imax = np.argmax(post_avg_mag2.V)
        Vmax = post_avg_mag2.V[imax]
        threshold = 0.5 * Vmax
        di_i = np.argwhere(post_avg_mag2.V[imax::-1] < threshold)[0,0]
        di_f = np.argwhere(post_avg_mag2.V[imax:] < threshold)[0,0]
        ti, tf = post_avg_mag2.t[imax - di_i], post_avg_mag2.t[imax + di_f]
        ti_pulse = ti - 200 * 1e-9
        tf_pulse = tf + 200 * 1e-9
        t_pulse = (di_i + di_f) * self.data.params.dt # length of pulse

        # Average phasors to estimate Rabi frequency
        # Note: using mag2 average instead adds in unwanted noise and messes up calibration
        post_avg_phasor = self.postmeasure.demod_atom_trace(ti).average_over(axis=0)
        post_mag2 = post_avg_phasor.mag2()

        # Apply calibration to estimate intracavity photon number and Rabi frequency
        Mc_est = post_mag2.V * alpha_a
        Omega_est = 2*g0689 * np.sqrt(Mc_est)
        # Omega_slope = 2*g0689 * np.sqrt(alpha_a)
        Omega_avg = np.max(Omega_est)/2

        idx = np.where((post_mag2.t > ti_pulse) & (post_mag2.t < tf_pulse))
        phase = np.sum(Omega_est[idx] * self.params.dt)
        print(f'{round(phase/pi,3)} pi')
        
        self.Omega0 = tp.Time_Multitrace(post_mag2.t[idx], Omega_est[idx])
        self.sd.theta0 = phase/pi
        
        """
        Calculate noise floor, align |S+|^2 data
        """
        # Calculate quantum noise floor of |S+| measurements
        # If z = z0 + δz, <|z|^2> = |z0|^2 + <|δz|^2>
        # Assume <|δz^2|> = <|z|^2> during premeasure when we expect |S+| = 0
        premeasure_demod = self.premeasure.demod_atom_trace(t_align=None)
        avg_mag2_premeasure = premeasure_demod.mag2().average_over(axis=0).truncate(t_min=(self.params.t_run*0.1), t_max=(self.params.t_run*0.9))
        avg_mag2_floor = np.average(avg_mag2_premeasure.V, weights=1/avg_mag2_premeasure.dV**2)
            
        # Find drive time empirically to get align time and t0 for plotting
        peak_finder = self.data.demod_atom_trace().mag2().average_over(axis=0)
        threshold = 0.3 * np.max(peak_finder.V)
        peaks, __ = find_peaks(peak_finder.V, prominence = threshold)
        
        t_drive_center = peak_finder.t[ peaks ][0]
        t_align = t_drive_center - t_pulse/2
        t0 = t_align \
            if (include_pulse_in_traces is True) \
            else (t_drive_center + t_pulse/2)
        
        """
        Cavity probe and estimate of ND
        """
        cav_freqs = self.data.track_cav_frequency_iq(t_bin)

        cav_freq_shift = cav_freqs.V
        dt_freq_avg = 2.5e-6
        n0_bins = int(dt_freq_avg / t_bin)
        f0 = np.average(cav_freqs.V[1:n0_bins+1])
        print(f'f0 = {round(f0/1e3)} kHz')

        # chi N dependent values
        chiN_div_2 = 2*pi * (-f0)
        if assert_chiN is not None:
            chiN_div_2 = 2*pi * (assert_chiN * 1e3)
        n_atoms = chiN_div_2 * 2 / chi

        # # Cavity frequency shift after drive
        # id0 = int(np.argmax(t_us_cav > 0))
        # fd0 = np.average(cav_freqs.V[id0:n0_bins+id0])

        # Convert cavity frequency jump to Ne (assumes chi -> chi/2 for inhom coupling)
        df_to_N = 1/2 / (chi/2/(2*pi))

        # # Exponential decay to cavity probe (in terms of atom number)
        # A = (fd0-f0) * df_to_N
        # exp_decay = lambda t: A * np.exp(-gamma689 * (t-t0-dt_freq_avg/2) )

        # # Exponential plus self-radiated decay (in terms of atom number)
        # # Assumes Gamma0 -> Gamma0/2 for inhom coupling
        # Gamma0 = C689 * gamma689 * kappa689**2 / Delta**2
        # B = Gamma0/2 * A / gamma689
        # exp_sr_decay = lambda t: exp_decay(t)/( 1 + B*(np.exp(gamma689*(t-t0-dt_freq_avg/2)) - 1) )

        # # Get NB, ND arrays
        cav_probe_N = (cav_freq_shift) * df_to_N # NB/2 - NG/2
        # ND = 2*(exp_sr_decay(cav_freqs.t) - cav_probe_N)
        # NB = N_scaled_array - ND/2
        # Ne = NB + ND
        
        self.cav_probe_shift = tp.Time_Multitrace(cav_freqs.t - t0, cav_freq_shift - f0, dV = cav_freqs.dV)
        self.cav_probe_N = tp.Time_Multitrace(cav_freqs.t - t0, cav_probe_N, dV = cav_freqs.dV * df_to_N)
        self.cav_f0 = f0

        """
        Atomic coherence probe
        """
        # OPTION: Use some number of shots from each sequence
        atom_demod_all = self.data.demod_atom_trace(collapse=False)
        atom_demod = atom_demod_all.set( atom_demod_all.V[:,:shots_used_per_seq,:] ) \
                        .collapse().truncate(t_min = t0)
        t_us_atom = (atom_demod.t-t0)*1e6

        # OPTION: Phasor or mag2 average of shots
        if use_mag2_average:
            avg_mag2 = atom_demod.mag2().average_over(axis=0)        
        else:
            avg_phasor = atom_demod.average_over(axis=0)
            avg_mag2 = avg_phasor.mag2()
            
        mag2_no_noise = avg_mag2.V - avg_mag2_floor

        # OPTION: Generate plottables for either mag or mag2 of coherence
        if plot_mag2:
            coherence_normalized = mag2_no_noise * JPerp_per_V**2/n_atoms**2 
            err_coherence = avg_mag2.dV * JPerp_per_V**2/n_atoms**2 
        else:
            coherence_normalized = np.sqrt(np.abs(mag2_no_noise)) * JPerp_per_V/n_atoms
            err_coherence = 1/2 * coherence_normalized * avg_mag2.dV/np.abs(mag2_no_noise)
        
        self.splus = tp.Time_Multitrace(atom_demod.t-t0, coherence_normalized, dV=err_coherence)

        # Other arrays
        higgs_finder = tp.Time_Multitrace(t_us_atom*1e-6, coherence_normalized).truncate(t_min = 2e-6, t_max = 20e-6)
        pOpt_higgs, pCov_higgs = curve_fit(fit_exp, higgs_finder.t, higgs_finder.V,
                                p0=(20, 1), bounds=([0,0],[100,np.inf]))
        higgs_finder = higgs_finder.set( higgs_finder.V - fit_exp(higgs_finder.t, *pOpt_higgs) )
        mag2_fft = higgs_finder.fft(t_pad = 250e-6)
        
        self.splus_fft = mag2_fft

        # idx_transient = np.argwhere((avg_mag2.t - t0 < 20e-6) & (avg_mag2.t - t0 > 0e-6)).flatten()
        # V_transient = coherence_normalized[idx_transient]
        # lin_correct = V_transient + (idx_transient-idx_transient)/(idx_transient-idx_transient) * (V_transient - V_transient)
        # mag_correct = np.average( V_transient - lin_correct )

        # coherence_transient = tp.Time_Multitrace(avg_mag2.t[idx_transient],
        #                                       V_transient - lin_correct - 0*mag_correct, dV=err_coherence[idx_transient])
        # mag2_fft = coherence_transient.fft(t_pad = 250 * 1e-6)

        # Fit decay
        x, y = (avg_mag2.t - t0)*1e6, coherence_normalized
        idx = np.where((x > 1) & (x < 10))
        x, y = x[idx], y[idx]
        
        fit_decay = fit_exp 
        p0_array = [5,0.1]
        bounds=(-np.inf,np.inf)
        def cos_exp(x, x0, A, b, omega, phi):
            return( A * np.exp(-x/x0) * (np.cos(b) + np.sin(b)*np.cos(omega*x+phi)))
        # fit_decay = lambda x, x0, A, b, phi: cos_exp(x, x0, A, b, 2*np.pi*set_val, phi)
        # p0_array = [1,0.1, 1, 0]
        # bounds = ([0.5, 0.001,0, 0],[np.inf, np.inf, pi/2, 2*pi])
        
        pOpt_decay, pCov_decay = curve_fit(fit_decay, x, y, p0=p0_array, bounds=bounds)
        self.splus_longdecay = tp.Time_Multitrace(atom_demod.t[idx]-t0, fit_decay((avg_mag2.t[idx] - t0)*1e6, *pOpt_decay))
        self.splus_longdecay_full = tp.Time_Multitrace(atom_demod.t-t0, fit_decay((avg_mag2.t - t0)*1e6, *pOpt_decay))


        """
        BODY: Postprocessing of total photon number, decay from finite temperature
        """
        # OPTION: Show decay from finite temperature (single particle)
        tau_temp_const = 2.49769*1e-11 # s*K; hbar/kB * 1/ep_alpha
        T_rad = Trad_dict[self.sd.heating]
        T_ax = Tax_dict[self.sd.heating]
        tau_rad = tau_temp_const / T_rad
        tau_ax = tau_temp_const / T_ax
        
        # optional
        delta = 2*pi * self.sd.deltaZ * 1e3
        phi0 = 2 * np.arctan( delta/2 / np.sqrt(Omega_avg**2 + delta**2/4) )
        
        def thermal_decay(t, tau, delta, n, phi0=0):
            return( (1+t**2/tau**2)**(-n/2) * np.cos(delta*t/2 + phi0)**2 )
        coherence_decay_n2 = thermal_decay(atom_demod.t - t0, tau_rad, delta, 2, phi0) * coherence_normalized[0]
        coherence_decay_n3 = coherence_decay_n2 * thermal_decay(atom_demod.t - t0, tau_ax, 0, 1, 0)
        self.splus_sp = tp.Time_Multitrace(atom_demod.t-t0, coherence_decay_n2)
        self.splus_sp3 = tp.Time_Multitrace(atom_demod.t-t0, coherence_decay_n3)
            
        # Rescale noise-subtracted |S+|^2 by calibration factor to get photon number
        Mc_est = mag2_no_noise * alpha_a
        M_emitted = np.cumsum(Mc_est) * kappa689 * self.params.dt
        self.M_emitted = tp.Time_Multitrace(atom_demod.t-t0, M_emitted)
    
    def get_label(self, param_type):
        if param_type == "chiN":
            return( fr"$\chi N$ = {self.sd.chiN} kHz" )
        elif param_type == "deltaZ":
            return( fr"$\delta_Z$ = {self.sd.deltaZ} kHz" )
        elif param_type == "theta0":
            return( fr"$\theta_0$ = {round(self.sd.theta0,2)} $\pi$" )
        elif param_type == "misc":
            return( fr"{self.sd.misc}" )
        else:
            return( fr"$\chi N$ = {self.sd.chiN} kHz" )