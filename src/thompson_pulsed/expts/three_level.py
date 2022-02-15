# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 18:00:35 2022

@author: dylan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit

from ..core import traces
from ..core.parsers import ni_pci5105
"""
Define experiment-specific functions
"""
def sinc(x):
    return(np.sinc(x/np.pi))
def sinc_symm(f, f0, A, T):
    return( A*(sinc(np.pi*(f-f0)*T)**2 + sinc(np.pi*(f+f0)*T)**2) )
def sinc_symm_fitter(T):
    """
    Returns a symmetric sinc function, with the sinc width fixed to T.
    """
    return( lambda f,f0,A: sinc_symm(f,f0,A,T) )
def cos(t, A, f, phi):
    return( A*np.cos(2*np.pi*f*t + phi) )
def moving_average(V, k):
    V_sum = np.cumsum(V, dtype=float, axis=-1)
    V_sum[..., k:] = V_sum[..., k:] - V_sum[..., :-k]
    shift = int((k-1)/2)
    V_avg_trunc = V_sum[..., k-1:]/k
    pad_width = ((0,0),) * (V_avg_trunc.ndim-1) + ((shift, k-shift-1),)
    return( np.pad(V_avg_trunc, pad_width) )
def notch(V, f_notch, Q, fs):
    b, a = signal.iirnotch(f_notch, Q, fs)
    return( signal.filtfilt(b,a,V,axis=-1) )

"""
Define object classes for experiment
"""
class Experiment:
    """
    Root object of a particular instance of the experiment.
    """
    def __init__(self, params):
        self.shots = []
        self.data = Data()
        
        if params._all_params_defined():
            self.params = params
        else:
            raise Exception("Error: not all parameters defined!")

    def load_shot(self, file):
        """
        Loads a text file from a single shot into the experiment.
        """
        self.shots.append(traces.Shot(file, ni_pci5105))
            
    def preprocess(self, n_shots = None, load = 'newest'):
        if not self.params:
            raise Exception("Error: need to set experiment parameters!")
        if not self.shots:
            raise Exception("Error: no shots loaded to experiment!")
        
        # Process optional args
        if not n_shots or n_shots > len(self.shots):
            n_shots = len(self.shots)
            
        if load=='oldest':
            shots = self.shots[:n_shots]
        elif load=='newest':
            shots = self.shots[-n_shots:]
        else:
            shots = self.shots[-n_shots:]
        
        # Define a single time array for all runs of the experiment
        self.params.dt = self.shots[0].t[1] - self.shots[0].t[0]
        i_run = round(self.params.t_run / self.params.dt)
        self.data.t = self.params.dt * np.arange(i_run)
        
        # TODO: preprocess partially if data already exists
        
        # Iterate over all shots
        atom_runs, cav_runs = np.array([]), np.array([])
        for shot in shots:
            if not shot.has_triggers:
                # Assume a single run for each shot and extract points
                shot_atom_runs = np.array([shot.atom.V[:i_run]])
                shot_cav_runs = np.array([shot.cav.V[:i_run]])
            else:
                # Extract runs from the single shot using triggers as markers
                shot_atom_runs = np.array(
                    [shot.atom.V[trig:trig+i_run] for trig in shot.triggers]
                    )
                shot_cav_runs = np.array(
                    [shot.cav.V[trig:trig+i_run] for trig in shot.triggers]
                    )
            if atom_runs.size > 0:
                atom_runs = np.concatenate((atom_runs, shot_atom_runs))
                cav_runs = np.concatenate((cav_runs, shot_cav_runs))
            else:
                atom_runs, cav_runs = shot_atom_runs, shot_cav_runs
        
        # Assign to data object
        self.data.atom_runs = traces.Time_Multitrace(self.data.t, atom_runs)
        self.data.cav_runs = traces.Time_Multitrace(self.data.t, cav_runs)
        self.data.params = self.params
            
class Parameters:
    def __init__(self):
        self.t_run = None
        self.t_bin = None
        self.t_drive = None
        self.t_fft_pad = None
        self.f0_cav = None
        self.f0_atom = None
        self.fft_fit = None
        self.demod_smoother = None
        
    def _all_params_defined(self):
        for attr in vars(self):
            if vars(self)[attr] is None:
                return(False)
        return(True)

class Data:
    """
    Stores preprocessed data. Accessed via the parent Experiment object. 
    """
    def __init__(self):
        self.t = None
        self.cav_runs = None
        self.atom_runs = None
        self.params = None
    
    def track_cav_frequency(self, out_fmt='MT'):
        """
        DEPRECATED METHOD (slower than track_cav_frequency_iq):
        Bins cavity time traces and fits the FFT of these bins to measure a
        cavity resonance frequency. Multiple shots give statistics on these
        bins.
        
        Parameters
        ----------
        out_fmt : str, optional
            Specifies what format the data is outputted in. All methods other than
            the default are deprecated.
            'trace' : returns t, V as two separate np arrays
            'MT' : returns t, V as a single tp.Time_Multitrace (see below)
            Default is 'MT'
                    
        Returns
        -------
        cav_freqs : tp.Time_Multitrace
            .t : 1D np array [bin]
            .V : 2D np array [run,bin]
        """
        # Bin cavity run traces, subtract mean from each bin, and do an FFT
        cav_bins = self.cav_runs.bin_trace(self.params.t_bin)
        cav_bins.V -= np.mean(cav_bins.V, axis=-1, keepdims=True)
        cav_ffts = cav_bins.fft(t_pad = self.params.t_fft_pad)
                                
        # Get bin times
        bin_times = self.params.t_bin * (0.5 + np.arange(cav_ffts.V.shape[1]))
        
        cav_freq_vals = np.zeros(cav_ffts.V.shape[:-1])
        for r in range(cav_ffts.V.shape[0]):
            run = cav_ffts.V[r,...]
            for b in range(run.shape[0]):
                bin = run[b,...]
                try:
                    [pOpt, pCov] = curve_fit(
                            self.params.fft_fit,
                            cav_ffts.f, np.abs(bin)**2,
                            p0 = [self.params.f0_cav, 1]
                            )
                    cav_freq_vals[r,b] = np.abs(pOpt[0])
                except RuntimeError:
                    # print('Fit ' + str(bin+1) + ' of ' + str(n_bins) + ' failed.')
                    # plt.figure()
                    # plt.plot(fs, np.abs(Vfs[bin,:])**2)
                    
                    # Set fitted frequency to a random value below the Nyquist frequency
                    cav_freq_vals[r,b] = np.random.uniform(low=0, high=np.max(cav_ffts.f))
            if (r+1) % 10 == 0:
                print(f"Run {r+1} of {cav_ffts.V.shape[0]} processed.")

        # Choose output format
        if out_fmt == 'array':
            # Deprecated: try returning multitrace (MT)
            return( bin_times, cav_freq_vals )
        elif out_fmt == 'MT':
            return( traces.Time_Multitrace(bin_times, cav_freq_vals) )
        else:
            # Default: return MT
            return( traces.Time_Multitrace(bin_times, cav_freq_vals) )
    
    def track_cav_frequency_iq(self, f_demod = None, out_fmt = 'MT'):
        """
        IQ demodulates cavity time traces, bins them, and fits their phase(t)
        with a linear regression to estimate instantaneous frequency. Multiple
        shots give statistics on these bins.
        
        Parameters
        ----------
        f_demod : float, optional
            Specifies what frequency to demodulate at. If None, f_demod is auto-set
            to self.params.f0_cav. The default is None.
        out_fmt : str, optional
            Specifies what format the data is outputted in. All methods other than
            the default are deprecated.
            'trace' : returns t, V as two separate np arrays
            'MT' : returns t, V as a single tp.Time_Multitrace (see below)
            Default is 'MT'

        Returns
        -------
        cav_freqs : tp.Time_Multitrace
            .t : 1D np array [bin]
            .V : 2D np array [run,bin]
        """
        if not f_demod:
            f_demod = self.params.f0_cav
            
        # Bin cavity run traces, subtract mean from each bin, and demodulate
        cav_runs = self.cav_runs
        cav_runs.V -= np.mean(cav_runs.V, axis=-1, keepdims=True)
        cav_phase = cav_runs.iq_demod(f_demod).phase()
        cav_bins = cav_phase.bin_trace(self.params.t_bin)
                                
        # Get bin times
        bin_times = self.params.t_bin * (0.5 + np.arange(cav_bins.V.shape[1]))
        
        # Estimate cavity frequency in bins using linear regression
        cav_freq_vals = cav_bins.frequency()
        
        # Choose output format
        if out_fmt == 'array':
            # Deprecated: try returning multitrace (MT)
            return( bin_times, cav_freq_vals )
        elif out_fmt == 'MT':
            return( traces.Time_Multitrace(bin_times, cav_freq_vals) )
        else:
            # Default: return MT
            return( traces.Time_Multitrace(bin_times, cav_freq_vals) )
    
    def demod_atom_trace(self):
        """
        # TODO
        """
        # Find the phase reference pulse (assume S/N > 1)
        trig_level = 0.6
        V = self.atom_runs.V - np.mean(self.atom_runs.V, axis=-1, keepdims=True)
        trig_idxs = np.argmax( 
            V > trig_level * V.max(axis=1, keepdims=True),
            axis = 1)
        
        # Get time bin for phase reference pulse using advanced indexing
        ref_pulse_frac = 0.8
        n_bin_pts = round(self.params.t_drive / self.params.dt * ref_pulse_frac)
        ref_pulse_slices = trig_idxs[:,None] + np.arange(n_bin_pts)
        ref_pulse_V = V[np.arange(V.shape[0])[:,None], ref_pulse_slices]
        ref_pulse_t = self.atom_runs.t[ref_pulse_slices]
        
        # plt.figure()
        # for i in range(5):
        #     trace = traces.Time_Multitrace(ref_pulse_t[i,...], ref_pulse_V[i,...]) \
        #                     .fft(t_pad = self.params.t_fft_pad)
        #     plt.plot(trace.f, np.abs(trace.V)**2)
        
        """
        The fit seems like the easiest way to get phase information
        The fit doesn't return a very reliable measure of the beat frequency
        *Idea 1: don't measure frequency, assume a frequency. We need to be
            accurate to much better than 1/t_run, but if t_run is 10us
            (realistic) then we only need a frequency accuracy of ~10 kHz
            which we need anyway to probe the atoms properly in MOT stages.
        Idea 2: start with assumed frequency and find max in fft^2, binned
            to a +/- tol around f0_atom (say +/- 50 kHz). This will avoid
            AM peaks (we wouldn't notice AM peaks at 50 kHz anyway).       
        """
        # For each run, fit a frequency and phase to the reference pulse
        # FIX atomic frequency
        ref_pulse_phi = np.zeros(ref_pulse_V.shape[0])
        for r in range(ref_pulse_V.shape[0]):
            try:
                [pOpt, pCov] = curve_fit(
                        lambda t,A,phi: cos(t,A,self.params.f0_atom,phi),
                        ref_pulse_t[r,...], ref_pulse_V[r,...],
                        p0 = [1, 0]
                        )
                ref_pulse_phi[r] = pOpt[1] if pOpt[0]>0 else pOpt[1]+np.pi
            except RuntimeError:
                print(f"Fit {r+1} of {ref_pulse_V.shape[0]} failed.")
                raise
            # if r == 0:
            #     plt.figure()
            #     plt.plot(ref_pulse_t[r], ref_pulse_V[r])
            #     plt.plot(ref_pulse_t[r], cos(ref_pulse_t[r], pOpt[0], self.params.f0_atom, pOpt[1]))
            #     print(pOpt)      
        
        ref_pulse_f = np.zeros(ref_pulse_V.shape[0]) + self.params.f0_atom
        
        # Demodulate traces using fitted (f,phi) on reference pulses
        t = self.atom_runs.t[None,:]
        f, phi = ref_pulse_f[:,None], ref_pulse_phi[:,None]
        
        local_osc = cos(t, 1, f, phi)
        
        V_demod_raw = local_osc * V
        V_demod = self.params.demod_smoother(V_demod_raw)
        
        return(traces.Time_Multitrace(self.atom_runs.t, V_demod))