# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 18:45:40 2022

@author: dylan
"""
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

import keyword

__all__ = ['Shot', 'Time_Multitrace', 'MT_Phasor', 'MT_Phase', 
           'Frequency_Multitrace']

class Shot:
    """
    Stores information for a single shot of an experiment. This is a generic
    class definition with variable-name attributes. 
    
    A Shot object is required to have a time array, with the name 't'. This
    array is detected and then used as the time record for all other arrays
    when packaged into Time Traces.
    
    Optionally, a Shot object may have a trigger array, with the name 'trig'.
    This array is detected and then used to mark indices where a trigger is
    found.
    """
    def __init__(self, file, parser):
        data, dataset_names = parser(file)
        
        # Find time array
        try:
            i_t = dataset_names.index('t')
        except ValueError:
            print('No time array with name "t" detected in file.')
            raise
        self.t = data[i_t]
        
        # Assign other arrays to variable-name attributes as Time Multitrace objects
        for i in range(len(dataset_names)):
            if i == i_t:
                continue
            dataset_name = dataset_names[i]
            if not dataset_name.isidentifier() or keyword.iskeyword(dataset_name) or dataset_name == 'None':
                raise SyntaxError(
                    f'"{dataset_name}" is not a valid variable name.'
                    )
            vars(self)[dataset_name] = Time_Multitrace(self.t, data[i])
        
        # Optional: find trigger array and record triggered indices
        self.has_triggers = ('trig' in dataset_names)
        if self.has_triggers:
            self.triggers = self._mark_triggers()
        else:
            self.triggers = None
    
    def _mark_triggers(self, slope=1):
        """
        Given a TTL Time Trace object stored in self.trig, detect all triggers
        within the trace and record their indices in a 1D np array.

        Parameters
        ----------
        slope : float, optional
            Specifies whether to look for a rising (1) or falling (-1) trigger.
            The default is 1.

        Returns
        -------
        triggers : 1D np array 
        """
        trig_level = np.max(self.trig.V) / 2
        trig_min_spacing = 1e-6 # 1 us minimum distance between triggers
        
        triggers = np.where( (self.trig.V[1:] > trig_level) & \
                            (self.trig.V[:-1] < trig_level) )[0]
        if triggers.size > 1 and np.min(triggers[1:]-triggers[:-1]) < trig_min_spacing / self.trig.dt:
            raise Exception(
                'Two triggers closer than 1 us detected. Possible bad triggers.'
                )
        
        return(triggers)

class Time_Multitrace:
    """
    Stores information for multiple time traces. These multiple traces are
    assumed to be packed in a single numpy array of arbitrary dimension, where
    the LAST index corresponds to time.
    """
    def __init__(self, t, V, dV=None):
        """
        Initializes a Time_Multitrace object.

        Parameters
        ----------
        t : 1D np array
            time points for data.
        V : np array [..., t]
            data points of arbitrary (d>=1) dimension. axis -1 corresponds to time.
        dV : np array, shape = self.V.shape, optional
            uncertainties for self.V. If None, assume no uncertainties. Default
            is None.

        Returns
        -------
        None.
        """
        # Check shape agreement and assign to object
        min_time_shape = min( t.shape[0], V.shape[-1] )
        if dV is not None:
            assert V.shape == dV.shape
        self.t = t[:min_time_shape]
        self.V = V[..., :min_time_shape]
        self.dV = dV[..., :min_time_shape] if (dV is not None) else None
        
        # Extract time parameters
        self.t0 = t[0]
        self.dt = t[1]-t[0]
        self.T = t[-1]-t[0] + self.dt
        self.dim = len(self.V.shape)

    def bin_trace(self, t_bin):
        """
        Given a time multitrace and a time t_bin, bins the multiple time traces
        into subtraces with length t_bin in time. Given a k+1 dimensional
        multitrace of the form [d1, ..., dk, t], returns a k+2 dimensional
        multitrace of the form [d1, ..., dk, bin, t].
        
        Parameters
        ----------
        t_bin : float
            Binning time in units of self.t

        Returns
        -------
        Class of type self (base class: Time_Multitrace)
        """
        if t_bin < self.dt or t_bin > self.T:
            raise ValueError("Error: t_bin is out of bounds!")
        
        # Calculate bin parameters
        n_pts_raw = self.V.shape[-1] 
        n_bin_pts = round( t_bin/self.dt )
        n_bins = int( n_pts_raw / n_bin_pts )
        n_pts = n_bins * n_bin_pts
        
        # Reshape array
        V = self.V[..., :n_pts]
        new_shape = V.shape[:-1] + (n_bins, n_bin_pts)
        V = np.reshape(V, new_shape)
        
        # Take only time values for first bin
        t = self.t[:n_bin_pts]
        
        return( type(self)(t, V) )
    
    def fft(self, t_pad=None):
        """
        Returns a Frequency Multitrace corresponding to the given time
        multitrace. Optionally, allows zero padding up to t_pad, which allows
        one to interpolate the raw fft to a resolution of 1/t_pad. Fundamental
        Fourier broadening of the spectrum is still limited by the time length
        of the original array, as expected of time-frequency uncertainty.
        
        Parameters
        ----------
        t_pad : float, optional
            Pad time in units of self.t. If None, no padding is performed.
            Default is None.

        Returns
        -------
        Frequency_Multitrace
        """
        if t_pad == None:
            t_pad = self.T
        
        pad_pts = round(t_pad / self.dt)
        pad_tuple = ((0,0),) * (self.dim-1) + ((0, pad_pts-self.t.size),)
        V_pad = np.pad(self.V, pad_tuple )
        
        f_raw = np.fft.fftfreq(pad_pts, d=self.dt)
        Vf_raw = np.fft.fft(V_pad)
        
        f = np.fft.fftshift(f_raw, axes=-1)
        Vf = np.fft.fftshift(Vf_raw, axes=-1)

        return( Frequency_Multitrace(f, Vf) )
    
    def iq_demod(self, f_demod, filt='butter', order=4, f_cutoff=None):
        """
        Performs an IQ demodulation on the multitrace with frequency f_demod.
        The scipy function filtfilt applies the given filter in the forward-
        time direction once, then applies it again in the backwards-time
        direction. This ensures the next applied filter has no net phase delay
        at the expense of preserving causality.

        Parameters
        ----------
        f_demod : float
            frequency to demodulate signal by
        filt : string, optional
            Describes which scipy filter to use. The default is 'butter'.
        order : int, optional
            Describes which order filter to use. The default is 4.
        f_cutoff : float, optional
            Describes the cutoff frequency of the filter. If None, function
            assumes f_cutoff to be f_demod. The default is None.

        Returns
        -------
        MT_Phasor(Time_Multitrace), representing demodulated phasors
        """
        LO_x = np.cos(2*np.pi*f_demod*self.t)
        LO_y = np.sin(2*np.pi*f_demod*self.t)
        
        # Generate 4th order Butterworth filter with cutoff at demod frequency
        if not f_cutoff:
            f_cutoff = f_demod
        f_nyquist = 1/(2 * self.dt)
        wn = f_cutoff / f_nyquist
        n = order
        # TODO: give option to use different filter
        b,a = signal.butter(n, wn)
        
        V_x = signal.filtfilt(b,a, self.V*LO_x)
        V_y = signal.filtfilt(b,a, self.V*LO_y)

        return( MT_Phasor(self.t, V_x + 1j * V_y) )
    
    def moving_average(self, t_avg, use_weights=True):
        """
        Assuming an evenly-spaced array, performs a moving average with window
        t_avg.

        Parameters
        ----------
        t_avg : float
            Time window for moving average. Units the same as self.t
        use_weights: bool, optional
            Determines whether or not to weight data points when averaging.
            If True, attempts to define w = 1/dV**2 and otherwise uses equal
            weighting. If False, uses equal weighting. Default is True.

        Returns
        -------
        Time_Multitrace, representing a moving average of self
        """
        # Determine number of points to average within t_avg
        k = int(t_avg/self.dt)
        if k < 1 or k > self.t.size:
            return( self )
        
        # Generate weights for average. If no errors exist, use equal weights
        w = (1/self.dV**2) if (self.dV is not None and use_weights) else (1 + 0*self.V)
        
        # Generate moving average
        V_sum = np.cumsum(self.V * w, dtype=float, axis=-1)
        V_sum[..., k:] = V_sum[..., k:] - V_sum[..., :-k]
        w_sum = np.cumsum(w, dtype=float, axis=-1)
        w_sum[..., k:] = w_sum[..., k:] - w_sum[..., :-k]
        shift = int((k-1)/2)
        
        # Normalize, then truncate the first k-1 points as they are not part of the average
        V_avg_trunc = V_sum[..., k-1:] / w_sum[..., k-1:]
        w_trunc = w_sum[..., k-1:]
        
        # Pad arrays with endpoint values to maintain shapes
        pad_width_V = ((0,0),) * (V_avg_trunc.ndim-1) + ((shift, k-shift-1),)
        V_avg = np.pad(V_avg_trunc, pad_width_V, mode='edge')
        dV_avg = np.pad(1/np.sqrt(w_trunc), pad_width_V, mode='edge')
        
        return( Time_Multitrace(self.t, V_avg, dV=dV_avg) )
        
class MT_Phasor(Time_Multitrace):
    def phase(self, unwrap=True):
        """
        Given a MT_Phasor trace or phasors, generates an MT_Phase trace of
        phases.
        
        Parameters
        ----------
        unwrap : bool, optional
            Describes whether or not to unwrap the phases.

        Returns
        -------
        MT_Phase(Time_Multitrace) with same shape as self
        """
        return( MT_Phase(self.t, np.angle(self.V), unwrap=unwrap) )

class MT_Phase(Time_Multitrace):
    def __init__(self, t, V, dV=None, unwrap=True):
        super().__init__(t,V,dV)
        
        if unwrap:
            self._unwrap()
            
    def _unwrap(self):
        """
        Unwraps the phase trace by detecting jumps in phase of larger than
        pi in a single time step, and correct.

        Returns
        -------
        None.
        """
        phase = self.V
        wraps = np.rint( (phase[..., 1:]-phase[..., :-1])/(2*np.pi) )
        change = np.concatenate((np.zeros( phase.shape[:-1] + (1,) ), 
                                 -np.cumsum(wraps, axis=-1)*2*np.pi), axis = -1)
        self.V += change
        
    def frequency(self, t_bin=None):
        """
        Calculates a linear regression slope in O(n) time, using a closed-form
        formula for least-squares that assumes evenly sampled points in time. 
        
        Parameters
        ----------
        t_bin : float, optional
            Describes how to bin 

        Returns
        -------
        freqs : np array of floats, shape = self.V.shape[:-1]
            Estimated frequencies of Phase traces, expressed in Hz (cycles/sec)
        """
        if t_bin:
            return( self.bin_trace(t_bin).frequency(t_bin=None) )
        else:
            y = self.V
            n = y.shape[-1]
            x = np.arange(1, n+1) - (n+1)/2
            dx = self.dt
            
            freqs = np.sum(x*y, axis=-1) / ( n*(n**2-1)/12 * dx ) * 1/(2*np.pi)
            return(freqs)

class Frequency_Multitrace:
    """
    Stores information for multiple frequency traces. These multiple traces are
    assumed to be packed in a single numpy array of arbitrary dimension, where
    the LAST index corresponds to frequency.
    """
    def __init__(self, f, V):
        self.f = f
        self.V = V
        
        # Extract frequency parameters
        self.df = f[1]-f[0]
        self.F = f[-1]-f[0] + self.df
        self.dim = len(self.V.shape)