# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 18:45:40 2022

@author: dylan
"""
import numpy as np
import keyword

__all__ = ['Shot', 'Time_Multitrace', 'Time_Trace', 'Frequency_Multitrace',
'Frequency_Trace']

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
        
        # Assign other arrays to variable-name attributes as Time Trace objects
        for i in range(len(dataset_names)):
            if i == i_t:
                continue
            dataset_name = dataset_names[i]
            if not dataset_name.isidentifier() or keyword.iskeyword(dataset_name) or dataset_name == 'None':
                raise SyntaxError(
                    f'"{dataset_name}" is not a valid variable name.'
                    )
            vars(self)[dataset_name] = Time_Trace(self.t, data[i])
        
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
        if np.min(triggers[1:]-triggers[:-1]) < trig_min_spacing / self.trig.dt:
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
    def __init__(self, t, V):
        self.t = t
        self.V = V
        
        # Extract time parameters
        self.t0 = t[0]
        self.dt = t[1]-t[0]
        self.T = t[-1]-t[0] + self.dt
        self.dim = len(self.V.shape)
    
    def _get_fft(self, t_pad=None):
        """
        Returns Fourier transform information to package into a Trace or
        Multitrace by the method Time_Trace.fft(). See documentation there.
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
        return(f, Vf)

    def bin_trace(self, t_bin):
        """
        Given a time multitrace and a time t_bin, bins the multiple time traces
        into subtraces with length t_bin in time. Given a k+1 dimensional
        multitrace of the form [d1, ..., dk, t], returns a k+2 dimensional
        multitrace of the form [d1, ..., dk, bin, t].
        
        Returns: Time_Multitrace
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
        
        return( Time_Multitrace(t, V) )
    
    def fft(self, t_pad=None):
        """
        Returns a Frequency Multitrace corresponding to the given time
        multitrace. Optionally, allows zero padding up to t_pad, which allows
        one to interpolate the raw fft to a resolution of 1/t_pad. Fundamental
        Fourier broadening of the spectrum is still limited by the time length
        of the original array, as expected of time-frequency uncertainty.
        
        Returns: Frequency_Multitrace
        """
        f, Vf = self._get_fft(t_pad)
        return( Frequency_Multitrace(f, Vf) )
    
class Time_Trace(Time_Multitrace):
    """
    Stores information for a single time trace. All arrays assumed to be
    numpy arrays.
    """
    def fft(self, t_pad=None):
        """
        Returns a Frequency Trace corresponding to the given time trace.
        Optionally, allows zero padding of the trace up to t_pad, which allows
        one to interpolate the raw fft to a resolution of 1/t_pad. Fundamental
        Fourier broadening of the spectrum is still limited by the time length
        of the original array, as expected of time-frequency uncertainty.
        
        Returns: Frequency_Trace
        """
        f, Vf = self._get_fft(t_pad)
        return( Frequency_Trace(f, Vf) )
        
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
        
class Frequency_Trace(Frequency_Multitrace):
    """
    Stores information for a single frequency trace. All arrays assumed to be
    numpy arrays.
    """