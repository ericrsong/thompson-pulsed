# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 12:28:42 2022

@author: dylan

This is an example watchdog script for the three_level experiment.
"""
# TESTING ONLY: Adds the local package temporarily to the path for testing
import sys
sys.path.insert(0, '../src')

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import time
import logging
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, LoggingEventHandler

# import thompson_pulsed as tp
import thompson_pulsed.expts.three_level as e3l

"""
Define experimental parameters
"""
params = e3l.Parameters()
params.t_run = 50 * 1e-6        # 50 us
params.t_bin = 0.5 * 1e-6       # 0.5 us
params.t_drive = 0.5 * 1e-6     # 0.5 us
params.t_fft_pad = 100 * 1e-6   # 100 us
params.f0_cav = 10 * 1e6        # 10 MHz
params.f0_atom = 10 * 1e6       # 10 MHz
params.fft_fit = e3l.sinc_symm_fitter(params.t_bin)
dt = 1/60 * 1e-6
params.demod_smoother = lambda V: \
    e3l.moving_average(V, round(1/params.f0_atom/dt/2))

class Custom_EventHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self.events = []

    # Record any non-directory filesystem events that occur in path
    def on_any_event(self, event):
        if not event.is_directory:
            self.events.append(event)
            
"""
Curve fit
"""
gamma = 2*np.pi * 7.5 * 1e3
def exponential(t, A, tau, C):
    return( A*np.exp(-t/tau) + C )
fit = lambda t,A,C: exponential(t, A, 1/gamma, C)

"""
Watchdog function
"""            
def watch_dir(path='.', keep_files=True):
    # Initialize experiment
    expt = e3l.Experiment(params)
    
    # # Remove all handlers associated with the root logger object.
    # for handler in logging.root.handlers[:]:
    #     logging.root.removeHandler(handler)
    # # Setup basic config for a root logger
    # logging.basicConfig(level=logging.INFO,
    #                     format='%(asctime)s - %(message)s',
    #                     datefmt='%Y-%m-%d %H:%M:%S')
    
    # If path doesn't exist, create path
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'Directory created: {os.path.abspath(path)}')
    
    # Initialize watchdog objects
    # event_handler = LoggingEventHandler()
    event_handler = Custom_EventHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    try:
        loaded_files = []
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 50)
        ax.set_ylim(-500, 500)
        ax.grid()
        cav_probe = None
        
        # Loop indefinitely to watch for file creation
        print('Press Ctrl+C to stop watchdog observer')
        while True:
            # Look for the second most recent FileCreatedEvent
            # Second most recent hopefully ensures it is completely loaded
            event, created_events = None, 0
            for i in range(1, len(event_handler.events)+1):
                if event_handler.events[-i].event_type == 'created':
                    created_events += 1
                if created_events == 2:
                    event = event_handler.events[-i]
                    break
            if not event:
                continue
            
            newest_file = event.src_path
            if newest_file not in loaded_files:
                # Load newest file into experiment
                expt.load_shot(newest_file)
                loaded_files.append(newest_file)
                print(f'File loaded: {newest_file}')
                
                # Delete old file
                if not keep_files:
                    try:
                        if len(loaded_files) > 2:
                            os.remove(loaded_files[0])
                            loaded_files = loaded_files[1:]
                    except:
                        if not os.path.exists(loaded_files[0]):
                            loaded_files = loaded_files[1:]
                
                # Process enough data to display demodulated atom trace
                expt.preprocess(n_shots = 1, load = 'newest')
                bin_times, cav_freqs = expt.data.track_cav_frequency_iq()
                cav_freqs_mean = np.mean(cav_freqs, axis=0)
                cav_freqs_stdmean = np.std(cav_freqs, axis=0) / np.sqrt(cav_freqs.shape[0])
                
                [pOpt, pCov] = curve_fit(fit,
                                         bin_times, cav_freqs_mean,
                                         sigma = cav_freqs_stdmean,
                                         p0=[1,0], 
                                         bounds=([0,-500e3],[np.inf, 500e3])               
                    )
                
                # Update plot with new trace
                if not cav_probe:
                    # No plot yet. Create one
                    cav_probe = ax.errorbar(bin_times * 1e6, cav_freqs_mean * 1e-3,
                                            cav_freqs_stdmean * 1e-3,
                                            fmt='ok', alpha=0.2)
                    cav_fit = ax.plot(bin_times * 1e6,
                                      fit(bin_times, *pOpt) * 1e-3, 'k',
                                      label=f'f0 = {round(pOpt[0]*1e-3)} +/- {round(np.sqrt(pCov[0,0])*1e-3)} kHz')
                    ax.set(xlabel=r'Time ($\mu s$)', ylabel=r'Frequency (kHz)')
                    ax.legend(loc='upper right')
                else:
                    cav_probe[0].remove()
                    for line in cav_probe[1]:
                        line.remove()
                    for line in cav_probe[2]:
                        line.remove()
                    cav_probe = ax.errorbar(bin_times * 1e6, cav_freqs_mean * 1e-3, 
                                            cav_freqs_stdmean * 1e-3,
                                            fmt='ok', alpha=0.2)
                    cav_fit[0].set(xdata=bin_times * 1e6,
                                   ydata=fit(bin_times, *pOpt) * 1e-3,
                                   label=f'f0 = {round(pOpt[0]*1e-3)} +/- {round(np.sqrt(pCov[0,0])*1e-3)} kHz')
                    ax.legend(loc='upper right')
                    
                fig.canvas.draw()
                fig.canvas.flush_events()
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

    
# TESTING ONLY: Remove package from path when done
sys.path.remove('../src')