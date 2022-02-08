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
            
def watch_dir(path='.'):
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
        atom_trace = None
        
        # Loop indefinitely to watch for file creation
        print('Press Ctrl+C to stop watchdog observer')
        while True:          
            # Look for an modify-file event followed by a create-file event
            # The create-file event implies the previous file is ready to read
            event = None
            for i in range(2, len(event_handler.events)+1):
                if event_handler.events[-i].event_type == 'modified' \
                    and event_handler.events[-(i-1)].event_type == 'created':
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
                
                # Process enough data to display demodulated atom trace
                expt.preprocess(n_shots = 1, load = 'newest')
                shot_atom_demod = expt.data.demod_atom_trace()
                t = shot_atom_demod.t
                shot_atom_demod_avg = np.mean(shot_atom_demod.V, axis=0)
                
                # Update plot with new trace
                if atom_trace:
                    atom_trace.set_ydata(shot_atom_demod_avg)
                else:
                    # No plot yet. Create one
                    atom_trace, = ax.plot(t, shot_atom_demod_avg)
                fig.canvas.draw()
                fig.canvas.flush_events()
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    
# TESTING ONLY: Remove package from path when done
sys.path.remove('../src')