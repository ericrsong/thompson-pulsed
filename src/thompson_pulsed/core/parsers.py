# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 17:49:15 2022

@author: dylan
"""
import numpy as np
import pandas as pd

__all__ = ['ni_oscilloscope_txt']

def ni_oscilloscope_txt(file):
    """
    Uses pandas to load data from a CSV-formatted data file, specifically as
    is formatted for data exported from the pulsed experiment NI PCI-5105
    oscilloscope board via our LabVIEW vi.

    Parameters
    ----------
    file : string corresponding to a valid file address

    Returns
    -------
    data : 2D numpy array of format np.array([t_array, ch0_array, ...])
    data_names : 1D list of the titles of all 1D arrays stored in data
    """
    df = pd.read_csv(file).T
    data = df.to_numpy()
    data_names = df.index.values.tolist()
    
    return( data, data_names )