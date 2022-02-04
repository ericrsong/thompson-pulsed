# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 17:49:15 2022

@author: dylan
"""
import numpy as np
import pandas as pd

__all__ = ['ni_oscilloscope_txt']

def ni_pci5105(file):
    """Uses pandas to load data from a CSV-formatted data file, specifically as
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

def ni_pcie7851r_ai(file):
    """Uses pandas to load data from a tab-delimited data file, as generated
    using the LabVIEW Master.vi for analog inputs from the NI-PCIe7851R board.

    Parameters
    ----------
    file : string corresponding to a valid file address

    Returns
    -------
    data : 2D numpy array of format np.array([t_array, x1_array, ...])
    data_names : 1D list of the titles of all 1D arrays stored in data
    """
    df = pd.read_csv(file, sep='\t', header=None)
    
    # Rename columns and store in dataset_names
    df = df.rename({i: 'x'+str(i) for i in df.columns if i!=0}, axis=1) \
        .rename({0: 't'}, axis=1)
    dataset_names = df.columns.tolist()
    
    # LabVIEW saves t array in units of ms. Convert to SI
    df['t'] *= 10**-3
    data = df.T.to_numpy()
    
    return( data, dataset_names )