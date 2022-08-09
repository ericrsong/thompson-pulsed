# -*- coding: utf-8 -*-
"""
These functions are designed to quickly pull information out of file generated
from common lab instruments and into a uniform datatype, specifically a 2d
numpy array alongside a list of dataset names.

For ease of use, the module was designed to allow for calling these functions
simply by typing, for instance, ``thompson_pulsed.ni_pci5105(file)``.
"""
import numpy as np
import pandas as pd

import os

__all__ = ['ni_pci5105', 'ni_pcie7851r_ai', 'keysight_hsa_csv','tektronix_oscilloscope_csv']

def ni_pci5105(file):
    """Uses pandas to load data from a CSV-formatted data file, specifically as
    is formatted for data exported from the pulsed experiment NI PCI-5105
    oscilloscope board via our LabVIEW vi.

    Parameters
    ----------
    file : str
        Path to a valid file

    Returns
    -------
    data : ndarray
        2D array with format np.array([t_array, x1_array, ...])
    dataset_names : list
        1D list of the titles of all 1D arrays stored in data

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
    file : str
        Path to a valid file

    Returns
    -------
    data : ndarray
        2D array with format np.array([t_array, x1_array, ...])
    dataset_names : list
        1D list of the titles of all 1D arrays stored in data

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

def keysight_hsa_csv(file):
    """Uses default python functions to load data from a CSV file, as
    generated using the Keysight Handheld Spectrum Analyzer.

    Parameters
    ----------
    file : str
        Path to a valid file

    Returns
    -------
    data : ndarray
        2D array with format np.array([t_array, x1_array, ...])
    dataset_names : list
        1D list of the titles of all 1D arrays stored in data

    """
    data, in_header, i = [], True, 0
    with open(file,'r') as f:
        for line in f:
            # Work through header
            line = line.replace('\x00','').split(',')
            if line[0] == 'Frequency(Hz)':
                num_cols = len(line) - 1
                in_header = False
                header_names = line[1:-1]
                continue
            elif in_header:
                i += 1
                continue
            
            # When not in header, add data
            data.append([float(x) for x in line[:num_cols] if len(line) >= num_cols])
    data = np.transpose(data)

    # Construct dataset names based on header
    channels = [[int(x) for x in name if x.isdigit()][0] for name in header_names]
    dataset_names = ['f'] + [f'ch{channel}' for channel in channels]

    return( data, dataset_names )

def tektronix_oscilloscope_csv(file):
    """Uses pandas to load data from a CSV file, as generated from a
    Tektronix oscilloscope. Tested on a TBS 2000B series scope.

    Parameters
    ----------
    file : str
        Path to a valid file

    Returns
    -------
    data : ndarray
        2D array with format np.array([t_array, x1_array, ...])
    dataset_names : list
        1D list of the titles of all 1D arrays stored in data

    """
    
    # Get header by reading through file for first numeric entry
    f = open(file, 'r')
    header_size = 0
    for line in f:
        line = line.strip().split(',')
        
        # Check if first element of line is float. If not, continue parsing
        try:
            float(line[0])
            break
        except ValueError:
            header_size += 1
            continue

    df = pd.read_csv(file, sep=',',
                     skiprows = (header_size-1) if (header_size > 0) else None)

    data, dataset_names = df.T.to_numpy(), np.char.lower( df.columns.to_numpy().astype(str) )
    for i in range(len(dataset_names)):
        if dataset_names[i] == 'time':
            dataset_names[i] = 't'
    
    return( data, dataset_names )

# INCOMPLETE
def labview_log(dataset_name, path='.'):
    file = os.path.join(path, f'{dataset_name}_XYGraphDataLog.txt')
    
    if os.path.exists(file):
        df = pd.read_csv(file, delimiter='\t')

    cols = df.columns.to_numpy().astype('str')

    # GenFit columns
    genfit_label_idx = np.argwhere( np.char.startswith(cols, 'GenFitLabel') ).flatten()
    genfit_err_idx = np.argwhere( np.char.startswith(cols, 'GenFitErr') ).flatten()
    genfit_idx = np.array([i for i in range(cols.size) if np.char.isnumeric(cols[i][6:])])
    
    # x Param
    xparam_idx = np.argwhere(cols == 'xParamValue').flatten()
    
    # File number
    filenum_idx = np.argwhere(cols == 'FileNumber').flatten()
    
    return( df )

# # INCOMPLETE
# def labview_graphs(dataset_name, path='.'):
#     file = os.path.join(path, f'{dataset_name}_XYGraphs.txt')
    
#     if os.path.exists(file):
#         df = pd.read_csv(file, delimiter='\t')
    
#     return( df )