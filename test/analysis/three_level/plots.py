# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

__all__ = ["plot_splus", "plot_cav_probe"]

def plot_splus(dsets, param_type = None, yscale="linear", 
               show_sp_decay = False, show_photons_lost = False):
    """
    Plot |S+|^2 vs. time for the given datasets. Returns figure and axis objects
    """
    spluses = [dset.splus for dset in dsets]
    spluses_sp = [dset.splus_sp for dset in dsets]
    Ms_emitted = [dset.M_emitted for dset in dsets]
    
    if len(dsets) == 0:
        return
    
    fig_splus, ax_splus = plt.subplots(1,1, figsize=(6,4.5))
    if show_photons_lost:
        ax_splus2 = ax_splus.twinx()
        
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    for s in range(len(dsets)):
        splus, splus_sp, M_emitted = spluses[s], spluses_sp[s], Ms_emitted[s]
        t_us = splus.t * 1e6
        label = dsets[s].get_label(param_type)
        
        # Plot dataset
        ax_splus.plot(t_us, splus.V,
                      label=label,color=colors[s], alpha=0.5)
        ax_splus.fill_between(t_us, splus.V + splus.dV, splus.V - splus.dV,
                              color=colors[s], alpha=0.2)
        ax_splus.set_xlim((0,20))
        ax_splus.set_xlabel(r"Time ($\mu s$)")
        ax_splus.set_ylabel(r"$|S^{+}|^2/N^2$")
        ax_splus.set_title(r"$|S^{+}|^2$ vs. time")
        ax_splus.grid()
        ax_splus.legend()
        
        # Y scale
        if yscale == "log":
            ax_splus.set_yscale("log")
            ax_splus.set_ylim((1e-3,1e-1))
        else:
            ax_splus.set_ylim((0,0.09))
        
        # Optional plotted objects
        if show_sp_decay:
            ax_splus.plot(t_us, splus_sp.V, color='k', alpha=0.7)
        if show_photons_lost:
            ax_splus2.semilogy(t_us, M_emitted, '--', color='chocolate', alpha=0.3)
            ax_splus2.set_ylim((1e1,1e5))
            ax_splus2.set_ylabel('Photons lost', color='chocolate')
             
    fig_splus.tight_layout()
    return( fig_splus, ax_splus  )

def plot_cav_probe(dsets, param_type=None, yscale="linear",
                   plot_vs_bare_cav = False):
    """
    Plot cavity shift vs. time for the given datasets. Returns figure and axis
    objects
    """
    fig_cav, ax_cav = plt.subplots(1, 1, figsize=(6, 4.5))
    
    cav_probes_shift = [dset.cav_probe_shift for dset in dsets]
    cav_probes_N = [dset.cav_probe_N for dset in dsets]
    cav_f0s = [dset.cav_f0 for dset in dsets]
           
    chiN_max = max([dset.sd.chiN for dset in dsets])
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for s in range(len(dsets)):
        cav_probe_shift, cav_probe_N, cav_f0 = cav_probes_shift[s], cav_probes_N[s], cav_f0s[s]
        t_us = cav_probe_shift.t * 1e6
        label = dsets[s].get_label(param_type)
        
        if (plot_vs_bare_cav) and (yscale != "log"):
            ax_cav.errorbar(t_us, (cav_probe_shift.V + cav_f0) * 1e-3, yerr=cav_probe_shift.dV * 1e-3,
                            fmt='.-', color=colors[s], alpha=0.5, label=label)
        else:
            ax_cav.errorbar(t_us, cav_probe_shift.V * 1e-3, yerr=cav_probe_shift.dV * 1e-3,
                            fmt='.-', color=colors[s], alpha=0.5, label=label)
    
    if yscale == "log":
        ax_cav.set_yscale("log")
        ax_cav.set_ylim((1e1, 1e3))
    else:
        ax_cav.set_yscale("linear")
        ax_cav.set_ylim((-chiN_max * 1.1, 0))
    
    ax_cav.set_xlim((-1,20))
    ax_cav.set_xlabel(r"Time ($\mu s$)")
    ax_cav.set_ylabel(r"$f_{cav}$ (kHz)")
    ax_cav.set_title(r"$f_{cav}$ vs. time")
    ax_cav.grid()
    ax_cav.legend()
    
    fig_cav.tight_layout()
    return( fig_cav, ax_cav )