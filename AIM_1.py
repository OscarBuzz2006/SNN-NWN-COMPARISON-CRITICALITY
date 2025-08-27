#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# IMPORTED: Neuromorphic Nanowire Network Simulation
Author: Yinhao Xu <yinhao.xu@sydney.edu.au>
"""

import numpy as np 
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm as pbar

class NanowireNetwork:
    """
    Nanowire Network Class
    """
    def __init__(self, 
                adjmtx,
                vset:float      = 0.01,
                vreset:float    = 0.005,
                ron:float       = 12.9e3,
                roff:float      = 12.9e6,
                fluxcrit:float  = 0.01,
                fluxmax:float   = 0.015,
                boost:float     = 10.0,
                eta:float       = 1.0
                 ) -> None:
        self.adjmtx     = adjmtx
        self.vset       = vset 
        self.vreset     = vreset 
        self.ron        = ron 
        self.roff       = roff 
        self.fluxcrit   = fluxcrit 
        self.fluxmax    = fluxmax 
        self.boost      = boost 
        self.eta        = eta
        self.nodenum:int= self.adjmtx.shape[0]
        self.edgenum:int= np.sum(np.triu(self.adjmtx)!=0)
        self.edgelist   = np.argwhere(np.triu(self.adjmtx))
        self.flux       = np.zeros(self.edgenum) #Assuming 0 flux initialisation
        self.conductance= np.zeros(self.edgenum) #Thus 0 cond initialisation

    def update_conductance(self):
        """
        Conductance updated with the tunnelling memristor model.
        This is an exact copy from the original Python implementation found in
        https://github.com/rzhu40/ASN_simulation
        """
        phi = 0.81
        C0 = 10.19
        J1 = 0.0000471307
        A = 0.17
        d = (self.fluxcrit - abs(self.flux))*5/self.fluxcrit
        d[d<0] = 0 
        rt = 2/A * d**2 / phi**0.5 * np.exp(C0*phi**2 * d)/J1
        self.conductance = 1/(rt + self.ron) +  1/self.roff
        return self.conductance
    
    def update_flux(self, edge_voltages, dt:float):
        """
        Updating the flux linkage of every memristive edge
        """
        dflux = (abs(edge_voltages) > self.vset) *\
                (abs(edge_voltages) - self.vset) *\
                np.sign(edge_voltages) 
        dflux = dflux + \
                (abs(edge_voltages) < self.vreset) *\
                (abs(edge_voltages) - self.vreset) *\
                np.sign(self.flux) * self.boost
        self.flux = self.flux + dt*dflux*self.eta
        self.flux[abs(self.flux) > self.fluxmax] = \
                np.sign(self.flux[abs(self.flux) > self.fluxmax])*self.fluxmax
        return self.flux
    
def get_node_voltages(nwn, signal):
    n = nwn.nodenum
    lhs = nwn.Gmtx 
    rhs = nwn.rhs
    lhs[nwn.edgelist[:,0], nwn.edgelist[:,1]] = -nwn.conductance 
    lhs[nwn.edgelist[:,1], nwn.edgelist[:,0]] = -nwn.conductance 
    lhs[range(n), range(n)] = 0 #needs to be here for correct sum below
    lhs[range(n), range(n)] = -np.sum(lhs[:n,:n], axis=0)
    rhs[n:] = signal
    sol = np.linalg.solve(lhs, rhs)
    return sol[:n]

def neuro_sim(
        nwn,
        electrodes:list,
        electrode_signals,
        dt:float    = 0.05,
        steps:int   = 10,
        sig_augment = None,
        return_flux:bool = False,
        disable_pbar:bool = False,
    ):
    """
    Input:
        nwn_params          (dict):     NanowireNetwork object
        electrodes          (list):     list of electrode node indices for the input and drain electrodes
        electrode_signals   (ndarray):  array of input signals
        dt                  (float):    timestep size of simulation
        steps               (int):      how many steps to simulate
        return_flux         (bool):     whether to return the edge flux vals
        disable_pbar        (bool):     whether to disable the progress bar
    Returns:
        (ndarray): all node voltage signals
    """
    n = nwn.nodenum
    m = len(electrodes)
    node_voltages = np.zeros((steps, n))
    edge_voltages = np.zeros(nwn.edgenum)
    if return_flux:
        edge_fluxes = np.zeros((steps, nwn.edgenum))
    update_signal = sig_augment is not None

    nwn.Gmtx = np.zeros((n+m, n+m))
    nwn.rhs = np.zeros(n+m)
    for i, this_elec in enumerate(electrodes):
        nwn.Gmtx[n+i, this_elec] = 1
        nwn.Gmtx[this_elec, n+i] = 1
    for t in pbar(
        range(steps), desc='|NWN Sim', 
        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', disable = disable_pbar):
        nwn.update_conductance()
        vs = get_node_voltages(
            nwn=nwn, signal=electrode_signals[t,:])
        edge_voltages = vs[nwn.edgelist[:,0]] - vs[nwn.edgelist[:,1]]
        nwn.update_flux(edge_voltages=edge_voltages, dt=dt)
        node_voltages[t,:] = vs
        if return_flux:
            edge_fluxes[t,:] = nwn.flux
        if update_signal and t+1<steps:
            electrode_signals = sig_augment(electrode_signals, node_voltages,nwn,t)

    if return_flux:
        return node_voltages, edge_fluxes  
    return node_voltages

def find_critical_voltage(nwn, electrodes, v_range=(0.0001, 0.2), n_voltages=100, steps=100, threshold=1e-6):
    """
    NEW IMPLEMENTATION/APLICATION: Find critical voltage using avalanche size distribution analysis
    """
    voltages = np.linspace(v_range[0], v_range[1], n_voltages)
    avalanche_sizes = []
    
    for v in voltages:
        # Apply DC voltage and measure switching avalanches
        initial_conductance = nwn.conductance.copy()
        
        # Single pulse or steady-state approach
        signal = np.array([[v, 0]])  # voltage between first two electrodes
        
        # Run simulation to equilibrium
        node_voltages = neuro_sim(nwn, electrodes, 
                                 np.repeat(signal, steps, axis=0), 
                                 dt=0.01, steps=100)
        
        # Count how many memristors switched
        final_conductance = nwn.conductance
        switches = np.sum(np.abs(final_conductance - initial_conductance) > threshold)
        avalanche_sizes.append(switches)

    max_idx = np.argmax(avalanche_sizes)
    critical_voltage = voltages[max_idx]
    max_avalanche_size = avalanche_sizes[max_idx]
    
    print(f"Critical voltage: {critical_voltage:.4f}V")
    print(f"Max avalanche size: {max_avalanche_size}")
    
    return voltages, avalanche_sizes


if __name__ == "__main__":
    # Simple NWN to start with 

    adjmtx = np.array([
        [0, 1, 0, 1, 1, 0],  # Node 0: connected to 1, 3, 4
        [1, 0, 1, 0, 1, 0],  # Node 1: connected to 0, 2, 4  
        [0, 1, 0, 0, 1, 1],  # Node 2: connected to 1, 4, 5
        [1, 0, 0, 0, 1, 0],  # Node 3: connected to 0, 4
        [1, 1, 1, 1, 0, 1],  # Node 4: central hub (connected to all)
        [0, 0, 1, 0, 1, 0],  # Node 5: connected to 2, 4
    ])
    nwn = NanowireNetwork(adjmtx=adjmtx)
    v, sizes = find_critical_voltage(nwn, electrodes=[0,1]) #IMPORTANT: OUTPUTS Critical voltage: 0.0506V; Max avalanche size: 

    
    plt.plot(v, sizes)
    plt.xlabel('Applied Voltage (V)')
    plt.ylabel('Avalanche Size (Number of Switches)')
    plt.savefig("avalanche_size_vs_voltage.png", dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
