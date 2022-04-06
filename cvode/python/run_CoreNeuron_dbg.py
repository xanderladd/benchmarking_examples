import numpy as np
import h5py
import os
import subprocess
import re
import time
import struct
import copy
import csv
from mpi4py import MPI
import multiprocessing
import logging
import sys
import pickle
import argparse
os.chdir('neuron_files/allen')
from neuron import h
import shutil
import glob
os.chdir('../../')
import subprocess



h.load_file("neuron_files/allen/run_model_cori.hoc")
h.dt = 0.02
# Number of timesteps for the output volt.
ntimesteps = 3168
tstop = ntimesteps*h.dt
stim_path = '../Data/Stim_raw0.csv'
times_path = '../Data/times0.csv'
# set up MPI
comm = MPI.COMM_WORLD
global_rank = comm.Get_rank()
size = comm.Get_size()


# importing MPI or h.nrnmpi_init() must come before the first instantiation of ParallelContext()
h.nrnmpi_init()
h.cvode.active(1)
pc = h.ParallelContext(64)



orig_params = [.00003, -75.0, 3.13796, 0.089259, 0.006827, 0.973538, 1.021945, 0.008752, 0.00099, 0.303472, 0.000994, 0.983955, 0.000333]
param_list = orig_params
nStims = 8


class Cell:
    def __init__(self, gid, run, stim_path):
        start = time.time()
        self.hoc_cell = h.cADpyr232_L5_TTPC1_0fb1ca4724()
        end = time.time()
        self.gid = gid
        # if run == 0:
            # self.pc.set_gid2node(gid, pc.id())
        curr_p = param_list  # TODO ----> #[gid]*gid
        self.update_params(curr_p)
        self.ic = h.IClamp(self.soma[0](.5))
        self.ic.amp = 0.5
        self.ic.dur = 1e9
        v = h.Vector().from_python(np.genfromtxt(stim_path, delimiter=','))
        v.play(self.ic, self.ic._ref_amp, True)
        self.rd = {k: h.Vector().record(v, sec=self.soma[0]) for k,v in zip(['t', 'v', 'stim_i', 'amp'],
                                                    [h._ref_t, self.soma[0](.5)._ref_v, self.ic._ref_i, self.ic._ref_amp])}
        
    def __getattr__(self, name):
        return getattr(self.hoc_cell, name)
    
    def update_stim(self, new_stim_path):
        v = h.Vector().from_python(np.genfromtxt(new_stim_path, delimiter=','))
        v.play(self.ic, self.ic._ref_amp, True)
        self.rd = {k: h.Vector().record(v, sec=self.soma[0]) for k,v in zip(['t', 'v', 'stim_i', 'amp'],
                                                    [h._ref_t, self.soma[0](.5)._ref_v, self.ic._ref_i, self.ic._ref_amp])}
        
    def update_params(self,p):
        #all
        for curr_sec in self.hoc_cell.all:
            curr_sec.g_pas = p[0]
            curr_sec.e_pas = p[1]
        # axonal
        for curr_sec in self.hoc_cell.axonal:
            # replicate
            curr_sec.gNaTa_tbar_NaTa_t = p[2]
            curr_sec.gK_Tstbar_K_Tst = p[3]
            curr_sec.gNap_Et2bar_Nap_Et2 = p[4]
            curr_sec.gK_Pstbar_K_Pst = p[5]
            curr_sec.gSKv3_1bar_SKv3_1 = p[6]
            curr_sec.gCa_LVAstbar_Ca_LVAst = p[7]
            curr_sec.gCa_HVAbar_Ca_HVA = p[8]
        # somatic
        for curr_sec in self.hoc_cell.somatic:
            curr_sec.gSKv3_1bar_SKv3_1 = p[9]
            curr_sec.gCa_HVAbar_Ca_HVA = p[10]
            curr_sec.gNaTs2_tbar_NaTs2_t = p[11]
            curr_sec.gCa_LVAstbar_Ca_LVAst = p[12]

            
def nrnMread(fileName):
    f = open(fileName, "rb")
    nparam = struct.unpack('i', f.read(4))[0]
    typeFlg = struct.unpack('i', f.read(4))[0]
    return np.fromfile(f,np.double)


def run_model(stim_ind=7): # pass stim as arg and set it with h?
    h.stimFile = f'../Data/{stim_ind}'
    h.finitialize(-65)
    h.tstop = tstop
    h.run()
    # pc.psolve(tstop)
    
def run_population(params, save_key):
    cell_ids = range(0, len(params)) 
# 
    cell_groups = []
    cell_count = 0
    params = params[:2]
    # chunklen = len(params) // size
    # params = params[int(global_rank*chunklen):int((global_rank+1)*chunklen)] # core neuron sees every 8 params due to MPI framing
    
    for param_set in params:
        # run num and chnl reps are 0
        curr_cells = []
        for stim_idx in range(nStims):
            stim_path = f'../Data/Stim_raw{stim_idx}.csv'
            curr_cell  = Cell(cell_count, 0, stim_path)
            curr_cell.update_params(param_set)
            curr_cells.append(curr_cell)
            cell_count += 1
        print(len(cell_groups), " cells instantiated")
        cell_groups.append(curr_cells) 
    start_time_sim = time.time()
    start = time.time()
    run_model()
    end = time.time()
    import pdb; pdb.set_trace()
    print(end-start, "NO CVODE")
    if global_rank == 0:
        with open('cvode_neuron_times_v2_yescvode', 'a+') as f:
            f.write(f'{save_key} : {end-start} \n')    # LOG RESULTS
    pc.done()
    



if __name__ == "__main__":
    
    
     #p = subprocess.Popen(["sh","watch_gpu_util.sh"])

    with open('rehash_history','rb') as f:
        data = pickle.load(f)
    for key,val in data.items():
        run_population(val,key)
        print('done after pop 1 patch')
    exit() 