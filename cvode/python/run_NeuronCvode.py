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
from multiprocessing import Process, Manager



h.load_file("neuron_files/allen/run_model_cori.hoc")
run_file = './neuron_files/run_model_cori.hoc'

# set up MPI
comm = MPI.COMM_WORLD
global_rank = comm.Get_rank()
size = comm.Get_size()

# h.cvode.active(1)
pc = h.ParallelContext()



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
        v = h.Vector().from_python(np.genfromtxt(new_stim_path, delimiter=',')[:3000])
        v.play(self.ic, self.ic._ref_amp, True)
        self.rd = {k: h.Vector().record(v, sec=self.soma[0]) for k,v in zip(['t', 'v', 'stim_i', 'amp', 'dt'],
                                                    [h._ref_t, self.soma[0](.5)._ref_v, self.ic._ref_i, self.ic._ref_amp, h._ref_dt])}
        
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


def run_model(params):
    stim_list = [f'../Data/Stim_raw{i}.csv' for i in range(8)]
    h.load_file(run_file)
    h.cvode.active(1)
    h.cvode.use_local_dt()
    # Number of timesteps for the output volt.
    ntimesteps = 3168
    tstop = ntimesteps*h.dt
    
    curr_cell  = Cell(0, 0, stim_list[0]) # gid = 0
    curr_cell.update_params(params)
    start = time.time()
    volts_list = []
    for elem in stim_list:
        curr_cell.update_stim(elem)
        h.finitialize(-65)
        h.tstop = tstop
        h.run()


    end = time.time()
    print(end-start,"TIME in map")
    print(h.cvode.statistics())
    with open('neuron_adaptive_indvs_64','a+') as f:
        f.write(f'{end - start} \n')
    runtime = end - start
    # import pdb; pdb.set_trace()
    
    curr_v = curr_cell.rd['v'].as_numpy()
    #debug plot DTS
    # curr_v = curr_cell.rd['dt'].as_numpy()
    # import matplotlib.pyplot as plt
    # plt.hist(curr_v)
    # plt.ylabel('count')
    # plt.xlabel('dt')
    # plt.savefig('tst.png')
    
    # debug check if neuron fires
    # with open('rdv','a+') as f:
    #     f.write(f"{curr_v.shape} max volt \n")
    # if np.max(curr_v) > 0:
    #     with open('rdv','a+') as f:
    #         f.write(f"{np.max(curr_v)} max volt \n")
    del curr_cell
    volts_list = [] # return empty list so multiprocessing overhead is not included in comparision
    return np.zeros(volts_list), runtime


    
    
    

def run_population(population, save_key):
    start = time.time()
    start = time.time()
    with multiprocessing.Pool(64) as p:
        res = np.array(p.map(run_model,population))
    end = time.time()
    lst = res[:,1] # get runtime which is second arg
    res = res[:,0] # get simulations which is first arg
    
    with open('neuron_adaptive_pops', 'a+') as f:
        f.write(f'{save_key} : {end-start} \n')    # LOG RESULTS

    return lst

    
    



if __name__ == "__main__":
    
    
     #p = subprocess.Popen(["sh","watch_gpu_util.sh"])

    with open('rehash_history','rb') as f:
        data = pickle.load(f)
    for key,val in data.items():
        run_population(val,key)
    exit() 
