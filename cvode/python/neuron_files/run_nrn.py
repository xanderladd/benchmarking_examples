import os
import numpy as np
import matplotlib.pyplot as plt
import struct
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pickle
# from extractModel_mappings import   allparams_from_mapping
import subprocess
import csv
import bluepyopt as bpop
import shutil, errno
import pandas as pd
import time
from neuron import h
#os.chdir("NeuroGPU/NeuroGPU_Base/VS/pyNeuroGPU_win2/NeuroGPU6/python/")


## set up filepaths
paramsCSV = '../params/params.csv'
data_dir = '../Data/'
run_dir = '../bin'
vs_fn = '/tmp/Data/VHotP'
nstims = 1
run_file='run_model_cori.hoc'
ntimestep=3168
stim_path = '../Data/Stim_raw.csv'
time_path = '../Data/times.csv'


             
def run_model():
    h.load_file(run_file)
    volts_list = []

    param_set = pd.read_csv('../Data/opt_table.csv').iloc[3].values
   
    curr_stim = np.genfromtxt(stim_path, delimiter=',')
    total_params_num = len(param_set)
    # dt = .02 # constant dt for this experiment is .02
    timestamps = np.genfromtxt(time_path, delimiter=',')  #np.array([dt for i in range(ntimestep)])
    h.curr_stim = h.Vector().from_python(curr_stim)
    h.transvec = h.Vector(total_params_num, 1).from_python(param_set)
    h.stimtime = h.Matrix(1, len(timestamps)).from_vector(h.Vector().from_python(timestamps))
    h.ntimestep = ntimestep
    h.runStim()
    out = h.vecOut.to_python()        
    return out



if not os.path.isdir('/tmp/Data'):
    os.mkdir('/tmp/Data')
    

def nrnMread(fileName):
    f = open(fileName, "rb")
    nparam = struct.unpack('i', f.read(4))[0]
    typeFlg = struct.unpack('i', f.read(4))[0]
    return np.fromfile(f,np.double)


def main():


    ###### CREATE MAPPING ################# 
    start = time.time()
    data = run_model()
    end = time.time()
    with open('results.txt','w') as f:
        f.write(str(end - start))
    print(np.max(data))


if __name__ == "__main__":
    main()
