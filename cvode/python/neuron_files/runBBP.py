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
#os.chdir("NeuroGPU/NeuroGPU_Base/VS/pyNeuroGPU_win2/NeuroGPU6/python/")


## set up filepaths
paramsCSV = '../params/params.csv'
data_dir = '../Data/'
run_dir = '../bin'
vs_fn = '/tmp/Data/VHotP'
nstims = 1

nGpus = len([devicenum for devicenum in os.environ['CUDA_VISIBLE_DEVICES'] if devicenum != ","])



if not os.path.isdir('/tmp/Data'):
    os.mkdir('/tmp/Data')
    

def nrnMread(fileName):
    f = open(fileName, "rb")
    nparam = struct.unpack('i', f.read(4))[0]
    typeFlg = struct.unpack('i', f.read(4))[0]
    return np.fromfile(f,np.double)

        
def run_model(stim_ind,real_ind):
    """
    Parameters
    -------------------------------------------------------
    stim_ind: index to send as arg to neuroGPU 
    params: DEPRECATED remove

    Returns
    ---------------------------------------------------------
    p_object: process object that stops when neuroGPU done
    """
    global_rank = 0
    volts_fn = vs_fn + str(stim_ind) + '.dat'
    if os.path.exists(volts_fn):
        print("removing ", volts_fn, " from ", global_rank)
        os.remove(volts_fn)
        pass
    #!{'../bin/neuroGPU'+str(global_rank),str(stim_ind), str(global_rank)}
    p_object = subprocess.Popen(['../bin/neuroGPU',str(stim_ind)],
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT,  # <-- redirect stderr to stdout
                    bufsize=1)
    
    with p_object.stdout:
        for line in iter(p_object.stdout.readline, b''):
            print(line),
    p_object.wait()
    print(p_object.stderr)
    #os.rename(volts_fn,'/tmp/Data/VHotP'+str(real_ind)+'.h5')

    return p_object



def main():
    
    p = subprocess.Popen(["sh","watch_gpu_util.sh"])
            
       

    ###### CREATE MAPPING ################# 
    start = time.time()
    run_model(0,0)
    end = time.time()
    with open('results.txt','w') as f:
        f.write(str(end - start))
    data = nrnMread("../Data/VHotP0.dat")
    print(np.max(data))


if __name__ == "__main__":
    main()
