import numpy as np
import h5py
import os
os.chdir("neuron_files/") # DO NOT keep this for when you want to run Allen
from neuron import h
os.chdir("../")
import pandas as pd
import time
import multiprocessing
import pickle
# DO NOT keep this for when you want to run Allen
run_file = './neuron_files/run_model_cori.hoc'
h.load_file(run_file)


orig_params = [.00003, -75.0, 3.13796, 0.089259, 0.006827, 0.973538, 1.021945, 0.008752, 0.00099, 0.303472, 0.000994, 0.983955, 0.000333]

# Number of timesteps for the output volt.
ntimestep = 3168

# Value of dt in miliseconds
dt = 0.02


def run_model(param_set):
    stim_list = [f'../Data/Stim_raw{i}.csv' for i in range(8)]
    h.cvode.active(0) # make sure cvode is off

    volts_list = []
    start = time.time()
    for elem in stim_list:
        curr_stim = np.genfromtxt(elem, delimiter=',')
        total_params_num = len(param_set)
        timestamps = np.array([dt for i in range(ntimestep)])
        h.curr_stim = h.Vector().from_python(curr_stim)
        h.transvec = h.Vector(total_params_num, 1).from_python(param_set)
        h.stimtime = h.Matrix(1, len(timestamps)).from_vector(h.Vector().from_python(timestamps))
        h.ntimestep = ntimestep
        h.runStim()
        out = h.vecOut.to_python()
        volts_list.append(out)
    end = time.time()
    
    print("\n \n")
    print(end - start)
    # print(h.cvode.statistics())
    print("\n \n")

    with open('neuron_fixed_indvs','a+') as f:
        f.write(f'{end - start} \n')
    
    volts_list = [] # return empty list so multiprocessing overhead is not included in comparision
    return np.array(volts_list)

def run_population(population, save_key):
    # run_model(population[0])
    start = time.time()
    with multiprocessing.Pool(64) as p:
        res = p.map(run_model,population)
    end = time.time()
    with open('neuron_fixed_pop', 'a+') as f:
        f.write(f'{save_key} : {end-start} \n')    # LOG RESULTS
                 
                                  
if __name__ == "__main__":
    with open('rehash_history','rb') as f:
        data = pickle.load(f)
    for key,val in data.items():
        run_population(val,key)
    exit() 


