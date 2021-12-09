import csv
import pandas as pd
import os
import numpy as np
import h5py
import nrnUtils
import multiprocessing
from mpi4py import MPI
# import utils


#nCpus =  multiprocessing.cpu_count()
comm = MPI.COMM_WORLD
global_rank = comm.Get_rank()
size = comm.Get_size()

# print("USING nGPUS: ", nGpus, " and USING nCPUS: ", nCpus)
# print("Rank: ", global_rank)
# CPU_name = MPI.Get_processor_name()
# print("CPU name", CPU_name)



run_volts_path ='../GPU_genetic_alg/' # archival: '../../run_volts_bbp_full_gpu_tuned/'
# original model
# objectives_file = h5py.File('./objectives/multi_stim_bbp_full_allen_gpu_tune_18_stims.hdf5', 'r')
# target_volts_path = './target_volts/allen_data_target_volts_10000.hdf5'
# stims_path = run_volts_path+'/stims/allen_data_stims_10000.hdf5'

# new mode
objectives_file = h5py.File('../GPU_genetic_alg/python/objectives/allen485835016_objectives.hdf5', 'r')
target_volts_path = '../GPU_genetic_alg/python/target_volts/target_volts_485835016.hdf5'
stims_path = run_volts_path+'/stims/stims_485835016.hdf5'
visualize=False




run_file = './neuron_files/allen/run_model_cori.hoc'
paramsCSV = run_volts_path+'params/params_bbp_full_gpu_tuned_10_based.csv'
orig_params = h5py.File(run_volts_path+'params/params_bbp_full_allen_gpu_tune.hdf5', 'r')['orig_full'][0]
scores_path = '../scores/'
opt_weight_list = objectives_file['opt_weight_list'][:]
opt_stim_name_list = objectives_file['opt_stim_name_list'][:]
opt_stim_list = [e for e in opt_stim_name_list if len(e) < 8]

score_function_ordered_list = objectives_file['ordered_score_function_list']
target_volts_hdf5 = h5py.File(target_volts_path, 'r')
ap_tune_stim_name = '18'
ap_tune_weight = 0
params_opt_ind = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
data_dir = '../Data/allenData/'
run_dir = '../bin'
vs_fn = '../Data/VHotP'
stim_file = h5py.File(stims_path, 'r')
target_volts_hdf5 = h5py.File(target_volts_path, 'r')

templateCSV = "../params/params_bbp_full_gpu_tuned_10_based.csv"

# Number of timesteps for the output volt.
ntimestep = 10000

stim_names = list([e for e in opt_stim_name_list if len(e) < 7])

custom_score_functions = [
                    'chi_square_normal',\
                    'traj_score_1',\
                    'traj_score_2',\
                    'traj_score_3',\
                    'isi',\
                    'rev_dot_product',\
                    'KL_divergence']


# orig_params = [0.0000022060, -66.5600000000, 4.0383444556,3.1263250613,0.0047147854,2.8515288117, 5.1781678079, 0.1096452854,0.0005345695, 0.1578655731,0.0016590198, 3.7929209114, 0.0036489355,0.0002609501]