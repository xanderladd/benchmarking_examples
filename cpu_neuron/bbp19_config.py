import csv
import pandas as pd
import os
import numpy as np
import h5py
import utils
import logging



model = "bbp"
peeling ="potassium"
logging.info("######## USING HARDCODE POTASSIUM in CONFIG #######")
peeling = "potassium"
logging.info("and a lot of other HARDCODED PARAMS")
params_opt_ind = [2,3,4,6,11,12,16,20,23]
logging.info("params opt ind = " + str(params_opt_ind))
date = "02_04_2021"
logging.info("DATE: " + str(date))
usePrev =True
orig_name = "orig_" + peeling
orig_params = h5py.File('../../params/params_' + model + '_' + peeling + '.hdf5', 'r')[orig_name][0]
if usePrev == "True":
    paramsCSV = '../../params/params_' + model + '_' + peeling + '_prev.csv'
else:
    paramsCSV = '../../params/params_' + model + '_' + peeling + '.csv'
    
if peeling == "sodium":
    templateCSV = "../../params/params_bbp_peeling_description.csv"
else:
    templateCSV = "../../params/params_bbp_{}.csv".format(peeling)
scores_path = '../../scores/'
objectives_file = h5py.File('../../objectives/multi_stim_without_sensitivity_bbp_' + peeling + "_" + date + '_stims.hdf5', 'r')
opt_weight_list = objectives_file['opt_weight_list'][:]
opt_stim_name_list = objectives_file['opt_stim_name_list'][:]
score_function_ordered_list = objectives_file['ordered_score_function_list'][:]
logging.info("stims : neg stims")
stims_path = '../../stims/neg_stims' + '.hdf5'
stim_file = h5py.File(stims_path, 'r')
#target_volts_path = './target_volts/allen_data_target_volts_10000.hdf5'
#target_volts_hdf5 = h5py.File(target_volts_path, 'r')
#params_opt_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
#params_opt_ind = np.arange(24) 
model_dir = '..'
data_dir = model_dir+'/Data/'
run_dir = '../bin'
vs_fn = '/tmp/Data/VHotP'

# Number of timesteps for the output volt.
ntimestep = 10000

