import struct
import os
import csv
import pandas as pd
import h5py
import numpy as np


nrnTypes = {
    2: np.short,
    3: np.float32,
    4: np.double,
    5: np.int,
}

#number of timesteps
ntimestep = 10000

def noSpikePen(volts, targetVolts):
    if np.max(volts) < 0 and  np.max(targetVolts) > 0:
        return 800
    else:
        return 0

def SpikePen(volts, targetVolts):
    if np.max(targetVolts) < 0 and  np.max(volts) > 0:
        return 400
    else:
        return 0


def nrnMread(fileName):
    f = open(fileName, "rb")
    nparam = struct.unpack('i', f.read(4))[0]
    typeFlg = struct.unpack('i', f.read(4))[0]
    return np.fromfile(f,np.double)

def nrnMreadH5(fileName): # W
    f = h5py.File(fileName,'r')
    dat = f['Data'][:][0]
    return np.array(dat)


def readParamsCSV(fileName):
    fields = ['Param name', 'Base value','Lower bound', 'Upper bound']
    df = pd.read_csv(fileName,skipinitialspace=True, usecols=fields)
    
    paramsList = [tuple(x) for x in df.values]
    return paramsList
    
def readCSV(fileName):
    df = pd.read_csv(fileName,header=None)
    paramsList = [tuple(x) for x in df.values]
    return paramsList



def stim_swap(idx, i):
    """
    Stim swap takes 'idx' which is the stim index % 8 and 'i' which is the actual stim idx
    and then deletes the one at 'idx' and replaces it with the stim at i so that 
    neuroGPU reads stims like 13 as stim_raw5 (13 % 8)
    """
#     old_stim = '../Data/Stim_raw' + str(idx) + '.csv'
#     old_time = '../Data/times' + str(idx) + '.csv'
#     if os.path.exists(old_stim):
#         os.remove(old_stim)
#         os.remove(old_time)
#     os.rename(r'../Data/Stim_raw' + str(i) + '.csv', r'../Data/Stim_raw' + str(idx) + '.csv')
#     os.rename(r'../Data/times' + str(i) + '.csv', r'../Data/times' + str(idx) + '.csv')
    old_stim = '../Data/Stim_raw' + str(idx) + '.csv'
    old_time = '../Data/times' + str(idx) + '.csv'
    if os.path.exists(old_stim):
        os.remove(old_stim)
        os.remove(old_time)
    os.rename(r'../Data/Stim_raw' + str(i) + '.csv', r'../Data/Stim_raw' + str(idx) + '.csv')
    os.rename(r'../Data/times' + str(i) + '.csv', r'../Data/times' + str(idx) + '.csv')

    
def get_first_zero(stim):
    """Kyung helper function to penalize AP where there should not be one"""
    for i in range(len(stim)-2, -1, -1):
        if stim[i] > 0 and stim[i+1] == 0:
            return i+1
    return None

def check_ap_at_zero(stim_ind, volts, opt_stim_name_list, stim_file):
    """
    Kyung function to check if a volt should be penalized for having an AP before there 
    should be one. Modified to take in "volts" as a list of individuals instead of "volt"
    """
    print(len(list([e for e in opt_stim_name_list])),int(stim_ind) )
    stim_name = list([e for e in opt_stim_name_list])[int(stim_ind)]
    stim = stim_file[stim_name][:]
    first_zero_ind = get_first_zero(stim)
    nindv =volts.shape[0]
    checks = np.zeros(nindv)
    for i in range(nindv):
        volt = volts[i,:]
        if first_zero_ind:
            if np.mean(stim[first_zero_ind:]) == 0:
                first_ind_to_check = first_zero_ind + 1000
                APs = [True if v > 0 else False for v in volt[first_ind_to_check:]]
                if True in APs:
                    #return 400 # threshold parameter that I am still tuning
                    #print("indv:",i, "stim ind: ", stim_ind)
                    checks[i] = 0
    return checks      


def getVolts(vs_fn, idx): # W
    '''Helper function that gets volts from data and shapes them for a given stim index'''
#     fn = vs_fn + str(idx) +  '.h5'    #'.h5' 
#     curr_volts =  nrnMreadH5(fn)
    fn = vs_fn + str(idx) +  '.dat'    #'.h5'
    curr_volts =  nrnMread(fn)
    Nt = int(len(curr_volts)/ntimestep)
    shaped_volts = np.reshape(curr_volts, [Nt,ntimestep])
    return shaped_volts



#def wrap_eval(self,pair):
#    result = [None]
#    pid = os.getpid()
#    print(pid)
#    cProfile.runctx("result[0] = self.eval_stim_sf_pair(pair)", globals(), locals(), 'prof%d.prof' % pid)
#    return result[0]



def top_SFs(run_num, score_function_ordered_list, weights, nGpus, threshold=0):
    """"    
    finds scoring functions w/ weight over 50 and pairs them with that stim and sends
    them to mapping function so that we will run so many processes
    Arguments
    --------------------------------------------------------------
    run_num: the number of times neuroGPU has ran for 8 stims,
    keep track of what stims we are picking out score functions for
    """
    
    all_pairs = []
    last_stim = (run_num+1) * nGpus # ie: 0th rank last_stim = (0+1)*ngpus = ngpus
    first_stim = last_stim - nGpus
    if last_stim > 18:
        last_stim = 18
    #print(first_stim,last_stim, "first and last...... rank: ", run_num)
    for i in range(first_stim, last_stim):#range(first_stim,last_stim):
        sf_len = len(score_function_ordered_list)
        curr_weights = weights[sf_len*i: sf_len*i + sf_len] #get range of sfs for this stim
        #top_inds = sorted(range(len(curr_weights)), key=lambda i: curr_weights[i], reverse=True)[:10] #finds top ten biggest weight indices
        top_inds = np.where(curr_weights > threshold)[0] # weights bigger than 50 #TODO: maybe this can help glitch
        pairs = list(zip(np.repeat(i,len(top_inds)), [ind for ind in top_inds])) #zips up indices with corresponding stim # to make sure it is refrencing a relevant stim
        all_pairs.append(pairs)
    flat_pairs = [pair for pairs in all_pairs for pair in pairs] #flatten the list of tuples
    return flat_pairs
    

# convert the allen data and save as csv
def convert_allen_data(opt_stim_name_list, stim_file, dts):
    """
    Function that sets up our new allen data every run. It reads and writes every stimi
    and timesi and removes previous ones. Using csv writer to write timesi so it reads well.
    """
    for i in range(len(opt_stim_name_list)):
        old_stim = "../Data/Stim_raw{}.csv".format(i)
        old_time = "../Data/times{}.csv".format(i)
        if os.path.exists(old_stim) :
            os.remove(old_stim)
            os.remove(old_time)
    for i in range(len(opt_stim_name_list)):
        # remove this somehow
        if len(opt_stim_name_list[i]) > 8:
            continue
        stim = opt_stim_name_list[i]#.decode('ASCII')
        dt = stim_file[stim+'_dt'][:][0]# refactor this later to be read or set to .02 if not configured
        dts.append(dt)
        f = open ("../Data/times{}.csv".format(i), 'w')
        wtr = csv.writer(f, delimiter=',', lineterminator='\n')
        current_times = [dt for i in range(ntimestep)]
        wtr.writerow(current_times)
        writer = csv.writer(open("../Data/Stim_raw{}.csv".format(i), 'w'))
        writer.writerow(stim_file[stim][:])
    return dts

# #     # convert the allen data and save as csv
# def convert_allen_data(opt_stim_name_list, stim_file, dts):
#         """
#         Function that sets up our new allen data every run. It reads and writes every stimi
#         and timesi and removes previous ones. Using csv writer to write timesi so it reads well.
#         """
#         for i in range(len(opt_stim_name_list)):
#             old_stim = "../Data/Stim_raw{}.csv".format(i)
#             old_time = "../Data/times{}.csv".format(i)
#             if os.path.exists(old_stim) :
#                 os.remove(old_stim)
#                 os.remove(old_time)
#         for i in range(len(opt_stim_name_list)):
#             stim = opt_stim_name_list[i].decode("utf-8")
#             dt = .02 # refactor this later to be read or set to .02 if not configured
#             dts.append(dt)
#             f = open ("../Data/times{}.csv".format(i), 'w')
#             wtr = csv.writer(f, delimiter=',', lineterminator='\n')
#             current_times = [dt for i in range(ntimestep)]
#             wtr.writerow(current_times)
#             writer = csv.writer(open("../Data/Stim_raw{}.csv".format(i), 'w'))
#             writer.writerow(stim_file[stim][:])