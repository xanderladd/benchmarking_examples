import numpy as np
import h5py
import nrnUtils
import score_functions as sf
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
from extractModel_mappings import allparams_from_mapping
import argparse




parser = argparse.ArgumentParser(description='control size of simulation - evaluation loop')
parser.add_argument('--offspring_size', type=int, required=False, default=100, help='size of simulation')
parser.add_argument('--trials', type=int, required=False, default=1, help='number of trials to run')

args, unknown = parser.parse_known_args()
# score function list
with open("../all_data/score_functions.pkl", "rb") as f:
    sf_list = pickle.load(f)['score_functions']
vs_fn = '../Data/VHotP'
nGpus = min(8,len([devicenum for devicenum in os.environ['CUDA_VISIBLE_DEVICES'] if devicenum != ","]))

#path to group of parameters
param_path = os.path.join('../','all_data','offspring.csv')
# Number of timesteps for the output volt.
ntimestep = 5000
# MPI setup
comm = MPI.COMM_WORLD
size = comm.Get_size()
total_rank = comm.Get_rank()
global_rank = comm.Get_rank() // nGpus
local_rank = comm.Get_rank() % nGpus
total_size = comm.Get_size()
global_size = size // nGpus
#print("Process is eligibl to run on:", affinity, "g rank ", global_rank, "len aff:", len(affinity))

if total_rank == 0:
    if os.path.isfile(f'logs/sim_size_{args.offspring_size}.log'):
        os.remove(f'logs/sim_size_{args.offspring_size}.log')
    logging.basicConfig(filename=f'logs/sim_size_{args.offspring_size}.log', level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    print(f"-------------------- output being directed to sim_size_{args.offspring_size}.log --------------------")



print("USING nGPUS: ", nGpus)
custom_score_functions = [
                    'chi_square_normal',\
                    'traj_score_1',\
                    'traj_score_2',\
                    'traj_score_3',\
                    'isi',\
                    'rev_dot_product',\
                    'KL_divergence']


def nrnMread(fileName):
    f = open(fileName, "rb")
    nparam = struct.unpack('i', f.read(4))[0]
    typeFlg = struct.unpack('i', f.read(4))[0]
    return np.fromfile(f,np.double)


def divide_params(param_values, size, rank):
    myChunk = [(len(param_values) // size) * rank , (len(param_values) // size) * (rank + 1)]
    # this is a bit hacky, just tacks on last ind if we need to because the split isn't great
    # cleaning
    if rank == size -1:
        myChunk[1] = len(param_values)

    myChunk = (myChunk[0],myChunk[1])
    param_values = param_values[myChunk[0]:myChunk[1],:]
    # revisit use of this var when you clean
    my_indvs = np.arange(myChunk[0], myChunk[1]).astype(int)

    return param_values, my_indvs

    
def top_SFs(run_num, max_sfs=50):
    """
    randomly choose `max_sfs` # of score functions for each stim and pair them up
    Arguments
    --------------------------------------------------------------
    run_num: the number of times neuroGPU has ran for 8 stims,
    keep track of what stims we are picking out score functions for
    """
    all_pairs = []
    last_stim = (run_num + 1) * nGpus # ie: 0th run last_stim = (0+1)*8 = 8
    first_stim = last_stim - nGpus # on the the last round this will be 24 - 8 = 16
    if last_stim > 18:
        last_stim = 18
    chosen_sfs = np.random.choice(np.arange(len(sf_list)), max_sfs)
    stim_correspondance = np.array([np.repeat(stim_ind, len(chosen_sfs)) for stim_ind in range(first_stim, last_stim)]).flatten()
    chosen_sfs = np.repeat(chosen_sfs, len(range(first_stim, last_stim)))
    all_pairs = zip(stim_correspondance,chosen_sfs) #zips up indices with corresponding stim # to make sure it is refrencing a relevant stim
    #flat_pairs = [pair for pairs in all_pairs for pair in pairs] #flatten the list of tuples
    return list(all_pairs)


def run_model(stim_ind, params):
    """
    Parameters
    -------------------------------------------------------
    stim_ind: index to send as arg to neuroGPU 
    params: DEPRECATED remove

    Returns
    ---------------------------------------------------------
    p_object: process object that stops when neuroGPU done
    """
    volts_fn = vs_fn + str(stim_ind) + "_" +  str(global_rank) + '.dat'  
    if os.path.exists(volts_fn):
        # print("removing ", volts_fn)#, " from ", global_rank)
        os.remove(volts_fn)
    # subprocess pipe hides neuroGPU outputs
    p_object = subprocess.Popen(['../bin/neuroGPU',str(stim_ind), str(global_rank)], stdout=subprocess.PIPE)
    return p_object





def map_par(run_num, data_volts_list):
    ''' 
    This function maps out what stim and score function pairs should be mapped to be evaluated in parallel
    first it finds the pairs with the highest weights, the maps them and then adds up the score for each stim
    for every individual.

    Parameters
    -------------------- 
    run_num: the amount of times neuroGPU has ran for 8 stims

    Return
    --------------------
    2d list of scalar scores for each parameter set w/ shape (nindv,nstims)
    '''
    fxnsNStims = top_SFs(run_num,n_sfs) # 52 stim-sf combinations (stim#,sf#)
    args = []
    for fxnNStim in fxnsNStims:
        i = fxnNStim[0]
        j = fxnNStim[1]

        argDict = {   "i": i,
            "j": j,
            "curr_data_volt" : data_volts_list[i % nGpus,:,:].astype(np.float32), # can be really big, like 1000,10000
            "curr_target_volt": target_volts_list[i].astype(np.float32), # 10k
            "curr_sf": sf_list[j],
            "weight": 1,
            "transformation": None,
            "dt": .02, # with bbp stims, dt is always .02
            "start": time.time(),
        }
        args.append(argDict)

    start = time.time()
    exs = []
    counter = 0
    # call score_functions.py parallel map
    res = sf.callPara(args)


    res = np.array(list(res)) ########## important: map returns results with shape (# of sf stim pairs, nindv)
    end = time.time()
    print("all evals took :" , end - start)
    res = res[:,:] 
    prev_sf_idx = 0 
    # look at key of each stim score pair to see how many stims to sum
    #num_selected_stims = len(set([pair[0] for pair in fxnsNStims])) # not always using 8 stims
    last_stim = (run_num + 1) * nGpus # ie: 0th run last_stim = (0+1)*8 = 8
    first_stim = last_stim - nGpus # on the the last round this will be 24 - 8 = 16
    if last_stim > 18:
        last_stim = 18
    for i in range(first_stim, last_stim):  # iterate stims and sum
        num_sfs = sum([1 for pair in fxnsNStims if pair[0]==i]) #find how many sf indices for this stim
        #print([pair for pair in fxnsNStims if pair[0]==i], "pairs from : ", run_num)
        #print(fxnsNStims[prev_sf_idx:prev_sf_idx+num_sfs], "Currently evaluating")

        if i % nGpus == 0:
            weighted_sums = np.reshape(np.sum(res[prev_sf_idx:prev_sf_idx+num_sfs, :], axis=0),(-1,1))
        else:
            #print(prev_sf_idx, "stim start idx", num_sfs, "stim end idx")
            curr_stim_sum = np.sum(res[prev_sf_idx:prev_sf_idx+num_sfs, :], axis=0)
            curr_stim_sum = np.reshape(curr_stim_sum, (-1,1))
            weighted_sums = np.append(weighted_sums, curr_stim_sum , axis = 1)
            #print(curr_stim_sum.shape," : cur stim sum SHAPE      ", weighted_sums.shape, ": weighted sums shape")
        prev_sf_idx = prev_sf_idx + num_sfs # update score function tracking index
    return weighted_sums




def getVolts(idx):
    '''Helper function that gets volts from data and shapes them for a given stim index'''
    fn = vs_fn + str(idx) + "_" +  str(global_rank) + '.dat'    #'.h5' 
    io_start = time.time()
    curr_volts =  nrnMread(fn)
    io_end = time.time()
    #fn = vs_fn + str(idx) +  '.dat'    #'.h5'
    #curr_volts =  nrnMread(fn)
    Nt = int(len(curr_volts)/ntimestep)
    shaped_volts = np.reshape(curr_volts, [Nt,ntimestep])
    return shaped_volts




if __name__ == "__main__":
        '''This function overrides the BPOP built in function. It is currently set up to run GPU tasks for each 
        stim in chunks based on number of GPU resources then stacks these results and sends them off to be
        evaluated. It runs concurrently so that while nGpus are busy, results ready for evaluation are evaluated.
        Parameters
        -------------------- 
        param_values: Population sized list of parameter sets to be ran through neruoGPU then scored and evaluated
        
        Return
        --------------------
        2d list of scalar scores for each parameter set w/ shape (nindv,1)
        '''
        dts = []
        nstims = 8
        n_sfs = 10
        trials = args.trials

        data_volts_list = np.array([])

        if total_rank == 0:
            ##### TODO: write a function to check for missing data?
            dts = np.repeat(.02, nstims)
            # insert negative param value back in to each set
            param_values = np.genfromtxt('../Data/offspring.csv', delimiter=',')[:args.offspring_size]
            param_values[:,:] = 0#param_values[:,1]
        else:
            param_values = None
            dts = None
            
        dts = comm.bcast(dts, root=0)
        param_values = np.array(comm.bcast(param_values, root=0))
        
        param_values, total_indvs = divide_params(param_values, global_size, global_rank) # TODO: is this the right size? should be # of nodes       
        # print(param_values.shape, "PARAM VALUES SHAPE AFTER GLOBAL DIVIDE")
        allparams = allparams_from_mapping(list(param_values))
        _, my_indvs = divide_params(param_values, nGpus, local_rank) 
        # print(my_indvs.shape, f"my indvs rank {total_rank}")
                
        if total_rank != 0:
            target_volts_list = None 
        else:
            target_volts_list = np.genfromtxt('../Data/exp_data.csv')
        target_volts_list = comm.bcast(target_volts_list, root=0)
        
        
        for trial in range(trials):
            p_objects = []
            score = []
            start_times = [] # a bunch of timers
            end_times = []
            eval_times = []
            all_volts = []
            all_params = []

       
            start_time_sim = time.time()

            #start running neuroGPU
            for i in range(0, nstims):
                start_times.append(time.time())
                if local_rank == 0:
                    p_objects.append(run_model(i, []))


             # evlauate sets of volts and     
            for i in range(0,nstims):
                if local_rank == 0:
                    p_objects[i].wait() #wait to get volts output from previous run then read and stack
                comm.barrier()
                end_times.append(time.time())
                shaped_volts = getVolts(i)


                if i == 0:
                    data_volts_list = shaped_volts #start stacking volts
                else:
                    data_volts_list = np.append(data_volts_list, shaped_volts, axis = 0) 

                if i == nstims - 1:
                    data_volts_list = np.reshape(data_volts_list, (nstims,len(total_indvs),ntimestep))[:,my_indvs,:] # ok
                    eval_start = time.time()
                    targV = target_volts_list[:nstims] # shifting targV and current dts
                    curr_dts = dts[:nstims] #  so that parallel evaluator can see just the relevant parts
                    score = map_par(0, data_volts_list) # call to parallel eval
                    eval_end = time.time()
                    eval_times.append(eval_end - eval_start)


            # PRINT RESULTS
            print("offspring (problem) size:",  args.offspring_size)
            print("average simulation (neuroGPU) runtime: ", np.mean(np.array(end_times) - np.array(start_times)))   
            print("simulation (neuroGPU) runtimes: " + str(np.array(end_times) - np.array(start_times)))
            print(f"{total_rank} , evaluation took: ", eval_times)
            print("everything took: ", eval_end - start_time_sim)
            
            score = np.reshape(np.sum(score,axis=1), (-1,1))
            score = comm.gather(score, root=0)
            score = comm.bcast(score, root=0)
            final_score = np.concatenate(score)
            final_score = np.array([item for sublist in final_score for item in sublist]).reshape(-1,1)


            # Minimum element indices in list 
            # Using list comprehension + min() + enumerate() 
            temp = min(final_score) 

            res = [i for i, j in enumerate(final_score) if j == temp] 

            # LOG RESULTS
            if total_rank ==0:
                logger.info("average simulation (neuroGPU) runtime: " +  str(np.mean(np.array(end_times) - np.array(start_times)))) 
            ####### ONLY USING GPU runtimes that don't intersect with eval
                logger.info("simulation (neuroGPU) runtimes: " + str(np.array(end_times) - np.array(start_times)))
                logger.info("evaluation took: " +  str(eval_times))
                logger.info("everything took: " + str(eval_end - start_time_sim))
                logger.info("gen size : " + str(args.offspring_size))
                logger.info("evaluation times: " + str(eval_times))
                logger.info("simulation times: " + str((np.array(end_times) - np.array(start_times))[:nGpus]))
                logger.info("simulation ends: " + str((np.array(end_times))))
                logger.info("simulation starts: " + str((np.array(start_times))))
                logger.info("whole gen took: " + str(time.time() - start_time_sim))
                logger.info("lap time: " + str(time.time()))
                logger.info(f'completed trial : {trial}')

            comm.barrier()

    

    
