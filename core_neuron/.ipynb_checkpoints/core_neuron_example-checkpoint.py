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
import argparse
# os.chdir('neuron_files/allen')
from neuron import h
from neuron import coreneuron
# os.chdir('../../')

from neuron.units import ms, mV

parser = argparse.ArgumentParser(description='control size of simulation - evaluation loop')
parser.add_argument('--offspring_size', type=int, required=False, default=100, help='size of simulation')
parser.add_argument('--trials', type=int, required=False, default=1, help='number of trials to run')

args, unknown = parser.parse_known_args()

# CoreNeuron Settings
coreneuron.gpu = True
coreneuron.enable = True
h.load_file("hoc_files/runModel.hoc")
h.cvode.cache_efficient(1)
h.cvode.use_fast_imem(1)
h.nrnmpi_init()
pc = h.ParallelContext()
h.cvode.cache_efficient(True)
h.dt = 0.02
# Number of timesteps for the output volt.
ntimesteps = 5000
tstop = ntimesteps*h.dt
pc.set_maxstep(10 * ms)

# set up MPI
comm = MPI.COMM_WORLD
global_rank = comm.Get_rank()
size = comm.Get_size()


# score function list
with open("../all_data/score_functions.pkl", "rb") as f:
    sf_list = pickle.load(f)['score_functions']
    
nGpus = min(8,len([devicenum for devicenum in os.environ['CUDA_VISIBLE_DEVICES'] if devicenum != ","]))

#path to group of parameters
param_path = os.path.join('../','all_data','offspring.csv')
orig_params = [1.418523367956126016e-05,4.125053776113334436e+01,1.139939936292646117e+01,7.221142819881146480e+00,3.484728506902474987e-02,1.938847092480697754e+01,4.625438639315531475e+01,1.053356163083602887e-01,2.286367813744082187e-03,6.235571302173457953e-02,3.756876749423531184e-03,1.935534122651551669e+01,1.323491215209898503e-03,5.397733400422616117e-04]
param_list = orig_params


if global_rank == 0:
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


# CORENEURON CELL

class Cell:
    def __init__(self, gid, pc, run):
        start = time.time()
        self.hoc_cell = h.cADpyr232_L5_TTPC1_0fb1ca4724()
        end = time.time()
        self.gid = gid
        self.pc = pc
        if run == 0:
            self.pc.set_gid2node(gid, pc.id())
        curr_p = param_list  # TODO ----> #[gid]*gid
        self.update_params(curr_p)
        self.ic = h.IClamp(self.soma[0](.5))
        self.ic.amp = 0.5
        self.ic.dur = 1e9
        v = h.Vector().from_python(np.repeat(2,10000))
        v.play(self.ic, self.ic._ref_amp, True)
        self.rd = {k: h.Vector().record(v, sec=self.soma[0]) for k,v in zip(['t', 'v', 'stim_i', 'amp'],
                                                    [h._ref_t, self.soma[0](.5)._ref_v, self.ic._ref_i, self.ic._ref_amp])}
        
    def __getattr__(self, name):
        return getattr(self.hoc_cell, name)
    
    def update_params(self,p):
        #all
        for curr_sec in self.hoc_cell.all:
            curr_sec.g_pas = p[0]
            curr_sec.e_pas = p[1]
        # axonal
        for curr_sec in self.hoc_cell.axonal:
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
        # basal
        for curr_sec in self.hoc_cell.basal:
            curr_sec.gIhbar_Ih = p[13]
        # apical
        for curr_sec in self.hoc_cell.apical:
            curr_sec.gIhbar_Ih = p[13]

            
def nrnMread(fileName):
    f = open(fileName, "rb")
    nparam = struct.unpack('i', f.read(4))[0]
    typeFlg = struct.unpack('i', f.read(4))[0]
    return np.fromfile(f,np.double)


def run_model(stim_ind=7): # pass stim as arg and set it with h?
    h.stimFile = f'../Data/{stim_ind}'
    h.finitialize(-65)
    h.tstop = tstop
    pc.psolve(tstop)
    
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





def map_par(run_num, target_volts_list):
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
            "curr_data_volt" : np.array(data_volts_list)[i % nGpus,:,:].astype(np.float32), # can be really big, like 1000,10000
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
    # print(multiprocessing.cpu_count())
    # print(1/0)
    # pool = multiprocessing.Pool(multiprocessing.cpu_count())
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
        num_gen = 0
        run_num = 0
        trials = args.trials

        data_volts_list = np.array([])
        # START TIMER
        start_time_sim = time.time()
 
        

        if global_rank == 0:
            param_values = np.genfromtxt(param_path, delimiter=',')[:args.offspring_size] # choose pop size
            target_volts_list = np.genfromtxt('../Data/exp_data.csv')

        else:
            param_values = None
            target_volts_list = None 

        ## with MPI we can have different populations so here we sync them up ##
        #full_params = comm.bcast(full_params, root=0)
        if global_rank != 0:
            param_values = None 
        
        param_values = comm.bcast(param_values, root=0)
        target_volts_list = comm.bcast(target_volts_list, root=0)
        param_values = np.array(param_values)
        
        
        score = []
        start_times = [] # a bunch of timers
        end_times = []
        eval_times = []
        all_volts = []
        all_params = []

        pc = h.ParallelContext()
        pc.set_maxstep(10 * ms)

        '''
        STEP 1 : if we can -- split up the population across nodes
        '''
        if size > 1: 
            myChunk = ((len(param_values) // size) * global_rank , (len(param_values) // size) * (global_rank + 1))
            if global_rank == size -1:
                myChunk = (myChunk[0],len(param_values))
            param_values = param_values[myChunk[0]:myChunk[1],:]
            old_pval_len = len(param_values) 
            ncell = len(param_values) * nstims
            # multiply by nstims to stretch out my chunk for all the stims we want to run
            gids = range(myChunk[0]  * nstims , myChunk[1]  * nstims)
            print(f"{gids} are from from rank {global_rank}, with pval len {len(param_values)} and chunk {myChunk}")
        else:
            old_pval_len = len(param_values) 
            ncell = len(param_values) * nstims
            gids = range(pc.id(), ncell,pc.nhost()) # round robin
        
        
        param_values = np.repeat(param_values, nstims)
        curr_pvals = np.array(param_values)#[gids]
        curr_pvals = np.tile(orig_params, ncell).reshape(ncell,-1)
        if num_gen == 0:
            '''
            STEP 2 : instantiate L5 TTPC Pyr. Cells
            '''
            cells = [Cell(gid, pc, num_gen) for gid in gids]
            # refresh start time
            
            
        
        '''
        STEP 3 : update cell parameters
        '''
        [cell.update_params(p) for p, cell in zip(curr_pvals, cells)]
    
        '''
        STEP 4 : run simulation-evaluation and time it
        '''
        for trial in range(trials):
            start_time_sim = time.time()
            start = time.time()
            run_model(pc)
            end = time.time()

            print("SIM TOOK: ", end - start)
            start_times.append(start)
            end_times.append(end)

            '''
            STEP 5 : exctract volts
            - note minor bug here for last rank node and 
            '''
            allvs = []
            allvs = [None]*len(gids)
            for gid in range(len(gids)):
                if len(cells) - 1 < gid: # MONKEY PATCH TO FIX LAST RANK HAVING 52 gids and 40 cells --> why is self.cells too small on last rank sometimes?
                    allvs[gid] = allvs[1]
                    continue
                curr_cell = cells[gid]
                curr_vs = curr_cell.rd['v'].to_python() # grab volatage from cell
                allvs[gid] = curr_vs
            '''
            STEP 5 : exctract volts
            '''

            curr_dvolts = np.array(allvs).reshape(nstims, old_pval_len, 5001)
            curr_dvolts = curr_dvolts[:,:5000]

            data_volts_list  = curr_dvolts

            data_volts_list = np.array(data_volts_list)

            data_volts_list = data_volts_list[:,:,:5000]
            eval_start = time.time()
            '''
            STEP 6 : score volts
            '''
            score = map_par(run_num, target_volts_list) # call to parallel eval
            eval_end = time.time()
            eval_times = eval_end - eval_start
            score = np.sum(score, axis=1)

            '''
            STEP 7 : reduce all volts to rank 1 ... ignore all further steps
            '''
            final_score = None
            if size > 1:
                final_score = np.concatenate(pc.py_allgather(score))
            else:
                final_score = score
            print(final_score.shape, "FINAL SCORE SHAPE")
            final_score = final_score.reshape(-1,1)


            # LOG RESULTS
            if global_rank == 0:
                logger.info("average simulation (coreNeuron) runtime: " +  str(np.mean(np.array(end_times) - np.array(start_times))))   
            ####### ONLY USING GPU runtimes that don't intersect with eval
                logger.info("simulation (coreNeuron) runtimes: " + str(np.array(end_times) - np.array(start_times)))
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

        
        pc.done()
        exit() # for some reason we need to exit here to end the code, something to do with launching python 
        #  using neuron