import numpy as np
import math
import efel
#import matplotlib.pyplot as plt
import time as timer
import os
import copy
import logging
import logging.config
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})
import pickle 
import utils
import re
import h5py
# These are here for efficiency. In order to avoid redundant computation, we cache the results for
# comp_width_helper, comp_height_helper and traj_score_helper. The name of stim and index as a string
# need to be passed in to do this.
import sys
#from concurrent.futures import ProcessPoolExecutor as Pool
# import tracemalloc
# tracemalloc.start()
# ... start your applic
# from joblib import Parallel, delayed
# import pprint
from mpi4py import MPI
import cProfile
import copy
from scipy.signal import argrelextrema
from ipfx.feature_extractor import SpikeTrainFeatureExtractor
from ipfx.feature_extractor import SpikeFeatureExtractor

nGpus = min(8,len([devicenum for devicenum in os.environ['CUDA_VISIBLE_DEVICES'] if devicenum != ","]))
comm = MPI.COMM_WORLD
size = comm.Get_size()
total_rank = comm.Get_rank()
global_rank = comm.Get_rank() // nGpus
local_rank = comm.Get_rank() % nGpus
total_size = comm.Get_size()
global_size = size // nGpus

#from concurrent.futures import ThreadPoolExecutor as Pool#
#multiprocessing.set_start_method('fork')

# os.environ["OPENBLAS_NUM_THREADS"] = "1" 
# os.environ["MKL_NUM_THREADS"] = "1" 
# os.environ["NUMEXPR_NUM_THREADS"] ="1"


from multiprocessing import Pool

#multiprocessing.set_start_method('fork')

#from torch.multiprocessing import Pool

#from ray.util.multiprocessing import Pool

# hack to remove eventually...imagine doing a getattr when you have the function
# and can just do an eval :O
thismodule = sys.modules[__name__]

comp_width_dict = {}
comp_height_dict = {}
traj_score_dict = {}
threshold = -10

#set environ


#if size == 1:
#cpu_str = os.environ['SLURM_JOB_CPUS_PER_NODE']
#SLURM_CPUS = int(re.search(r'\d+', cpu_str).group() )
# else:
#     SLURM_CPUS = int(os.environ['SLURM_JOB_CPUS_PER_NODE'][:1])\
pid=0
affinity = os.sched_getaffinity(pid)
nCpus = len(affinity)
# These constants exist for efel features
time_stamps =  5000
starting_time_stamp = 1000
ending_time_stamp = 7000

custom_score_functions = [
                    'chi_square_normal',\
                    'traj_score_1',\
                    'traj_score_2',\
                    'traj_score_3',\
                    'isi',\
                    'rev_dot_product',\
                    'KL_divergence']



########################################################################
# These functions are util functions.
def zero_pad(list1, list2):
    if isinstance(list1, int) or isinstance(list2, int) or isinstance(list1, float) or isinstance(list2, float):
        return list1, list2
    if len(list1) > len(list2):
        list2 = list2 + [0 for i in range(0, len(list1) - len(list2))]
    elif len(list2) > len(list1):
        list1 = list1 + [0 for i in range(0, len(list2) - len(list1))]
    return list1, list2

def safe_mean(lis):
    if np.size(lis) == 0:
        return 0
    return np.mean(lis)

def find_positive_inds(lis):
    return [i for i in range(0, len(lis)) if lis[i] > 0]

def find_positive_inds(lis):
    return [i for i in range(0, len(lis)) if lis[i] > 0]

def find_peaks(lis, min_value):
    inds = [i for i in range(1, len(lis) - 1) if lis[i] >= lis[i - 1] and lis[i] >= lis[i + 1] and lis[i] >= min_value]
    return [lis[i] for i in inds], inds

# def find_peaks(lis, min_value):
#     lis = np.array(lis)
#     inds = np.argwhere(lis[argrelextrema(lis, np.greater)] >= min_value)
#     vals = lis[inds]
#     return vals, inds

def diff(lis):
    return [lis[i + 1] - lis[i] for i in range(0, len(lis) - 1)]

########################################################################

# chi_square_normal takes one dimensional lists target and data and return mean of (x_1 - x_2)^2 for
# each timepoints.
def chi_square_normal(target, data, dt=0.02, stims=None, index=None):
    diff = np.square(np.array(target) - np.array(data))
    score = safe_mean(diff)
    return score

# abs_cumsum_diff takes one dimensional lists target and data and return abs(x_1 - x_2) for
# each timepoints in the cumsum.
def abs_cumsum_diff(target, data, dt=0.02, stims=None, index=None):
    cumsum_chi = np.absolute(np.cumsum(np.array(target)) - np.cumsum(np.array(data)))
    score = safe_mean(cumsum_chi)
    return score

# comp_rest_potential takes one dimensional lists target and data and compares the resting
# potential by squaring the difference of last elements in lists.
def comp_rest_potential(target, data, dt=0.02, stims=None, index=None):
    v_rest_ind = len(target) - 1
    v_rest_target = target[v_rest_ind]
    v_rest_data = data[v_rest_ind]

    return (v_rest_target - v_rest_data) ** 2

# comp_width takes one dimensional lists target and data and compares the widths of action potentials
# comp_width_avg does very similar thing but takes average. The two functions share very similar code
# but it was just copy pasted because I was lazy.
def comp_width_helper(target, data):
    def AP_inds(positive_ind_vec):
        ind_when_change = [i for i in range(0, len(positive_ind_vec) - 1) if positive_ind_vec[i + 1] - positive_ind_vec[i] > 1]
        AP_ends = [positive_ind_vec[i] for i in ind_when_change] + [positive_ind_vec[len(positive_ind_vec) - 1]]
        AP_starts = [positive_ind_vec[0]] + [positive_ind_vec[i + 1] for i in ind_when_change]
        return AP_starts, AP_ends
    postive_target_inds = find_positive_inds(target)
    postive_data_inds = find_positive_inds(data)

    if postive_target_inds:
        target_AP_start, target_AP_end = AP_inds(postive_target_inds)
        curr_target_widths = [target_AP_end[i] - target_AP_start[i] for i in range(0, min(len(target_AP_end), len(target_AP_start)))]
        target_width_avg = safe_mean(curr_target_widths)
    else:
        curr_target_widths = []
        target_width_avg = 0

    if postive_data_inds:
        data_AP_start, data_AP_end = AP_inds(postive_data_inds)
        curr_data_widths = [data_AP_end[i] - data_AP_start[i] for i in range(0, min(len(data_AP_start), len(data_AP_end)))]
        data_width_avg = safe_mean(curr_data_widths)
    else:
        curr_data_widths = []
        data_width_avg = 0


    if postive_target_inds or postive_data_inds:
        curr_target_widths, curr_data_widths = zero_pad(curr_target_widths, curr_data_widths)
        target_width_avg, data_width_avg = zero_pad(target_width_avg, data_width_avg)
    else:
        curr_target_widths = []
        curr_data_widths = []
        target_width_avg = 0
        data_width_avg = 0

    result = [sum([(curr_target_widths[i] - curr_data_widths[i])**2 for i in range(0, len(curr_data_widths))]), (target_width_avg - data_width_avg)**2]
    return [result[0], result[1]]

def comp_width(target, data, dt=0.02, stims=None, index=None):
    global comp_width_dict
    if stims and index:
        stim_ind = stims + index
        if stim_ind in comp_width_dict:
            return comp_width_dict[stim_ind][0]
        else:
            comp_width_dict[stim_ind] = comp_width_helper(target, data)
        return comp_width_dict[stim_ind][0]
    else:
        return comp_width_helper(target, data)[0]

def comp_width_avg(target, data, dt=0.02, stims=None, index=None):
    global comp_width_dict
    if stims and index:
        stim_ind = stims + index
        if stim_ind in comp_width_dict:
            return comp_width_dict[stim_ind][1]
        else:
            comp_width_dict[stim_ind] = comp_width_helper(target, data)
        return comp_width_dict[stim_ind][1]
    else:
        return comp_width_helper(target, data)[1]


# comp_height takes one dimensional lists target and data and compares the heights of action potentials
# comp_height_avg does very similar thing but takes average.
def comp_height_helper(target, data):
    if find_positive_inds(target):
        orig_target_peaks, target_peaks_locs = find_peaks(target, 0.1)
    else:
        orig_target_peaks = []
        target_peaks_locs = []
    if find_positive_inds(data):
        orig_data_peaks, data_peaks_locs = find_peaks(data, 0.1)
    else:
        orig_data_peaks = []
        data_peaks_locs = []

    peaks_target, peaks_data = zero_pad(orig_target_peaks, orig_data_peaks)

    #These always follows same scheme where the stims are complete.
    #At the end, it is really close to the rest membrane potential

    v_rest_ind = len(target) - 1
    v_rest_target = target[v_rest_ind]
    v_rest_data = data[v_rest_ind]
    height_target = [peaks_target[i] - v_rest_target for i in range(0, len(peaks_target))]
    height_data = [peaks_data[i] - v_rest_data for i in range(0, len(peaks_data))]
    target_height_avg = 0
    if len(height_target) > 0:
        target_height_avg = safe_mean(np.array(height_target))
    data_height_avg = 0
    if len(height_data) > 0:
        data_height_avg = safe_mean(np.array(height_data))

    result = [sum([(height_target[i] - height_data[i]) ** 2 for i in range(0, len(height_target))]), (target_height_avg - data_height_avg) ** 2]

    return [result[0], result[1]]

def comp_height(target, data, dt=0.02, stims=None, index=None):
    global comp_height_dict
    if stims and index:
        stim_ind = stims + index
        if stim_ind in comp_height_dict:
            return comp_height_dict[stim_ind][0]
        else:
            comp_height_dict[stim_ind] = comp_height_helper(target, data)
        return comp_height_dict[stim_ind][0]
    else:
        return comp_height_helper(target, data)[0]

def comp_height_avg(target, data, dt=0.02, stims=None, index=None):
    global comp_height_dict
    if stims and index:
        stim_ind = stims + index
        if stim_ind in comp_height_dict:
            return comp_height_dict[stim_ind][1]
        else:
            comp_height_dict[stim_ind] = comp_height_helper(target, data)
        return comp_height_dict[stim_ind][1]
    else:
        return comp_height_helper(target, data)[1]

# traj_score takes one dimensional lists target and data and counts the traces of sweep per grid in phase domain.
def traj_score_helper(target, data):
    n_bins = 100
    min_v = -80
    max_v = 70
    min_dv = -6
    max_dv = 40

    v_edges = np.linspace(min_v,max_v,n_bins)
    dv_edges = np.linspace(min_dv,max_dv,n_bins)
    edges = [v_edges, dv_edges]

    target_dvs = diff(target)
    target_hists = np.histogram2d(target[1:], target_dvs, bins = edges)[0]
    #Inserting 2d list in a list so target_hists bocomes 3d list but for now it's 2d list
    c_target_hists = np.cumsum(np.cumsum(target_hists, axis = 0), axis = 1)
    c_target_hists_max = max([max(c_target_hists[j]) for j in range(0, len(c_target_hists))])
    c_target_hists = [[float(c_target_hists[j][k])/c_target_hists_max for k in range(0, len(c_target_hists[j]))] for j in range(0, len(c_target_hists))]
    target_hists_sum = sum([sum(target_hists[j]) for j in range(0, len(target_hists))])
    target_hists = [[float(target_hists[j][k])/target_hists_sum for k in range(0, len(target_hists[j]))] for j in range(0, len(target_hists))]
    #Here, I am assuming that these are 2d arrays

    curr_dvs = diff(data)
    curr_hists =  np.histogram2d(data[1:], curr_dvs, bins = edges)[0]
    curr_hists_sum = sum([sum(curr_hists[k]) for k in range(0, len(curr_hists))])
    curr_hists = [[float(curr_hists[k][l])/curr_hists_sum for l in range(0, len(curr_hists[k]))] for k in range(0, len(curr_hists))]
    t_score = safe_mean([safe_mean([(target_hists[k][l] - curr_hists[k][l])**2 for l in range(0, len(curr_hists[k]))]) for k in range(0, len(curr_hists))])
    c_curr_hists =  np.cumsum(np.cumsum(curr_hists, axis = 0), axis = 1)
    c_curr_hists_max = max([max(c_curr_hists[k]) for k in range(0, len(c_curr_hists))])
    c_curr_hists = [[float(c_curr_hists[k][l])/c_curr_hists_max for l in range(0, len(c_curr_hists[k]))] for k in range(0, len(c_curr_hists))]
    c_diff = [[abs(c_curr_hists[k][l] - c_target_hists[k][l]) for l in range(0, len(c_curr_hists[k]))] for k in range(0, len(c_curr_hists))]
    fast_ks_score = max([max(lis) for lis in c_diff])
    fast_emd_score = safe_mean([safe_mean(lis) for lis in c_diff])


    scores = [(t_score**0.5), fast_ks_score, fast_emd_score]
    return scores

def traj_score_1(target, data, dt=0.02, stims=None, index=None):
    global traj_score_dict
    if stims and index:
        stim_ind = stims + index
        if stim_ind in traj_score_dict:
            return traj_score_dict[stim_ind][0]
        else:
            traj_score_dict[stim_ind] = traj_score_helper(target, data)
        return traj_score_dict[stim_ind][0]
    else:
        return traj_score_helper(target, data)[0]

def traj_score_2(target, data, dt=0.02, stims=None, index=None):
    global traj_score_dict
    if stims and index:
        stim_ind = stims + index
        if stim_ind in traj_score_dict:
            return traj_score_dict[stim_ind][1]
        else:
            traj_score_dict[stim_ind] = traj_score_helper(target, data)
        return traj_score_dict[stim_ind][1]
    else:
        return traj_score_helper(target, data)[1]

def traj_score_3(target, data, dt=0.02, stims=None, index=None):
    global traj_score_dict
    if stims and index:
        stim_ind = stims + index
        if stim_ind in traj_score_dict:
            return traj_score_dict[stim_ind][2]
        else:
            traj_score_dict[stim_ind] = traj_score_helper(target, data)
        return traj_score_dict[stim_ind][2]
    else:
        return traj_score_helper(target, data)[2]

# isi takes one dimensional lists target and data and computes rms of inter spike intervals.
def isi(target, data, dt=0.02, stims=None, index=None):
    times = np.cumsum([dt for i in range(len(target))])

    def get_isi(target, times):
        peak, inds = find_peaks(target, 0)
        isis =  [times[j] for j in inds]
        return isis, inds

    def compare_isi(target, input):
        curr_target, curr_input = target, input
        curr_target, curr_input = zero_pad(curr_target, curr_input)
        curr_score = safe_mean([(curr_target[j] - curr_input[j])**2 for j in range(0, len(curr_target))])**0.5
        norm_factor = safe_mean([elem**2 for elem in diff(curr_target)])**0.5
        if not norm_factor or norm_factor == 0:
            norm_factor = 1
        return float(safe_mean(curr_score))/norm_factor

    target_isi, target_isi_inds = get_isi(target, times)
    curr_isi, curr_isi_inds = get_isi(data, times)
    scores =  compare_isi(target_isi, curr_isi)

    if np.isnan(scores):
        scores = 0

    return scores

# Computing the 1 - dot product of 2 vectors.
def rev_dot_product (target, data, dt=0.02, stims=None, index=None):
    def vectorSize(v):
        return np.sqrt(np.sum(np.square(np.array(v))))

    stimMag = vectorSize(target)
    pSetMag = vectorSize(data)
    dot = np.dot(np.array(target), np.array(data))
    score = 1 - (float(dot) / (stimMag * pSetMag))

    return score

# # work if target & data are discrete probability distribution
def KL_divergence (target, data, dt=0.02, stims=None, index=None):
    def shift(v):
        base = np.absolute(np.amin(np.array(v)))
        return np.array(v) + base + 0.1
    def normalize (v):
        return np.array(v)/float(np.sum(np.array(v)))
    def divergence (v1, v2):
        v1_array = np.array(v1)
        v2_array = np.array(v2)
        d = 0
        v1_masked = v1_array[v1_array != 0]
        v2_masked = v2_array[v1_array != 0]
        d = np.sum(np.multiply(v2_masked, np.log(np.divide(v2_masked, v1_masked)))) 
        return d
    score = divergence(normalize(shift(target)), normalize(shift(data)))
    
    return score

# # work if target & data are discrete probability distribution
# def KL_divergence (target, data, dt=0.02, stims=None, index=None): 
#     def shift(v):
#         base = np.absolute(np.amin(np.array(v)))
#         return np.array(v) + base + 0.1

#     def normalize (v):
#         return np.array(v)/float(np.sum(np.array(v)))

#     def divergence (v1, v2):
#         v1_array = np.array(v1)
#         v2_array = np.array(v2)
#         d = 0
#         for i in range(0, len(v1_array)):
#             if (v1_array[i] != 0 and v2_array[i] != 0):
#                 d += float(v2_array[i]) * np.log(float(v2_array[i]) / v1_array[i])
#         return d

#     score = divergence(normalize(shift(target)), normalize(shift(data)))

#     return score


def traj_score_single_peak(target, data, dt=0.02, stims = None, index = None):
    X_size = 100
    Y_size = 100
    M_size = X_size*Y_size

    # initial half width time = 15ms
    half_width_time = 25

    def NaN_pad(list1, list2):
        if len(list1) > len(list2):
            list2 = list2 + [math.nan for i in range(0, len(list1) - len(list2))]
        elif len(list2) > len(list1):
            list1 = list1 + [math.nan for i in range(0, len(list2) - len(list1))]
        return list1, list2

    def pairs_peak_half_width(peak_indices, max_half_width_time, dt):
        pair_index_width = []
        max_half_width = max_half_width_time/dt #25ms/0.1ms = 250
        for i in range(len(peak_indices)):
            # the number of data points of half_width = 15/0.1 = 150
            half_width = max_half_width
            new_width = max_half_width

            if i == 0:
                new_width = (peak_indices[i+1]-peak_indices[i])/2
            elif i == len(peak_indices)-1 or math.isnan(peak_indices[i+1]):
                new_width = (peak_indices[i] - peak_indices[i-1])/2
            else:
                new_width = min((peak_indices[i+1]-peak_indices[i])/2,(peak_indices[i] - peak_indices[i-1])/2)

            if new_width < half_width:
                half_width = new_width

            pair_index_width.append((peak_indices[i], math.floor(half_width)))

        return pair_index_width

    def phase_plane(signal, pair_peak_half_width, dt):
        peak_index, half_width = pair_peak_half_width
        if (math.isnan(peak_index)):
            return [math.nan],[math.nan]
        lowBound = peak_index-half_width
        upBound = peak_index+half_width
        one_peak = signal[lowBound:upBound]

    #     x = [(one_peak[i]+one_peak[i+1])/2 for i in range(len(one_peak)-1)]
        x = one_peak[1:]
        dV = diff(one_peak)
        dT = [dt for i in range(len(dV))]
        y = np.array(dV)/np.array(dT).tolist()
        return x, y

    # find peaks above threshold, corresponding indices
    target_peak_volts, target_peak_indices = find_peaks(target, threshold)
    data_peak_volts, data_peak_indices = find_peaks(data, threshold)

    target_peak_indices, data_peak_indices = NaN_pad(target_peak_indices, data_peak_indices)

    nPeaks = len(target_peak_indices)

    #find half-width that covers only one peak at a time
    target_pairs = pairs_peak_half_width(target_peak_indices, half_width_time, dt)
    data_pairs = pairs_peak_half_width(data_peak_indices, half_width_time, dt)

    total_score = 0

    for i in range(nPeaks):
        target_pair = target_pairs[i]
        target_phase_plane = phase_plane(target, target_pair, dt)
        target_x = target_phase_plane[0]
        target_y = target_phase_plane[1]

        data_pair = data_pairs[i]
        data_phase_plane = phase_plane(data, data_pair, dt)
        data_x = data_phase_plane[0]
        data_y = data_phase_plane[1]


        x_bins = np.linspace(np.nanmin([min(target_x), min(data_x)]),np.nanmax([max(target_x),max(data_x)]), X_size+1)
        y_bins = np.linspace(np.nanmin([min(target_y),min(data_y)]),np.nanmax([max(target_y),max(data_y)]), Y_size+1)

        target_hist2d = np.histogram2d(target_x, target_y, bins=(x_bins, y_bins))[0]
        data_hist2d = np.histogram2d(data_x, data_y, bins=(x_bins, y_bins))[0]

        score = np.sum((target_hist2d-data_hist2d)**2)
        total_score += score
    return (total_score/M_size) ** 0.5

def DTWDistance(target, data, dt=0.02, stims = None, index = None):
   DTW = np.zeros((len(target)+1, len(data)+1))

   for i in range(len(target)+1):
       DTW[i][0] = np.inf

   for i in range(len(data)+1):
       DTW[0][i] = np.inf
   DTW[0][0]=0

   for i in range(1,len(target)+1):
       for j in range(1,len(data)+1):
           cost = abs(target[i-1]-data[j-1])
           DTW[i][j] = cost + min(DTW[i-1][j], DTW[i][j-1], DTW[i-1][j-1])
   return DTW[len(target)-1,len(data)-1]/len(target)

def testMAP(i):
    print("launched pid ", os.getpid())
    timer.sleep(.8)
        
def eval_function(target, data, function, dt):
    """
    function that sends target and simulated volts to scorefunctions.py so they
    can be evaluated, then adds Kyung AP penalty function at the end

    Parameters
    -------------------------------------------------------------
    target: target volt for this stim
    data: set of volts with shape (nindvs, ntimsteps)
    function: string containing score function to use
    dt: dt of the stim as it is a parameter for some sfs
    i: index of the stim

    Returns
    ------------------------------------------------------------
    score: scores for each individual in an array corresponding to this score function
    with shape (nindv, 1)
    """
    
    scorestart = timer.time()
   

    #logging.info("process {} is computing at {}".format(os.getpid(), scorestart))
    num_indvs = data.shape[0]
    if function in custom_score_functions:
        score = [getattr(thismodule, function)(target, data[indv,:], dt) for indv in range(num_indvs)]
    else:
        score = eval_efel(function, target, data, dt)

    return score

def normalize_scores(curr_scores, transformation):
    '''changed from hoc eval so that it returns normalized score for list of indvs, not just one
    TODO: not sure what transformation[6] does but I changed return statement to fit our 
    dimensions'''
    # transformation contains: [bottomFraction, numStds, newMean, std, newMax, addFactor, divideFactor]
    # indices for reference:   [      0       ,    1   ,    2   ,  3 ,    4  ,     5    ,      6      ]
    for i in range(len(curr_scores)):
        if curr_scores[i] > transformation[4]:
            curr_scores[i] = transformation[4]        # Cap newValue to newMax if it is too large
    normalized_single_score = (curr_scores + transformation[5])/transformation[6]  # Normalize the new score
    if transformation[6] == 0:
        return np.ones(len(curr_scores)) 
    return normalized_single_score




def eval_stim_sf_pair(args):
    """ 
    function that evaluates a stim and score function pair on line 252. Sets i as the actual 
    index and and mod_i as it's adjusted index (should get 15th target volt but that will 
    be 7th in the data_volts_list). transform then normalize and then multiply by weight
    and then SENT BACK to MAPPER. uses self. for weights, data_volts, and target volts because
    it is easy information transfer instead of passing arguments into map.

    Arguments
    --------------------------------------------------------------------
    perm: pair of ints where first is the stim and second is the score function label index
    to run

    Returns
    ---------------------------------------------------------------------
    scores: normalized+weighted scores with the shape (nindv, 1), and sends them back to map
    to be stacked then summed.

    """
    final_score = 0
    if type(args)== dict:
        args = [args]
    for arg in args:
        i = arg["i"]
        j = arg["j"]
        #time = arg['start']
        start = arg["start"]
        curr_sf = arg["curr_sf"]
        curr_data_volt =arg["curr_data_volt"]
        curr_target_volt =arg["curr_target_volt"]
        # if global_rank == 0:
        #     logging.info("process {} is {} and started at {}".format(os.getpid(), curr_sf, timer.time()))
#         f =h5py.File("../Data/tmp/{}.hdf5".format(global_rank),"r")
#         curr_data_volt =f["data_volt{}{}".format(i,j)][:]
#         curr_target_volt =f["target_volt{}{}".format(i,j)][:]
        #logging.info("IO:: " + str(io_end - io_start))
        curr_weight = arg["weight"]
        transformation = arg["transformation"]
        dt = arg["dt"]
        computation_time_start = timer.time()

        if curr_weight == 0:
            curr_scores = np.zeros(len(curr_data_volt))

        else:
            strt = timer.time()
            curr_scores = eval_function(curr_target_volt, curr_data_volt, curr_sf, dt)
            endd = timer.time()
#             print(endd - strt, ": proc took", os.getpid(),  " was  :", curr_sf, 'dvolt shape :', curr_data_volt.shape)

        norm_scores = np.array(curr_scores)#normalize_scores(curr_scores, transformation)
#         for k in range(len(norm_scores)):
#             if np.isnan(norm_scores[k]):
#                 norm_scores[k] = 1

        computation_time_end = timer.time()
        # if global_rank == 0:
        #     logging.info("process {} returning at {}".format(os.getpid(), timer.time()))
        final_score += norm_scores * curr_weight 

    return final_score

def wrap_para(arg):
#     cProfile.runctx('eval_stim_sf_pair(arg)', globals(), locals(), 'profiles/prof%d.prof' % os.getpid())
    datafn = 'profiles/prof%d.prof' % os.getpid()
    prof = cProfile.Profile()
    retval = prof.runcall(eval_stim_sf_pair, arg)
    prof.dump_stats(datafn)
    return retval

    
def callPara(args):
    # using either nCpus  or 20 
    if total_rank == 0:
        logging.info("************ launched {} PIDS at {} ************".format(len(args),timer.time()))
    
    # exit here to avoid bug
    #exit()
    with Pool(nCpus) as p:
        start = timer.time()
        res = p.map(eval_stim_sf_pair,args)

    end = timer.time()
    print(end-start)
    if total_rank == 0:
        logging.info("************ finished {} PIDS at {} ************".format(len(args),timer.time()))
    return res


def callParaIpfx(p,args):
    # using either nCpus  or 20 
    logging.info("************ launched PIDS at {} ************".format(timer.time()))
    
    # exit here to avoid bug
    #exit()
#     with Pool(35) as p:
    start = timer.time()
    real_args = []
    unique_stims = []
    for arg in args:
        for i in range(len(arg['curr_data_volt'])):
            curr_arg = copy.deepcopy(arg)
            curr_arg['curr_data_volt'] = curr_arg['curr_data_volt'][i,:]
            real_args.append(curr_arg)
            unique_stims.append(curr_arg['i'])
    num_unique_stims = len(np.unique(np.array(unique_stims)))
    res = p.map(eval_ipfx,real_args)
    res = np.array(res).reshape(-1, num_unique_stims)
    end = timer.time()
    print(end-start)



    logging.info("************ finished PIDS at {} ************".format(timer.time()))
    return np.array(res)

def clean_flatten_score(score):
    if len(list(score.keys())) == 1:
        return 100000
    else:
        total = 0
        for key in score.keys():
            if np.isnan(score[key]):
                continue
            total += score[key]
    return total
                

def eval_ipfx(arg):
    target = arg['curr_target_volt']
    data =  arg['curr_data_volt']
    curr_stim = arg['curr_stim']
    dt=0.02
    index=None
    time_stamps =  10000
    time = np.cumsum([dt for i in range(time_stamps)])
        
    time = time / 1000
    ext = SpikeFeatureExtractor(start=.00002, end=.2)
#     import pdb; pdb.set_trace()
#     simVolts = nrnMread("./VHotP9.dat")
    start = timer.time()
    try:
        spikes = ext.process(time, data, curr_stim)
        ext = SpikeTrainFeatureExtractor(start=.00002, end=.2)
        features = ext.process(time, data, curr_stim, spikes) # re-using spikes from above

    except Exception as e:
        features = {'adapt':0.000004}
        print(e)
    end = timer.time()
    return clean_flatten_score(features)
    
    
    

def eval_efel(feature_name, target, data, dt=0.02, stims=None, index=None):
    def diff_lists(lis1, lis2):
        if lis1 is None and lis2 is None:
            return 0
        if lis1 is None:     
            lis1 = [0]
        if lis2 is None:
            lis2 = [0]
        len1, len2 = len(lis1), len(lis2)
        if len1 > len2:
            lis2 = np.concatenate((lis2, np.zeros(len1 - len2)), axis=0)
        if len2 > len1:
            lis1 = np.concatenate((lis1, np.zeros(len2 - len1)), axis=0)
        return np.sqrt(safe_mean((lis1 - lis2)**2))
    all_features = []

    time = np.cumsum([dt for i in range(time_stamps)])
    curr_trace_target, curr_trace_data = {}, {}
    stim_start, stim_end = starting_time_stamp*dt, ending_time_stamp*dt
    curr_trace_target['T'] = time
    curr_trace_target['V'] = target
    curr_trace_target['stim_start'] = [stim_start]
    curr_trace_target['stim_end'] = [stim_end]
    traces = [curr_trace_target]
    #testing
    # print(len(data), "LEN DATA")
    for i in range(len(data)):
        curr_trace_data = {}
        curr_trace_data['T'] = time
        curr_trace_data['V'] = data[i,:]
        curr_trace_data['stim_start'] = [stim_start]
        curr_trace_data['stim_end'] = [stim_end]
        traces.append(curr_trace_data)
    efelstart = timer.time()
    #with Pool(2) as p2:

    traces_results = efel.getFeatureValues(traces, [feature_name], raise_warnings=False)
    # print("EFEL eval took: ", timer.time()-efelstart)
    diff_features = []
    for i in range(len(data)): #testing
        diff_features.append(diff_lists(traces_results[0][feature_name], traces_results[i+1][feature_name]))
    return diff_features

