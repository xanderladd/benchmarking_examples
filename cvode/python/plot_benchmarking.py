import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pickle
from matplotlib import cycler
import argparse
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter
from matplotlib.text import Text
from matplotlib.ticker import AutoLocator
import struct
import os
import re
import matplotlib.patches as mpatches
from scalebary import add_scalebar
from matplotlib.transforms import Bbox




data_dir ='../Data'
orig_volts_fn = data_dir + '/exp_dataold.csv' #ORIG volts
target_volts = np.genfromtxt(orig_volts_fn)
interested_stim = 7
gens_to_plot = [1,100,200,300,400]
gens_to_plot = [100,200,300,400]
gens_to_plot = [60,100,200,400]
gens_to_plot = [1,16,25]
ignore_list =  [ '.ipynb_checkpoints','archive']
colors = ['red', 'green','blue', 'orange', 'brown']

def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)


def nrnMread(fileName):
    f = open(fileName, "rb")
    nparam = struct.unpack('i', f.read(4))[0]
    typeFlg = struct.unpack('i', f.read(4))[0]
    return np.fromfile(f,np.double)

def set_custom_params_plt():

    colors = cycler('color',
                    ['#EE6666', '#3388BB', '#9988DD',
                    '#EECC55', '#88BB44', '#FFBBBB'])
    # plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
    #     axisbelow=True, grid=True)
    # plt.rc('grid', color='w', linestyle='solid')
    # plt.rc('patch', edgecolor='#E6E6E6')
    plt.rc('lines', linewidth=2)
    #plt.rcParams['font.family'] = 'serif'
    #plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams.update({'font.size': 20})

    
    
    
# def load_files(prefix=''):
#     hof_path = os.path.join(prefix, 'hof.pkl')
#     log_path = os.path.join(prefix, 'log.pkl')
#     with open(hof_path, 'rb') as f:
#         hof = pickle.load(f)

#     with open(log_path, 'rb') as f:
#         log = pickle.load(f)
    
#     return hof, log
def load_files(prefix=''):
    if prefix != '':
        if 'seed' in prefix:
            nodes, pop, seed = re.findall(r'\d+', prefix)
        else:
            nodes, pop = re.findall(r'\d+', prefix)
            seed = 1010
        hof_path = os.path.join(prefix,  f'indv{nodes}N_{pop}O.pkl')
        log_path = os.path.join(prefix,  f'log{nodes}N_{pop}O.pkl')
    else:
        hof_path = os.path.join(prefix, 'hof.pkl')
        log_path = os.path.join(prefix, 'log.pkl')
        
#     import pdb; pdb.set_trace()
    try:
        with open(hof_path, 'rb') as f:
            hof = pickle.load(f)
    except Exception as e:
        print(e)
        print(f'opening hof {hof_path} failed')
        hof = None
    try:
        with open(log_path, 'rb') as f:
            log = pickle.load(f)
    except Exception as e:
        print(e)
        print(f'opening log {log_path} failed')
        log = None
    
    return hof, log
    

def make_scores_plot(scores, fig, label='', color='red',style='solid', marker='*', stderr=[], label_best=False):
    argmin = np.argmin(scores)
    
    if label_best:
        plt.scatter(argmin,scores[argmin], marker="o", s=350, color=color, label=' Best Score')
    else:
        plt.scatter(argmin,scores[argmin], marker="o", s=350, color=color)
    plt.plot(scores, color=color, linestyle=style)#, marker=marker)
    if len(stderr):
        # plt.fill_between(np.arange(len(stderr)), scores - stderr, scores + stderr, color=color,alpha=.35)
        ci_lower = scores - .95*(stderr / np.sqrt(9))
        ci_upper = scores + .95*(stderr / np.sqrt(9)) 

        plt.fill_between(np.arange(len(stderr)), ci_lower, ci_upper, color=color,alpha=.35)
    plt.ylabel("Score")
    plt.xlabel("Generation")
    return fig
    

def save_scores_plot(fig, save_path='benchmark_plots'):
    L = plt.legend(loc=(.4,.75), fontsize=14)
    try:
        L.legendHandles[-1].set_color('black') 
    except:
        print('legend not found')
    save_path_full = os.path.join(save_path, 'scores.png')
    plt.savefig(save_path_full,bbox_inches='tight')
    plt.close(fig)

    
def scores_from_log(log):
    scores = []
    for i in range(len(log)):
        scores.append(log[i]['min'])
    return scores

def scientific(x, pos):
    # x:  tick value - ie. what you currently see in yticks
    # pos: a position - ie. the index of the tick (from 0 to 9 in this example)
    return '%.0f' % x

def score_plot_gauntlet():
    filepaths = []
    colors = plt.cm.jet(np.linspace(0,1,len(os.listdir('outputs'))))# Initialize holder for trajectories
    styles = ['dashed','solid','dashdot', 'dotted']
    markers = [".", ",", "o", "v","^","<",">"]

    dark_jet = cmap_map(lambda x: x*0.75, matplotlib.cm.jet)(np.linspace(0,1,len(os.listdir('outputs'))))
    colors = ['blue', 'green','red', 'orange', 'brown']
    qual_map = plt.cm.Dark2(np.linspace(0,1,len(os.listdir('outputs'))))
    fig = plt.figure()
    pop_sizes = [1000,5000,10000]
    seen = []       
    color_count = 0
    label_best = False # set to true to add label o fig legend
    score_dict = {pop:[] for pop in pop_sizes}
    # collect scores 
    for i,path in enumerate(os.listdir('outputs')):
        if path in ignore_list: continue
        path = os.path.join('outputs', path)
        try:
            nodes, offspring = re.findall(r'\d+', path)
            seed = 1010
        except:
            print(path)
            nodes, offspring, seed = re.findall(r'\d+', path)
        # if offspring in seen:
        #     continue
        # else:
        #     seen.append(offspring)

        hof, log = load_files(prefix=path)
        scores = scores_from_log(log)
        if len(scores) < 76:
            continue
        score_dict[int(offspring)].append(scores)
        
        
    def patch_score(score_list):
        max_len = max([len(score) for score in score_list])
        for i in range(len(score_list)):
            if len(score_list[i] ) < max_len:
                print(f'patched {i} for ', max_len - len(score_list[i] ))
                score_list[i] = np.pad(score_list[i] , (0,max_len - len(score_list[i] )), 'constant',constant_values= np.nan)
        return score_list
    
    # plot scores
    for i, pop_size in enumerate(pop_sizes):
        scores = score_dict[pop_size]
        scores = np.vstack(patch_score(scores)) # fillout missing scores with nan and stack
        min_score_idx = np.unravel_index(scores.argmin(), scores.shape)
        min_score = scores[min_score_idx[0],min_score_idx[1]]
        print(pop_size, f' min score is {min_score} @ idx {min_score_idx}')
        stderr = np.nanstd(scores,axis=0)
        mean_score = np.nanmean(scores,axis=0)
        curr_style = styles[i % len(styles)]
        curr_marker = markers[i % len(markers)]
        fig = make_scores_plot(mean_score, fig, label=str(offspring) + " individuals", color=colors[color_count], style=curr_style, marker=curr_marker, stderr=stderr, label_best=label_best)
        label_best = False
        color_count += 1
        
    ax = fig.gca()
    ax.set_yscale('log')
#     plt.yscale('log', subsx=[2, 3, 4, 5, 6, 7, 8, 9])
    plt.minorticks_off()

    tcks = ax.get_yticks()
    tcks = [400,450,500,550]
    # tcks[-1] = max(scores)
    # tcks = np.insert(tcks,0,min(scores))

    # ax.set_yticks(tcks)
    # ax.set_xticks(np.arange(0,65,15))
    # ax.set_xticklabels(np.arange(15,80,15))
    ax.set_ylim(400, 600)
    # ax.set_ylim(min(scores)-100, max(scores))
    # ax.set_ylim(min(score_dict)-100, max(score_dict))
    scientific_formatter = FuncFormatter(scientific)
    ax.yaxis.set_major_formatter(scientific_formatter)
    
    
    save_scores_plot(fig, save_path='benchmark_plots')
        
        


def subplot_voltage(ax, curr_volts, target_volts, interested_stim=7, dt=.02, color='red'):
    volts2plot = target_volts[interested_stim, :3000]
    time = np.arange(len(volts2plot)) * dt
    ax.plot(time, volts2plot, color='black' , label='target')
    ax.plot(time, curr_volts[:3000], color=color, label='simulated')
    

    
def compare_voltages():
 
    rows = len(os.listdir('outputs')) // 2
    folders = os.listdir('outputs')
    fig, axs  = plt.subplots(ncols=3, nrows=1, figsize=(20,3))
#     plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.5)
    axs = axs.flatten()
    gens_to_plot = np.repeat(75, len(folders))
    seen = []
    i = 0
    folders = [folder for folder in folders if not folder in ignore_list]
    folders = sorted(folders, key=lambda x: int(re.findall(r'\d+', x)[-1]))

    for folder in  folders:
        if folder in ignore_list:
            continue
            
        path = os.path.join('outputs', folder)
        if not os.path.isdir(f'outputs/{folder}/voltages'):
            continue
            
        nodes, offspring = re.findall(r'\d+', path)
        if offspring in seen:
            continue
        else:
            seen.append(offspring)
            
        gen = gens_to_plot[i]
        curr_volts = nrnMread(f'outputs/{folder}/voltages/VHotP{interested_stim}_gen{gen}')
        subplot_voltage(axs[i], curr_volts, target_volts, interested_stim=interested_stim, color=colors[i])
        axs[i].axis('off')
        if i == 0 :
            add_scalebar(axs[i],   labelx="mS", labely="mV", loc=1, pad=0.1, y_sep=12, barwidth=3, bbox_to_anchor=Bbox.from_bounds(0,0, .12, .6), bbox_transform=axs[i].figure.transFigure)
#         red_patch = mpatches.Patch(color='red', label='Target Voltage')
#         blue_patch = mpatches.Patch(color='blue', label='Simulated Voltage')
#         plt.legend(loc=(1.1,1.1),handles=[red_patch,blue_patch], fontsize=20)
        if i == 2 :
            axs[i].legend(loc=(1.1,1.1), fontsize=20)
        
        
        axs[i].set_title(f'{offspring} Offspring')
        i += 1

    # fig.text(0.5, -.2, 'Time (mS)', ha='center', fontsize=25)
    # fig.text(-.045, 0.5, 'Vm', va='center', rotation='vertical', fontsize=25)

    plt.savefig(f"benchmark_plots/gen_compare_{interested_stim}.png",  bbox_inches='tight')
    plt.close(fig)

    
def basic_compare_voltages():
 
    gens_to_plot = [1,16,25, 44]
    fig, axs  = plt.subplots(ncols=2, nrows=2)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.5)
    axs = axs.flatten()
    for i, ax in enumerate(axs):
        gen = gens_to_plot[i]
        curr_volts = nrnMread(f'plot_data/VHotP{interested_stim}_gen{gen}')
        subplot_voltage(ax, curr_volts, target_volts)
        ax.set_title(f'Generation {gen}')

    fig.text(0.5, 0, 'Timestep (ms)', ha='center', fontsize=20)
    fig.text(0, 0.5, 'Neuron Potential (mV)', va='center', rotation='vertical', fontsize=20)

    plt.savefig("benchmark_plots/gen_compare.png",  bbox_inches='tight')
    plt.close(fig)
    
def parse_logs(logpath):
    with open(logpath,'r') as f:
        lines = f.readlines()
    exec_times = []
    for line in lines:
        if "everything took: " in line:
            exec_time = re.findall(r'\d+\.\d+', line)[0]
            exec_times.append(exec_time)
            
    exec_times = np.array(exec_times, dtype=float)
    return {'runtimes': exec_times}
            
def plot_time_in_evaluator():
    folders = os.listdir('outputs')
    total_runtimes = []
    labels = []
    all_nodes = []
    for i,folder in enumerate(folders):
        if folder in ignore_list:
            continue
        folder_path = os.path.join('outputs', folder)
        nodes, pop_size = re.findall(r'\d+', folder)
        if int(pop_size) != 5000 or int(nodes) == 5:
            continue
        all_nodes.append(nodes)
        log_path = os.path.join(folder_path, f'{nodes}N42C{pop_size}O.out')
        log_res = parse_logs(log_path)
        total_runtimes.append(np.sum(log_res['runtimes']))
        print(np.mean(log_res['runtimes']), len(log_res['runtimes']), folder)
        labels.append(str(nodes) + "N")
        
    inds = np.argsort(np.array(all_nodes, dtype=int))
    all_nodes = np.array(all_nodes, dtype=int)[inds]
    labels = np.array(labels)[inds]
    total_runtimes = np.array(total_runtimes, dtype=float)[inds]
    exp = total_runtimes[0] / all_nodes
    plt.scatter(labels,total_runtimes, color='blue', zorder=10, s=70)
    plt.plot(labels, total_runtimes, color='blue', label='observed')
    plt.plot(labels,exp, color='darkred', zorder=10, linewidth=2, label='ideal', linestyle='dashdot')
    plt.scatter(labels,exp, color='darkred', zorder=10, s=70)

    plt.legend()

    plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
    ax = plt.gca()
#     ax.set_yscale('log')
#     tcks = ax.get_yticks()
#     tcks[-1] = max(max(total_runtimes), max(exp))
#     ax.set_yticks(tcks)
#     scientific_formatter = FuncFormatter(scientific)
#     ax.yaxis.set_major_formatter(scientific_formatter)
    plt.xlabel("Nodes")
    plt.ylabel('Seconds')
    plt.savefig('benchmark_plots/runtime_bar',  bbox_inches='tight')
        
        
    
if __name__ == "__main__":
    set_custom_params_plt()
    score_plot_gauntlet()
    # compare_voltages()
#     basic_compare_voltages()
#     plot_time_in_evaluator()


