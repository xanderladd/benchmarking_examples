import numpy as np
import matplotlib.pyplot as plt
import re
import os
import pstats
import sys
import io
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil
import argparse
import matplotlib
from matplotlib import cycler
import pickle
import matplotlib.ticker as mticker
import matplotlib


# font = {'family' : 'normal',
#         'size'   : 20}

# matplotlib.rc('font', **font)
# # plt.rcParams["font.weight"] = "bold"
# plt.rcParams["axes.labelweight"] = "bold"

default_sfs = str(20)
default_stims = str(1)
default_pop = str(500)
POP_SCALING_FACTOR=500
# with open("std_dev_backup.pkl",'rb') as f:
#     backup_stddev = pickle.load(f)


def title_and_save(fig,title, pdf):
    global fig_count
    if title:
        plt.title("Fig {}: ".format(fig_count) + title, fontsize=20)
    else:
        plt.title("Fig {}: ".format(fig_count), fontsize=20)
    pdf.savefig(fig, bbox_inches='tight')
    fig_count += 1
    plt.close(fig)

def list_other_logs(f):
    path = os.path.dirname(f)
    files = [file for file in os.listdir(path) if ".log" in file and "gpu" not in file and 'std' not in file]
    return os.path.join(path,files[0])

def set_custom_params_plt():

    colors = cycler('color',
                    ['#EE6666', '#3388BB', '#9988DD',
                    '#EECC55', '#88BB44', '#FFBBBB'])
    plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
        axisbelow=True, grid=True)
    plt.rc('grid', color='w', linestyle='solid')
    plt.rc('patch', edgecolor='#E6E6E6')
    plt.rc('lines', linewidth=2)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


    
set_custom_params_plt()

def restore_default_mpl_params():
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)

def format_logname(node, pop, nCpu, stim, sf, path, how, title=None):
    if how == 'vanilla':
        stim = default_stims
        sf = default_sfs
        title = "Population"
    elif how == 'stims':
        pop = default_pop
        sf = default_sfs
        title = "Stims"
    elif how == "sfs":
        pop = default_pop
        stim = default_stims
        title = "Sfs"
    elif how == "full":
        assert title
        pass
    else:
        raise NotImplementedError
#         title=" PLOTTING METHOD NOT RECOGNIZED"
        
    if os.path.isfile("{}/{}N_{}C_{}O_{}S_{}SF/{}N_{}C_{}O_{}S_{}SF.log".format(path, node,nCpu, pop, stim, sf, node,nCpu,pop, stim, sf)):
        return "{}/{}N_{}C_{}O_{}S_{}SF/{}N_{}C_{}O_{}S_{}SF.log".format(path, node,nCpu, pop, stim, sf, node,nCpu,pop, stim, sf), title
    else:
        print("NO FILE ", "{}/{}N_{}C_{}O_{}S_{}SF/{}N_{}C_{}O_{}S_{}SF.log".format(path, node,nCpu, pop, stim, sf, node,nCpu,pop, stim, sf))
        return "{}/{}N_{}C_{}O_{}S_{}SF/{}N_{}C_{}O_{}S_{}SF.log".format(path, node,nCpu, pop, stim, sf, node,nCpu,pop, stim, sf), title



def processLog(f):
    with open(f, "r") as file:
        startEndPairs = []
        runtimes = []
        sfs = []
        neuroGPUStartTimes = []
        neuroGPUEndTimes = []
        procToSf = {}
        procStartDict = {}
        procEndDict = {}
        compStartDict = {}
        io_times = []
        file_lines =file.readlines()
        readingEnds = False
        readingStarts = False
        evalTimes = []
        neuronTimes = []
        for line in file_lines:
            if "Date:" in line:
                print(line)
            if "absolute start" in line:
                numbers = re.findall(r'\d+', line)
                abs_start = [ '.'.join(x) for x in zip(numbers[0::2], numbers[1::2]) ][0]
            if "nCpus" in line:
                nCpus = int(re.match('.*?([0-9]+)$', line).group(1))
                #assert nCpus  == cpu, "expected {} but got {} cpus in log".format(cpu,nCpus)
            if "Generation took" in line:
                line = line.split(":")
                hours, mins, seconds =  [float(re.findall(r"[-+]?\d*\.\d+|\d+",elem)[0]) for elem in line[2:]]
                runtime= hours*360 + mins*60 + seconds
                #print(runtime)
                runtimes.append(runtime)
            if "launched PIDS" in line:
                start = re.findall(r'\d+', line)[0] # second half is in miliseconds, don't need that precision
                
            if "finished PIDS" in line:
                end = re.findall(r'\d+', line)[0] 
                startEndPairs.append((start,end))
            if "process"  in line and "started" in line:
                stSplit = line.split(" ")
                sf = [stSplit[i] for i in range(2,len(stSplit)-2) if stSplit[i-1] == "is" and  stSplit[i+1] == "and"][0]
                sfs.append(sf)
                line = re.sub(r'(?<=is)(.*)(?=and)', "", line)
                numbers = re.findall(r'\d+', line)

                procToSf[numbers[0]] = sf
                if numbers[0] in procStartDict.keys():
                    procStartDict[numbers[0]].append(numbers[1])
                else:
                    procStartDict[numbers[0]] = [numbers[1]]
            if "returning" in line:
                numbers = re.findall(r'\d+', line)
                if numbers[0] in procEndDict.keys():
                    procEndDict[numbers[0]].append(numbers[1])
                else:
                    procEndDict[numbers[0]] = [numbers[1]]
            if "computing" in line:
                numbers = re.findall(r'\d+', line)
                if numbers[0] in compStartDict.keys():
                    compStartDict[numbers[0]].append(numbers[1])
                else:
                    compStartDict[numbers[0]] = [numbers[1]]
            if "evaluation:" in line:
                numbers = re.findall(r'\d+', line)
                numbers = [ '.'.join(x) for x in zip(numbers[0::2], numbers[1::2]) ]
                if "evalTimes" in locals():
                    evalTimes = np.append(evalTimes,  np.array(list(numbers), dtype=np.float32))
                else:
                    evalTimes = np.array(list(numbers), dtype=np.float32)
                avgEval = np.mean(evalTimes)
            if "neuron" in line and "starts" not in line and "ends" not in line:
                numbers = re.findall(r'\d+', line)
                numbers = [ '.'.join(x) for x in zip(numbers[0::2], numbers[1::2]) ]
                
                if "neuronTimes" in locals():
                    neuronTimes = np.append(neuroGPUTimes,  np.array(list(numbers), dtype=np.float32))
                else:
                    neuronTimes = np.array(list(numbers),dtype=np.float32)
                avgNrn = np.mean(neuronTimes)
            if ("neuroGPU" in line and "starts" in line and "ends" not in line) or readingEnds:
                readingEnds = True
                numbers = re.findall(r'\d+', line)
                numbers = [ '.'.join([x1,x2]) + "e+" + str(x3) for x1,x2,x3 in zip(numbers[0::3], numbers[1::3], numbers[2::3]) ]
                neuroGPUStartTimes += numbers
                if "]" in line:
                    readingEnds = False
            if ("neuroGPU" in line and "starts" not in line and "ends" in line) or readingStarts:
                readingStarts = True
                numbers = re.findall(r'\d+', line)
                numbers = [ '.'.join([x1,x2]) + "e+" + str(x3) for x1,x2,x3 in zip(numbers[0::3], numbers[1::3], numbers[2::3]) ]
                neuroGPUEndTimes += numbers
                if "]" in line:
                    readingStarts = False
            if "IO:" in line:
                numbers = re.findall(r'\d+', line)
                numbers = [ '.'.join([x1,x2]) + "e+" + str(x3) for x1,x2,x3 in zip(numbers[0::3], numbers[1::3], numbers[2::3]) ]
                io_times.append(numbers)
#             if "gen1 took" in line:
#                 break

    try:
        
        res = {#"procStartDict": procStartDict,"procEndDict": procEndDict,\
               "runtimes": runtimes,\
#                "compStartDict": compStartDict,"sfs": sfs,\
               "evalTimes": evalTimes,"neuronTimes": neuronTimes,\
#               "procToSf": procToSf, "absStart": abs_start, \
#                "neuronStartTimes": neuroGPUStartTimes, \
                   #"startEndPairs": startEndPairs,  
#                "neuronEndTimes": neuroGPUEndTimes}
        }
    except UnboundLocalError as e:
        print(e)
        print("MISREAD LOG : ", f, "  but I am in PERMISSIVE mode so it's ok")
#         raise e
        return {"procStartDict": {},"procEndDict": {},\
               "startEndPairs": [],"runtimes": [],\
               "compStartDict": {},"sfs": [],\
               "evalTimes": [],"neuroGPUTimes": [],\
              "procToSf": {}, "absStart": 0, \
               "neuroGPUStartTimes": [], \
               "neuroGPUEndTimes": [], "ioTimes": []}
    return res

    


def compare_scaling(strong_df, weak_df, path):
    #f  = "runTimeLogs/runTime.log"
    #assert (strong_df['offspring'].values == weak_df['offspring'].values).all()
    fig = plt.figure()
    plt.title("Population Scaling Comparison")
    labels = strong_df['offspring'].values
    y = strong_df['Runtime'].values
    err = strong_df['Runtime Stddev'].values
    plt.plot(labels, y, color='blue', label="strong scaling")
    plt.fill_between(labels, y - err, y+ err, color='blue', alpha=.4)
    # revisit this line
    labels = weak_df['offspring'].values
    y = weak_df['Runtime'].values
    err = weak_df['Runtime Stddev'].values
    plt.plot(labels, y, color='red', label="weak scaling")
    plt.fill_between(labels, y - err, y+ err, color='red', alpha=.4)
    
    plt.ylabel("time (s)")
    plt.xlabel("pop size")
    plt.legend()
    plt.ylim(bottom=0)
    plt.savefig(os.path.join(path, "scaling_time_compare"),  bbox_inches='tight')
    plt.close(fig)
    
    fig2 = plt.figure()
    plt.title("FOM comparison where FOM = pop size / nGPUs / runtime ")
    labels = strong_df['offspring'].values
    y = strong_df['FOM'].values
    plt.plot(labels, y, color='blue', label="strong scaling")
    plt.ylim(bottom=0)

    # revisit this line
    labels = weak_df['offspring'].values
    y = weak_df['FOM'].values
    err = weak_df['Runtime Stddev'].values
    plt.plot(labels, y, color='red', label="weak scaling")
    plt.ylabel("FOM")
    plt.xlabel("pop size")
    plt.legend()
    plt.savefig(os.path.join(path, "scaling_fom_compare"),  bbox_inches='tight')
    plt.close(fig)
    
    
def compare_stim_scaling(strong_df, weak_df, path):
    #f  = "runTimeLogs/runTime.log"
    strong_df = strong_df[strong_df['score functions'] == 20.0]
    strong_df = strong_df.sort_values(by='stims')
    fig = plt.figure()
    plt.title("Stim Scaling Comparison")
    labels = strong_df['stims'].values
    y = strong_df['Runtime'].values
    err = strong_df['Runtime Stddev'].values
    plt.plot(labels, y, color='blue', label="strong scaling")
    plt.fill_between(labels, y - err, y+ err, color='blue', alpha=.4)
    
    
    # MONKEY PATCH
    weak_df =weak_df[~(weak_df['nodes'] >  weak_df['stims'])]
    
    y = weak_df['Runtime'].values
    err = weak_df['Runtime Stddev'].values
    plt.plot(labels, y, color='red', label="weak scaling")
    plt.fill_between(labels, y - err, y+ err, color='red', alpha=.4)
    
    plt.ylabel("time (s)")
    plt.xlabel("number of stims")
    plt.legend()
    plt.ylim(bottom=0)
    plt.savefig(os.path.join(path, "stim_scaling_time_compare"),  bbox_inches='tight')
    plt.close(fig)

    
def compare_sf_scaling(strong_df, weak_df, path):
    #f  = "runTimeLogs/runTime.log"
    strong_df = strong_df[strong_df['stims'] == 1.0]
    strong_df = strong_df[strong_df['offspring'] == 500.0]
    strong_df = strong_df[strong_df['score functions'] < 71]
    weak_df = weak_df[~((weak_df['score functions'] == 20.0) & (weak_df['nodes'] == 1.0))]
    strong_df = strong_df.sort_values(by='score functions')
    fig = plt.figure()
    plt.title("Score Function Scaling Comparison")
    labels = strong_df['score functions'].values
    y = strong_df['Runtime'].values
    err = strong_df['Runtime Stddev'].values
    plt.plot(labels, y, color='blue', label="strong scaling")
    plt.fill_between(labels, y - err, y+ err, color='blue', alpha=.4)
    
    y = weak_df['Runtime'].values
    err = weak_df['Runtime Stddev'].values
    plt.plot(labels, y, color='red', label="weak scaling")
    plt.fill_between(labels, y - err, y+ err, color='red', alpha=.4)
    
    plt.ylabel("time (s)")
    plt.xlabel("# of score functions")
    plt.legend()
    plt.ylim(bottom=0)
    plt.savefig(os.path.join(path, "sf_scaling_time_compare"),  bbox_inches='tight')
    plt.close(fig)

    
    
    
    
    
    
    
    


def read_exps(exp_names, condition="vanilla", args=None):
    nodes = []
    pops = []
    nCpus = []
    sfs = []
    stims = []
    version_dict = {}
    use_constraint = False
    for exp_name in exp_names:
        try:
            curr_node, curr_core, curr_pop, curr_stims, curr_sfs, version = re.findall(r'\d+', exp_name) # TODO: use version appropriately
        except ValueError as e:
            print(exp_name, " doesn't confrom")
            continue
            
        shared_exp_name = exp_name[:-2]
        if shared_exp_name not in version_dict:
            version_dict[shared_exp_name] = [version]
        else:
            continue
            
        # here we can filter to only use relevant experiments
        # TODO: if one wanted to see scaling in multiple dimensions this will not work
        # need to expand conditions to allow something like "stims_sfs"
        if args and args.constraint_file:
            use_constraint = True
            constraints = {}
            with open(args.constraint_file, "r") as f: 
                lines = f.readlines()
            for line in lines:
                line = line.replace("\n", "")
                line = line.split("=")
                constraints[line[0]] = line[1].split(",")
        if use_constraint and ((curr_stims not in constraints['n_stims']) or  (curr_node not in constraints['N']) \
        or (curr_pop not in constraints['offspring']) or (curr_sfs not in constraints['n_sfs']) or (not len( np.where(np.array(constraints['offspring'])==curr_pop)[0]) > 1 and (np.where(np.array(constraints['offspring'])==curr_pop)[0] != np.where(np.array(constraints['N'])==curr_node)[0]).all())):
            continue
        elif not use_constraint:
            if "vanilla" in condition and (curr_stims != default_stims or curr_sfs != default_sfs):
                continue
           # MONKEY PATCH IF STATEMENT, ex; if you have a study with 2N 500 pop
            elif "vanilla" in condition and int(curr_pop) < (int(curr_node)  * POP_SCALING_FACTOR):
                continue
            elif condition == "stims" and (curr_pop != default_pop or curr_sfs != default_sfs):
                continue
            elif condition == "sfs" and (curr_pop != default_pop or curr_stims != default_stims):
                continue
        
        print("consuming ", curr_node, curr_pop, curr_stims, curr_sfs)
        nodes.append(curr_node), pops.append(curr_pop), nCpus.append(curr_core)
        sfs.append(curr_sfs), stims.append(curr_stims)
#         constraints['N'].remove(curr_node)

    max_version_list = [max(version_dict[key]) for key in version_dict]
    # TODO: fix this monkey patch on which version to use... this always use 0 versino
    #max_version_list = np.zeros(shape=len(nodes)).astype(int).astype(str)
    sort_inds = np.argsort(np.array(nodes).astype(int))
    if len(nodes)< 1:
        print(' NO EXPERIMENTS FOUND')
        print(1/0)
    if (nodes[0] == np.array(nodes)).all():
        sort_inds = np.argsort(np.array(pops).astype(int))
    if 'strong' in condition:
        sort_inds = np.argsort(np.array(pops).astype(int))
#     if condition == "vanilla":
#         sort_inds = np.argsort(np.array(pops).astype(int))
#     elif condition == "stims":
#         sort_inds = np.argsort(np.array(stims).astype(int))
#     elif condition == 'sfs':
#         sort_inds = np.argsort(np.array(sfs).astype(int))
        
    nodes = np.array(nodes)[sort_inds]
    pops = np.array(pops)[sort_inds]
    nCpus = np.array(nCpus)[sort_inds]
    sfs = np.array(sfs)[sort_inds]
    stims = np.array(stims)[sort_inds]
    max_version_list = np.zeros(shape=len(nodes)).astype(int).astype(str)
    max_version_listmax_version_list = np.array(max_version_list)[sort_inds]
    return nodes, pops, nCpus, sfs, stims, max_version_list
        


def plotScaling(nCpus,nodes,pops, sfs, stims, versions, path, how='vanilla', title=None, show=True):
    #f  = "runTimeLogs/runTime.log"
    runtimes = []
    labels = []
    stds = []
    if title:
        figname = title + "_scaling.png"
        title = title #+ " Scaling"
    for idx,(node,pop,nCpu,stim, sf, vers) in enumerate(zip(nodes,pops,nCpus, stims, sfs, versions)):
        if not title:
            f, title = format_logname(node,pop,nCpu,stim, sf, path, how=how, title=title)
            figname = title + "_scaling.png"
            title = title + " Scaling"
        else:
             f, _ = format_logname(node,pop,nCpu,stim, sf, path, how=how, title=title)
        try:
            print("READING",f)

            logRes = processLog(f)
        except:
            print("found no master log for ", f, " using first")
            prev_f = f
            f =  list_other_logs(prev_f)#re.sub(".log","_0.log", f)
            logRes = processLog(f)
            if len(logRes['runtimes']) == 0:
                f = f[:-4] + '_0' + f[-4:]
                if os.path.isfile(f):
                    logRes = processLog(f)
            shutil.copyfile(f, prev_f)
        if len(logRes['runtimes']) < 1:
            continue
        runtime = np.mean(logRes['runtimes'])
        if len(logRes['runtimes']) > 1:
            stds.append(np.std(logRes['runtimes']))
            print("not using back up", node)
        else:
            stds.append(0)#np.mean(backup_stddev[node]))
            print(" using back up standard deviation for {}.... get more trials".format(node))
        if (nodes[0] == nodes).all():
            label = "{}".format(pop)
        else:
            label = "{}".format(node)
        runtimes.append(runtime)
        labels.append(label)
    if (pops[0] == pops).all():
        lin_decr = runtimes[0]/ np.array([label.replace("N","") for label in labels]).astype(int)
        bench_name = 'Ideal'
    elif (nodes[0] == nodes).all():
        lin_decr = [runtimes[i] * (i+1) for i in range(len(runtimes))]
        bench_name = 'Exponential'
    else:
        bench_name = 'Ideal'
        lin_decr = np.repeat(runtimes[0],len(runtimes)) 
        
    fig = plt.figure()  
    
    plt.scatter(np.arange(len(runtimes)), lin_decr, color='orange', label=bench_name, s=15)
    plt.plot(np.arange(len(runtimes)), lin_decr,  color='orange')
    
    ax = fig.axes[0]
    plt.scatter(np.arange(len(runtimes)), runtimes, color='blue', label="Observed", s=15)
    plt.plot(np.arange(len(runtimes)), runtimes,  color='blue')
    runtimes, stds = np.array(runtimes), np.array(stds)
    plt.fill_between(np.arange(len(runtimes)), runtimes-stds, runtimes+stds, alpha=.5)

   
    plt.yscale("log")
    
    if (pops[0] == pops).all():
        plt.ylim(bottom=1)
        plt.xlabel("Nodes")
        #ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())

    elif (nodes[0] == nodes).all():
        
        plt.xlabel("Population")
    else:
        plt.ylim(bottom=10, top=100)
        plt.xlabel("Nodes")
        ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())

    plt.xticks(ticks=np.arange(len(runtimes)), labels=labels, rotation=45)
    

    plt.ylabel("Log(Total Runtime (s))")
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(path, figname),  bbox_inches='tight')
    
    
          
def generate_result_table(nCpus,nodes,pops, sfs, stims, versions, path, title=None, how='vanilla'):
    df = pd.DataFrame(columns=['Nodes','Total Cpus', 'Total Gpus',\
                               'Offspring', 'Stimuli', 'Score Functions', \
                               'Runtime', 'Runtime Stddev', 'FOM', 'FOM Std Dev','GPU Utilization'])
    if title:
        figname = title + "_scale.tex"
        df_name =  title + "_scale.csv"
    
    fn_to_gpu_df = {}
    for idx,(node,pop,nCpu,stim, sf, vers) in enumerate(zip(nodes,pops,nCpus, stims, sfs, versions)):
        if not title:
            f, title = format_logname(node,pop,nCpu,stim, sf, path, how=how, title=title)
            figname = title + "_scale.tex"
            df_name =  title + "_scale.csv"
        else:
            f, _ = format_logname(node,pop,nCpu,stim, sf, path, how=how, title=title)
        
        
           
        def drop_constant(df, preserve_list=[]):
            res = df.loc[:, (df != df.iloc[0]).any()] 
            if len(preserve_list) > 0:
                preserved_cols = df.loc[:,preserve_list]
                dropped = list(preserved_cols.columns)
                res.loc[:,preserve_list] = preserved_cols
                cols = list(res)
                if 'Nodes' in dropped:
                    cols.insert(0, cols.pop(cols.index('Nodes')))
                    cols.insert(1, cols.pop(cols.index('Offspring')))
                elif 'Offspring' in dropped:
                    cols.insert(2, cols.pop(cols.index('Offspring')))
                

                # move the column to head of list using index, pop and insert
#                 cols.insert(2, cols.pop(cols.index(preserve_list[0])))
                # use ix to reorder
                res = res.loc[:, cols]
            return res
        
        logRes = processLog(f)
        if len(logRes['runtimes']) == 0:
            f = f[:-4] + '_0' + f[-4:]
            if os.path.isfile(f):
                logRes = processLog(f)

        mean_runtime, std_runtime = np.mean(logRes['runtimes']), np.std(logRes['runtimes'])
        mean_eval, std_eval = 0,0#np.mean(logRes['evalTimes']), np.std(logRes['evalTimes'])
        mean_neuroGPU, std_neuroGPU =  0,0# np.mean(logRes['neuroGPUTimes']), np.std(logRes['neuroGPUTimes'])
        FOM = int(pop)/(int(nCpu))/np.array(logRes['runtimes'], dtype=np.float64)
        fom_mean = np.mean(FOM)
        fom_dev = np.std(FOM)
        pct_util = 0#float(percent_utilization)
#         import pdb; pdb.set_trace()

        if np.isnan(mean_runtime):
            continue
        df.loc[idx] = [int(node),int(nCpu), 0, \
                     int(pop),int(stim), int(sf), float(mean_runtime), float(std_runtime), fom_mean, fom_dev, pct_util]
        
        

    df = df.sort_values('Nodes', ascending=True) 
    
    
    
    # SAVE CSV
    df = df.drop_duplicates()
    # WHY DOES THAT HAPPEN ^
    df.to_csv(os.path.join(path, df_name))
    
    skip_latex=False
    if not skip_latex:
    # SAVE LATEX
        if title == 'Compute Scales and Problem Fixed':
            latex_df = df
        elif title == 'Compute Fixed and Problem Scales':
            latex_df = df
        else:
            latex_df = drop_constant(df, preserve_list=['Nodes','Offspring'] )
#         import pdb; pdb.set_trace()

        def plus_minus_cols(df, main, std, drop=True, precision=5):
            df[main] = df[main].astype(str).apply(lambda x: x[:precision]) \
            + " ± " + df[std].astype(str).apply(lambda x: x[:precision])
            if drop:
                df = df.drop(std,axis=1)
            return df


        latex_df = plus_minus_cols(latex_df, main='Runtime',std='Runtime Stddev')
        latex_df = plus_minus_cols(latex_df, main='FOM',std='FOM Std Dev', precision=8)
    #     formaters =  {"Runtime": "{:0.2f}".format, "Runtime Stddev":  "{:0.4f}".format,   "cori fom" : "{:0.2f}".format, "fom std dev" : "{:0.3f}".format,  'gpu_util': "{:0.2f}".format }
    #     df.to_latex(os.path.join(path, figname), formatters=formaters, float_format="%.0f", index=False)
        col_fmt = "|".join(np.repeat('c', len(df.columns)))
        col_fmt = "|" + col_fmt + "|"
        latex_df.to_latex(os.path.join(path, figname), float_format="%.0f", index=False, column_format=col_fmt)
    else:
        print("WARNING: skipped latex")

        
    print("WARNING: assumed 6 gpus, WARNING: made a bunch of gpu dfs but not doing much with em .. could plot")
    return df

    
    
def write_all_files(dest, srcs):
    with open(dest, 'w') as outfile:
        for fname in srcs:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
                
    
def merge_experiments(src, dest, version, path):
    src_path = os.path.join(path,src)
    curr_node, curr_core, curr_pop, curr_stims, curr_sfs, version = re.findall(r'\d+', src) # TODO: use version appropriately
    prof_name = "{}N_{}C_{}O_{}S_{}SF.prof".format(curr_node, curr_core, curr_pop, curr_stims, curr_sfs)
    new_prof_name = "{}N_{}C_{}O_{}S_{}SF.{}.prof".format(curr_node, curr_core, curr_pop, curr_stims, curr_sfs, version)
    # move profile
    if not os.path.isdir(dest):
        os.makedirs(dest)

# NOT CURRENTLY USING PROFILES OR GPU LOGS (won't use gpu logs ever...)
#     try:
#         shutil.copy(os.path.join(src_path,prof_name), os.path.join(dest,new_prof_name))
#     except FileNotFoundError:
#         print("no profile for ", src_path)
#     # move gpu util log
#     gpu_util_logname = "gpu_utillization.log"
#     new_gpu_util_logname = "gpu_utillization.{}.log".format(version)
#     try:
#         shutil.copy(os.path.join(src_path,gpu_util_logname), os.path.join(dest,new_gpu_util_logname))
#     except FileNotFoundError:
#         print("no gpu util for ", src_path)
    # comebine regular log
    log_path = os.path.join(dest, "{}N_{}C_{}O_{}S_{}SF.log".format(curr_node, curr_core, curr_pop, curr_stims, curr_sfs, version))
    old_log = os.path.join(src_path, "{}N_{}C_{}O_{}S_{}SF_{}.log".format(curr_node, curr_core, curr_pop, curr_stims, curr_sfs, version))
    
     
    if os.path.isfile(log_path) and os.path.isfile(old_log) :
        write_all_files(log_path, [log_path, old_log])
    elif  os.path.isfile(old_log):
        #assert int(version) == 1, "no master log and version is not 1?"
        write_all_files(log_path, [old_log])
    else:
        print("WARNING: No log merge for ", log_path)
        print("not deleting ... could be though")
        print(src_path, log_path)
#         print(1/0)
#         shutil.rmtree(src_path)

    
def collapse_exps(exp_names, path):
    nodes = []
    pops = []
    nCpus = []
    stims = []
    sfs = []
    exp_names = sorted(exp_names)
    for exp_name in exp_names:
        try:
            curr_node, curr_core, curr_pop, curr_stims, curr_sfs, version = re.findall(r'\d+', exp_name) # TODO: use version appropriately
        except ValueError as e:
            print(exp_name, " doesn't confrom")
            continue
        
        nodes.append(curr_node), pops.append(curr_pop), nCpus.append(curr_core)
        sfs.append(curr_sfs), stims.append(curr_stims)
        agg_exp_path =  os.path.join(path,"{}N_{}C_{}O_{}S_{}SF".format(curr_node, curr_core, curr_pop, curr_stims, curr_sfs))
        

        if int(version) == 0 or not os.path.isdir(agg_exp_path):
            if os.path.isdir(agg_exp_path):
                shutil.rmtree(agg_exp_path)
            shutil.copytree(os.path.join(path,exp_name), agg_exp_path)
            log_name = [file for file in os.listdir(agg_exp_path) if ".log" in file and "gpu" not in file]
            try:
                os.rename(os.path.join(agg_exp_path,log_name[0]), os.path.join(agg_exp_path,log_name[0][:-6] + log_name[0][-4:] ))
                print(log_name)
            except:
                print("no log for : ", agg_exp_path)
                print(os.path.dirname(agg_exp_path))
#                 shutil.rmtree(os.path.dirname(agg_exp_path)

                continue
            
        else:
            merge_experiments(exp_name, agg_exp_path, version, path )
       
    return 
        
    
def sf_plot_strategy(exp_names, args, collapse=False):
    if collapse:
        collapse_exps(exp_names, args.path)
    nodes, pops,  nCpus, sfs, stims, versions = read_exps(exp_names, condition='sfs', args=args)
    plt.title("Population Size Scaling w. Nodes")
    # step 1
    plotScaling(nCpus,nodes,pops, sfs, stims, versions, args.path, how='sfs')
    # step 2
    #wrapProfileMaker(nCpus, nodes, pops, versions)
    
    # step 3

    # step 4
    generate_result_table(nCpus,nodes,pops, sfs, stims, versions, args.path, how='sfs')
    
def stim_plot_strategy(exp_names, args, collapse=False):
    
    if collapse:
        collapse_exps(exp_names, args.path)
    nodes, pops,  nCpus, sfs, stims, versions = read_exps(exp_names, condition='stims', args=args)
    plt.title("Population Size Scaling w. Nodes")
    # step 1
    how ='vanilla'
    if args.constraint_file:
        how = 'full'
    plotScaling(nCpus,nodes,pops, sfs, stims, versions, args.path, how=how)
    # step 2
    #wrapProfileMaker(nCpus, nodes, pops, versions)
    
    # step 3

    # step 4
    generate_result_table(nCpus,nodes,pops, sfs, stims, versions, args.path, how=how)
    
def vanilla_plot_strategy(exp_names, args, collapse=False):
    
    if collapse:
        collapse_exps(exp_names,args.path)
    print("NOT COLLAPSING CHANGE L8R")
    nodes, pops,  nCpus, sfs, stims, versions = read_exps(exp_names, args=args)
    plt.title("Population Size Scaling w. Nodes")
    set_custom_params_plt()
    print("CUSTOMING PARAMS")
    # step 1
    how ='vanilla'
    title = None
    figname='population'
    if args.constraint_file:
        how = 'full'
        title =  os.path.basename(args.constraint_file)
        figname =  os.path.basename(args.constraint_file)
    plotScaling(nCpus,nodes,pops, sfs, stims, versions, args.path, how=how, title=title)
    # step 2
   # wrapProfileMaker(nCpus, nodes, pops, stims, sfs,  versions,  args.path)
    
    # step 3

    # step 4
    df = generate_result_table(nCpus,nodes,pops, sfs, stims, versions, args.path, how=how, title=title)
    restore_default_mpl_params()
    # step 5      
    #plot_gpu_pies(df,figname)
        
    
def strong_plot_strategy(exp_names, args, collapse=False):
    
    if collapse:
        collapse_exps(exp_names, args.path)
        
    how = "strong_vanilla" 
    weak_name = 'pop_scale.csv'
    if args.stims:
        how = "strong_stims" 
        weak_name= "stim_scale.csv"
    elif args.sfs:
        how = "strong_sfs" 
        weak_name  = 'sf_scale.csv'
    nodes, pops,  nCpus, sfs, stims, versions = read_exps(exp_names, condition=how, args=args)
    plt.title("Population Size Scaling w. Nodes")

    # step 4
    strong_df = generate_result_table(nCpus,nodes,pops, sfs, stims, versions, args.path, how=how)
    weak_df = pd.read_csv(os.path.join("weak_outputs",weak_name))
    if not args.stims and not args.sfs:
        compare_scaling(strong_df, weak_df)
    elif args.stims:
        compare_stim_scaling(strong_df, weak_df)
    elif args.sfs:
        compare_sf_scaling(strong_df, weak_df)
        
def check_collapse(exp_names, path):
    for exp_name in exp_names:
        if os.path.isfile(os.path.join(path, exp_name)):
            continue
        if not os.path.isdir(os.path.join(path, exp_name.split("SF")[0] + "SF")):
            return True
    return False

def find_largest_std(exp_names, args):
    stds = {}
    nodes, pops,  nCpus, sfs, stims, versions = read_exps(exp_names, condition="permissive")
    for node, pop, nCpu, sf, stim, version in zip(nodes, pops,  nCpus, sfs, stims, versions):
        f, title = format_logname(node, pop, nCpu, stim, sf, args.path, 'howwww')
        try:
            logRes = processLog(f)
        except:
            continue
#             print("found no master log for ", f, " using first")
#             prev_f = f
#             f = re.sub(".log","_0.log", f)
#             logRes = processLog(f)
#             shutil.copyfile(f, prev_f)
        mean_runtime, std_runtime = np.mean(logRes['runtimes']), np.std(logRes['runtimes'])
        # ignore an obvious case where algorithm glitched for some reason and took 300 seconds or something
        if node == '1' and std_runtime > 20:
            continue
        if node in stds:
            stds[node].append(std_runtime)
        else:
            stds[node] = [std_runtime]
    for key in stds:
        print(key, np.mean(stds[key]))
    with open("std_dev_backup.pkl",'wb') as f:
        pickle.dump(stds,f)
    exit()
    
    def delete_basic():
        root = 'outputw'
        folders = list(os.walk(root))[1:]

        for folder in folders:
            # folder example: ('FOLDER/3', [], ['file'])
            if not folder[2]:
                #os.rmdir(folder[0])
                print('rm dir folder[0]')


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Benchmarking viz')


    parser.add_argument('--stims', action="store_true")
    parser.add_argument('--sfs', action="store_true")
    parser.add_argument('--strong', action="store_true")
    parser.add_argument('--constraint_file', type=str, required=False, default=None)
    parser.add_argument('--path', type=str, required=False, default="outputs")

    args = parser.parse_args()
    
    exp_names = [dirname for dirname in os.listdir(args.path) if "_" in dirname and "ipynb" not in dirname] # make this more strict later --> should match coresnodes_POPSIZE_iteration
    #find_largest_std(exp_names, args)
 
    collapse = check_collapse(exp_names, args.path)
    collapse = True
#     if args.path != 'outputs':
#         collapse = False
    print(collapse, "SHOULD I COLLAPSE ?? IM SPITTING THESE RAPS TIL THE DAY THAT I DROP")
#     if args.stims:
#         stim_plot_strategy(exp_names, args, collapse=coargsllapse)
#     elif args.sfs:
#         sf_plot_strategy(exp_names, args, collapse=collapse)
#     else:
    vanilla_plot_strategy(exp_names, args, collapse=collapse)
        
    if args.strong:
        strong_plot_strategy(exp_names, args, collapse=collapse)
    
        
    
    
    