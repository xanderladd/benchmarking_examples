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



parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Benchmarking viz')


parser.add_argument('--path', type=str, required=True, default="log path")

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
        for line in file_lines:
            if "Date:" in line:
                print(line)
            if "absolute start" in line:
                numbers = re.findall(r'\d+', line)
                abs_start = [ '.'.join(x) for x in zip(numbers[0::2], numbers[1::2]) ][0]
            if "nCpus" in line:
                nCpus = int(re.match('.*?([0-9]+)$', line).group(1))
                #assert nCpus  == cpu, "expected {} but got {} cpus in log".format(cpu,nCpus)
            if "took:" in line:
                runtime = float(re.findall(r"[-+]?\d*\.\d+|\d+",line)[1])
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
            if "neuroGPU" in line and "starts" not in line and "ends" not in line:
                numbers = re.findall(r'\d+', line)
                numbers = [ '.'.join(x) for x in zip(numbers[0::2], numbers[1::2]) ]
                
                if "neuroGPUTimes" in locals():
                    neuroGPUTimes = np.append(neuroGPUTimes,  np.array(list(numbers), dtype=np.float32))
                else:
                    neuroGPUTimes = np.array(list(numbers),dtype=np.float32)
                avgNGPU = np.mean(neuroGPUTimes)
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
    #try:

    res = {"procStartDict": procStartDict,"procEndDict": procEndDict,\
           "startEndPairs": startEndPairs,"runtimes": runtimes,\
           "compStartDict": compStartDict,"sfs": sfs,\
           "evalTimes": evalTimes,"neuroGPUTimes": neuroGPUTimes,\
          "procToSf": procToSf, "absStart": abs_start, \
           "neuroGPUStartTimes": neuroGPUStartTimes, \
           "neuroGPUEndTimes": neuroGPUEndTimes, "ioTimes": io_times}
#     except UnboundLocalError as e:
#         print("MISREAD LOG : ", f, "  but I am in PERMISSIVE mode so it's ok")
# #         raise e
#         return {"procStartDict": {},"procEndDict": {},\
#                "startEndPairs": [],"runtimes": [],\
#                "compStartDict": {},"sfs": [],\
#                "evalTimes": [],"neuroGPUTimes": [],\
#               "procToSf": {}, "absStart": 0, \
#                "neuroGPUStartTimes": [], \
#                "neuroGPUEndTimes": [], "ioTimes": []}
    return res

    

def makeCustomProfile(percent_utilization=None):
    f  = "runTimeLogs/runTime.log"
    logRes = processLog(f)
    absStart = float(logRes['absStart'])
    start_data = np.array([float(start) for start in logRes["neuroGPUStartTimes"]]) 
    end_data = np.array(logRes["neuroGPUEndTimes"]).astype(float)
    print(len(start_data))
    times = logRes["neuroGPUTimes"]
    # bugged timer
    end_data = np.mean(times) + start_data
    procEndDict = logRes['procEndDict']
    sfsMap = logRes['procToSf']
    sfsMapMap = {}
    counter = 0
    for val in set(list(sfsMap.values())):
        sfsMapMap[val] = counter
        counter +=1

    nGpus = 8# THIS SHOULD BE IN LOG RES logRes['nGpus']
    compStartDict = logRes['compStartDict']
    procStartDict = logRes['procStartDict']
    
    startEndPairs = logRes['startEndPairs']
    #print(absStart)

    #print(startEndPairs)

    startEndPairs = [(float(pair0) - float(absStart), float(pair1) - float(absStart)) for pair0, pair1 in startEndPairs]
    #print(procStartDict)
    #print(startEndPairs)
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(13,9))
    
    # YLIM XLIM
    plt.ylim(0,600)
    #plt.xlim(0,120)
    
    x_anchors= []
    x_ends = []

    def calc_y_anchor(x_anchor,width,x_anchors,x_ends):
        curr_ht = 0
        for x_anc, x_end in zip(x_anchors,x_ends):
            if x_anchor > x_anc and x_anchor < x_end:
                curr_ht += 10
            elif x_anc > x_anchor and x_anc < x_anchor+width:
                curr_ht += 10
        return curr_ht

    cur_start = 0
    y_base = 0
    firstGo = True
    for start, end in startEndPairs:
        if firstGo:
            firstGo = False
            plt.axvline(x=start,color="blue", label="CPU Eval Start")
            plt.axvline(x=end,color="red", label="CPU Eval Done")
        else:
            plt.axvline(x=start,color="blue")
            plt.axvline(x=end,color="red")

    idx = 1
    labels = list(compStartDict.keys())
    box_ht = 10
    for procStart,procEnd,proc in \
    zip(list(procStartDict.values()),list(procEndDict.values()), list(procEndDict.keys()) ):
        for pStart,pEnd in zip(procStart,procEnd):
            x_anchor = float(pStart)  - float(absStart)#float(procStart) - float(absStart)
            y_anchor = y_base + 10
            if x_anchor > float(startEndPairs[cur_start][1]) and cur_start < len(startEndPairs) - 1 :
                cur_start += 1
                y_base = 0
            else:
                y_base += 10
            
            if y_base > 1200: # MAX HEIGHT EXCEEDED
                #print('max height exc.')
                break
            width =(float(pEnd) - float(absStart)) - (float(pStart) - float(absStart)) #(float(procEnd) - float(absStart)) - (float(procStart) - float(absStart))
            #y_anchor = calc_y_anchor(x_anchor,width,x_anchors,x_ends)
            x_anchors.append(float(x_anchor)), x_ends.append(width)
            rect = patches.Rectangle((x_anchor, y_anchor), width, box_ht, \
                                     linewidth=2, edgecolor='black', facecolor='lightblue', fill=True, zorder=0)
            curr_sf = sfsMapMap[sfsMap[proc]]
            ax.annotate(curr_sf, (x_anchor + 2.5, y_anchor + 5), color='black', weight='bold', \
                        fontsize=7, ha='center', va='center', zorder=4)
            # Add the patch to the Axes
            ax.add_patch(rect)
            idx += 1
#         if y_base > 1200:
#             print("max height exceeded")
#             break

    # Create a Rectangle patches
    box_ht = 15 # constant box height
    cur_start = 0
    y_base = 0
    for start,end,idx in zip(start_data,end_data, np.arange(len(end_data))):
        x_anchor = start-absStart
        if x_anchor > float(startEndPairs[cur_start][1]):
            cur_start += 2
            y_base = 0
        else:
            y_base += 15
        y_anchor = y_base
        width =  end - start
        x_anchors.append(float(x_anchor)), x_ends.append(width)
        rect = patches.Rectangle((x_anchor, y_anchor), width, box_ht, \
                                 linewidth=2.5, edgecolor='black', facecolor='palegreen', fill=True, zorder=10)
        ax.annotate("GPU {}".format(idx %  nGpus), (x_anchor + 10, y_anchor + 8), color='black', weight='bold', 
                    fontsize=10, ha='center', va='center', zorder=20)

        # Add the patch to the Axes
        ax.add_patch(rect)
    #plt.title("Custom Profile for {} CPUs, {} Pop Size and {} Nodes".format(nCpus,nodes,popSize))
    plt.legend()
    if percent_utilization:
        plt.title("GPU Utilization : {}".format(percent_utilization))
    plt.xlabel("time (s)")
    #plt.show()
    print("TODO: add legend later")
    out_dir = os.path.dirname(f)
    plt.savefig(os.path.join(out_dir,"custom_profile"), bbox_inches='tight')
    plt.close()
    sfsMap = logRes['procToSf']
    sfsMapMap = {}
    counter = 0
    for val in set(list(sfsMap.values())):
        sfsMapMap[val] = counter
        counter +=1

    
    plt.savefig(os.path.join(out_dir,"legend"), bbox_inches='tight')
    plt.close()




def read_gpu_logs(fn):
    with open(fn, 'r') as gpu_f: 
        lines = gpu_f.readlines()
    gpu_df = pd.DataFrame([sub.split(",") for sub in lines])
    gpu_df.columns = gpu_df.iloc[0]
    gpu_df = gpu_df[1:]
    gpu_df = gpu_df.rename({' name': 'name', ' utilization.gpu [%]' : 'utilization'}, axis=1)
    # remove label rows
    gpu_df = gpu_df[gpu_df['name'] != ' name']
    gpu_df['timestamp'] = gpu_df['timestamp'].astype('datetime64[ns]')
    # WE GET ONE MEASUREMENT PER SECOND
    total_elapsed = (max(gpu_df.timestamp) - min(gpu_df.timestamp)).seconds 
    gpu_df['utilization'] = gpu_df['utilization'].str.replace(" \%","").astype(int)
    import pdb; pdb.set_trace()
    percent_utilization = (np.sum(gpu_df['utilization'] ) / (6 * total_elapsed))
    #gpu_df[['Date','Time']] = gpu_df.timestamp.str.split(expand=True)
    return percent_utilization, gpu_df

if __name__ == "__main__":
    
    args = parser.parse_args()
    percent_utilization,gpu_df =  read_gpu_logs(args.path)
    makeCustomProfile(percent_utilization=percent_utilization)
    print("PCT UTIL : ", percent_utilization)
    