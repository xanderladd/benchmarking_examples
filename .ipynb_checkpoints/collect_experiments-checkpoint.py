import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re


experiment_dirs = ['neuroGPU', 'core_neuron']
cwd = os.getcwd()

results = pd.DataFrame(columns=['method', 'runtime', 'stddev', 'population size'])

for exp_dir in experiment_dirs:
    log_dir = os.path.join(cwd, exp_dir, 'logs')
    experiments = os.listdir(log_dir)
    for exp in experiments:
        if '.log' not in exp:
            continue
        log_path = os.path.join(log_dir, exp)
        with open(log_path,'r') as f:
            lines = f.readlines()
        curr_runtimes = []
        for line in lines:
            if 'everything took' in line:
                runtime = float(re.findall(r'\d+\.\d+',line)[0])
                curr_runtimes.append(runtime)
                
                
        curr_runtimes = np.array(curr_runtimes)
        pop_size = re.findall(r'\d+', exp)[0]
        
        if len(curr_runtimes) < 2:
            print(f'{pop_size}  for {exp_dir} needs more trials')
            continue
        # add result to dataframe
        row = {'method': exp_dir, 'runtime': np.mean(curr_runtimes), 'stddev': np.std(curr_runtimes), \
               'population size': pop_size}
        results = results.append(row, ignore_index=True)
results = results.sort_values('population size').reset_index(drop=True)
print(results)