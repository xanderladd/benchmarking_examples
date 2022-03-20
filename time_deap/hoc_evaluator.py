import numpy as np
import h5py
import bluepyopt as bpop
import time
import bluepyopt.deapext.algorithms as algo
import logging
from mpi4py import MPI
import pandas as pd


old_eval = algo._evaluate_invalid_fitness



comm = MPI.COMM_WORLD
global_rank = comm.Get_rank()
size = comm.Get_size()
end = None
times = []

class hoc_evaluator(bpop.evaluators.Evaluator):
    def __init__(self, title=None, logger=None):
        """Constructor"""
        super(hoc_evaluator, self).__init__()
        self.objectives = [bpop.objectives.Objective('Weighted score functions')]
        params_df = pd.read_csv('params_bbp_full_gpu_tuned_10_based.csv')
        self.params = []
        i = 0
        self.pmin = params_df['Lower bound']
        self.pmax = params_df['Upper bound']
        for row in params_df.to_dict(orient="records"):
            base_value = row['Base value']
            lb = row['Lower bound']
            ub = row['Upper bound']
            # print(base_value, lb, ub)
            self.params.append(bpop.parameters.Parameter(base_value, bounds=(lb,ub)))
            print(i)
            i += 1
        self.weights = np.random.normal(0,scale=1, size=1000)
        self.logger = logger
        self.title = title
        
  
    def my_evaluate_invalid_fitness(toolbox, population):
        '''Evaluate the individuals with an invalid fitness
        Returns the count of individuals with invalid fitness
        '''
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        invalid_ind = [population[0]] + invalid_ind 
        fitnesses = toolbox.evaluate(invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        return len(invalid_ind)
 

    def evaluate_with_lists(self, param_values):
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

        global end
        start = time.time()
        if end:
            print(f'end - start : {start - end}')
            times.append(start - end)
        if len(times) > 49:
            np.savetxt(f'deap_runtimes_{self.title}.csv', times, delimiter=',')
        if global_rank == 0:
            self.logger.info("gen start: " + str(time.time()))

        param_values = np.array(param_values)
        self.data_volts_list = np.array([])
        if global_rank == 0:
            fake_scores = np.random.normal(loc=0, scale=1,  size=param_values.shape[0])
            fake_scores = fake_scores.reshape(-1,1)
        else:
            fake_scores = None
        fake_scores = comm.bcast(fake_scores, root = 0)
        comm.barrier()
        if global_rank == 0:
            self.logger.info("gen end: " + str(time.time()))
        end = time.time()
        return fake_scores

    

    
algo._evaluate_invalid_fitness = hoc_evaluator.my_evaluate_invalid_fitness
