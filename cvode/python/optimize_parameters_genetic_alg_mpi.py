import bluepyopt as bpop
import bluepyopt.deapext.algorithms as algo
from bluepyopt.deapext.tools.selIBEA import selIBEA

import neurogpu_multistim_evaluator_par_mpi as hoc_ev
import pickle
import time
import numpy as np
from deap import tools
from mpi4py import MPI

# attempts to speed up DEAP
from concurrent.futures import ProcessPoolExecutor as Pool
import dill 
import torch

import argparse
gen_counter = 0
best_indvs = []
cp_freq = 1
old_update = algo._update_history_and_hof
#cp_file = 'C:/pyNeuroGPU_win55/x64/cp.pkl'
cp_true = False
parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='L5PC example')
parser.add_argument('--offspring_size', type=int, required=False, default=2,
                    help='number of individuals in offspring')
parser.add_argument('--max_ngen', type=int, required=False, default=2,
                    help='maximum number of generations')
parser.add_argument('--seed', type=int, required=True, default=1178,
                    help='maximum number of generations')

comm = MPI.COMM_WORLD
global_rank = comm.Get_rank()
size = comm.Get_size()

abs_start = time.time()

global fn
fn =None
    
def my_update(halloffame, history, population):
    global gen_counter,cp_freq, best_indvs, fn
    old_update(halloffame, history, population)
    if halloffame:
        best_indvs.append(halloffame[0])
#     if gen_counter%cp_freq == 0:
#         save_logs(fn,best_indvs,population)

    gen_counter = gen_counter+1
    print("Current generation: ", gen_counter)
    print("@ my update, took: ", time.time() - abs_start)


# @jit(nopython=True)
# def np_apply_along_axis(func1d, axis, arr):
#     assert arr.ndim == 2
#     assert axis in [0, 1]
#     if axis == 0:
#         result = np.empty(arr.shape[1])
#         for i in range(len(result)):
#             result[i] = func1d(arr[:, i])
#     else:
#         result = np.empty(arr.shape[0])
#     for i in range(len(result)):
#         result[i] = func1d(arr[i, :])
#     return result
        
def my_record_stats(stats, logbook, gen, population, invalid_count):
    '''Update the statistics with the new population'''
    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=gen, nevals=invalid_count, **record)
    logbook = comm.bcast(logbook, root=0)
    if global_rank == 0:
        print('log: ', logbook, '\n')
        output = open("log"+fn, 'wb')
        pickle.dump(logbook, output)
        output.close()

def save_logs(fn, best_indvs, hof):
    if global_rank == 0:
        print(best_indvs)
        output = open("indv"+fn, 'wb')
        pickle.dump(best_indvs, output)
        output.close()
        output = open("hof"+fn, 'wb')
        pickle.dump(hof, output)
        output.close()
        
        
def fast_calc_fitness_components(population, kappa):
    """returns an N * N numpy array of doubles, which is their IBEA fitness """
    # DEAP selector are supposed to maximise the objective values
    # We take the negative objectives because this algorithm will minimise
    population_matrix = np.fromiter(
        iter(-x for individual in population
             for x in individual.fitness.wvalues),
        dtype=np.float)
    pop_len = len(population)
    feat_len = len(population[0].fitness.wvalues)
    population_matrix = population_matrix.reshape((pop_len, feat_len))

    # Calculate minimal square bounding box of the objectives
    box_ranges = (np.max(population_matrix, axis=0) -
                  np.min(population_matrix, axis=0))

    # Replace all possible zeros to avoid division by zero
    # Basically 0/0 is replaced by 0/1
    box_ranges[box_ranges == 0] = 1.0

    numba_start = time.time()

    components_matrix = torch.zeros((pop_len, pop_len)).cuda()
    population_matrix = torch.from_numpy(population_matrix).cuda()
    box_ranges = torch.from_numpy(box_ranges).cuda()
    
    for i in range(0, pop_len):
        diff = population_matrix - population_matrix[i, :]
        components_matrix[i, :] = torch.amax(torch.divide(diff, box_ranges), dim=1)

    # @jit(nopython=True) 
    # def fast_compenent(population_matrix, box_ranges, pop_len): 
    #     components_matrix = np.zeros((pop_len, pop_len))
    #     for i in range(0, pop_len):
    #         diff = population_matrix - population_matrix[i, :]
    #         components_matrix[i, :] = np.max(
    #         np.divide(diff, box_ranges),
    #         axis=1)
            

    # components_matrix = fast_compenent(population_matrix, box_ranges, pop_len)
    components_matrix = components_matrix.cpu().numpy()
    numba_end = time.time()
    print(numba_end - numba_start, " IS HOW LNOG NUMBA TOOK")

    # Calculate max of absolute value of all elements in matrix
    max_absolute_indicator = np.max(np.abs(components_matrix))

    # Normalisation
    if max_absolute_indicator != 0:
        components_matrix = np.exp(
            (-1.0 / (kappa * max_absolute_indicator)) * components_matrix.T)

    return components_matrix

import random


def selIBEA(population, mu, alpha=None, kappa=.05, tournament_n=4):
    """IBEA Selector"""

    if alpha is None:
        alpha = len(population)

    # Calculate a matrix with the fitness components of every individual
    components = fast_calc_fitness_components(population, kappa=kappa)

    # Calculate the fitness values
    _calc_fitnesses(population, components)

    # Do the environmental selection
    population[:] = _environmental_selection(population, alpha)

    # Select the parents in a tournament
    parents = _mating_selection(population, mu, tournament_n)

    return parents


# def _calc_fitness_components(population, kappa):
#     """returns an N * N numpy array of doubles, which is their IBEA fitness """
#     # DEAP selector are supposed to maximise the objective values
#     # We take the negative objectives because this algorithm will minimise
#     population_matrix = np.fromiter(
#         iter(-x for individual in population
#              for x in individual.fitness.wvalues),
#         dtype=np.float)
#     pop_len = len(population)
#     feat_len = len(population[0].fitness.wvalues)
#     population_matrix = population_matrix.reshape((pop_len, feat_len))

#     # Calculate minimal square bounding box of the objectives
#     box_ranges = (np.max(population_matrix, axis=0) -
#                   np.min(population_matrix, axis=0))

#     # Replace all possible zeros to avoid division by zero
#     # Basically 0/0 is replaced by 0/1
#     box_ranges[box_ranges == 0] = 1.0

#     components_matrix = np.zeros((pop_len, pop_len))
#     for i in range(0, pop_len):
#         diff = population_matrix - population_matrix[i, :]
#         components_matrix[i, :] = np.max(
#             np.divide(diff, box_ranges),
#             axis=1)

#     # Calculate max of absolute value of all elements in matrix
#     max_absolute_indicator = np.max(np.abs(components_matrix))

#     # Normalisation
#     if max_absolute_indicator != 0:
#         components_matrix = np.exp(
#             (-1.0 / (kappa * max_absolute_indicator)) * components_matrix.T)

#     return components_matrix


def _calc_fitnesses(population, components):
    """Calculate the IBEA fitness of every individual"""

    # Calculate sum of every column in the matrix, ignore diagonal elements
    column_sums = np.sum(components, axis=0) - np.diagonal(components)

    # Fill the 'ibea_fitness' field on the individuals with the fitness value
    for individual, ibea_fitness in zip(population, column_sums):
        individual.ibea_fitness = ibea_fitness


def _choice(seq):
    """Python 2 implementation of choice"""

    return seq[int(random.random() * len(seq))]


def _mating_selection(population, mu, tournament_n):
    """Returns the n_of_parents individuals with the best fitness"""

    parents = []
    for _ in range(mu):
        winner = _choice(population)
        for _ in range(tournament_n - 1):
            individual = _choice(population)
            # Save winner is element with smallest fitness
            if individual.ibea_fitness < winner.ibea_fitness:
                winner = individual
        parents.append(winner)

    return parents


def _environmental_selection(population, selection_size):
    """Returns the selection_size individuals with the best fitness"""

    # Sort the individuals based on their fitness
    population.sort(key=lambda ind: ind.ibea_fitness)

    # Return the first 'selection_size' elements
    return population[:selection_size]


def main():
    global fn

    print("*****")
    #pool = multiprocessing.Pool(processes=64)
    print('Init evaluator')
    args = parser.parse_args()
    start = time.time()
    evaluator = hoc_ev.neurogpu_multistim_evaluator()
    algo._update_history_and_hof = my_update
    algo._record_stats = my_record_stats
    
    fn = f"{size}N_{args.offspring_size}O.pkl"
    # first seed 1178
    
    opt = bpop.optimisations.DEAPOptimisation(evaluator,seed=args.seed, offspring_size=args.offspring_size,  eta=20, mutpb=0.3, cxpb=0.7)#, hof = tools.ParetoFront())
    opt.toolbox.register("select", selIBEA)
    #
    if (cp_true ==True):
        pop, hof, log, hst = opt.run(max_ngen=args.max_ngen, cp_filename=cp_file, cp_frequency=1,continue_cp=True)
    else:
        pop, hof, log, hst = opt.run(max_ngen=args.max_ngen)
        # pop, hof, log, hst = opt.run(max_ngen=args.max_ngen, cp_filename=f'cp_{args.seed}.pkl', cp_frequency=20)
#     fn = time.strftime("%d_%m_%Y_%H_%M")
    #scipy.io.savemat(fn+'.mat',mdict={'hof':hof})
#     fn = fn + f"_{args.offspring_size}.pkl"
    

    if global_rank == 0:
        end = time.time()
        save_logs(fn, best_indvs, hof)
        output = open("indv" + fn, 'wb')
        pickle.dump(best_indvs, output)
        output.close()
        output = open("log"+fn, 'wb')
        pickle.dump(log,output)
        output.close()
        output = open("hst"+fn, 'wb')
        pickle.dump(hst,output)
        output.close()
        with open("speed.txt", "a+") as f:
            f.write(f"{size} took {end - start} \n")
            
        print("WHOLE RUN TOOK : ", end-start)
        #print('Hall of fame: ', hof, '\n')
        print('log: ', log, '\n')
        print('History: ', hst, '\n')
        print('Best individuals: ', best_indvs, '\n')
if __name__ == '__main__':
    main()
