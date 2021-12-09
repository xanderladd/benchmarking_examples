import bluepyopt as bpop
import bluepyopt.deapext.algorithms as algo
import pickle
import time
import numpy as np
from datetime import datetime
import argparse
import sys
import argparse
import time
import textwrap
import os
import cProfile
import shutil
import glob
from mpi4py import MPI
import multiprocessing
# set up environment variables for MPI

comm = MPI.COMM_WORLD
global_rank = comm.Get_rank()
size = comm.Get_size()

import logging.config
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})

import logging.handlers
import os
if global_rank == 0:
    filename = "runTimeLogs/runTime.log"
    # your logging setup
    should_roll_over = os.path.isfile(filename)
    handler = logging.handlers.RotatingFileHandler(filename, mode='w', backupCount=15, delay=True)
    if should_roll_over:  # log already exists, roll over!
        handler.doRollover()
else:
    filename = None
    
filename = comm.bcast(filename, root=0)
    
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename=filename, level=logging.DEBUG)
logging.info("absolute start : " + str(time.time()) + " from rank" + str(global_rank))


#if size > 1:
logging.info("USING MPI : TRUE")
import hoc_evaluatorGPU_MPI3_CN as hoc_ev
# else:
#     logging.info("USING MPI : FALSE")
#     import hoc_evaluatorGPU_BBP_par as hoc_ev

    #testing
nGpus = len([devicenum for devicenum in os.environ['CUDA_VISIBLE_DEVICES'] if devicenum != ","])
logging.info("nGPUS :" + str(nGpus))
    #assert nGpus == 8 # this only works if you have 8 gpus, if you are using 6 run the the tests instead
# if global_rank == 0 and os.path.isfile("gpu_utillization.log"):
#     os.remove("gpu_utillization.log")
open('gpu_utillization.log', 'a+').close()


gen_counter = 0
best_indvs = []
cp_freq = 1
old_update = algo._update_history_and_hof

import psutil
def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='L5PC example',
        epilog=textwrap.dedent('''\
The folling environment variables are considered:
    L5PCBENCHMARK_USEIPYP: if set, will use ipyparallel
    IPYTHON_PROFILE: if set, used as the path to the ipython profile
    BLUEPYOPT_SEED: The seed used for initial randomization
        '''))
    parser.add_argument('--start', action="store_true")
    parser.add_argument('--continu', type=bool, default=False)
    parser.add_argument('--checkpoint', required=False, default=None,
                        help='Checkpoint pickle to avoid recalculation')
    parser.add_argument('--offspring_size', type=int, required=False, default=2,
                        help='number of individuals in offspring')
    parser.add_argument('--max_ngen', type=int, required=False, default=1,
                        help='maximum number of generations')
    parser.add_argument('--n_stims', type=int, required=False, default=1,
                        help='number of stims to optimize over')
    parser.add_argument('--n_sfs', type=int, required=False, default=0,
                        help='number of score functions to use')
    parser.add_argument('--n_cpus', type=int, required=False, default=0,
                        help='number of cpu cores to use')
    parser.add_argument('--sf_module', type=str, required=False, default='efel',
                        help='number of cpu cores to use')
    parser.add_argument('--responses', required=False, default=None,
                        help='Response pickle file to avoid recalculation')
    parser.add_argument('--analyse', action="store_true")
    parser.add_argument('--compile', action="store_true")
    parser.add_argument('--hocanalyse', action="store_true")
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed to use for optimization')
    parser.add_argument('--ipyparallel', action="store_true", default=False,
                        help='Use ipyparallel')
    parser.add_argument(
        '--diversity',
        help='plot the diversity of parameters from checkpoint pickle file')
    parser.add_argument('-v', '--verbose', action='count', dest='verbose',
                        default=0, help='-v for INFO, -vv for DEBUG')

    return parser

def my_update(halloffame, history, population):
    global gen_counter, cp_freq
    old_update(halloffame, history, population)
    best_indvs.append(halloffame[0])
    gen_counter = gen_counter+1
    print("Current generation: ", gen_counter)

    if gen_counter%cp_freq == 0:
        fn = '.pkl'
        save_logs(fn, best_indvs, population)

def save_logs(fn, best_indvs, population):
    output = open("./best_indv_logs/best_indvs_gen_"+str(gen_counter)+fn, 'wb')
    pickle.dump(best_indvs, output)
    output.close()
    
def my_record_stats(stats, logbook, gen, population, invalid_count):
    '''Update the statistics with the new population'''
    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=gen, nevals=invalid_count, **record)
    logbook = comm.bcast(logbook, root=0)
    if global_rank == 0:
        print('log: ', logbook, '\n')
        output = open("log.pkl", 'wb')
        pickle.dump(logbook, output)
        output.close()

def main(pool):
    args, unk = get_parser().parse_known_args()
    algo._update_history_and_hof = my_update
    algo._record_stats = my_record_stats

#     logging.basicConfig(level=(logging.WARNING,
#                                 logging.INFO,
#                                 logging.DEBUG)[args.verbose],
#                                 stream=sys.stdout)
#     opt = create_optimizer(args)
#     print(args.max_ngen, "MAX NGEN")
#     if glob.glob(f"outputs/{size}N_{args.n_cpus}C_{args.offspring_size}O_{args.n_stims}S_{args.n_sfs}SF*"):
#         print(f"outputs/{size}N_{args.n_cpus}C_{args.offspring_size}O_{args.n_stims}S_{args.n_sfs}SF*)", "exists... exiting....")
#         exit()
#     else:
#         print(f"outputs/{size}N_{args.n_cpus}C_{args.offspring_size}O_{args.n_stims}S_{args.n_sfs}SF*)")
    evaluator = hoc_ev.hoc_evaluator(pool, args.n_stims, args.n_sfs, args.sf_module)

#     seed = os.getenv('BLUEPYOPT_SEED', args.seed)
    seed = os.getenv('BLUEPYOPT_SEED', 100)

    opt = bpop.optimisations.DEAPOptimisation(
        evaluator=evaluator,
        #map_function=map_function,
        seed=seed,
        eta=20,
        mutpb=0.3,
        cxpb=0.7)
    pop, hof, log, hst = opt.run(max_ngen=args.max_ngen,
        offspring_size=args.offspring_size)
#         continue_cp=args.continu,
#         cp_filename=args.checkpoint,
#         cp_frequency=1)
    if global_rank == 0: # only record root process
        fn = time.strftime("_%d_%b_%Y")
        fn = fn + ".pkl"
        output = open("best_indvs_final"+fn, 'wb')
        pickle.dump(best_indvs, output)
        output.close()
        output = open("log"+fn, 'wb')
        pickle.dump(log, output)
        output.close()
        output = open("hst"+fn, 'wb')
        pickle.dump(hst, output)
        output.close()
        output = open("hof"+fn, 'wb')
        pickle.dump(hof, output)
        output.close()
        print ('Hall of fame: ', hof, '\n')
        print ('last log: ', log, '\n')
        print ('History: ', hst, '\n')
        print ('Best individuals: ', best_indvs, '\n')
if __name__ == '__main__':
    import multiprocessing as mp
    pool = mp.Pool(10)
#     pool = None
    import subprocess
    if global_rank == 0:
        p = subprocess.Popen(["sh","watch_gpu_util.sh"])#subprocess.Popen(["python", "/gpfs/alpine/scratch/zladd/nro106/axonproj/benchmarking/GPU_genetic_alg/python/monitor_gpu.py"])
    datafn = 'main.prof'
    prof = cProfile.Profile()
    retval = prof.runcall(main, pool)
    prof.dump_stats(datafn)
#     main(pool)
#     if global_rank == 0:
#         p.kill()
#         kill(p.pid)
    logging.info("absolute end : " + str(time.time()))
    exit()
