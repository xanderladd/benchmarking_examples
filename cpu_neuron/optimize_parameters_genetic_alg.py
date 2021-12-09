import bluepyopt as bpop


# when things are setup, make this a conditional import
import hoc_evaluator_allen as hoc_ev
#import hoc_evaluator as hoc_ev

import bluepyopt.deapext.algorithms as algo
import pickle
import time
import numpy as np
from datetime import datetime
import argparse
import os
import sys
import argparse
import textwrap
import logging
import logging

from mpi4py import MPI

comm = MPI.COMM_WORLD
# global_rank = comm.Get_rank()
size = int(os.environ['SLURM_NNODES'])
global_rank = int(os.environ['SLURM_PROCID'])


import logging.config
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})

import logging.handlers
import os
print(global_rank,  ": MY GRANK")
if global_rank == 0:
    os.makedirs('/global/cscratch1/sd/zladd/benchmarking/neuron_genetic_alg/runTimeLogs/', exist_ok=True)
    filename = "runTimeLogs/runTime.log"
    # your logging setup
    should_roll_over = os.path.isfile(filename)
    handler = logging.handlers.RotatingFileHandler(filename, mode='w', backupCount=15, delay=True)
    if should_roll_over:  # log already exists, roll over!
        handler.doRollover()
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=filename, level=logging.DEBUG)
    #logging.config.dictConfig({'filename':filename, 'level': logging.DEBUG, 'disable_existing_loggers': True })

    logging.info("absolute start : " + str(time.time()) + " from rank" + str(global_rank))

else:
    filename = None
    logger = None
    
    


#if size > 1:
logging.info("USING MPI : TRUE")

gen_counter = 0
best_indvs = []
cp_freq = 1
old_update = algo._update_history_and_hof





def create_optimizer(args, logger):
    '''returns configured bluepyopt.optimisations.DEAPOptimisation'''
    if args.ipyparallel:
        from ipyparallel import Client
        rc = Client(profile=os.getenv('IPYTHON_PROFILE'))
        logger.debug('Using ipyparallel with %d engines', len(rc))

        lview = rc.load_balanced_view()
        dview = rc.direct_view()
        def mapper(func, it):
            dview.map(os.chdir, [os.getcwd()]*len(rc.ids))
            start_time = datetime.now()
            ret = dview.map_sync(func, it)
            print('Generation took', datetime.now() - start_time)
            if global_rank == 0:
                logging.info(f'Generation took {datetime.now() - start_time}')
            return ret

        map_function = mapper
    else:
        map_function = None

    evaluator = hoc_ev.hoc_evaluator(args, logger)
    seed = os.getenv('BLUEPYOPT_SEED', args.seed)
    opt = bpop.optimisations.DEAPOptimisation(
        evaluator=evaluator,
        map_function=map_function,
        seed=seed,
        eta=20,
        mutpb=0.3,
        cxpb=0.7)
    print(map_function)

    return opt


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
    parser.add_argument('--continu', action="store_false", default=False)
    parser.add_argument('--checkpoint', required=False, default=None,
                        help='Checkpoint pickle to avoid recalculation')
    parser.add_argument('--offspring_size', type=int, required=False, default=2,
                        help='number of individuals in offspring')
    parser.add_argument('--max_ngen', type=int, required=False, default=2,
                        help='maximum number of generations')
    parser.add_argument('--responses', required=False, default=None,
                        help='Response pickle file to avoid recalculation')
    parser.add_argument('--analyse', action="store_true")
    parser.add_argument('--compile', action="store_true")
    parser.add_argument('--hocanalyse', action="store_true")
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed to use for optimization')
    parser.add_argument('--ipyparallel', action="store_true", default=False,
                        help='Use ipyparallel')
    parser.add_argument('--n_stims', type=int, required=False, default=1,
                        help='number of stims to optimize over')
    parser.add_argument('--n_sfs', type=int, required=False, default=0,
                        help='number of score functions to use')
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

def main():
    args = get_parser().parse_args()
    algo._update_history_and_hof = my_update
#     logging.basicConfig(level=(logging.WARNING,
#                                 logging.INFO,
#                                 logging.DEBUG)[args.verbose],
#                                 stream=sys.stdout)
    opt = create_optimizer(args, logger)
    pop, hof, log, hst = opt.run(max_ngen=args.max_ngen,
        offspring_size=args.offspring_size,
        continue_cp=args.continu,
        cp_filename=args.checkpoint,
        cp_frequency=1)

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
    print ('log: ', log, '\n')
    print ('History: ', hst, '\n')
    print ('Best individuals: ', best_indvs, '\n')
if __name__ == '__main__':
    main()
