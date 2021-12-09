from mpi4py import MPI
import os

comm = MPI.COMM_WORLD
global_rank = comm.Get_rank()
size = comm.Get_size()
print(os.environ['SLURM_PROCID'])
print(global_rank,  ": MY GRANK")