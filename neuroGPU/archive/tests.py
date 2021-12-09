from mpi4py import MPI
import os
import multiprocessing
import time
import numpy as np
import tst2
import subprocess
comm = MPI.COMM_WORLD
global_rank = comm.Get_rank()
size = comm.Get_size()

#print(os.environ)

print(global_rank,size, "GRANK AND SIZE")
# importing os module 
import os
  
# Get the number of CPUs
# in the system
# using os.cpu_count() method
print("Number of CPUs:", os.cpu_count())
  
# Get the set of CPUs
# on which the calling process
# is eligible to run. uing
# os.sched_getaffinity() method
# 0 as PID represnts the
# calling proces
pid = 0
affinity = os.sched_getaffinity(pid)
  
# Print the result
print("Process is eligibl to run on:", affinity)

data = np.repeat(np.ones(shape=(1,10000,8)), 100, axis=0)

p_objects = []
for i in range(10):
    p_objects.append(subprocess.Popen(['ls']))
for p_object in p_objects:
    p_object.wait()
tst2.main()