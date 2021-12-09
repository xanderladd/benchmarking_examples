#!/bin/bash



offspring_size=$1
trials=$2
proccesses=$(( $3 * 8 )) # salloc 8 processes per node -- 1 for every gpu. Then will launch method across each gpu


: '
NOTE: Here is my command to launch core neuron, I was running into issues with paths, so I use this convoluted command
you should put your own in here. youll want to keep the -- core_neuron_example.py --offspring_size $offspring_size --trials $trials -- part




YOUR CODE GOES HERE (below)
'


cd core_neuron

module purge
module load cgpu
module load gcc cuda cmake/3.21.3 openmpi hpcsdk/20.11 
module load cray-python/3.7.3.2
module load craype



export PYTHONPATH=/global/project/projectdirs/m2043/$USER/install/lib/python:$PYTHONPATH
export PATH=/global/project/projectdirs/m2043/$USER/install/bin:$PATH
export CC=pgcc
export LD_LIBRARY_PATH=/global/project/projectdirs/m2043/$USER/install/lib:$LD_LIBRARY_PATH
export CXX=pgc++



srun -n $proccesses --export=PATH=/global/project/projectdirs/m2043/$USER/install/bin:$PATH,PYTHONPATH=/global/project/projectdirs/m2043/$USER/install/lib/python:$PYTHONPATH,LD_LIBRARY_PATH=/global/project/projectdirs/m2043/$USER/install/lib:$LD_LIBRARY_PATH,CC=pgcc,CXX=pgc++  x86_64/special -python  -mpi core_neuron_example.py --offspring_size $offspring_size --trials $trials