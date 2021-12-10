# benchmarking_examples

## Installation
 - have CoreNeuron installed 
 - `cd src` then `make clean` then `make`
  - **prerequsites** g++ > 7.5.0 , Cuda > 10.1.3.1 (I used 11.1)
  - this will give you neuroGPU executable in `benchmarking_examples/bin`
 - other **prequsites**
  - pip install -r requirements.txt
  
## Setup
  - there are many different way to launch neuroGPU and core_neuron. In order to configure these experiments you'll need to edit two files:
      - `launch_neuroGPU.sh`
      - `launch core_neuron.sh`
  - you can see that I had these configured to use a complex version of `Srun` due to some problems linking core_neuron with the local version of python on CORI. 
  - you'll want to delete that and add something of the order `mpiexec -n <num_process> x86_64/special -mpi -python core_neuron_example.py` or   `mpiexec -n <num_process> nrniv -mpi -python neuroGPU_example.py`.
  - $num_process and $cores will be specific to your machine.

## Launching Experiments
  - Set `method` in experiment_design.txt to be `core_neuron` or `neuroGPU` and then do `sh launch_experiments.sh`. **Note**, this will overwrite my benchmarking results in  `neuroGPU/logs` or `core_neuron/logs`.
     - you can either move those logs or just check them out from main branch if you'd still like to see them.
     
## Gathering results
  - just run `python collect_experiments.py` to parse the logs.
 
 
 
 
 
 
