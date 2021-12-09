import time
from neuron import coreneuron
from neuron import h
from neuron.units import ms, mV
coreneuron.gpu = True


coreneuron.enable = True


h.load_file("hoc_files/runModel.hoc")

h.cvode.cache_efficient(1)
# h.cvode.use_fast_imem(1)

# h.nrnmpi_init()

start = time.time()
h.cADpyr232_L5_TTPC1_0fb1ca4724()
end =time.time()
print(f'time to create a single mode: {end-start}')

start = time.time()
[h.cADpyr232_L5_TTPC1_0fb1ca4724() for _ in range(20)]
end =time.time()

print(f'time to create 20 models: {end-start}')


exit()