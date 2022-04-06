from neuron import h

# importing MPI or h.nrnmpi_init() must come before the first instantiation of ParallelContext()
h.nrnmpi_init()

pc = h.ParallelContext()

def f(x):
    """a function with no context that changes except its argument"""
    print('hi!')
    return x * x

pc.runworker() # master returns immediately, workers in an
               # infinite loop running jobs from bulletin board

s = 0
if pc.nhost() == 1:          # use the serial form
    for i in range(20):
        s += f(i)
else:                        # use the bulleting board form
    for i in range(20):      # scatter processes
        pc.submit(f, i)      # any context needed by f had better be
                             # the same on all hosts
    while pc.working():      # gather results
        s += pc.pyret()      # the return value for the executed function

print(s)
pc.done()                    # tell workers to quit