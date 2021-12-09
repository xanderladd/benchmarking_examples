import GPUtil
import sys
import logging
import time
from mpi4py import MPI
from io import StringIO
comm = MPI.COMM_WORLD
global_rank = comm.Get_rank()
size = comm.Get_size()

exit()
import logging.config
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})

import logging.handlers
import os
if global_rank == 0:
    filename = "GPUTilLogs/GPUTil.log"
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
all_time = 0
if global_rank == 0:
        while all_time < 200:
            time.sleep(5)
            old_stdout = sys.stdout
            mystdout = StringIO()
            sys.stdout = mystdout
            GPUtil.showUtilization()
            sys.stdout = old_stdout
            logging.info(str(time.time())+" : GPU UTIL : " + mystdout.getvalue())
            all_time += 5

