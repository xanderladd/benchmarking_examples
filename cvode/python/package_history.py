import pickle

with open('outputs/trial_4N_1000O_seed33/hst4N_1000O.pkl','rb') as f:
    history = pickle.load(f)
    
    
import pdb; pdb.set_trace()

pop_size=1000
generations_to_save = [2,3,4,5,6,145,145,146,147,148,149]
res = {}
for gen in generations_to_save:
    curr_params = []
    for pop in range(pop_size):
        idx = (gen * pop_size) + pop
        curr_params.append(history.genealogy_history[idx])
    res[gen] = curr_params

import pdb; pdb.set_trace()

with open('rehash_history','wb') as f:
    pickle.dump(res,f)