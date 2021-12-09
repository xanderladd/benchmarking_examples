import struct
import numpy as np
import pandas as pd
import os 
import shutil

bad_dirs=['.ipynb_checkpoints']
def check_and_delete(curr_path):
    try:
        ls = os.listdir( curr_path)
    except FileNotFoundError:
        print(" NO such folder ", curr_path)
        return 
    y_or_n = input("DELETING {} it has {} in it -- is that ok (Y/N) ?".format(curr_path, ls))
    free = False
    while not free:
        if y_or_n == 'Y':
            print("deleting (seriously) ..", curr_path)
            shutil.rmtree(curr_path)
            free = True
        elif y_or_n == 'N':
            print("ignoring...", curr_path)
            free = True
        else:
            y_or_n = input("choose either Y or N")

for folder in os.listdir("outputs"):
    folder_bad = np.array([True for bad_dir in bad_dirs if folder in bad_dir]).any()
    folder_path = os.path.join("outputs", folder)

    if not os.path.isdir(folder_path) or folder_bad:
        continue
    slurms = [file for file in os.listdir(folder_path) if ".out" in file]
    logs = [file for file in os.listdir(folder_path) if ".log" in file and "gpu" not in file]

    
    if len(slurms) < 1 and len(logs) < 1:
        print("DELETING: ",folder_path, os.listdir(folder_path))
        check_and_delete(folder_path)
        continue
    
    try:
        curr_slurm = os.path.join(folder_path,slurms[0])

        with open(curr_slurm, "r") as fp:
            count = 0
            for line in fp:
                if "exists... exiting...." in line:
                    print("DELETING REPEAT -->", curr_slurm )

                    check_and_delete(folder_path)

                count += 1
                if count > 100:
                    break
    except IndexError:
        print("no slurm for ", folder_path)
        continue
#     exists... exiting....


