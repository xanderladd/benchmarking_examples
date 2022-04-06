import os 
import shutil

ignore = ['.ipynb_checkpoints']
for folder in os.listdir('outputs'):
    folder_path = os.path.join('outputs', folder)
    if (not 'seed' in folder) or (not os.path.isdir(folder_path) or folder in ignore): continue
    files = os.listdir(folder_path)
    delete=True
    for file in files:
        if 'hof' in file: delete = False
    if delete:        
        print(files)
        response = input(f'\n are you sure you want to delete {folder}? y/n \n')
        if response == 'y':
            # print('rm -r folder_path')
            shutil.rmtree(folder_path)
        