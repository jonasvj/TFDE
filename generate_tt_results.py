#!/usr/bin/env python3
import time
import subprocess
from itertools import product

def write_bsub(dataset, mb_size, lr, K, n_epochs, run, result_file, 
    sys_mem='16GB', hours='05', minutes='00', gpu_queue='gpuv100'):

    model_name = 'TT_{}_{}_{}_{}_{}_{}'.format(
        dataset, mb_size, lr, K, n_epochs, run)
    
    bsub_string = \
        '#!/bin/sh\n' \
        '### General options\n' \
        '### –- specify queue --\n' \
        '#BSUB -q {gpu_queue}\n' \
        '### -- set the job Name --\n' \
        '#BSUB -J {job_name}\n' \
        '### -- ask for number of cores (default: 1) --\n' \
        '#BSUB -n 1\n' \
        '### -- Select the resources: 1 gpu in exclusive process mode --\n' \
        '#BSUB -gpu "num=1:mode=exclusive_process"\n' \
        '### -- set walltime limit: hh:mm --\n' \
        '#BSUB -W {hours}:{minutes}\n' \
        '# request system-memory\n' \
        '#BSUB -R "rusage[mem={sys_mem}]"\n' \
        '### -- Specify the output and error file. %J is the job-id --\n' \
        '### -- -o and -e mean append, -oo and -eo mean overwrite --\n' \
        '#BSUB -o {out_file}\n' \
        '#BSUB -e {error_file}\n' \
        '# -- end of LSF options --\n' \
        '# -- start of user input --\n' \
        'source ~/.virtualenvs/tfde/bin/activate\n' \
        'cd ~/TFDE\n' \
        './train_tt.py {dataset} {mb_size} {lr} {K} {epochs} {run} {result_file}\n' \
        ''.format(
            gpu_queue=gpu_queue,
            job_name=model_name,
            sys_mem=sys_mem,
            hours=hours,
            minutes=minutes,
            out_file='misc_files/' + model_name + '.out',
            error_file='misc_files/' + model_name + '.err',
            dataset=dataset,
            mb_size=mb_size,
            lr=lr,
            K=K,
            epochs=epochs,
            run=run,
            result_file=result_file)
    
    bsub_file_name = 'misc_files/' + model_name + '.bsub'
    bsub_file = open(bsub_file_name, 'w')
    bsub_file.write(bsub_string)
    bsub_file.close()
    time.sleep(1)

    return bsub_file_name

if __name__ == '__main__':
    datasets = ['gas']
    mini_batch_sizes = [8192]
    learning_rates = [3e-4]
    K_range = range(10, 11)
    n_epochs = [500]
    n_runs = range(1)

    # The directory ~/TFDE/misc_files must exist
    # The results file (or another specified results file) must also exist
    # (~/TFDE/results/tt_results.txt)
    result_file = 'results/tt_results.txt'

    all_runs = product(
        datasets, mini_batch_sizes, learning_rates, K_range, n_epochs, n_runs)

    for dataset, mb_size, lr, K, epochs, run in all_runs:

        bsub_file_name = write_bsub(
            dataset, mb_size, lr, K, epochs, run, result_file, hours='02')
        
        bsub_file = open(bsub_file_name, 'rb')
        subprocess.run(['bsub'], stdin=bsub_file)
        bsub_file.close()
