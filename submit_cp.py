#!/usr/bin/env python3
import time
import subprocess
from itertools import product

def write_bsub(command, model_name, sys_mem='64GB', hours='05', minutes='00',
    gpu_queue='gpuv100'):

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
        '###BSUB -R "select[gpu32gb]"\n' \
        '### -- Specify the output and error file. %J is the job-id --\n' \
        '### -- -o and -e mean append, -oo and -eo mean overwrite --\n' \
        '#BSUB -o {out_file}\n' \
        '#BSUB -e {error_file}\n' \
        '# -- end of LSF options --\n' \
        '# -- start of user input --\n' \
        'source ~/.virtualenvs/tfde/bin/activate\n' \
        'cd ~/TFDE\n' \
        '{command}\n' \
        ''.format(
            gpu_queue=gpu_queue,
            job_name=model_name,
            sys_mem=sys_mem,
            hours=hours,
            minutes=minutes,
            out_file='misc_files/' + model_name + '.out',
            error_file='misc_files/' + model_name + '.err',
            command=command)

    bsub_file_name = 'misc_files/' + model_name + '.bsub'
    bsub_file = open(bsub_file_name, 'w')
    bsub_file.write(bsub_string)
    bsub_file.close()
    time.sleep(1)

    return bsub_file_name

if __name__ == '__main__':

    #datasets = ['hepmass']
    #K_range = [2, 28, 55, 82, 109, 136, 163, 190, 217, 244]
    datasets = ['miniboone']
    K_range = [2, 15, 28, 41, 54, 67, 80, 93, 106, 120]
    mini_batch_sizes = [64]
    learning_rates = [3e-4]
    n_epochs = [300]
    subsample_sizes = [28000]
    optimal_order = [1]
    early_stopping = [1]
    n_runs = range(3)

    train_time = '10'
    queue = 'gpuv100'

    all_runs = product(
        datasets, K_range, mini_batch_sizes, learning_rates, n_epochs, 
        subsample_sizes, optimal_order, early_stopping, n_runs)

    for dataset, K, mb_size, lr, epochs, subsample_size, order, early_stop, run in all_runs:
        model_name = 'CP_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
            dataset, K, mb_size, lr, epochs, subsample_size, order, early_stop, run)

        command = './train_cp.py --dataset {dataset} --K {K} ' \
            '--mb_size {mb_size} --lr {lr} --epochs {epochs} ' \
            '--subsample_size {subsample_size} --optimal_order {order} ' \
            '--early_stopping {early_stop} {model_name}' \
            ''.format(
                dataset=dataset, K=K, mb_size=mb_size, lr=lr, epochs=epochs,
                subsample_size=subsample_size, order=order,
                early_stop=early_stop, model_name=model_name)
        
        bsub_file_name = write_bsub(
            command, model_name, gpu_queue=queue, hours=train_time)
    
        bsub_file = open(bsub_file_name, 'rb')
        subprocess.run(['bsub'], stdin=bsub_file)
        bsub_file.close()
