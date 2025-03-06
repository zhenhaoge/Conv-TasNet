#!/usr/bin/env python
# Created on 2025-01-28
# Author: Zhenhao Ge

import argparse
import os
from pathlib import Path
import numpy as np

# set paths
# home_dir = str(Path.home())
# work_dir = os.path.join(home_dir, 'code', 'repo', 'tasnet', 'egs', 'wsj0')
# if os.getcwd() != work_dir:
#     os.chdir(work_dir)
print('current path: {}'.format(os.getcwd()))

def parse_args():

    usage = "get statistics after training and evaluation"
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--train-log', type=str, default=None, help='training log')
    parser.add_argument('--evaluate-log', type=str, default=None, help='evaluation log')

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # interactive mode
    args = argparse.ArgumentParser()

    # for self-trained models
    work_dir = os.getcwd()
    exp_dir = 'train_r8000_N256_L20_B256_H512_P3_X8_R4_C2_gLN_causal0_relu_epoch100_half1_norm5_bs6_worker12_adam_lr1e-3_mmt0_l20_train'
    args.train_log = os.path.join(work_dir, 'exp', exp_dir, 'train.log.20250305')
    args.evaluate_log = os.path.join(work_dir, 'exp', exp_dir, 'evaluate.log.20250305')

    # check file existence
    if args.train_log:
        assert os.path.isfile(args.train_log), f'train log: {args.train_log} does not exist!'
    if args.evaluate_log:    
        assert os.path.isfile(args.evaluate_log), f'evaluate log: {args.evaluate_log} does not exist!'

    # read from the train log
    if args.train_log:
        lines = open(args.train_log, 'r').readlines()
        lines = [l for l in lines if 'Valid Summary' in l]
        nepochs = len(lines)
        print(f'{nepochs} epochs completed')

        durations = [0.0 for _ in range(nepochs)]
        valloss = [0.0 for _ in range(nepochs)]
        for i in range(nepochs):
            parts = lines[i].split('|')
            durations[i] = float(parts[2].strip()[5:-1])
            valloss[i] = float(parts[3].strip().split()[-1])

        total_dur_hrs = sum(durations)/3600

        print(f'total duration (hrs): {total_dur_hrs:.2f}')
        print(f'final val loss: {valloss[-1]}')

    # read from the evaluation log
    SDRi, SISNRi = {}, {}
    if args.evaluate_log:
        lines = open(args.evaluate_log, 'r').readlines()
        for i in range(len(lines)):
            if lines[i][:3] == 'Utt':
                uid = lines[i].split()[-1].rstrip()
                sdri = float(lines[i+1].split('=')[-1].strip())
                sisnri = float(lines[i+2].split('=')[-1].strip())
                SDRi[uid] = sdri
                SISNRi[uid] = sisnri

    SDRi_mean = np.mean(list(SDRi.values()))
    SISNRi_mean = np.mean(list(SISNRi.values()))
    print(f'mean SDRi: {SDRi_mean:.2f}')
    print(f'mean SISNRi: {SISNRi_mean:.2f}')