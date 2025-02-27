#!/usr/bin/env python
#
# check the duration of train/val/test dataset
#
# Created on 2025-01-28
# Author: Zhenhao Ge

import argparse
import os
import json
from pathlib import Path

# set paths
home_dir = str(Path.home())
# work_dir = os.path.join(home_dir, 'code', 'repo', 'tasnet', 'egs', 'wsj0')
work_dir = os.path.join(home_dir, 'code', 'repo', 'tasnet', 'egs', 'Echo2Mix')
if os.getcwd() != work_dir:
    os.chdir(work_dir)    
print('current path: {}'.format(os.getcwd()))

def get_dur(jsonfile, fs=8000):

    with open(jsonfile, 'r') as f:
        data = json.load(f)

    nentries = len(data)
    print(f'#entries: {nentries}')
    
    dur_hrs = sum([d[1] for d in data]) / fs / 3600
    print(f'{dur_hrs:.2f} hours from {jsonfile}')

    return dur_hrs, nentries

def parse_args():

    usage = "check the duration of tranining, validation and testing datasets"
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--train-file', type=str, help='mixed speaker json file for training')
    parser.add_argument('--val-file', type=str, help='mixed speaker json file for validation')
    parser.add_argument('--test-file', type=str, help='mixed speaker json file for testing')

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # # interactive mode
    # args = argparse.ArgumentParser()

    # args.train_file = os.path.join(work_dir, 'data', 'train', 'mix.json')
    # args.val_file = os.path.join(work_dir, 'data', 'val', 'mix.json')
    # args.test_file = os.path.join(work_dir, 'data', 'test', 'mix.json')

    # # check file existence
    # assert os.path.isfile(args.train_file), f'train file: {args.train_file} does not exist!'
    # assert os.path.isfile(args.val_file), f'val file: {args.val_file} does not exist!'
    # assert os.path.isfile(args.test_file), f'test file: {args.test_file} does not exist!'

    fs = 16000

    dur_hrs_train, nentries_train = get_dur(args.train_file, fs=fs)
    dur_hrs_val, nentries_val = get_dur(args.val_file, fs=fs)
    dur_hrs_test, nentries_test = get_dur(args.test_file, fs=fs)

    print(f'train: {dur_hrs_train:.2f} hours with {nentries_train} files')

 

