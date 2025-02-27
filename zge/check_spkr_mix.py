#!/usr/bin/env python
#
# conclusions:
#   - training and validation use the same speaker set from si_tr_s
#   - testing uses speaker set from ['si_dt_05', 'si_et_05']
#   - there are no overlap between the speakers in (train, val) and the speaker in test
#
# Created on 2025-01-28
# Author: Zhenhao Ge

import argparse
import os
from pathlib import Path

# set paths
home_dir = str(Path.home())
work_dir = os.path.join(home_dir, 'code', 'repo', 'tasnet', 'egs', 'wsj0')
if os.getcwd() != work_dir:
    os.chdir(work_dir)
print('current path: {}'.format(os.getcwd()))

def get_cat_n_spkr(spkr_file):

    lines = open(spkr_file, 'r').readlines()
    nlines = len(lines)

    cats0 = ['' for _ in range(nlines)]
    cats1 = ['' for _ in range(nlines)]
    spkrs0 = ['' for _ in range(nlines)]
    spkrs1 = ['' for _ in range(nlines)]
    for i in range(nlines):

        parts = lines[i].rstrip().split()
        wavfile0, wavfile1 = parts[0], parts[2]
        cats0[i], spkrs0[i] = wavfile0.split(os.sep)[1:3]
        cats1[i], spkrs1[i] = wavfile1.split(os.sep)[1:3]

    return cats0, cats1, spkrs0, spkrs1

def parse_args():

    usage = "check speaker mix in training, validation and testing"
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--train-file', type=str, help='speaker mix file for training')
    parser.add_argument('--val-file', type=str, help='speaker mix file for validation')
    parser.add_argument('--test-file', type=str, help='speaker mix file for testing')

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # # interactive mode
    # args = argparse.ArgumentParser()

    # root_dir = os.path.dirname(os.path.dirname(work_dir))
    # args.train_file = os.path.join(root_dir, 'tools', 'create-speaker-mixtures', 'mix_2_spk_tr.txt')
    # args.val_file = os.path.join(root_dir, 'tools', 'create-speaker-mixtures', 'mix_2_spk_cv.txt')
    # args.test_file = os.path.join(root_dir, 'tools', 'create-speaker-mixtures', 'mix_2_spk_tt.txt')

    # check file existence
    assert os.path.isfile(args.train_file), f'train file: {args.train_file} does not exist!'
    assert os.path.isfile(args.train_file), f'val file: {args.val_file} does not exist!'
    assert os.path.isfile(args.train_file), f'test file: {args.test_file} does not exist!'

    cats0_train, cats1_train, spkrs0_train, spkrs1_train = get_cat_n_spkr(args.train_file)
    cats0_val, cats1_val, spkrs0_val, spkrs1_val = get_cat_n_spkr(args.val_file)
    cats0_test, cats1_test, spkrs0_test, spkrs1_test = get_cat_n_spkr(args.test_file)

    spkr0_list_train = sorted(set(spkrs0_train))
    spkr1_list_train = sorted(set(spkrs1_train))
    
    spkr0_list_val = sorted(set(spkrs0_val))
    spkr1_list_val = sorted(set(spkrs1_val))

    spkr0_list_test = sorted(set(spkrs0_test))
    spkr1_list_test = sorted(set(spkrs1_test))

    print(f'categories for speaker 0 in training: {sorted(set(cats0_train))}')
    print(f'categories for speaker 1 in training: {sorted(set(cats1_train))}')

    print(f'categories for speaker 0 in validation: {sorted(set(cats0_val))}')
    print(f'categories for speaker 1 in validation: {sorted(set(cats1_val))}')

    print(f'categories for speaker 0 in test: {sorted(set(cats0_test))}')
    print(f'categories for speaker 1 in test: {sorted(set(cats1_test))}')

    assert spkr0_list_train == spkr1_list_train, 'list of spkr0 and list of spkr1 in train are not the same!'
    assert spkr0_list_val == spkr1_list_val, 'list of spkr0 and list of spkr1 in val are not the same!'
    assert spkr0_list_test == spkr1_list_test, 'list of spkr0 and list of spkr1 in test are not the same!'

    spkr_list_train = spkr0_list_train
    spkr_list_val = spkr0_list_val
    spkr_list_test = spkr0_list_test

    del spkr0_list_train, spkr1_list_train
    del spkr0_list_val, spkr1_list_val
    del spkr0_list_test, spkr1_list_test

    assert spkr_list_train == spkr_list_val, 'train and val speakers are not the same!'

    print(f'#spkrs in train/val: {len(spkr_list_train)}')
    print(f'#spkrs in test: {len(spkr_list_test)}')
