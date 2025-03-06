#!/usr/bin/env python
# Created on 2018/12
# Author: Kaituo XU

import argparse
import json
import os
from pathlib import Path

import librosa

# set paths
home_dir = str(Path.home())
work_dir = os.path.join(home_dir, 'code', 'repo', 'tasnet', 'egs', 'wsj0')
if os.getcwd() != work_dir:
    os.chdir(work_dir)
print('current path: {}'.format(os.getcwd()))

def preprocess_one_dir(in_dir, out_dir, out_filename, sample_rate=8000):
    file_infos = []
    in_dir = os.path.abspath(in_dir)
    wav_list = os.listdir(in_dir)
    for wav_file in wav_list:
        if not wav_file.endswith('.wav'):
            continue
        wav_path = os.path.join(in_dir, wav_file)
        samples, _ = librosa.load(wav_path, sr=sample_rate)
        file_infos.append((wav_path, len(samples)))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, out_filename + '.json'), 'w') as f:
        json.dump(file_infos, f, indent=4)

def preprocess(args):

    sr_type = {8000:'8k', 16000: '16k'}
    for data_type in ['tr', 'cv', 'tt']:
        for speaker in ['mix', 's1', 's2']:
            sr_folder = sr_type[args.sample_rate]
            print(f'{data_type}:{sr_folder}:{speaker} ...')
            in_dir = os.path.join(args.in_dir, data_type, speaker)
            out_dir = os.path.join(args.out_dir, sr_folder, data_type)
            preprocess_one_dir(in_dir, out_dir, speaker, sample_rate=args.sample_rate)

def parse_args():

    usage = "WSJ0 data preprocessing"
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--in-dir', type=str, default=None,
                        help='Directory path of wsj0 including tr, cv and tt')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Directory path to put output files')
    parser.add_argument('--sample-rate', type=int, default=8000,
                        help='Sample rate of audio file')
    args = parser.parse_args()
    return args                           


if __name__ == "__main__":

    # runtime mode
    args = parse_args()

    # # interactive mode
    # args = argparse.ArgumentParser()
    # args.in_dir = '/home/users/zge/data1/datasets/WSJ0/wav16k/min'
    # args.out_dir = 'data'
    # args.sample_rate = 16000

    print(args)
    preprocess(args)
