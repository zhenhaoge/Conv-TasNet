#!/usr/bin/env python
# Created on 2018/12
# Author: Kaituo XU

import argparse
import json
import os
import glob
from pathlib import Path

import librosa

# set paths
home_dir = str(Path.home())
work_dir = os.path.join(home_dir, 'code', 'repo', 'tasnet', 'egs', 'Echo2Mix')
if os.getcwd() != work_dir:
    os.chdir(work_dir)
print('current path: {}'.format(os.getcwd()))

def preprocess(args):
    spks = ['mix', 'spk1', 'spk2']
    for data_type in ['train', 'val', 'test']:

        print(f'data type: {data_type}')

        # set input and output dirs
        in_dir = os.path.abspath(os.path.join(args.in_dir, data_type))
        out_dir = os.path.abspath(os.path.join(args.out_dir, data_type))
        os.makedirs(out_dir, exist_ok=True)

        # get wav file list
        wav_list = sorted(glob.glob(os.path.join(in_dir, '**', '*.wav'), recursive=True))

        # sanity check
        nwavs = len(wav_list)
        assert nwavs % len(spks) == 0, 'wav files are not evenly distributed across speakers!'

        file_infos = {spk:[] for spk in spks}
        for i in range(nwavs):

            if i % 1000 == 0:
                print(f'processing wav file [{i}, {min(i+1000, nwavs)}), {nwavs} total ...')
            
            wav_file = wav_list[i]
            wav_filename = os.path.basename(wav_file)
            spk = os.path.splitext(wav_filename)[0].split('_')[0]
            samples, _ = librosa.load(wav_file, sr=args.sample_rate)
            file_infos[spk].append((wav_file, len(samples)))

        for spk in spks:
            jsonfile = os.path.join(out_dir, f'{spk}.json')
            with open(jsonfile, 'w') as f:
                json.dump(file_infos[spk], f, indent=4)

def parse_args():

    usage = "Echo2Mix data preprocessing"
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--in-dir', type=str, default=None,
                        help='Directory path of Echo2Mix including train, val and test')
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
    # args.in_dir = '/home/users/zge/data1/datasets/Echo2Mix'
    # args.out_dir = 'data'
    # args.sample_rate = 16000

    print(args)
    preprocess(args)
