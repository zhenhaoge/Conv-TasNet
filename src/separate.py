#!/usr/bin/env python

# Created on 2018/12
# Author: Kaituo XU

import argparse
import os
from pathlib import Path
import librosa
import torch
import soundfile as sf
import numpy as np
import shutil
import pandas as pd

# set paths
# home_dir = str(Path.home())
# work_dir = os.path.join(home_dir, 'code', 'repo', 'tasnet', 'egs', 'wsj0')
# if os.getcwd() != work_dir:
#     os.chdir(work_dir)
work_dir = os.getcwd()
print('current path: {}'.format(work_dir))

from data import EvalDataLoader, EvalDataset, AudioDataLoader, AudioDataset
from pit_criterion import cal_loss
from conv_tasnet import ConvTasNet
from utils import remove_pad
from metrics import cal_SDRi, cal_SISNRi

def parse_args():

    parser = argparse.ArgumentParser('Separate speech using Conv-TasNet')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model file created by training')
    parser.add_argument('--mix_dir', type=str, default=None,
                        help='Directory including mixture wav files')
    parser.add_argument('--mix_json', type=str, default=None,
                        help='Json file including mixture wav files')
    parser.add_argument('--out_dir', type=str, default='exp/result',
                        help='Directory putting separated wav files')
    parser.add_argument('--use_cuda', type=int, default=0,
                        help='Whether use GPU to separate speech')
    parser.add_argument('--sample_rate', default=8000, type=int,
                        help='Sample rate')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size')
    parser.add_argument('--num_batches', default=1, type=int,
                        help='Number of batches to be separated')
    args = parser.parse_args()
    return args


def separate(args):
    if args.mix_dir is None and args.mix_json is None:
        print("Must provide mix_dir or mix_json! When providing mix_dir, "
              "mix_json is ignored.")

    # Load model
    model = ConvTasNet.load_model(args.model_path)
    # print(model)
    model.eval()
    if args.use_cuda:
        model.cuda()

    # Load data
    eval_dataset = EvalDataset(args.mix_json,
                               batch_size=args.batch_size,
                               sample_rate=args.sample_rate)
    eval_loader =  EvalDataLoader(eval_dataset, batch_size=1)

    data_dir = os.path.dirname(args.mix_json)
    dataset = AudioDataset(data_dir, args.batch_size,
                           sample_rate=args.sample_rate, segment=-1, cv_maxlen=float('inf'))
    data_loader = AudioDataLoader(dataset, batch_size=1, num_workers=2)

    os.makedirs(args.out_dir, exist_ok=True)

    # get dataset name from mixed json file name
    dataset_name = args.mix_json.split(os.sep)[-4]
    print(f'dataset name: {dataset_name}')

    # def write(inputs, filename, sr=args.sample_rate):
    #     librosa.output.write_wav(filename, inputs, sr)# norm=True)

    metrics = {'SDRi':[], 'SISNRi':[]}
    with torch.no_grad():
        for i, (eval_data, data) in enumerate(zip(eval_loader, data_loader)):

            # # check the first batch
            # print('stop at the first batch')
            # break

            if i >= args.num_batches:
                print(f'{args.num_batches} batches separated ({args.batch_size} samples per batch).')
                break

            # Get batch data
            padded_mixture, mix_lengths, filenames = eval_data
            padded_mixture2, _, padded_source = data
            assert torch.equal(padded_mixture, padded_mixture2), "data not aligned!"

            if args.use_cuda:
                padded_mixture = padded_mixture.cuda()
                mix_lengths = mix_lengths.cuda()
                padded_source = padded_source.cuda()

            # Remove padding and flat
            mixture = remove_pad(padded_mixture, mix_lengths)
            source = remove_pad(padded_source, mix_lengths)

            # Forward
            estimate_source = model(padded_mixture)  # [B, C, T]
            loss, max_snr, estimate_source, reorder_estimate_source = \
                cal_loss(padded_source, estimate_source, mix_lengths)

            # # use regular estimate source
            # flat_estimate = remove_pad(estimate_source, mix_lengths)
            # estimate_source_nograd = estimate_source.detach()
            # flat_estimate = remove_pad(estimate_source_nograd, mix_lengths)

            # use reorder estimate source (needed for evaluation, so s{x}-original matched with s{x}-separated)
            # flat_estimate = remove_pad(reorder_estimate_source, mix_lengths)
            reorder_estimate_source_nograd = reorder_estimate_source.detach()
            flat_estimate = remove_pad(reorder_estimate_source_nograd, mix_lengths)

            # Write result
            for j, filename in enumerate(filenames):

                filename_ori = filenames[j]
                idx = i * args.batch_size + j
                fid = '{:04d}'.format(idx)
                filename = os.path.join(args.out_dir, f'{fid}_{os.path.basename(filename_ori).strip(".wav")}')

                # compute the evaluation metrics
                mix, src_ref, src_est = mixture[j], source[j], flat_estimate[j]
                avg_SDRi = cal_SDRi(src_ref, src_est, mix)
                avg_SISNRi = cal_SISNRi(src_ref, src_est, mix)
                print(f'{filename}, SDRi: {avg_SDRi:.2f}, SISNRi: {avg_SISNRi:.2f}')

                # append metrics
                metrics['SDRi'].append(avg_SDRi)
                metrics['SISNRi'].append(avg_SISNRi)
                assert len(metrics['SDRi']) == idx + 1, 'duplicated SDRi exist!'
                assert len(metrics['SISNRi']) == idx + 1, 'duplicated SISNRi exist!'

                ## write the mixed file
                # sf.write(filename + '.wav', mixture[j], args.sample_rate) # new method
                # write(mixture[j], filename) # old method (no longer compatible)

                # copy the mixed file
                shutil.copy(filename_ori, filename + '.wav')

                C = src_est.shape[0]
                for c in range(C):

                    # copy the source speaker file before mix (ground truth) for reference
                    if dataset_name == 'wsj0':
                        filename_spk_gt = filename_ori.replace('mix', f's{c+1}')
                        filename_spk_gt2 = filename + f'_s{c+1}_ori.wav'
                    elif dataset_name == 'Echo2Mix':
                        filename_spk_gt = os.path.join(filenames[j].replace('mix.wav', f'spk{c+1}_reverb.wav'))
                        filename_spk_gt2 = filename.replace('mix', f's{c+1}') + '.wav'
                    shutil.copy(filename_spk_gt, filename_spk_gt2)

                    # write the separated speaker file
                    normalized_audio = src_est[c] / np.max(np.abs(src_est[c]))
                    sf.write(filename+'_s{}.wav'.format(c+1), normalized_audio, args.sample_rate)

    df = pd.DataFrame(metrics)
    csvfile = os.path.join(args.out_dir, 'metrics.csv')
    df.to_csv(csvfile, index=True, index_label='idx')


if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # # interactive mode
    # args = argparse.ArgumentParser()

    # exp_folder = 'exp/train_r16000_N256_L20_B256_H512_P3_X8_R4_C2_gLN_causal0_relu_epoch100_half1_norm5_bs6_worker4_adam_lr1e-3_mmt0_l20_train'
    # args.model_path = os.path.join(work_dir, exp_folder, 'final.pth')
    # args.batch_size = 6

    # exp_folder = 'exp/train_titan12'
    # args.model_path = os.path.join(work_dir, exp_folder, 'final.pth')
    # args.batch_size = 6

    # args.mix_dir = None
    # # args.mix_json = os.path.join(work_dir, 'data', 'test', 'mix.json')
    # args.mix_json = os.path.join(work_dir, 'data', 'tt', 'mix.json')
    # args.out_dir = os.path.join(exp_folder, 'separate')
    # args.use_cuda = 0
    # args.sample_rate = 8000 # 8000, 16000, etc.
    # args.num_batches = 1

    # check file existence
    assert os.path.isfile(args.model_path), f'model file: {args.model_path} does not exist!'
    assert os.path.isfile(args.mix_json), f'mixed json file: {args.mix_json} does not exist!'

    # print arguments
    print(f'model: {args.model_path}')
    print(f'batch size: {args.batch_size}')
    print(f'mix dir: {args.mix_dir}')
    print(f'out dir: {args.out_dir}')
    print(f'use cuda: {args.use_cuda}')
    print(f'sample rate: {args.sample_rate}')
    print(f'num of batches: {args.num_batches}')

    print(args)
    separate(args)

