#!/usr/bin/env python

# Created on 2018/12
# Author: Kaituo XU

import argparse
import os, sys
from pathlib import Path
import librosa
import numpy as np
import torch

# set paths
# home_dir = str(Path.home())
# work_dir = os.path.join(home_dir, 'code', 'repo', 'tasnet', 'egs', 'wsj0')
# if os.getcwd() != work_dir:
#     os.chdir(work_dir)
work_dir = os.getcwd()
print('current path: {}'.format(work_dir))

src_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'src')
sys.path.insert(0, src_dir)
from mir_eval.separation import bss_eval_sources
from data import AudioDataLoader, AudioDataset
from pit_criterion import cal_loss
from conv_tasnet import ConvTasNet
from utils import remove_pad
from metrics import cal_SDRi, cal_SISNRi

def parse_args():

    usage = 'Evaluate separation performance using Conv-TasNet'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model file created by training')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='directory including mix.json, s1.json and s2.json')
    parser.add_argument('--cal_sdr', type=int, default=0,
                        help='Whether calculate SDR, add this option because calculation of SDR is very slow')
    parser.add_argument('--use_cuda', type=int, default=0,
                        help='Whether use GPU')
    parser.add_argument('--sample_rate', default=8000, type=int,
                        help='Sample rate')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size')
    args = parser.parse_args()
    return args

def evaluate(args):
    total_SISNRi = 0
    total_SDRi = 0
    total_cnt = 0

    # Load model
    model = ConvTasNet.load_model(args.model_path)
    # print(model)
    model.eval()
    if args.use_cuda:
        model.cuda()

    # Load data
    dataset = AudioDataset(args.data_dir, args.batch_size,
                           sample_rate=args.sample_rate, segment=-1, cv_maxlen=float('inf'))
    data_loader = AudioDataLoader(dataset, batch_size=1, num_workers=2)

    with torch.no_grad():
        for i, (data) in enumerate(data_loader):

            # # check the first batch
            # print('stop at the first batch')
            # break

            # Get batch data
            padded_mixture, mixture_lengths, padded_source = data

            if args.use_cuda:
                padded_mixture = padded_mixture.cuda()
                mixture_lengths = mixture_lengths.cuda()
                padded_source = padded_source.cuda()

            # Remove padding and flat
            mixture = remove_pad(padded_mixture, mixture_lengths)
            source = remove_pad(padded_source, mixture_lengths)

            # Forward
            estimate_source = model(padded_mixture)  # [B, C, T]
            loss, max_snr, estimate_source, reorder_estimate_source = \
                cal_loss(padded_source, estimate_source, mixture_lengths)

            # NOTE: use reorder estimate source
            # estimate_source = remove_pad(reorder_estimate_source,
            #                              mixture_lengths)
            reorder_estimate_source_nograd = reorder_estimate_source.detach()
            estimate_source = remove_pad(reorder_estimate_source_nograd, mixture_lengths)

            # for each utterance
            bs_current = padded_mixture.shape[0]
            for j in range(bs_current):
                mix, src_ref, src_est = mixture[j], source[j], estimate_source[j]
            # for mix, src_ref, src_est in zip(mixture, source, estimate_source):
                print("Utt", total_cnt + 1)
                # Compute SDRi
                if args.cal_sdr:
                    avg_SDRi = cal_SDRi(src_ref, src_est, mix)
                    print("\tSDRi={0:.2f}".format(avg_SDRi))
                    total_SDRi += avg_SDRi
                # Compute SI-SNRi
                avg_SISNRi = cal_SISNRi(src_ref, src_est, mix)
                print("\tSI-SNRi={0:.2f}".format(avg_SISNRi))
                total_SISNRi += avg_SISNRi
                total_cnt += 1
    if args.cal_sdr:
        print("Average SDR improvement: {0:.2f}".format(total_SDRi / total_cnt))
    print("Average SISNR improvement: {0:.2f}".format(total_SISNRi / total_cnt))

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # # interactive mode
    # args = argparse.ArgumentParser()

    # # for the pretrained model
    # args.model_path = os.path.join(work_dir, 'models', 'pretrained', 'final.pth.tar')
    # args.batch_size = 3

    # # for the self-trained model

    # # wsj0 - model 1
    # args.exp_folder = 'exp/train_r16000_N256_L20_B256_H512_P3_X8_R4_C2_gLN_causal0_relu_epoch100_half1_norm5_bs4_worker4_adam_lr1e-3_mmt0_l20_tr'
    # args.data_dir = os.path.join(work_dir, 'data', '16k', 'tt')
    # args.sample_rate = 16000

    # # wsj0 - model 2
    # args.exp_folder = 'exp/train_titan12'
    # args.data_dir = os.path.join(work_dir, 'data', 'tt')
    # args.sample_rate = 8000

    # # Echo2Mix
    # args.exp_folder = 'exp/train_r16000_N256_L20_B256_H512_P3_X8_R4_C2_gLN_causal0_relu_epoch100_half1_norm5_bs6_worker4_adam_lr1e-3_mmt0_l20_train'
    # args.exp_folder = 'exp/train_r8000_N256_L20_B256_H512_P3_X8_R4_C2_gLN_causal0_relu_epoch100_half1_norm5_bs6_worker12_adam_lr1e-3_mmt0_l20_train'
    # args.data_dir = os.path.join(work_dir, 'data', '8k', 'test')
    # args.sample_rate = 8000

    # args.model_path = os.path.join(work_dir, args.exp_folder, 'final.pth')
    # args.batch_size = 6

    # args.cal_sdr = 1
    # args.use_cuda = 0

    # check file/dir existence
    assert os.path.isfile(args.model_path), f'model file: {args.model_path} does not exist!'
    assert os.path.isdir(args.data_dir), f'data dir: {args.data_dir} does not exist!'

    print(args)
    evaluate(args)
