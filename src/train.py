#!/usr/bin/env python

# Created on 2018/12
# Author: Kaituo XU

import argparse
import os, sys
from pathlib import Path

import torch

# set paths
# dataset = 'wsj0'
# home_dir = str(Path.home())
# work_dir = os.path.join(home_dir, 'code', 'repo', 'tasnet', 'egs', dataset)
# if os.getcwd() != work_dir:
#     os.chdir(work_dir)
print('current path: {}'.format(os.getcwd()))

src_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'src')
sys.path.insert(0, src_dir)
from data import AudioDataLoader, AudioDataset
from solver import Solver
from conv_tasnet import ConvTasNet

def parse_args():

    parser = argparse.ArgumentParser(
        "Fully-Convolutional Time-domain Audio Separation Network (Conv-TasNet) "
        "with Permutation Invariant Training")

    # General config

    # Task related
    parser.add_argument('--train_dir', type=str, default=None,
                        help='directory including mix.json, s1.json and s2.json')
    parser.add_argument('--valid_dir', type=str, default=None,
                        help='directory including mix.json, s1.json and s2.json')
    parser.add_argument('--sample_rate', default=8000, type=int,
                        help='Sample rate')
    parser.add_argument('--segment', default=4, type=float,
                        help='Segment length (seconds)')
    parser.add_argument('--cv_maxlen', default=8, type=float,
                        help='max audio length (seconds) in cv, to avoid OOM issue.')

    # Network architecture
    parser.add_argument('--N', default=256, type=int,
                        help='Number of filters in autoencoder')
    parser.add_argument('--L', default=20, type=int,
                        help='Length of the filters in samples (40=5ms at 8kHZ)')
    parser.add_argument('--B', default=256, type=int,
                        help='Number of channels in bottleneck 1 × 1-conv block')
    parser.add_argument('--H', default=512, type=int,
                        help='Number of channels in convolutional blocks')
    parser.add_argument('--P', default=3, type=int,
                        help='Kernel size in convolutional blocks')
    parser.add_argument('--X', default=8, type=int,
                        help='Number of convolutional blocks in each repeat')
    parser.add_argument('--R', default=4, type=int,
                        help='Number of repeats')
    parser.add_argument('--C', default=2, type=int,
                        help='Number of speakers')
    parser.add_argument('--norm_type', default='gLN', type=str,
                        choices=['gLN', 'cLN', 'BN'], help='Layer norm type')
    parser.add_argument('--causal', type=int, default=0,
                        help='Causal (1) or noncausal(0) training')
    parser.add_argument('--mask_nonlinear', default='relu', type=str,
                        choices=['relu', 'softmax'], help='non-linear to generate mask')

    # Training config
    parser.add_argument('--use_cuda', type=int, default=1,
                        help='Whether use GPU')
    parser.add_argument('--epochs', default=30, type=int,
                        help='Number of maximum epochs')
    parser.add_argument('--half_lr', dest='half_lr', default=0, type=int,
                        help='Halving learning rate when get small improvement')
    parser.add_argument('--early_stop', dest='early_stop', default=0, type=int,
                        help='Early stop training when no improvement for 10 epochs')
    parser.add_argument('--max_norm', default=5, type=float,
                        help='Gradient norm threshold to clip')

    # minibatch
    parser.add_argument('--shuffle', default=0, type=int,
                        help='reshuffle the data at every epoch')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to generate minibatch')

    # optimizer
    parser.add_argument('--optimizer', default='adam', type=str,
                        choices=['sgd', 'adam'],
                        help='Optimizer (support sgd and adam now)')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Init learning rate')
    parser.add_argument('--momentum', default=0.0, type=float,
                        help='Momentum for optimizer')
    parser.add_argument('--l2', default=0.0, type=float,
                        help='weight decay (L2 penalty)')

    # save and load model
    parser.add_argument('--save_folder', default='exp/temp',
                        help='Location to save epoch models')
    parser.add_argument('--checkpoint', dest='checkpoint', default=0, type=int,
                        help='Enables checkpoint saving of model')
    parser.add_argument('--continue_from', default='',
                        help='Continue from checkpoint model')
    parser.add_argument('--model_path', default='final.pth.tar',
                        help='Location to save best validation model')

    # logging
    parser.add_argument('--print_freq', default=10, type=int,
                        help='Frequency of printing training infomation')
    parser.add_argument('--visdom', dest='visdom', type=int, default=0,
                        help='Turn on visdom graphing')
    parser.add_argument('--visdom_epoch', dest='visdom_epoch', type=int, default=0,
                        help='Turn on visdom graphing each epoch')
    parser.add_argument('--visdom_id', default='TasNet training',
                        help='Identifier for visdom run')

    args = parser.parse_args()
    return args


def main(args):

    # Construct Solver
    # data
    tr_dataset = AudioDataset(args.train_dir, args.batch_size,
                              sample_rate=args.sample_rate, segment=args.segment)
    cv_dataset = AudioDataset(args.valid_dir, batch_size=1,  # 1 -> use less GPU memory to do cv
                              sample_rate=args.sample_rate,
                              segment=-1, cv_maxlen=args.cv_maxlen)  # -1 -> use full audio
    tr_loader = AudioDataLoader(tr_dataset, batch_size=1,
                                shuffle=args.shuffle,
                                num_workers=args.num_workers)
    cv_loader = AudioDataLoader(cv_dataset, batch_size=1,
                                num_workers=0)
    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}
    # model
    model = ConvTasNet(args.N, args.L, args.B, args.H, args.P, args.X, args.R,
                       args.C, norm_type=args.norm_type, causal=args.causal,
                       mask_nonlinear=args.mask_nonlinear)
    print(model)

    if args.use_cuda:
        # device = torch.device(f"cuda:0")
        # model = model.to(device)
        model = torch.nn.DataParallel(model)
        model.cuda()

    # optimizer
    if args.optimizer == 'sgd':
        optimizier = torch.optim.SGD(model.parameters(),
                                     lr=args.lr,
                                     momentum=args.momentum,
                                     weight_decay=args.l2)
    elif args.optimizer == 'adam':
        optimizier = torch.optim.Adam(model.parameters(),
                                      lr=args.lr,
                                      weight_decay=args.l2)
    else:
        print("Not support optimizer")
        return

    # solver
    solver = Solver(data, model, optimizier, args)
    solver.train()

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # # interactive mode
    # args = argparse.ArgumentParser()

    # # task related
    # work_dir = os.getcwd()
    # args.train_dir = os.path.join(work_dir, 'data', '16k', 'tr')
    # args.valid_dir = os.path.join(work_dir, 'data', '16k', 'cv')
    # args.sample_rate = 16000
    # args.segment = 4 # segment length in seconds
    # args.cv_maxlen = 6 # max audio length (seconds) in cv, to avoid OOM issue

    # # network architecture
    # args.N = 256 # number of filters in autoencoder
    # args.L = 20 # length of the filters in samples (40=5ms at 8kHZ)
    # args.B = 256 # number of channels in bootlenect 1X1-conv blocks
    # args.H = 512 # number of channels in conv blocks
    # args.P = 3 # kernel size in conv blocks
    # args.X = 8 # number of conv-blocks in each repeat
    # args.R = 4 # number of repeats
    # args.C = 2 # number of speakers
    # args.norm_type = 'gLN' # layer norm type
    # args.causal = 0 # causal (1) or noncausal (0) training
    # args.mask_nonlinear = 'relu' # non-linear to generate mask

    # # training config
    # args.use_cuda = 1
    # args.epochs = 100
    # args.half_lr = 1
    # args.early_stop = 0
    # args.max_norm = 5 # gradient norm threshold to clip

    # # minibatch
    # args.shuffle = 1 # shuffle the data at every epoch
    # args.batch_size = 4
    # args.num_workers = 4 # number of workers to generate minibatch

    # # optimizer
    # args.optimizer = 'adam'
    # args.lr = 1e-3
    # args.momentum = 0.0
    # args.l2 = 0.0

    # # save and load model
    # basename = os.path.basename(args.train_dir)
    # exp_folder = 'train_r{}_N{}_L{}_B{}_H{}_P{}_X{}_R{}_C{}_{}_causal{}_{}_epoch{}_half{}_norm{}_bs{}_worker{}_{}_lr{}_mmt{}_l2{}_{}'.format(
    #     args.sample_rate, args.N, args.L, args.B, args.H, args.P, args.X, args.R, args.C, args.norm_type, args.causal, args.mask_nonlinear,
    #     args.epochs, args.half_lr, args.max_norm, args.batch_size, args.num_workers, args.optimizer, f'{args.lr:.0e}'.replace("03", "3"), int(args.momentum), int(args.l2), basename)
    # exp_dir = os.path.join(work_dir, 'exp', exp_folder)
    # args.save_folder = exp_dir
    # args.checkpoint = 0
    # # args.continue_from = os.path.join(args.save_folder, 'ckpt.ep11.pth')
    # args.continue_from = os.path.join(work_dir, 'models', '16k', 'ckpt.ep15.pth')
    # # args.continue_from = ""
    # args.model_path = 'final.pth'

    # # logging
    # args.print_freq = 10
    # args.visdom = 0
    # args.visdom_epoch = 0
    # args.visdom_id = 'Conv-TasNet Training'

    print(args)
    main(args)

