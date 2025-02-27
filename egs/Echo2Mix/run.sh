#!/bin/bash

# Created on 02/06/2025
# Author: Zhenhao Ge

# # setup envionment and working directory
# source ~/.zshrc
# conda activate py37
# cd /home/users/zge/code/repo/tasnet/egs/Echo2Mix

data=/home/users/zge/data1/datasets/Echo2Mix

stage=1

dumpdir=data

# -- START Conv-TasNet Config
train_dir=$dumpdir/train
valid_dir=$dumpdir/val
evaluate_dir=$dumpdir/test
separate_dir=$dumpdir/test
sample_rate=16000
segment=4  # seconds
cv_maxlen=6  # seconds
# Network config
N=256
L=20
B=256
H=512
P=3
X=8
R=4
norm_type=gLN
causal=0
mask_nonlinear='relu'
C=2
# Training config
use_cuda=1
id=0,1,2
epochs=100
half_lr=1
early_stop=0
max_norm=5
# minibatch
shuffle=1
batch_size=6 # original: 3
num_workers=4
# optimizer
optimizer=adam
lr=1e-3
momentum=0
l2=0
# save and visualize
checkpoint=0
continue_from=""
model_path="final.pth"
print_freq=10
visdom=0
visdom_epoch=0
visdom_id="Conv-TasNet Training"
# evaluate
ev_use_cuda=0
cal_sdr=1
# -- END Conv-TasNet Config

# exp tag
tag="" # tag for managing experiments.

ngpu=1  # always 1

. utils/parse_options.sh;
. ./cmd.sh
. ./path.sh

if [ $stage -le 1 ]; then
  echo "Stage 1: Generating json files including wav path and duration (#samples)"
  [ ! -d $dumpdir ] && mkdir $dumpdir
  preprocess.py \
    --in-dir $data \
    --out-dir $dumpdir \
    --sample-rate $sample_rate
fi

tag=''
if [ -z ${tag} ]; then
  expdir=exp/train_r${sample_rate}_N${N}_L${L}_B${B}_H${H}_P${P}_X${X}_R${R}_C${C}_${norm_type}_causal${causal}_${mask_nonlinear}_epoch${epochs}_half${half_lr}_norm${max_norm}_bs${batch_size}_worker${num_workers}_${optimizer}_lr${lr}_mmt${momentum}_l2${l2}_`basename $train_dir`
else
  expdir=exp/train_${tag}
fi
echo "exp dir: ${expdir}"
continue_from="${expdir}/checkpoint.epoch14.pth"

if [ $stage -le 2 ]; then
  echo "Stage 2: Training"
  touch ${expdir}/train.log
  CUDA_VISIBLE_DEVICES=0,1 \
    train.py \
    --train_dir $train_dir \
    --valid_dir $valid_dir \
    --sample_rate $sample_rate \
    --segment $segment \
    --cv_maxlen $cv_maxlen \
    --N $N \
    --L $L \
    --B $B \
    --H $H \
    --P $P \
    --X $X \
    --R $R \
    --C $C \
    --norm_type $norm_type \
    --causal $causal \
    --mask_nonlinear $mask_nonlinear \
    --use_cuda $use_cuda \
    --epochs $epochs \
    --half_lr $half_lr \
    --early_stop $early_stop \
    --max_norm $max_norm \
    --shuffle $shuffle \
    --batch_size $batch_size \
    --num_workers $num_workers \
    --optimizer $optimizer \
    --lr $lr \
    --momentum $momentum \
    --l2 $l2 \
    --save_folder ${expdir} \
    --checkpoint $checkpoint \
    --continue_from "$continue_from" \
    --model_path ${model_path} \
    --print_freq ${print_freq} \
    --visdom $visdom \
    --visdom_epoch $visdom_epoch \
    --visdom_id "$visdom_id" | tee ${expdir}/train.log
fi

if [ $stage -le 3 ]; then
  echo "Stage 3: Evaluate separation performance"
  ${decode_cmd} --gpu ${ngpu} ${expdir}/evaluate.log \
    evaluate.py \
    --model_path ${expdir}/final.pth \
    --data_dir $evaluate_dir \
    --cal_sdr $cal_sdr \
    --use_cuda $ev_use_cuda \
    --sample_rate $sample_rate \
    --batch_size $batch_size

    # evaluate.py \
    #   --model_path ${expdir}/final.pth \
    #   --data_dir $evaluate_dir \
    #   --cal_sdr $cal_sdr \
    #   --use_cuda $ev_use_cuda \
    #   --sample_rate $sample_rate \
    #   --batch_size $batch_size | tee ${expdir}/evaluate.log

fi

if [ $stage -le 4 ]; then
  echo "Stage 4: Separate speech using Conv-TasNet"
  separate_out_dir=${expdir}/separate
  ${decode_cmd} --gpu ${ngpu} ${separate_out_dir}/separate.log \
    separate.py \
    --model_path ${expdir}/final.pth \
    --mix_json $separate_dir/mix.json \
    --out_dir ${separate_out_dir} \
    --use_cuda $ev_use_cuda \
    --sample_rate $sample_rate \
    --batch_size $batch_size
fi