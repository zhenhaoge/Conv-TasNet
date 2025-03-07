#!/bin/bash

# Created on 2018/12/08
# Author: Kaituo XU

# work_dir=/home/users/zge/code/repo/tasnet/egs/wsj0
# cd $work_dir

stage=1

# data=/home/ktxu/workspace/data/CSR-I-WSJ0-LDC93S6A
# wav_dir=/home/ktxu/workspace/data/wsj0-wav/wsj0

data=/home/sprats/data/WSJ0
wav_dir=/home/users/zge/data1/datasets/WSJ0/wav

. utils/parse_options.sh || exit 1;

if [ $stage -le 1 ]; then
  echo "Convert sphere format to wav format"

  sph2pipe=../../tools/sph2pipe_v2.5/sph2pipe
  if [ ! -x $sph2pipe ]; then
    echo "Could not find (or execute) the sph2pipe program at $sph2pipe";
    exit 1;
  fi

  tmp=data/local
  mkdir -p $tmp

  [ ! -f $tmp/sph.list ] && find $data/ -iname '*.wv*' | grep -e 'si_tr_s' -e 'si_dt_05' -e 'si_et_05' > $tmp/sph.list

  if [ ! -d $wav_dir ]; then
    while read line; do
      # line=/home/sprats/data/WSJ0/csr_1_wsj0_complete_d2/csr_1_d2/11-9.1/wsj0/si_tr_s/40m/40mo030g.wv2
      wav=`echo "$line" | sed "s:wv[12]:wav:g" | awk -v dir=$wav_dir -F'/' '{printf("%s/%s/%s/%s", dir, $(NF-2), $(NF-1), $NF)}'`
      echo $wav
      mkdir -p `dirname $wav`
      $sph2pipe -f wav $line > $wav
    done < $tmp/sph.list > $tmp/wav.list
  else
    echo "Do you already get wav files? if not, please remove $wav_dir"
  fi
fi
