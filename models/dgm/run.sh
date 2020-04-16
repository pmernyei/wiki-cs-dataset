#!/bin/bash

# input
GPU=${1-0}
DATA="${2-DD}"  # DD, PROTEINS, COLLAB, REDDIT-BINARY
mode="${3-mpr}"  # mpr, diffpool, mincut
fold=${4-0}  # evaluation fold; set to 0 if want to run on all folds

# model and optimisation
cluster_dims="20 5"
epochs=30
hidden_dims="128 128"
interval_overlap=0.1
lrate=0.001
pooling_ratio=0.25
sim_batch_size=32
std_hidden_dim=32

case ${mode} in
mpr)
  lrate=0.0005

  # task hyperparameters
  case ${DATA} in
  PROTEINS)
    cluster_dims="8 2"
    interval_overlap=0.25
    sim_batch_size=128
    ;;
  REDDIT-BINARY)
    interval_overlap=0.25
    ;;
  esac
  ;;
diffpool)
  pooling_ratio=0.1

  # task hyperparameters
  case ${DATA} in
  COLLAB)
    std_hidden_dim=128
    ;;
  REDDIT-BINARY)
    std_hidden_dim=128
    ;;
  esac
  ;;
mincut)
  # task hyperparameters
  case ${DATA} in
  DD)
    pooling_ratio=0.1
    ;;
  esac
  ;;
esac

cd mpr/;
CUDA_VISIBLE_DEVICES=${GPU} python eval.py \
    --data $DATA \
    --fold $fold \
    --mode $mode \
    --cluster_dims $cluster_dims \
    --interval_overlap $interval_overlap \
    --hidden_dims $hidden_dims \
    --lrate $lrate \
    --epochs $epochs \
    --sim_batch_size $sim_batch_size \
    --pooling_ratio $pooling_ratio \
    --std_hidden_dim $std_hidden_dim
