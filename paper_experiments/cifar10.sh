#!/usr/bin/env bash

pushd ../models

declare -a alphas=("1000")

function run_fedMD() {
  echo "############################################## Running FedMD ##############################################"
  alpha="$1"
  python main.py --num-rounds 10000 --eval-every 100 --batch-size 64 --batch-size-digest 256 --batch-size-revisit 5 --num-epochs-digest 1 --num-epochs-revisit 4 --num-epochs 1 --clients-per-round 10 -lr 0.01 --weight-decay 0.0004 -device cuda:0 -algorithm fedopt --server-lr 1 --server-opt sgd --num-workers 0 --where-loading init -alpha ${alpha}
}


echo "####################### EXPERIMENTS ON CIFAR10 #######################"
for alpha in "${alphas[@]}"; do
  run_fedMD "${alpha}"
done