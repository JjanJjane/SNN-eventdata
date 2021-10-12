#!/bin/bash

#
source ../00_SNN/venv/bin/activate

####

model=${1}
dataset=${2}

#{ unbuffer time kernprof -l main.py \
#{ CUDA_VISIBLE_DEVICES=0 unbuffer time python main.py \
#{ unbuffer time python -m line_profiler main.py \
{ unbuffer time python -u main_hp_tune.py \
    -model=${model} \
    -dataset=${dataset} \
    ; }

