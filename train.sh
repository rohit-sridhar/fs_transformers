#!/bin/bash

ROOT=/data/deep_learning/fs_transformers

############################## for test run and debugging ############################### 
### Already tuned on n100/n1000
# lrs=(3e-3 1e-2 3e-3 1e-1 1e-5 1e-4 3e-4 1e-3)
# nls=(2 3 4)
# hss=(64 128 256)
# 
# lrs=(1e-3)
# nls=(2)
# hss=(256)
# dps=(0.2 0.4 0.6)

# bp=0
# pid=()

# Without preprocessing
python ${ROOT}/train.py \
        -df ${ROOT}/data/data_supplemental_gen_drop-na_lininterp0_sd1248_rh.pkl.train \
        -msr first_pass \
        -se 50 -ep 500 -bs 16 -lr 5e-4 \
        -lf CEL -mt ByT5 -op Adam \
        -bp 0 # -std
# ((bp+=3))
# pid+=("$!")

# wait "${pid[@]}"
#########################################################################################

