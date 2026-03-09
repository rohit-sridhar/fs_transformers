#!/bin/bash

ROOT=/data/deep_learning/sltorch

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
        -df ${ROOT}/datasets/data_supplemental_gen_drop-na_lininterp0_sd1248_rh_smp3.pkl.train \
        -msr run_tests \
        -se 1000 -ep 15000 -bs 2 -lr 1e-4 \
        -lf CEL -mt Transformer -op Adam \
        -bp 0
# ((bp+=3))
# pid+=("$!")

# wait "${pid[@]}"
#########################################################################################

