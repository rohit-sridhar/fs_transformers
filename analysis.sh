#!/usr/bin/env bash
# -*- coding: utf-8 -*-
set -euo pipefail

# File: analysis.sh
# Date: 12-06-2026
# Last Modified:

# Description
#   

seeds=(1248 2248 3248 4248 5248)
ns=(50 100 500 1000 2000 5000 10000)

for seed in ${seeds[@]}; do
for n in ${ns[@]}; do
    ./analysis.py \
        -at sample_classification -ss stratified \
        -ds main_train -df ./data/data_main_train_drop-na_lininterp0_rh.pq.all \
        -n ${n} -sd ${seed}
done
done

