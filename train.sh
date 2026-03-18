#!/bin/bash

ROOT=/data/deep_learning/fs_transformers

############################## SUPPLEMENTAL_GEN TRAIN ############################### 

# bp=0
# pid=()

##### General split (USER DEPENDENT) #####
# python ${ROOT}/train.py \
#         -df ${ROOT}/data/data_supplemental_gen_drop-na_lininterp0_sd1248_rh.pkl.train \
#         -msr run_test \
#         -se 5 -ep 100 -bs 16 -lr 5e-4 \
#         -lf CEL -mt ByT5 -op Adam \
#         -bp 0 # -std

##### General split (USER INDEPENDENT) #####
# python ${ROOT}/train.py \
#         -df ${ROOT}/data/data_supplemental_gen_drop-na_lininterp0_sd1248_pt-split_rh.pkl.train \
#         -msr pt_split \
#         -se 5 -ep 100 -bs 16 -lr 5e-4 \
#         -lf CEL -mt ByT5 -op Adafactor \
#         -bp 0 # -std

# pid+=("$!")
# wait "${pid[@]}"

#################################################################################
############################## MAIN DATASET TRAIN ############################### 

# bp=0
# pid=()

##### General split (USER INDEPENDENT) #####
python ${ROOT}/train.py \
        -df ${ROOT}/data/data_main_train_drop-na_lininterp0_rh.pkl.all \
        -msr main_train \
        -se 10 -ep 200 -bs 16 -lr 5e-4 \
        -lf CEL -mt ByT5 -op Adafactor \
        -bp 0 # -std

# pid+=("$!")
# wait "${pid[@]}"
