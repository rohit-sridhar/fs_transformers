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
# python ${ROOT}/train.py \
#         -df ${ROOT}/data/data_main_train_drop-na_lininterp0_rh.pkl.all \
#         -msr main_train \
#         -se 10 -ep 200 -bs 16 -lr 5e-4 \
#         -lf CEL -mt ByT5 -op Adafactor \
#         -bp 0 # -std

# pid+=("$!")
# wait "${pid[@]}"

##### Debug on single category datasets #####
clss=(NA phone address url name)
ns=(20 50)

for cls in ${clss[@]}; do
for n in ${ns[@]}; do
    python ${ROOT}/train.py \
            -df ${ROOT}/data/data_main_train_drop-na_lininterp0_rh_cls-${cls}${n}.pq.all \
            -msr run_test \
            -se 1500 -ep 3000 -bs 16 -lr 5e-4 \
            -mt ByT5 -op Adafactor \
            -bp 0 -oft # -std
done
done

##### Debug on stratified datasets #####

# python ${ROOT}/train.py \
#         -df ${ROOT}/data/data_main_train_drop-na_lininterp0_rh_smp3.pq.all \
#         -msr run_test \
#         -se 50 -ep 200 -bs 16 -lr 5e-4 \
#         -mt ByT5 -op Adafactor \
#         -bp 0 -std
# 
# python ${ROOT}/train.py \
#         -df ${ROOT}/data/data_main_train_drop-na_lininterp0_rh_cls20.pq.all \
#         -msr run_test \
#         -se 500 -ep 1000 -bs 16 -lr 5e-4 \
#         -mt ByT5 -op Adafactor \
#         -bp 0 -oft # -std
# 
# python ${ROOT}/train.py \
#         -df ${ROOT}/data/data_main_train_drop-na_lininterp0_rh_cls50.pq.all \
#         -msr run_test \
#         -se 500 -ep 1000 -bs 16 -lr 5e-4 \
#         -mt ByT5 -op Adafactor \
#         -bp 0 -oft # -std
# 
# python ${ROOT}/train.py \
#         -df ${ROOT}/data/data_main_train_drop-na_lininterp0_rh_cls100.pq.all \
#         -msr run_test \
#         -se 200 -ep 1000 -bs 16 -lr 5e-4 \
#         -mt ByT5 -op Adafactor \
#         -bp 0 -oft # -std
# 
# python ${ROOT}/train.py \
#         -df ${ROOT}/data/data_main_train_drop-na_lininterp0_rh_cls100.pq.all \
#         -msr run_test \
#         -se 200 -ep 1000 -bs 16 -lr 5e-4 \
#         -mt ByT5 -op Adafactor \
#         -bp 0 # -std
# 
# python ${ROOT}/train.py \
#         -df ${ROOT}/data/data_main_train_drop-na_lininterp0_rh_cls1000.pq.all \
#         -msr run_test \
#         -se 100 -ep 1000 -bs 16 -lr 5e-4 \
#         -mt ByT5 -op Adafactor \
#         -bp 0 # -std

