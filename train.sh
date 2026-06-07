#!/bin/bash

ROOT=/data/deep_learning/fs_transformers

############################## SUPPLEMENTAL_GEN TRAIN ############################### 
# 
# bp=0
# pid=()
# 
# ##### General split (USER DEPENDENT) #####
# python ${ROOT}/train.py \
#         -df ${ROOT}/data/data_supplemental_gen_drop-na_lininterp0_sd1248_rh.pkl.train \
#         -msr run_test \
#         -se 5 -ep 100 -bs 16 -lr 5e-4 \
#         -lf CEL -mt ByT5 -op Adam \
#         -bp 0 # -std
# 
# ##### General split (USER INDEPENDENT) #####
# python ${ROOT}/train.py \
#         -df ${ROOT}/data/data_supplemental_gen_drop-na_lininterp0_sd1248_pt-split_rh.pkl.train \
#         -msr pt_split \
#         -se 5 -ep 100 -bs 16 -lr 5e-4 \
#         -lf CEL -mt ByT5 -op Adafactor \
#         -bp 0 # -std
# 
# pid+=("$!")
# wait "${pid[@]}"
# 
#################################################################################
############################## MAIN DATASET TRAIN ############################### 
# bp=0
# pid=()

##### General split (USER INDEPENDENT) #####
python ${ROOT}/train.py \
        -df ${ROOT}/data/data_supplemental_gen_drop-na_lininterp0_sd1248_pt-split_rh.pq.train \
        -msr pt_split \
        -se 20 -ep 500 -bs 16 -lr 5e-4 \
        -mt ByT5 -op Adafactor \
        -bp 0 # -std
# pid+=("$!")
# wait "${pid[@]}"

# ##### Debug on stratified datasets #####
# 
# python ${ROOT}/train.py \
#         -df ${ROOT}/data/data_main_train_drop-na_lininterp0_rh_smp3.pq.all \
#         -msr debug_stratified \
#         -se 50 -ep 200 -bs 16 -lr 5e-4 \
#         -mt ByT5 -op Adafactor \
#         -bp 0 -oft -std
# 
# ns=(20 50)
# for n in ${ns[@]}; do
#     python ${ROOT}/train.py \
#             -df ${ROOT}/data/data_main_train_drop-na_lininterp0_rh_cls${n}.pq.all \
#             -msr debug_stratified \
#             -se 500 -ep 1000 -bs 16 -lr 5e-4 \
#             -mt ByT5 -op Adafactor \
#             -bp 0 -oft # -std
# done
# 
# ns=(100 1000)
# for n in ${ns[@]}; do
#     python ${ROOT}/train.py \
#             -df ${ROOT}/data/data_main_train_drop-na_lininterp0_rh_cls${n}.pq.all \
#             -msr debug_stratified \
#             -se 200 -ep 1000 -bs 16 -lr 5e-4 \
#             -mt ByT5 -op Adafactor \
#             -bp 0 # -std
# done

##### debug on stratified dataset with larger learning rate and larger n
# 
# ns=(500 1000)
# python ${ROOT}/train.py \
#         -df ${ROOT}/data/data_main_train_drop-na_lininterp0_rh_cls${ns[0]}.pq.all \
#         -msr debug_stratified2 \
#         -se 750 -ep 1500 -bs 16 -lr 5e-3 \
#         -mt ByT5 -op Adafactor \
#         -bp 0 -oft # -std
# 
# python ${ROOT}/train.py \
#         -df ${ROOT}/data/data_main_train_drop-na_lininterp0_rh_cls${ns[1]}.pq.all \
#         -msr debug_stratified2 \
#         -se 750 -ep 1500 -bs 16 -lr 5e-3 \
#         -mt ByT5 -op Adafactor \
#         -bp 0 # -std
# 
# python ${ROOT}/train.py \
#         -df ${ROOT}/data/data_main_train_drop-na_lininterp0_rh_cls-address100.pq.all \
#         -msr run_test \
#         -se 750 -ep 1500 -bs 16 -lr 5e-3 \
#         -mt ByT5 -op Adafactor \
#         -bp 0 # -std

# ##### Debug on single category datasets #####
# ##### run on Mar 19
# ns=(20 50)
# 
# for cls in ${clss[@]}; do
# for n in ${ns[@]}; do
#     python ${ROOT}/train.py \
#             -df ${ROOT}/data/data_main_train_drop-na_lininterp0_rh_cls-${cls}${n}.pq.all \
#             -msr debug_categorical \
#             -se 1500 -ep 3000 -bs 16 -lr 5e-4 \
#             -mt ByT5 -op Adafactor \
#             -bp 0 -oft # -std
# done
# done
# 

##### run on Mar 20
# clss=(NA phone address url name)
# ns=(20 100)
# for cls in ${clss[@]}; do
# for n in ${ns[@]}; do
#     python ${ROOT}/train.py \
#             -df ${ROOT}/data/data_main_train_drop-na_lininterp0_rh_cls-${cls}${n}.pq.all \
#             -msr debug_categorical2 \
#             -se 750 -ep 1500 -bs 16 -lr 5e-3 \
#             -mt ByT5 -op Adafactor \
#             -bp 0 -oft # -std
# done
# done
# 

####################!!!!!!!!!!!!!!!!!!!! DELETED debug_categorical3 !!!!!!!!!!!!!!!!!!!!####################
##### run on Mar 20
# ns=(1000 2000)
# for cls in ${clss[@]}; do
# for n in ${ns[@]}; do
#     python ${ROOT}/train.py \
#             -df ${ROOT}/data/data_main_train_drop-na_lininterp0_rh_cls-${cls}${n}.pq.all \
#             -msr debug_categorical3 \
#             -se 500 -ep 1000 -bs 16 -lr 5e-3 \
#             -mt ByT5 -op Adafactor \
#             -bp 0 # -std
# done
# done
# 
