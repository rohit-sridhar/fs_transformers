#!/bin/bash

ROOT=/data/deep_learning/fs_transformers

bp=0
pid=()
data_splits=(train val)

######################################## SUPPLEMENTAL_GEN ########################################
# for data_split in ${data_splits[@]}; do
#     python ${ROOT}/test.py \
#         -cp ${ROOT}/models/pt_split/261503_0403_28/83ecffe2eb/85_of_100.chkpt \
#         -df ${ROOT}/data/data_supplemental_gen_drop-na_lininterp0_sd1248_rh.pq.${data_split} \
#         -tt gen -bp ${bp} -n 0 -gpu &
#     ((bp+=1))
#     pid+=("$!")
# done
# wait "${pid[@]}"

######################################## MAIN DATASET ########################################
# ##### This model only tests on 10 data points
# data_splits=(train)
# for data_split in ${data_splits[@]}; do
#     python ${ROOT}/test.py \
#         -cp ${ROOT}/models/main_train/261703_1614_49/b25820bc8a/100_of_200.chkpt \
#         -df ${ROOT}/data/data_main_${data_split}_drop-na_lininterp0_rh.pq.all \
#         -tt gen -bp ${bp} -n 10 &
#     ((bp+=1))
#     pid+=("$!")
# done
# wait "${pid[@]}"

######################################## DEBUG ########################################
##### Test overfitting to 3 samples
# python ${ROOT}/test.py \
#     -cp ${ROOT}/models/run_test/261903_1637_19/b143711d24/200_of_200.chkpt \
#     -df ${ROOT}/data/data_main_train_drop-na_lininterp0_rh_smp3.pq.all \
#     -tt gen -bp ${bp} -n 0 -dbg

typeset -A model_data_map
model_data_map["${ROOT}/models/debug_stratified/262003_1255_57/dc6cb6e16b/1000_of_1000.chkpt"]="${ROOT}/data/data_main_train_drop-na_lininterp0_rh_cls20.pq.all"
model_data_map["${ROOT}/models/debug_stratified/262003_1303_33/4b8656dd3c/1000_of_1000.chkpt"]="${ROOT}/data/data_main_train_drop-na_lininterp0_rh_cls50.pq.all"
model_data_map["${ROOT}/models/debug_stratified/262003_1319_35/95df2cde5f/1000_of_1000.chkpt"]="${ROOT}/data/data_main_train_drop-na_lininterp0_rh_cls100.pq.all"
model_data_map["${ROOT}/models/debug_stratified/262003_1356_30/515636e548/1000_of_1000.chkpt"]="${ROOT}/data/data_main_train_drop-na_lininterp0_rh_cls100.pq.all"
model_data_map["${ROOT}/models/debug_categorical/262003_1630_48/4458cf6735/3000_of_3000.chkpt"]="${ROOT}/data/data_main_train_drop-na_lininterp0_rh_cls-NA20.pq.all"
model_data_map["${ROOT}/models/debug_categorical/262003_1657_14/716e3e796d/3000_of_3000.chkpt"]="${ROOT}/data/data_main_train_drop-na_lininterp0_rh_cls-NA50.pq.all"
model_data_map["${ROOT}/models/debug_categorical/262003_1747_05/e0ac892398/3000_of_3000.chkpt"]="${ROOT}/data/data_main_train_drop-na_lininterp0_rh_cls-phone20.pq.all"
model_data_map["${ROOT}/models/debug_categorical/262003_1818_25/54b735ea44/3000_of_3000.chkpt"]="${ROOT}/data/data_main_train_drop-na_lininterp0_rh_cls-phone50.pq.all"
model_data_map["${ROOT}/models/debug_categorical/262003_1915_56/38c4afbbad/3000_of_3000.chkpt"]="${ROOT}/data/data_main_train_drop-na_lininterp0_rh_cls-address20.pq.all"
model_data_map["${ROOT}/models/debug_categorical/262003_1942_42/57aeb0957d/3000_of_3000.chkpt"]="${ROOT}/data/data_main_train_drop-na_lininterp0_rh_cls-address50.pq.all"
model_data_map["${ROOT}/models/debug_categorical/262003_2032_24/abc319390a/3000_of_3000.chkpt"]="${ROOT}/data/data_main_train_drop-na_lininterp0_rh_cls-url20.pq.all"
model_data_map["${ROOT}/models/debug_categorical/262003_2101_31/61db3c776f/3000_of_3000.chkpt"]="${ROOT}/data/data_main_train_drop-na_lininterp0_rh_cls-url50.pq.all"
model_data_map["${ROOT}/models/debug_categorical/262003_2157_47/c4a7d8cc17/3000_of_3000.chkpt"]="${ROOT}/data/data_main_train_drop-na_lininterp0_rh_cls-name20.pq.all"
model_data_map["${ROOT}/models/debug_categorical/262003_2217_00/f063a5dda2/3000_of_3000.chkpt"]="${ROOT}/data/data_main_train_drop-na_lininterp0_rh_cls-name50.pq.all"

##### Test overfitting to 20,50,100 samples (stratified/categorized by label type)
##### Last model of stratified case doesn't overfit to 100 samples
for key in "${!model_data_map[@]}"; do
    python ${ROOT}/test.py \
        -cp ${key} \
        -df ${model_data_map[$key]} \
        -tt gen -bp ${bp} -n 0
done


