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

##### Test overfitting to 20 samples (stratified by text output)
python ${ROOT}/test.py \
    -cp ${ROOT}/models/run_test/262003_1255_57/dc6cb6e16b/1000_of_1000.chkpt \
    -df ${ROOT}/data/data_main_train_drop-na_lininterp0_rh_cls20.pq.all \
    -tt gen -bp ${bp} -n 0

##### Test overfitting to 50 samples (stratified by text output)
python ${ROOT}/test.py \
    -cp ${ROOT}/models/run_test/262003_1303_33/4b8656dd3c/1000_of_1000.chkpt \
    -df ${ROOT}/data/data_main_train_drop-na_lininterp0_rh_cls50.pq.all \
    -tt gen -bp ${bp} -n 0

##### Test overfitting to 100 samples (stratified by text output)
python ${ROOT}/test.py \
    -cp ${ROOT}/models/run_test/262003_1303_33/4b8656dd3c/1000_of_1000.chkpt  \
    -df ${ROOT}/data/data_main_train_drop-na_lininterp0_rh_cls100.pq.all \
    -tt gen -bp ${bp} -n 0

##### Test fitting (not over) to 100 samples (stratified by text output)
python ${ROOT}/test.py \
    -cp ${ROOT}/models/run_test/262003_1356_30/515636e548/1000_of_1000.chkpt \
    -df ${ROOT}/data/data_main_train_drop-na_lininterp0_rh_cls100.pq.all \
    -tt gen -bp ${bp} -n 0

##### Test model trained on all data
# python ${ROOT}/test.py \
#     -cp ${ROOT}/models/main_train/261703_1614_49/b25820bc8a/100_of_200.chkpt \
#     -df ${ROOT}/data/data_main_train_drop-na_lininterp0_rh.pq.all \
#     -tt gen -bp ${bp} -n 10 -dbg
