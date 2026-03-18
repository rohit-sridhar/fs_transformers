#!/bin/bash

ROOT=/data/deep_learning/fs_transformers

bp=0
pid=()
data_splits=(train val)

######################################## SUPPLEMENTAL_GEN ########################################
# for data_split in ${data_splits[@]}; do
#     python ${ROOT}/test.py \
#         -cp ${ROOT}/models/pt_split/261503_0403_28/83ecffe2eb/85_of_100.chkpt \
#         -df ${ROOT}/data/data_supplemental_gen_drop-na_lininterp0_sd1248_rh.pkl.${data_split} \
#         -tt gen -bp ${bp} -n 0 -gpu &
#     ((bp+=1))
#     pid+=("$!")
# done
# wait "${pid[@]}"

######################################## MAIN DATASET ########################################
data_splits=(train)
for data_split in ${data_splits[@]}; do
    python ${ROOT}/test.py \
        -cp ${ROOT}/models/main_train/261703_1614_49/b25820bc8a/40_of_200.chkpt \
        -df ${ROOT}/data/data_main_${data_split}_drop-na_lininterp0_rh.pkl.all \
        -tt gen -bp ${bp} -n 0 &
    ((bp+=1))
    pid+=("$!")
done
wait "${pid[@]}"
