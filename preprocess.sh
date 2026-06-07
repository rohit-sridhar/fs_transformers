#!/bin/bash
# 
# python preprocess.py \
#     -ds supplemental_gen \
#     -ip 0 -tr 1.0 \
#     -dna
# 
# python preprocess.py \
#     -ds supplemental_gen \
#     -ip 0 -tr 0.8 \
#     -dna
# 
# python preprocess.py \
#     -ds supplemental_gen \
#     -ptgrp pt-split \
#     -ip 0 -tr 0.72 \
#     -dna
# 
# python preprocess.py \
#     -ds main_train \
#     -ip 0 -tr 1.0 \
#     -dna
# 
# python preprocess.py \
#     -ds main_train \
#     -ip 0 -tr 0.7 \
#     -dna
# 

./preprocess.py \
    -ds supplemental_gen \
    -pt ab12 -ip 0 \
    -dna -so

