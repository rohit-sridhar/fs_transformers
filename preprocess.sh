#!/bin/bash
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

supplemental_gen_participants=(3f8b 13e3 494d b2d1 c0df d3ab 8e3b fe96 8c4d a3d4 3a6e 3d12 f9ea 2ff7 e0f7 ed8e 51f5 a362 a6ed 0ba8 812c 03ad a021 a442 1d72 711d a95b fa10 1bd5 6b92 5b63 bd21 1f91 917d fbb7 4ddc ab12 dbf9 99cb 39e5 4f1e 63a1 163a c82a f418 9d2b b718 39a6 4c3d 675f 9b23 9ed9 d478 f066 e3c0 fede 0a77 0bea d05c 9ff4 f760 7f32 80fe 19d3 6f68 a3e7 cf84 d69c 1f86 2f35 e4fa 5d33)

./preprocess.py \
    -ds supplemental_gen \
    -ip 0 -tr 1.0 \
    -dna

./preprocess.py \
    -ds supplemental_gen -ptgrp pt-split \
    -ip 0 -tr 1.0 \
    -dna

./preprocess.py \
    -ds supplemental_gen \
    -pt ${supplemental_gen_participants} -ip 0 \
    -dna -so

