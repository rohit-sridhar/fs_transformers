#!/bin/bash

model_dir=${1}

if [[ ! -d ./${model_dir} ]]; then
    echo "Directory ${model_dir} does not exist."
fi

