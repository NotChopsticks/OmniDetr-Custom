#!/usr/bin/env bash

set -x

#EXP_DIR=results/gamod/1kG8kA/
#EXP_DIR=results/gamod/6kG8kA/
#EXP_DIR=results/gamod/4kG8kA/
#EXP_DIR=results/gamod/3kG8kA/
#EXP_DIR=results/gamod/1kG4kA_synced/
EXP_DIR=results/gamod/8kG8kA_20ep/

PY_ARGS=${@:1}

# export LD_LIBRARY_PATH=/work/mconda3/envs/deformable_detr/lib:$LD_LIBRARY_PATH

python3 -u main.py \
    --output_dir ${EXP_DIR} \
    --dataset_file 'gamod' \
    --data_path '/data/jlorray1/SDU-GAMODv4' \
    --num_queries 900 \
    --nheads 16 \
    --epochs 20 \
    --num_feature_levels 4 \
    ${PY_ARGS}

    #--epochs 39 \
    #--finetune True \
    # GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/
    # GPUS_PER_NODE=2 ./tools/run_dist_launch.sh 2 ./configs/gamod_r50_deformable_detr.sh
    # --resume 'results/gamod/6kG8kA/checkpoint0014.pth' \