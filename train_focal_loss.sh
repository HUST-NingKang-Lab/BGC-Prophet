#!/bin/bash
export CUDA_VISIBLE_DEVICES='1'
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
echo $PYTORCH_CUDA_ALLOC_CONF

source ~/anaconda3/etc/profile.d/conda.sh
# source /mnt/hdd0/qllai/miniconda3/etc/profile.d/conda.sh
conda activate pt


nohup python -u ./src/trainer_focal_loss.py --datasetPath ./data/BGC_train_dataset_plus.csv \
    --max_len 128 --nhead 4 --seed 42 --num_encoder_layers 6 --dropout 0.2 \
    --batch_size 64 --learning_rate 0.001 \
    --distribute_epochs 1000 --warmup_epochs 30 --lmdbPath ./lmdb_BGC \
    >nohup_transformerEncoder_BGC_focal_loss.out &