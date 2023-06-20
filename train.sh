#!/bin/bash
export CUDA_VISIBLE_DEVICES='0'
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
echo $PYTORCH_CUDA_ALLOC_CONF

source ~/anaconda3/etc/profile.d/conda.sh
# source /mnt/hdd0/qllai/miniconda3/etc/profile.d/conda.sh
conda activate pt

max_len=128
nhead=5
seed=42
num_encoder_layers=2
transformer_dropout=0.1
mlp_dropout=0.1
batch_size=32
learning_rate=0.0005
distribute_epochs=200
warmup_epochs=0

nohup python -u ./src/trainer.py --datasetPath ./data/BGC_train_dataset_inbalence.csv \
    --max_len $max_len --nhead $nhead --seed $seed --num_encoder_layers $num_encoder_layers --transformer_dropout $transformer_dropout \
    --batch_size $batch_size --learning_rate $learning_rate --mlp_dropout $mlp_dropout \
    --distribute_epochs $distribute_epochs --warmup_epochs $warmup_epochs --lmdbPath ./lmdb_BGC \
    >./nohup/transformerEncoder_BGC_loss$max_len$nhead$seed$num_encoder_layers$transformer_dropout$mlp_dropout$batch_size$learning_rate$distribute_epochs$warmup_epochs.out &
