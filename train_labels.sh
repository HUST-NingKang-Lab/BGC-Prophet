#!/bin/bash
export CUDA_VISIBLE_DEVICES='1'
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
mlp_dropout=0.5
batch_size=64
learning_rate=0.01
epochs=150
alpha=0.1
gamma=1


nohup python -u ./src/classifyTrainer.py --datasetPath ./data/BGC_train_dataset_classify.csv \
    --max_len $max_len --nhead $nhead --seed $seed --num_encoder_layers $num_encoder_layers --transformer_dropout $transformer_dropout \
    --batch_size $batch_size --learning_rate $learning_rate --mlp_dropout $mlp_dropout \
    --epochs $epochs  --lmdbPath ./lmdb_BGC --alpha $alpha --gamma $gamma \
    >./nohup/transformerClassifier7_mean_unmask_$max_len$nhead$seed$num_encoder_layers$transformer_dropout$mlp_dropout$batch_size$learning_rate$epochs$alpha$gamma.out &
