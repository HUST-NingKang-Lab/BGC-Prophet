export CUDA_VISIBLE_DEVICES='1'
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
echo $PYTORCH_CUDA_ALLOC_CONF

source ~/anaconda3/etc/profile.d/conda.sh
# source /mnt/hdd0/qllai/miniconda3/etc/profile.d/conda.sh
# conda activate pytorch
conda activate pt

nohup python -u src/classifyEval.py --lmdbPath ./lmdb_train/ \
    --datasetPath ./data/BGC_train_dataset_classify.csv --max_len 128 \
    --batch_size 64 \
    --modelPath ./modelSave/transformerClassifier/transformerClassifier_128_5_2_0.5_0.1_0.01_150_0.1_1.0/transformerClassifier_50.pt \
    --name classify_last_7——unmask \
    >eval_classify_last_7_unmask.out &
