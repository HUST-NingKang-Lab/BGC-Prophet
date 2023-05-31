export CUDA_VISIBLE_DEVICES='0'
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
echo $PYTORCH_CUDA_ALLOC_CONF

source ~/anaconda3/etc/profile.d/conda.sh
# source /mnt/hdd0/qllai/miniconda3/etc/profile.d/conda.sh
# conda activate pytorch
conda activate pt

nohup python -u src/eval.py --lmdbPath ./lmdb_val/ \
    --datasetPath ./data/Nine_genomes_test_dataset.csv --max_len 128 \
    --nhead 4 --num_encoder_layers 4 --dropout 0.2 --batch_size 32 \
    --transformerEncoderPath ./modelSave/transformerEncoder_TD/bS_32_dE_1000_lR_0.01_mL_128_d_320_nH_4_nEL_6_dP_0.2_TD/transformerEncoder_Model_TD_500.pt \
    --name NineGenomes_128_test \
    >nohup_transformerEncoder_BGC.out &
