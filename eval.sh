export CUDA_VISIBLE_DEVICES='0'
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
echo $PYTORCH_CUDA_ALLOC_CONF

source ~/anaconda3/etc/profile.d/conda.sh
# source /mnt/hdd0/qllai/miniconda3/etc/profile.d/conda.sh
# conda activate pytorch
conda activate pt

nohup python -u src/eval.py --lmdbPath ./lmdb_val/ \
    --datasetPath ./data/Nine_genomes_test_dataset.csv --max_len 128 \
    --nhead 4 --num_encoder_layers 4 --dropout 0.4 --batch_size 64 \
    --transformerEncoderPath ./modelSave/transformerEncoder_TD/bS_64_dE_200_lR_0.0005_mL_128_d_320_nH_5_nEL_8_dP_0.5_alpha_0.95_gamma_2.0_TD/transformerEncoder_Model_TD_70.pt \
    --name NineGenomes_128_test_no_insert_shared \
    >eval_transformerEncoder_shared_BGC_9_test.out &
