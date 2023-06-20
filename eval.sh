export CUDA_VISIBLE_DEVICES='0'
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
echo $PYTORCH_CUDA_ALLOC_CONF

source ~/anaconda3/etc/profile.d/conda.sh
# source /mnt/hdd0/qllai/miniconda3/etc/profile.d/conda.sh
# conda activate pytorch
conda activate pt

nohup python -u src/eval.py --lmdbPath ./lmdb_val/ \
    --datasetPath ./data/Nine_genomes_test_dataset.csv --max_len 128 \
    --nhead 4 --num_encoder_layers 2 --dropout 0.4 --batch_size 64 \
    --transformerEncoderPath ./modelSave/transformerEncoder_TD_focal/bS_64_dE_200_lR_0.0005_mL_128_d_320_nH_5_nEL_2_tdP_0.1_mdP_0.1_alpha_0.1_gamma_1.0_TD/transformerEncoder_Model_TD_28.pt \
    --name NineGenomes_128_test_28 \
    >eval_transformerEncoder_shared_BGC_9_test_28.out &
