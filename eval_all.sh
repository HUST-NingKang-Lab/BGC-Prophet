export CUDA_VISIBLE_DEVICES='1'
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
echo $PYTORCH_CUDA_ALLOC_CONF

source ~/anaconda3/etc/profile.d/conda.sh
# source /mnt/hdd0/qllai/miniconda3/etc/profile.d/conda.sh
# conda activate pytorch
conda activate pt

nohup python -u src/eval_all.py --lmdbPath ./lmdb_val/ \
    --models_folder ./modelSave/transformerEncoder_TD_focal/bS_64_dE_200_lR_0.0005_mL_128_d_320_nH_5_nEL_2_tdP_0.1_mdP_0.1_alpha_0.9_gamma_0.0_TD/ \
    --datasetPath ./data/Nine_genomes_test_dataset.csv --max_len 128 \
    --batch_size 32 --epochs 200\
    >./eval_nohup/eval_transformerEncoder_eval_all_BGC_9_test_2layer_alpha_0.9_gamma_0.out &
