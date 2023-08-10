#!/bin/bash
export CUDA_VISIBLE_DEVICES='1'
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
echo $PYTORCH_CUDA_ALLOC_CONF

source ~/anaconda3/etc/profile.d/conda.sh
# source /mnt/hdd0/qllai/miniconda3/etc/profile.d/conda.sh
conda activate pt


nohup python -u src/utils/pipline.py --genomesDir /data4/yaoshuai/Aspergillus/faa/ \
    --lmdbPath /data4/yaoshuai/Aspergillus/lmdb/ --threads 50 --batch_size 1536\
    --outputPath /data4/yaoshuai/Aspergillus/output/ --device cuda \
    --modelPath ./modelSave/transformerEncoder_TD_loss/bS_32_dE_200_lR_0.0005_mL_128_d_320_nH_5_nEL_2_tdP_0.1_mdP_0.1_TD/transformerEncoder_Model_TD_28.pt \
    --saveIntermediate --name Aspergillus --threshold 0.5 --max_gap 2 --min_count 2 \
    --classifierPath ./modelSave/transformerClassifier/transformerClassifier_128_5_2_0.5_0.1_0.01_150_0.1_1.0/transformerClassifier_50.pt \
    --classify_t 0.5 \
    >./nohup/Aspergillus.out &
 