python -u predict.py --datasetPath ../output/Aspergillus.csv \
--modelPath \../modelSave/transformerEncoder_TD_loss/bS_32_dE_200_lR_0.0005_mL_128_d_320_nH_5_nEL_2_tdP_0.1_mdP_0.1_TD/transformerEncoder_Model_TD_28.pt \
--outputPath ../output/ --lmdbPath /data4/yaoshuai/Aspergillus/lmdb \
--name Aspergillus --device cuda --saveIntermediate