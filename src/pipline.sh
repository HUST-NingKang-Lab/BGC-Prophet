 python -u src/utils/pipline.py --genomesDir ~/data/validation_data/9genomes/formal/faa/ \
 --lmdbPath ./output/lmdb_Nine_0.5/ \
 --modelPath ./modelSave/transformerEncoder_TD_loss/bS_32_dE_200_lR_0.0005_mL_128_d_320_nH_5_nEL_2_tdP_0.1_mdP_0.1_TD/transformerEncoder_Model_TD_28.pt \
 --saveIntermediate --name Nine_0.03 --threshold 0.03 --max_gap 3 --min_count 2 \
 --classifierPath ./modelSave/transformerClassifier/transformerClassifier_128_5_2_0.5_0.1_0.01_200_0.05_1.0/transformerClassifier_100.pt \
 --classify_t 0.3
 