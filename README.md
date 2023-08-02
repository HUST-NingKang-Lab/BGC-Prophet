# BGC-Prophet

Introduction.

## Publications

...

## Installation

using pip

to do

## Using

### BGC-Prophet Pipline

BGC-Prophet can detect and classify BGCs in server genomic sequence. The process involves several steps:

1. Utilizing the ESM2 model to extract word embeddings for each gene in the sequences.

2. Organizing multiple genomes and split them into gene sequences of length 128.

3. Using a trained detection model to identify BGC gene at a given threshold.

4. Finally, applying a classification model to categorize the detected BGCs and outputting the results in a CSV file.


```shell
BGC-Prophet pipline  --genomesDir ./genomesFastaDirectory/ \
 --modelPath ./transformerEncoder_Model_TD_28.pt \
 --saveIntermediate --name results --threshold 0.03 --max_gap 3 --min_count 2 \
 --classifierPath ./transformerClassifier_100.pt \
 
```

command parameters meaning:


### Download models

You can download trained models from Github releases page:

```shell
wget 
```


### Get Embedding

We use ESM2-8M model to get genes' embedding vector, and the last layer output of the model is selected as the final word embedding vector for the sequences. You can use the following command:

```shell
BGC-Prophet extract esm2_t6_8M_UR50D ./genomes.fasta ./lmdb_genomes --toks_per_batch 40960 --include mean
```

This operation will get a lmdb dataset folder, containing all sequences embeddings.

### Organize Genomes

Organize multiple genomes into a csv file ,which can be used to split sequences.
```shell
BGC-Prophet organize --genomesDir ./genomesFastaDirectory/ --outputPath ./output/ --name organize --threads 10
```
This operation will generate a csv file(organize.csv) organizing all genomes and their sequences' ids.

### Split Sequences

Split the genome into gene sequences of length 128.
```shell
BGC-Prophet split --genomesPath ./output/organize.csv --outputPath ./output/ --name split -threads 10
```
This operatione will get a csv file(split.csv), all genomes will be split into gene id sequences of length 128.

### Gene Prediction

Use a trained dectection model to indetify BGC genes at a given threshold.

```shell
BGC-Prophet predict --datasetPath ./output/split.csv \
--modelPath ./transformerEncoder_Model_TD_28.pt \
--outputPath ./output/ --lmdbPath ./lmdb_genomes \
--name prediction --device cuda --saveIntermediate
```
This command will use GPU to detect BGCs' gene, and the 'TDlabels' column of dataframe loaded from split.csv will be updated, then will output a new csv file named prediction.csv. If 'saveIntermediate' parameter is specified, results of prediction will be saved as a numpy file.

### Biosynthetic Classification

Apply a trained classifier to categorize the detected BGCs.

```shell
BGC-Prophet classify --datasetPath ./prediction.csv \
--classifierPath ./transformerClassifier_100.pt \
--outputPath ./output/ --lmdbPath ./lmdb_genomes \
--name classify --device cuda 
```

The finall output will be save as a csv file, containing dection and classification results.

## Training and evaluating



### training dataset

