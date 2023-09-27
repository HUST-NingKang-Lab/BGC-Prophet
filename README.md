# BGC-Prophet

BGC-Prophet, a deep learning approach that leverages language processing neural networkmodel to accurately identify known BGCs and extrapolate novel ones. 

![figure1](images/figure1.png?raw=true "figure1")

## Publications

...

## Installation

Install BGC-Prophet using pip:

```shell
pip install bgc_prophet
```

BGC-Prophet is developed under the environment of Python3, and uses Pytroch to build the model, GPU devices are recommended to accelerate model infernece.

## Using

### BGC-Prophet Pipline

BGC-Prophet can detect and classify BGCs in server genomic sequence. The process involves several steps:

1. Utilizing the ESM2 model to extract word embeddings for each gene in the sequences.

2. Organizing multiple genomes and split them into gene sequences of length 128.

3. Using a trained detection model to identify BGC gene at a given threshold.

4. Finally, applying a classification model to categorize the detected BGCs and outputting the results in a CSV file.


```shell
bgc_prophet pipline --genomesDir ./pathtogenomesdir/ --modelPath ./pathto/annotator.pt --saveIntermediate --name nameoftask --threshold 0.03 --max_gap 3 --min_count 2 --classifierPath ./pathto/classifier.pt  --classify_t 0.5
 
```

use `bgc_prophet pipline --help` command for more help


### Download models

You can download trained models from Github releases page:

```shell
wget 
```

`annoator.pt` model is used to dectct BGC genes, and `classifier.pt` model is used to classify BGCs.

---
### Step by Step Operation

#### Get Embedding

We use ESM2-8M model to get genes' embedding vector, and the last layer output of the model is selected as the final word embedding vector for the sequences. You can use the following command:

```shell
bgc_prophet extract esm2_t6_8M_UR50D ./genomes.fasta ./lmdb_genomes --toks_per_batch 40960 --include mean
```

This operation will get a lmdb dataset folder, containing all sequences embeddings.

#### Organize Genomes

Organize multiple genomes into a csv file ,which can be used to split sequences.
```shell
bgc_prophet organize --genomesDir ./genomesFastaDirectory/ --outputPath ./output/ --name organize --threads 10
```
This operation will generate a csv file(organize.csv) organizing all genomes and their sequences' ids.

#### Split Sequences

Split the genome into gene sequences of length 128.
```shell
bgc_prophet split --genomesPath ./output/organize.csv --outputPath ./output/ --name split -threads 10
```
This operatione will get a csv file(split.csv), all genomes will be split into gene id sequences of length 128.

#### Gene Prediction

Use a trained dectection model to indetify BGC genes at a given threshold.

```shell
bgc_prophet predict --datasetPath ./output/split.csv --modelPath ./annotator.pt --outputPath ./output/ --lmdbPath ./lmdb_genomes --name prediction --device cuda --saveIntermediate
```
This command will use GPU to detect BGCs' gene, if 'saveIntermediate' parameter is specified, results of prediction will be saved as a numpy file.

#### Output format

Merge genes within a distance of 'max_gap' to form a single BGC, and filter out BGCs composed of fewer than 'min_count' genes, predict and output the BGC with the highest confidence and broadest coverage.

```shell
bgc_prophet output --datasetPath ./output/split.csv \
--outputPath ./output/ --loadIntermediate ./output/intermediate_rediction.npy \
--name output --threshold 0.03 --max_gap 3 --min_count 2
```

The 'TDlabels' column of dataframe loaded from split.csv will be updated, then will output a new csv file named prediction.csv.

#### Biosynthetic Classify

Apply a trained classifier to categorize the detected BGCs.

```shell
bgc_prophet classify --datasetPath ./prediction.csv \
--classifierPath ./transformerClassifier_100.pt \
--outputPath ./output/ --lmdbPath ./lmdb_genomes \
--name classify --device cuda 
```

The finall output will be save as a csv file, containing dection and classification results.



