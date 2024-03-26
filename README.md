# BGC-Prophet

BGC-Prophet, a deep learning approach that leverages language processing neural networkmodel to accurately identify known BGCs and extrapolate novel ones. 

![Figure1](images/Figure1.svg?raw=true "Figure1")


## Installation

Install BGC-Prophet using pip:

```shell
pip install bgc_prophet
```

Or you can download the offline installation package from the GitHub release page and install BGC-Prophet using the following command:

```shell
pip install bgc_prophet-0.1.0-py3-none-any.whl
```

BGC-Prophet is developed under the environment of Python3, and uses Pytroch to build the model, GPU devices are recommended to accelerate model infernece.

## Usage

### BGC-Prophet Pipline

BGC-Prophet can detect and classify BGCs in several genomic sequences. The process involves some steps:

1. Utilizing the ESM2 model to extract word embeddings for each gene in the sequences.

2. Organizing multiple genomes and split them into gene sequences of length 128.

3. Using a trained detection model to identify BGC gene at a given threshold.

4. Finally, applying a classification model to categorize the detected BGCs and outputting the results in a CSV file.


```shell
bgc_prophet pipeline --genomesDir ./pathtogenomesdirectory/ --modelPath ./pathto/annotator.pt --saveIntermediate --name nameoftask --threshold 0.5 --max_gap 3 --min_count 2 --classifierPath ./pathto/classifier.pt  --classify_t 0.5
 
```

use `bgc_prophet pipline --help` command for more explanation of parameters.


### Download models

You can download trained models from Github releases page:

```shell
wget https://github.com/HUST-NingKang-Lab/BGC-Prophet/files/12733164/model.tar.gz
```

`annoator.pt` model is used to dectct BGC genes, and `classifier.pt` model is used to classify BGCs.

---
### Step by Step Operation

#### Get Embedding

We use ESM2-8M model to get genes' embedding vector, and the last layer output of the model is selected as the final word embedding vector for the sequences. You can use the following command:

```shell
bgc_prophet extract esm2_t6_8M_UR50D ./genome.fasta ./lmdb_genomes --toks_per_batch 40960 --include mean
```

This operation takes a gene context to be explored as input, with each gene represented by an amino acid sequence, and outputs a folder in the LMBD format, storing the corresponding gene's word embedding vectors. 

If you need to obtain multiple FASTA files, you can specify the "--directory" or "-d" parameter, and the FASTA location parameter should be specified as a folder.

Special amino acid symbols like "J" should be replaced with "L" or "I" manually. This operation has a minor impact on the overall generation of gene embeddings.

#### Organize Genomes

Organize multiple genomes into a csv file ,which can be used to split sequences.
```shell
bgc_prophet organize --genomesDir ./genomesFastaDirectory/ --outputPath ./output/ --name organize --threads 10
```
This operation will generate a csv file(organize.csv) organizing all genomes and their sequences' ids.

#### Split Sequences

Split the genome into gene sequences of length 128.
```shell
bgc_prophet split --genomesPath ./output/organize.csv --outputPath ./output/ --name split --threads 10
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
--outputPath ./output/ --loadIntermediate ./output/intermediate_prediction.npy \
--name output --threshold 0.5 --max_gap 3 --min_count 2
```

The 'TDlabels' column of dataframe loaded from split.csv will be updated, then will output a new csv file named output.csv.

#### Biosynthetic Classify

Apply a trained classifier to categorize the detected BGCs.

```shell
bgc_prophet classify --datasetPath ./output.csv \
--classifierPath ./pathto/classifier.pt \
--outputPath ./output/ --lmdbPath ./lmdb_genomes \
--name classify --device cuda 
```

The finall output will be save as a csv file, containing dection and classification results.

## Publications

Deciphering the Biosynthetic Potential of Microbial Genomes Using a BGC Language Processing Neural Network Model [[bioRxiv]](https://doi.org/10.1101/2023.11.30.569352)

## Maintainer

| Name       | Email                                                     | Organization                                                 |
| ---------- | --------------------------------------------------------- | ------------------------------------------------------------ |
| Qilong Lai | [laiqilong@hust.edu.cn](mailto:laiqilong@hust.edu.cn)     | PhD student, Institute of Neuroscience, Chinese Academy of Sciences |
| Shuai Yao  | [yaoshuai@stu.pku.edu.cn](mailto:yaoshuai@stu.pku.edu.cn) | PhD student, Academy for Advanced interdisciplinary Studies, Peking University |
| Yuguo Zha  | [hugozha@hust.edu.cn](mailto:hugozha@hust.edu.cn)         | PhD student, School of Life Science and Technology, Huazhong University of Science & Technology |
| Kang Ning  | [ningkang@hust.edu.cn](mailto:ningkang@hust.edu.cn)       | Professor, School of Life Science and Technology, Huazhong University of Science & Technology |


