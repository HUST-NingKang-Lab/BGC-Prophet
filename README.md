# BGC-Prophet

Introduction.

## Publications

...

## Installation

using pip

to do

## Using

### End2End Pipline

```shell
BGC-Prophet xxx.fasta --outputPath ./output/ --threshold 0.03 --minGap 3 ...
```

command parameters meaning:


### Download models

You can download trained models from Github releases page:

```shell
wget 
```

### Preprocess

organize some genome into a csv file.

```shell
BGC-Prophet preprocess ./folder/
```


### Get Embedding

We use ESM2-8M model to get genes' embedding vector, and the last layer output of the model is selected as the final word embedding vector for the sequences. You can use the following command:

```shell

BGC-Prophet embed xxx.fasta

```



### Split Sequences

Split the genome into gene sequences of length 128.
```shell
BGC-Prophet split --genomesPath ./genomes.csv --outputPath ./output/ --name split -threads 10
```


### Gene Prediction



### Biosynthetic Classification



## Training and evaluating



### training dataset

