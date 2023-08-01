import pandas as pd
import argparse
from tqdm import tqdm
from multiprocessing import Pool
import os

def create_split_parser():
    parser = argparse.ArgumentParser(
        prog='Split',
        description='Split genomes into 128 gene sequences',
    )
    parser.add_argument('--genomesPath', type=str, required=True, help='genomes path')
    parser.add_argument('--outputPath', type=str, required=False, default='./output/', help='output path')
    parser.add_argument('--name', type=str, required=False, default='split', help='name of the output file')
    parser.add_argument('--threads', type=int, required=False, default=10, help='number of threads')
    return parser

def split_genome(input):
    index, series = input
    genome = series['Genome']
    gene_seq = eval(series['Gene_sequence'])
    dataFrame = pd.DataFrame(columns=['ID', 'sentence', 'labels', 'isBGC', 'TDsentence', 'TDlabels'])
    if len(gene_seq) <128:
        while len(gene_seq) < 128:
            gene_seq += gene_seq
        sentence = gene_seq[:128]
        ID = genome + '_' + str(0)
        dataFrame.loc[len(dataFrame)] = [ID, None, 'Unknown', 'No', sentence, [0]*2]
        return dataFrame
    for i in range(0, len(gene_seq)//128+1):
        if (i+1)*128 > len(gene_seq):
            sentence = gene_seq[-128:]
        else:
            sentence = gene_seq[i*128:(i+1)*128]
        ID = genome + '_' + str(i)
        dataFrame.loc[len(dataFrame)] = [ID, None, 'Unknown', 'No', sentence, [0]*2]
        # len(gene_seq) == n*128, deduplicate
        if (i+1)*128 == len(gene_seq):
            break
    return dataFrame

class splitModule:
    def __init__(self, args) -> None:
        self.args = args
        self.genomesPath = args.genomesPath
        self.outputPath = args.outputPath
        self.name = args.name
        self.threads = args.threads

    def split_genomes(self):
        dataFrame = pd.read_csv(self.genomesPath)
        
        with Pool(self.threads) as p:
            dataFrames = list(tqdm(p.imap(split_genome, dataFrame.iterrows()), total=len(dataFrame), desc='Splitting', leave=True))
        splitDataFrame = pd.concat(dataFrames)

        # for index, row in tqdm(dataFrame.iterrows(), desc='Splitting', total=len(dataFrame), leave=True):
        #     genome = row['Genome']
        #     gene_seq = row['Gene_sequence']
        #     self.__split(genome, gene_seq, dataset)
        print("Saving to csv...")
        # splitDataFrame['sentence'] = splitDataFrame['sentence'].apply(lambda x: ' '.join(x))
        splitDataFrame['TDsentence'] = splitDataFrame['TDsentence'].apply(lambda x: ' '.join(x))
        splitDataFrame['TDlabels'] = splitDataFrame['TDlabels'].apply(lambda x:' '.join([str(i) for i in x]))
        os.makedirs(os.path.dirname(self.outputPath), exist_ok=True)
        splitDataFrame.to_csv(self.outputPath + self.name + '_split.csv', index=False)

if __name__=='__main__':
    parser = create_split_parser()
    args = parser.parse_args()
    split = splitModule(args)
    split.split_genomes()

