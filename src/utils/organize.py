import argparse
import pandas as pd
import os
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial

def organize_genome(genome, genomesDir):
    genome_name = genome.split('.')[0]
    gene_seq = []
    with open(genomesDir+genome, 'r') as f:
        for line in f:
            if not line.startswith('>'):
                continue
            else:
                gene_seq.append(line.strip()[1:])
    return pd.Series([genome_name, gene_seq], index=['Genome', 'Gene_sequence'])

def create_organize_parser():
    parser = argparse.ArgumentParser(
        prog="organize",
        description="Organize the input genomes of BGC-Prophet",
    )
    parser.add_argument('--genomesDir', type=str, required=True, help='genomes fasta directory')
    parser.add_argument('--outputPath', type=str, required=False, default='./output/', help='output path')
    parser.add_argument('--name', type=str, required=False, default='output', help='name of the output file')
    parser.add_argument('--threads', type=int, required=False, default=10, help='number of threads')
    return parser

class organizeModule:
    def __init__(self, args) -> None:
        self.args = args
        self.genomesDir = args.genomesDir
        self.outputPath = args.outputPath
        self.name = args.name
        self.threads = args.threads
        
    def organize_genomes(self, ):
        self.dataFrame = pd.DataFrame(columns=['Genome', 'Gene_sequence'])
        files = os.listdir(self.genomesDir)
        files = [file for file in files if file.endswith('.fasta') or file.endswith('.faa')]
        organize_genome_Dir = partial(organize_genome, genomesDir=self.genomesDir)
        with Pool(self.threads) as p:
            dataFrames = list(tqdm(p.imap(organize_genome_Dir, files), total=len(files), desc='Organizing', leave=True))
        self.dataFrame = pd.DataFrame(dataFrames)

    def save(self, ):
        print("Saving to csv...")
        os.makedirs(os.path.dirname(self.outputPath), exist_ok=True)
        self.dataFrame.to_csv(self.outputPath + self.name + '.csv', index=False)


if __name__=='__main__':
    parser = create_organize_parser()
    args = parser.parse_args()
    organize = organizeModule(args)
    organize.organize_genomes()
    organize.save()
