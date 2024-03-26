import argparse
import pandas as pd
import os
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
from .baseCommand import baseCommand
from pathlib import Path
from Bio import SeqIO


def organize_genome(genome, genomesDir):
    genome_name = os.path.splitext(genome)[0]
    # gene_seq = []
    genome_seq = list(SeqIO.parse(genomesDir.joinpath(genome), 'fasta'))
    gene_names = [seq.id for seq in genome_seq]
    # with open(genomesDir.joinpath(genome), 'r') as f:
    #     for line in f:
    #         if not line.startswith('>'):
    #             continue
    #         else:
    #             gene_seq.append(line.strip()[1:])
    return pd.Series([genome_name, gene_names], index=['Genome', 'Gene_sequence'])

class organizeCommand(baseCommand):
    name = 'organize'
    description = 'Organize the input genomes of BGC-Prophet'

    def add_arguments(self, parser):
        parser.add_argument('--genomesDir', type=Path, required=True, help='genomes fasta directory')
        parser.add_argument('--outputPath', type=Path, required=False, default=Path('./output/'), help='output path')
        parser.add_argument('--name', type=str, required=False, default='output', help='name of the output file')
        parser.add_argument('--threads', type=int, required=False, default=10, help='number of cpu threads to use in organizing')

    def handle(self, args):
        organize = organizeModule(args)
        organize.organize_genomes()
        organize.save()

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
        os.makedirs(self.outputPath, exist_ok=True)
        self.dataFrame.to_csv(self.outputPath.joinpath(self.name+'.csv'), index=False)


# if __name__=='__main__':
#     parser = create_organize_parser()
#     args = parser.parse_args()
#     organize = organizeModule(args)
#     organize.organize_genomes()
#     organize.save()
