import argparse
import os

from extract_BGC import *
from organize import organizeModule
from split import splitModule
from predict import genePredicter
from output import outputFormatter
from classify import geneClassifier

class piplineModule:
    def __init__(self, args) -> None:
        self.args = args
        self.genomesDir = args.genomesDir
        self.lmdbPath = args.lmdbPath
        self.toks_per_batch = args.toks_per_batch
        self.device = args.device
        self.modelPath = args.modelPath
        self.batch_size = args.batch_size
        self.outputPath = args.outputPath
        self.saveIntermediate = args.saveIntermediate
        self.name = args.name

    def process(self, ):
        # organize genomes
        self.organize = organizeModule(self.args)
        self.organize.organize_genomes()
        self.organize.save()
        # split genomes
        self.args.genomesPath = self.outputPath + self.name + '.csv'
        self.split = splitModule(self.args)
        self.split.split_genomes()
        # predict BGC
        args.datasetPath = self.outputPath + self.name + '_split.csv'
        self.predict = genePredicter(self.args)
        self.predict.predict()
        self.predict.save()
        # output
        self.results = self.predict.results
        self.args.loadIntermediate = None
        self.output = outputFormatter(self.args)
        self.output.load()
        self.output.save(self.results)
        # classify BGC
        self.classifier = geneClassifier(self.args)
        self.classifier.classify()
        self.classifier.process()
        self.classifier.save()
        # clear intermediate results
        if not self.saveIntermediate:
            os.remove(self.outputPath + f'intermediate_{self.name}.npy')
            os.remove(self.outputPath + self.name + '_split.csv')
            os.remove(self.outputPath + self.name + '.csv')
            os.remove(self.outputPath + self.name + '_results.npy')
            os.remove(self.outputPath + self.name + '_classified.csv')
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BGC-Prophet process pipline.')
    # parser.add_argument('--genomesPath', type=str, required=True, help='genomes csv path')
    parser.add_argument('--genomesDir', type=str, required=True, help='genomes fasta directory')
    parser.add_argument('--lmdbPath', type=str, required=False, default=None, help='lmdb path')
    parser.add_argument('--toks_per_batch', type=int, required=False, default=40960, help='toks per batch')
    parser.add_argument('--device', type=str, required=False, default='cpu', help='device to use')
    parser.add_argument('--modelPath', type=str, required=True, help='model path')
    parser.add_argument('--batch_size', type=int, required=False, default=512, help='batch size')
    parser.add_argument('--outputPath', type=str, required=False, default='./output/', help='output path')
    parser.add_argument('--saveIntermediate', action='store_true', required=False, default=False, help='save intermediate results')
    parser.add_argument('--name', type=str, required=False, default='output', help='name of the output file')
    parser.add_argument('--threads', type=int, required=False, default=10, help='number of threads')
    parser.add_argument('--threshold', type=float, required=True, default=0.5, help='threshold of the prediction')
    parser.add_argument('--max_gap', type=int, required=True, default=3, help='max geme gap for the prediction')
    parser.add_argument('--min_count', type=int, required=True, default=2, help='min gene count for the prediction')
    parser.add_argument('--classifierPath', type=str, required=True, help='classifier path')
    parser.add_argument('--classify_t', type=float, required=False, default=0.5, help='threshold for classification')
    # parser.add_argument('--input', '-i', type=str, help='Input fasta file path.')

    args = parser.parse_args()
    if args.lmdbPath is None:
        parser_lmdb = create_parser()
        for file in os.listdir(args.genomesDir):
            if file.endswith('.fasta') or file.endswith('.faa'):
                args_lmdb = parser_lmdb.parse_args('esm2_t6_8M_UR50D {genomesDir}{filename} {outputDir}lmdb_{name} --toks_per_batch {toksNum} --include mean'.format(
                    genomesDir=args.genomesDir, filename=file, outputDir=args.outputPath, name=args.name, toksNum=args.toks_per_batch).split())
                run(args_lmdb)
        args.lmdbPath = args.outputPath + 'lmdb_' + args.name
    pipline = piplineModule(args)
    pipline.process()

