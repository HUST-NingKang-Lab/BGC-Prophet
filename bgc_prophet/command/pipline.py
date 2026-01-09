import argparse
import os
from pathlib import Path

from .baseCommand import baseCommand
from .extract import ExtractCommand, run
from .organize import organizeModule
from .split import splitModule
from .predict import genePredicter
from .output import outputFormatter
from .classify import geneClassifier

class piplineCommand(baseCommand):
    name = "pipeline"
    description = "Run the whole pipline of BGC-Prophet"

    def add_arguments(self, parser):
        parser.add_argument('--genomesDir', type=Path, required=True, help='genomes fasta directory')
        parser.add_argument('--lmdbPath', type=Path, required=False, default=None, help='lmdb path')
        parser.add_argument('--toks_per_batch', type=int, required=False, default=40960, help='toks per batch')
        parser.add_argument('--device', type=str, required=False, default='cpu', choices=['cpu', 'cuda'],help='device to use')
        parser.add_argument('--modelPath', type=Path, required=True, help='model path')
        parser.add_argument('--batch_size', type=int, required=False, default=512, help='batch size')
        parser.add_argument('--outputPath', type=Path, required=False, default=Path('./output/'), help='output path')
        parser.add_argument('--saveIntermediate', action='store_true', required=False, default=False, help='save intermediate results')
        parser.add_argument('--name', type=str, required=False, default='output', help='name of the output file')
        parser.add_argument('--threads', type=int, required=False, default=10, help='number of threads')
        parser.add_argument('--threshold', type=float, required=True, default=0.03, help='threshold of the prediction')
        parser.add_argument('--max_gap', type=int, required=True, default=3, help='max geme gap for the prediction')
        parser.add_argument('--min_count', type=int, required=True, default=2, help='min gene count for the prediction')
        parser.add_argument('--classifierPath', type=Path, required=True, help='classifier path')
        parser.add_argument('--classify_t', type=float, required=False, default=0.3, help='threshold for classification')

    def handle(self, args):
        # --------- Pre-validate parameters and paths to avoid long invalid computations ---------
        if not args.modelPath.is_file():
            raise FileNotFoundError(f"Model file not found: {args.modelPath}")
        if not args.classifierPath.is_file():
            raise FileNotFoundError(f"Classifier file not found: {args.classifierPath}")

        if not args.genomesDir.is_dir():
            raise NotADirectoryError(f"Genomes directory not found: {args.genomesDir}")

        genomes_files = [
            f for f in os.listdir(args.genomesDir)
            if f.endswith('.fasta') or f.endswith('.faa')
        ]
        if not genomes_files:
            raise FileNotFoundError(
                f"No .fasta or .faa files found in genomesDir: {args.genomesDir}"
            )

        # --------- LMDB reuse and feature extraction logic ---------
        # If user explicitly provides lmdbPath, use it directly (avoid duplicate extraction)
        if args.lmdbPath is not None:
            pipline = piplineModule(args)
            pipline.process()
            return

        # User did not provide lmdbPath, try to auto-detect default LMDB directory
        default_lmdb = args.outputPath.joinpath('lmdb_' + args.name)
        if default_lmdb.exists() and default_lmdb.is_dir() and any(default_lmdb.iterdir()):
            print(f"Found existing LMDB at {default_lmdb}, reuse it and skip feature extraction.")
            args.lmdbPath = default_lmdb
        else:
            # No reusable LMDB found, perform feature extraction once
            parser_lmdb = argparse.ArgumentParser(
                prog="BGC-Prophet extract",
                description="Extract gene representations from genomes"
            )
            ExtractCommand.add_arguments(ExtractCommand(), parser_lmdb)
            output_name = default_lmdb
            for file in genomes_files:
                file_path = args.genomesDir.joinpath(file)
                args_lmdb = parser_lmdb.parse_args(
                    'esm2_t6_8M_UR50D {file_path} {output_name} --toks_per_batch {toksNum} --include mean'.format(
                        file_path=file_path,
                        output_name=output_name,
                        toksNum=args.toks_per_batch,
                    ).split()
                )
                run(args_lmdb)
            args.lmdbPath = str(output_name)

        pipline = piplineModule(args)
        pipline.process()

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
        self.args.genomesPath = self.outputPath.joinpath(self.name + '.csv')
        self.split = splitModule(self.args)
        self.split.split_genomes()
        # predict BGC
        self.args.datasetPath = self.outputPath.joinpath(self.name + '_split.csv')
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
        self.args.datasetPath = self.outputPath.joinpath(self.name + '.csv')
        self.classifier = geneClassifier(self.args)
        self.classifier.classify()
        self.classifier.process()
        self.classifier.save()
        # clear intermediate results
        if not self.saveIntermediate:
            os.remove(self.outputPath.joinpath(f'intermediate_{self.name}.npy'))
            os.remove(self.outputPath.joinpath(self.name + '_split.csv'))
            os.remove(self.outputPath.joinpath(self.name + '.csv'))
            os.remove(self.outputPath.joinpath(self.name + '_results.npy'))
        


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
    parser.add_argument('--threshold', type=float, required=True, default=0.03, help='threshold of the prediction')
    parser.add_argument('--max_gap', type=int, required=True, default=3, help='max geme gap for the prediction')
    parser.add_argument('--min_count', type=int, required=True, default=2, help='min gene count for the prediction')
    parser.add_argument('--classifierPath', type=str, required=True, help='classifier path')
    parser.add_argument('--classify_t', type=float, required=False, default=0.3, help='threshold for classification')
    # parser.add_argument('--input', '-i', type=str, help='Input fasta file path.')

    args = parser.parse_args()

    # Keep behavior consistent with piplineCommand.handle:
    # 1) Validate paths first; 2) Try to reuse LMDB; 3) Perform feature extraction only when necessary.
    if not os.path.isfile(args.modelPath):
        raise FileNotFoundError(f"Model file not found: {args.modelPath}")
    if not os.path.isfile(args.classifierPath):
        raise FileNotFoundError(f"Classifier file not found: {args.classifierPath}")

    if not os.path.isdir(args.genomesDir):
        raise NotADirectoryError(f"Genomes directory not found: {args.genomesDir}")

    genomes_files = [
        f for f in os.listdir(args.genomesDir)
        if f.endswith('.fasta') or f.endswith('.faa')
    ]
    if not genomes_files:
        raise FileNotFoundError(
            f"No .fasta or .faa files found in genomesDir: {args.genomesDir}"
        )

    # If user has provided lmdbPath, use it directly
    if args.lmdbPath is None:
        default_lmdb = os.path.join(args.outputPath, 'lmdb_' + args.name)
        if os.path.isdir(default_lmdb) and os.listdir(default_lmdb):
            print(f"Found existing LMDB at {default_lmdb}, reuse it and skip feature extraction.")
            args.lmdbPath = default_lmdb
        else:
            parser_lmdb = argparse.ArgumentParser(
                prog="BGC-Prophet extract",
                description="Extract gene representations from genomes"
            )
            ExtractCommand.add_arguments(ExtractCommand(), parser_lmdb)
            output_name = default_lmdb
            for file in genomes_files:
                file_path = os.path.join(args.genomesDir, file)
                args_lmdb = parser_lmdb.parse_args(
                    'esm2_t6_8M_UR50D {genomesDir}{filename} {outputDir} --toks_per_batch {toksNum} --include mean'.format(
                        genomesDir="",
                        filename=file_path,
                        outputDir=output_name,
                        name=args.name,
                        toksNum=args.toks_per_batch,
                    ).split()
                )
                run(args_lmdb)
            args.lmdbPath = output_name

    pipline = piplineModule(args)
    pipline.process()

