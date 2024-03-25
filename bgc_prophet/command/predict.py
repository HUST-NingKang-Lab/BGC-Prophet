import torch
import argparse
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import pickle
import random
import sys
directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(directory))
# print(sys.path)
from bgc_prophet.train.data import DataReader, BGCLabelsDataset
from bgc_prophet.train.model import transformerEncoderNet
from .baseCommand import baseCommand
from pathlib import Path
import lmdb

class predictCommand(baseCommand):
    name = "predict"
    description = "Predict the gene cluster of given genomes"

    def add_arguments(self, parser):
        parser.add_argument('--datasetPath', type=Path, required=True, help='dataset path')
        parser.add_argument('--modelPath', type=Path, required=True, help='model path')
        parser.add_argument('--outputPath', type=Path, required=False, default=Path('./output/'), help='output path')
        parser.add_argument('--lmdbPath', type=Path, required=True, help='gene representations lmdb path')
        parser.add_argument('--name', type=str, required=False, default='predict', help='name of the output file')
        parser.add_argument('--device', type=str, required=True, choices=["cuda", "cpu"], help='device to use')
        parser.add_argument('--batch_size', type=int, required=False, default=512, help='batch size')
        parser.add_argument('--saveIntermediate', action='store_true', required=False, default=False, help='save intermediate results')

    def handle(self, args):
        predict = genePredicter(args)
        predict.predict()
        if args.saveIntermediate:
            predict.save()

def create_predict_parser():
    parser = argparse.ArgumentParser(
        prog="predict",
        description="Predict the gene cluster of given genomes",
    )
    parser.add_argument('--datasetPath', type=str, required=True, help='dataset path')
    parser.add_argument('--modelPath', type=str, required=True, help='model path')
    parser.add_argument('--outputPath', type=str, required=False, default='./output/', help='output path')
    parser.add_argument('--lmdbPath', type=str, required=True, help='lmdb path')
    parser.add_argument('--name', type=str, required=False, default='predict', help='name of the output file')
    parser.add_argument('--device', type=str, required=True, help='device to use')
    parser.add_argument('--batch_size', type=int, required=False, default=512, help='batch size')
    parser.add_argument('--saveIntermediate', action='store_true', required=False, default=False, help='save intermediate results')
    return parser

class genePredicter:
    def __init__(self, args,):
        self.args = args
        self.datasetPath = args.datasetPath
        self.modelPath = args.modelPath
        self.lmdbPath = args.lmdbPath
        self.outputPath = args.outputPath
        self.device = args.device
        self.batch_size = args.batch_size
        self.name = args.name
        self.saveIntermediate = args.saveIntermediate

        # if not self.saveIntermediate:
        #     self.output = outputFormatter()
        print(self.outputPath)
        os.makedirs(self.outputPath, exist_ok=True)

        self.model = transformerEncoderNet(d_model=320, nhead=5, num_encoder_layers=2, max_len=128,
                                           dim_feedforward=320*4, transformer_dropout=0.1, mlp_dropout=0.5, batch_first=True)
        state_dict = torch.load(self.modelPath, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.data = DataReader(self.datasetPath, test_ratio=0)
        self.dataset = BGCLabelsDataset(self.data, str(self.lmdbPath), 'eval')
        self.dataLoader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=5)

    def predict(self):
        self.results = []
        for i, data in tqdm(enumerate(self.dataLoader), desc='Predict', leave=True, total=len(self.dataLoader)):
            sentence, labels, distribution = data[0], data[1], data[2]
            sentence = sentence.to(self.device)
            labels = labels.to(self.device)
            distribution = distribution.to(self.device)

            outputsTD = self.model(sentence) 
            outputsTD = outputsTD.cpu().detach().numpy()
            self.results.append(outputsTD)

        self.results = np.concatenate(self.results, axis=0)

    def save(self):
        np.save(self.outputPath.joinpath(f'intermediate_{self.name}.npy'), self.results)

    

# if __name__=='__main__':

#     parser = create_predict_parser()
#     args = parser.parse_args()
#     predict = genePredicter(args)
#     predict.predict()
#     if args.saveIntermediate:
#         predict.save()
