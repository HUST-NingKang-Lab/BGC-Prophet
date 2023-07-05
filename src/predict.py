import torch
import argparse
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import pickle
import random
from data import DataReader, BGCLabelsDataset
import lmdb

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

        # if self.saveIntermediate:
        #     self.env = lmdb.open(self.outputPath+f'lmdb_{self.name}', map_size=307374182400, readonly=False, meminit=False, map_async=True)

        os.makedirs(os.path.dirname(self.outputPath), exist_ok=True)

        self.model = torch.load(self.modelPath)
        self.model.to(self.device)
        self.model.eval()

        self.data = DataReader(self.datasetPath, test_ratio=0)
        self.dataset = BGCLabelsDataset(self.data, self.lmdbPath, 'eval')
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
            if self.saveIntermediate:
                self.results.append(outputsTD)

        self.results = np.concatenate(self.results, axis=0)

    def saveIntermediate(self):
        np.save(self.outputPath+f'intermediate_{self.name}.npy', self.results)

    

if __name__=='__main__':
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

    args = parser.parse_args()

    predict = genePredicter(args)
    predict.predict()
    if args.saveIntermediate:
        predict.saveIntermediate()
