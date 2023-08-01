import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

import sys
dicrectory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(dicrectory))
from data import DataReader, BGCLabelsDataset
from classifier import transformerClassifier
import lmdb

def create_classify_parser():
    parser = argparse.ArgumentParser(
        prog="classify",
        description="Classify the gene cluster of given genomes",
    )
    parser.add_argument('--datasetPath', type=str, required=True, help='dataset path')
    parser.add_argument('--classifierPath', type=str, required=True, help='classifier path')
    parser.add_argument('--outputPath', type=str, required=False, default='./output/', help='output path')
    parser.add_argument('--lmdbPath', type=str, required=True, help='lmdb path')
    parser.add_argument('--name', type=str, required=False, default='classify', help='name of the output file')
    parser.add_argument('--device', type=str, required=True, help='device to use')
    parser.add_argument('--batch_size', type=int, required=False, default=512, help='batch size')
    parser.add_argument('--classify_t', type=float, required=False, default=0.5, help='threshold for classification')

    return parser

class geneClassifier:
    def __init__(self, args) -> None:
        self.args = args
        self.datasetPath = args.datasetPath
        self.classifierPath = args.classifierPath
        self.lmdbPath = args.lmdbPath
        self.outputPath = args.outputPath
        self.device = args.device
        self.batch_size = args.batch_size
        self.name = args.name
        self.classify_t = args.classify_t

        self.model = torch.load(self.classifierPath)
        self.model.to(self.device)
        self.model.eval()
        
        self.data = DataReader(self.datasetPath, test_ratio=0)
        self.dataset = BGCLabelsDataset(self.data, self.lmdbPath, 'eval')
        self.dataLoader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=5)
        
    def classify(self):
        self.results = []
        for i, data in tqdm(enumerate(self.dataLoader), desc='Classify', leave=True, total=len(self.dataLoader)):
            sentence, labels, distribution = data[0], data[1], data[2]
            sentence = sentence.to(self.device)
            labels = labels.to(self.device)
            distribution = distribution.to(self.device)

            output = self.model(src=sentence, src_key_padding_mask=distribution)
            classes = output.detach().cpu().numpy()
            self.results.append(classes)
        self.results = np.concatenate(self.results, axis=0)

    def process(self):
        self.classes = self.results[:]
        self.classes[self.classes>=self.classify_t] = 1
        self.classes[self.classes<self.classify_t] = 0
        dataFrame = pd.read_csv(self.datasetPath)
        labels_list = self.data.labels_list
        for i, row in dataFrame.iterrows():
            prediction = self.classes[i]
            if 1 in prediction:
                dataFrame.loc[i, 'labels'] = ' '.join([labels_list[j] for j in range(len(labels_list)) if prediction[j]==1])
            else:
                dataFrame.loc[i, 'labels'] = 'Unknown'
        self.dataFrame = dataFrame

    def save(self):
        np.save(self.outputPath+self.name+'_results.npy', self.results)
        self.dataFrame.to_csv(self.outputPath+self.name+'_classified.csv', index=False)

if __name__ == "__main__":
    parser = create_classify_parser()
    args = parser.parse_args()
    classifier = geneClassifier(args)
    classifier.classify()
    classifier.process()
    classifier.save()
