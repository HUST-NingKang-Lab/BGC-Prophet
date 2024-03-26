import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

import sys
from pathlib import Path
# dicrectory = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(dicrectory))
from bgc_prophet.train.data import DataReader, BGCLabelsDataset
from bgc_prophet.train.classifier import transformerClassifier
from .baseCommand import baseCommand
import lmdb

class classifyCommand(baseCommand):
    name = "classify"
    description = "Classify the gene cluster of given genomes"

    def add_arguments(self, parser):
        parser.add_argument('--datasetPath', type=Path, required=True, help='dataset path')
        parser.add_argument('--classifierPath', type=Path, required=True, help='classifier path')
        parser.add_argument('--outputPath', type=Path, required=False, default=Path('./output/'), help='output path')
        parser.add_argument('--lmdbPath', type=Path, required=True, help='lmdb path')
        parser.add_argument('--name', type=str, required=False, default='classify', help='name of the output file')
        parser.add_argument('--device', type=str, required=True, help='device to use')
        parser.add_argument('--batch_size', type=int, required=False, default=512, help='batch size')
        parser.add_argument('--classify_t', type=float, required=False, default=0.5, help='threshold for classification')

    def handle(self, args):
        classifier = geneClassifier(args)
        classifier.classify()
        classifier.process()
        classifier.save()

# def create_classify_parser():
#     parser = argparse.ArgumentParser(
#         prog="classify",
#         description="Classify the gene cluster of given genomes",
#     )
#     parser.add_argument('--datasetPath', type=str, required=True, help='dataset path')
#     parser.add_argument('--classifierPath', type=str, required=True, help='classifier path')
#     parser.add_argument('--outputPath', type=str, required=False, default='./output/', help='output path')
#     parser.add_argument('--lmdbPath', type=str, required=True, help='lmdb path')
#     parser.add_argument('--name', type=str, required=False, default='classify', help='name of the output file')
#     parser.add_argument('--device', type=str, required=True, help='device to use')
#     parser.add_argument('--batch_size', type=int, required=False, default=512, help='batch size')
#     parser.add_argument('--classify_t', type=float, required=False, default=0.5, help='threshold for classification')

#     return parser

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

        self.model = transformerClassifier(d_model=320, nhead=5, num_encoder_layers=2, max_len=128, 
                                           dim_feedforward=320*4, labels_num=7, transformer_dropout=0.1, 
                                           mlp_dropout=0.5)
        state_dict = torch.load(self.classifierPath, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        # print(self.datasetPath)
        self.data = DataReader(self.datasetPath, test_ratio=0)
        self.dataset = BGCLabelsDataset(self.data, str(self.lmdbPath), 'eval')
        self.dataLoader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=5)
        self.OtherIndex = self.data.labels_list.index('Other')
        
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
        self.classes = self.results.copy()
        dataFrame = pd.read_csv(self.datasetPath)
        labels_list = self.data.labels_list
        for i, row in dataFrame.iterrows():
            if row['isBGC'] == 'No':
                dataFrame.loc[i, 'labels'] = 'NonBGC'
                dataFrame.loc[i, 'probability'] = 1.
                continue
            result = self.results[i]
            prediction = result.copy()
            threshold = max(result[self.OtherIndex], self.classify_t)
            prediction[result>=threshold] = 1
            prediction[result<threshold] = 0
            if 1 in prediction:
                labels = [labels_list[j] for j in range(len(labels_list)) if prediction[j]==1]
                probability = [result[j] for j in range(len(labels_list)) if prediction[j]==1]
                if 'Other' in labels and len(labels)>1:
                    oindex = labels.index('Other')
                    labels.pop(oindex)
                    probability.pop(oindex)
                dataFrame.loc[i, 'labels'] = ' '.join(labels)
                dataFrame.loc[i, 'probability'] = sum(probability)/len(probability)
            else:
                dataFrame.loc[i, 'labels'] = 'Unknown'
                dataFrame.loc[i, 'probability'] = 1 - np.max(result)
            
        self.dataFrame = dataFrame

    def save(self):
        np.save(self.outputPath.joinpath(self.name+'_results.npy'), self.results)
        self.dataFrame['probability'] = self.dataFrame['probability'].apply(lambda x: '%.3f' % x)
        self.dataFrame.to_csv(self.outputPath.joinpath(self.name+'_classified.csv'), index=False)

# if __name__ == "__main__":
#     parser = create_classify_parser()
#     args = parser.parse_args()
#     classifier = geneClassifier(args)
#     classifier.classify()
#     classifier.process()
#     classifier.save()
