import numpy as np
import pandas as pd
import argparse
from pathlib import Path

from .baseCommand import baseCommand

class outputCommand(baseCommand):
    name = "output"
    description = "Output the results of the prediction"

    def add_arguments(self, parser):
        parser.add_argument('--datasetPath', type=Path, required=True, help='dataset path')
        parser.add_argument('--outputPath', type=Path, required=False, default=Path('./output/'), help='output path')
        parser.add_argument('--loadIntermediate', type=str, required=True, default=None, help='load intermediate results')
        parser.add_argument('--name', type=str, required=False, default='output', help='name of the output file')
        parser.add_argument('--threshold', type=float, required=True, default=0.5, help='threshold for the prediction')
        parser.add_argument('--max_gap', type=int, required=True, default=3, help='max gene gap for the prediction')
        parser.add_argument('--min_count', type=int, required=True, default=2, help='min gene count for the prediction')
    
    def handle(self, args):
        output = outputFormatter(args)
        output.load()
        output.save(output.results)


class outputFormatter:
    def __init__(self, args) -> None:
        self.datasetPath = args.datasetPath
        self.outputPath = args.outputPath
        self.loadIntermediate = args.loadIntermediate
        self.name = args.name
        self.threshold = args.threshold
        self.max_gap = args.max_gap
        self.min_count = args.min_count
        # self.dataFrame = pd.DataFrame(columns=['ID', 'sentence', 'labels', 'isBGC', 'TDsentence', 'TDlabels'])

    def load(self):
        self.dataset = pd.read_csv(self.datasetPath)
        if self.loadIntermediate:
            self.results = np.load(self.loadIntermediate)

    def organize(self, results):
        self.dataset['result'] = results.tolist()
        # print(self.dataset.head())
        outputDataset = self.dataset.apply(self.output, axis=1, result_type='expand')
        self.dataFrame = self.dataset.loc[:, ['ID', 'TDsentence', 'labels', 'isBGC']]
        # print(self.dataFrame.head())
        self.dataFrame['sentence'] = outputDataset['sentence'].copy()
        self.dataFrame['TDlabels'] = outputDataset['TDlabels'].copy()
        self.dataFrame['isBGC'] = outputDataset['isBGC'].copy()
        # print(self.dataFrame.head())
        self.dataFrame['sentence'] = self.dataFrame['sentence'].apply(lambda x: ' '.join(x) if x!=None else None)
        self.dataFrame['TDlabels'] = self.dataFrame['TDlabels'].apply(lambda x: ' '.join([str(i) for i in x]))
        # print(self.dataFrame.head())

    def output(self, series):
        TDlabels = []
        result = series['result']
        TDsentence = series['TDsentence'].split()
        result = np.array(result)
        result[result>self.threshold] = 1.
        result[result<=self.threshold] = 0.
        TDlabels = result.tolist()
        if not 1. in TDlabels:
            TDlabels = [0]*len(TDlabels)
            return pd.Series([None, 'No', TDlabels], index=['sentence', 'isBGC', 'TDlabels'])
        current_start = TDlabels.index(1.)
        current_end = current_start
        output_start = 0
        output_end = 0
        max_range = 0
        for i in range(current_start+1, len(TDlabels)):
            if TDlabels[i]==1:
                if i-current_end<=self.max_gap:
                    current_end = i
                else:
                    if current_end-current_start>max_range:
                        output_start = current_start
                        output_end = current_end
                        max_range = current_end-current_start
                    current_start = i
                    current_end = i
        if current_end-current_start>max_range:
            output_start = current_start
            output_end = current_end
            max_range = current_end-current_start
        if max_range<self.min_count:
            isBGC = 'No'
            sentence = None
            TDlabels = [0]*len(TDlabels)
        else:
            sentence = TDsentence[output_start:output_end+1]
            isBGC = 'Yes'
            TDlabels = [0]*output_start+[1]*(output_end-output_start+1)+[0]*(len(TDlabels)-output_end-1)            
        return pd.Series([sentence, isBGC, TDlabels], index=['sentence', 'isBGC', 'TDlabels'])

    def save(self, results):
        self.organize(results)
        self.dataFrame.to_csv(self.outputPath.joinpath(f'{self.name}.csv'), index=False, header=True)

# if __name__=='__main__':
#     parser = argparse.ArgumentParser(
#         prog='output',
#         description='Output the results of the prediction',
#     )
#     parser.add_argument('--datasetPath', type=str, required=True, help='dataset path')
#     parser.add_argument('--outputPath', type=str, required=False, default='./output/', help='output path')
#     parser.add_argument('--loadIntermediate', type=str, required=True, default=None, help='load intermediate results')
#     parser.add_argument('--name', type=str, required=False, default='output', help='name of the output file')
#     parser.add_argument('--threshold', type=float, required=True, default=0.5, help='threshold for the prediction')
#     parser.add_argument('--max_gap', type=int, required=True, default=3, help='max gene gap for the prediction')
#     parser.add_argument('--min_count', type=int, required=True, default=2, help='min gene count for the prediction')

#     args = parser.parse_args()

#     output = outputFormatter(args)
#     # output = outputFormatter(args.datasetPath, args.outputPath, args.loadIntermediate, args.name, args.threshold, args.max_gap, args.min_count)
#     output.load()
#     output.save(output.results)
