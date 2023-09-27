import torch
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter   
from tqdm import tqdm
import os
import pickle
import numpy as np
import random

from data import DataReader, BGCLabelsDataset
from loss import trainLoss
from utils import evaluate
# from model import transformerEncoderNet
# from classifyTrainer import TransformerClassifier
from classifier import transformerClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix

class classifyEval:
    def __init__(self, args, writer, data, model, loss) -> None:
        self.args = args
        self.writer = writer
        self.data = data
        self.model = model
        self.loss = loss

        self.lmdbPath = args.lmdbPath
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.modelPath = args.modelPath
        self.name = args.name
        self.results = []
        self.labels = []

        self.dataset = BGCLabelsDataset(self.data, self.lmdbPath, 'eval')
        self.dataLoader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=5)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        os.makedirs(self.save_dir, exist_ok=True)

    def eval(self):
        self.model.eval()
        total_loss = 0
        total_acc = 0
        with torch.no_grad():
            for i, data in tqdm(enumerate(self.dataLoader), desc='Evaluate', leave=True):
                sentence, labels, distribution = data[0], data[1], data[2]
                sentence = sentence.to(self.device)
                labels = labels.to(self.device)
                distribution = distribution.to(self.device)

                outputs = self.model(sentence, distribution)
                lloss = self.loss(outputs, labels)
                total_loss += lloss.item()
                correct = evaluate(outputs.clone().detach(), labels)
                predict = outputs.clone().detach()
                predict[predict>=0.5] = 1
                predict[predict<0.5] = 0
                acc = torch.sum(torch.eq(predict, labels)).item() / labels.numel()
                total_acc += acc

                if (i%10==0):
                    print(f'{i}/{len(self.dataLoader)}: loss: {lloss.item()}, acc: {acc}')
                self.results.append(outputs.numpy(force=True))
                self.labels.append(labels.numpy(force=True))
        self.total_loss = total_loss / len(self.dataLoader)
        self.total_acc = total_acc / len(self.dataLoader)
        print(f'loss: {self.total_loss}, acc: {self.total_acc}')
        self.results = np.vstack(self.results)
        self.labels = np.vstack(self.labels)
        fpr, tpr, thresholds = roc_curve(self.labels.flatten().astype('int'), self.results.flatten())
        roc_auc = auc(fpr, tpr)
        print(f'roc_auc: {roc_auc}')
        predicts = self.results.copy()
        predicts[predicts>=0.5] = 1
        predicts[predicts<0.5] = 0
        tn, fp, fn, tp = confusion_matrix(self.labels.flatten().astype('int'), predicts.flatten()).ravel()
        print(f'tn: {tn}, fp: {fp}, fn: {fn}, tp: {tp}')
        accuracy = (tp+tn)/(tp+tn+fp+fn)
        recall = tp/(tp+fn)
        precision = tp/(tp+fp)
        f1 = 2*precision*recall/(precision+recall)
        print(f'accuracy: {accuracy}, recall: {recall}, precision: {precision}, f1: {f1}')

    def saveResults(self):
        np.save(self.save_dir + self.name + '_results.npy', self.results)
        np.save(self.save_dir + self.name + '_labels.npy', self.labels)

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        prog='classifyEval',
        description='Evaluate the classification model',
    )
    parser.add_argument('--datasetPath', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--lmdbPath', type=str, required=True, help='Path to the lmdb')
    parser.add_argument('--modelPath', type=str, required=True, help='Path to the model')
    parser.add_argument('--max_len', type=int, default=128, required=True)
    parser.add_argument('--save_dir', type=str, default='./resultSave/', required=False)
    parser.add_argument('--name', type=str, default='classifyEval', required=False)
    parser.add_argument('--batch_size', type=int, default=64, required=False)

    args = parser.parse_args()
    print('--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('--------args----------\n')
    writer = SummaryWriter('./log/classifyEval/')
    data = DataReader(args.datasetPath, args.max_len)
    embedding_dim = data.embedding_dim
    model = torch.load(args.modelPath)
    loss = trainLoss()
    
    evaluator = classifyEval(args=args, writer=writer, data=data, model=model, loss=loss)
    evaluator.eval()
    evaluator.saveResults()

    