

import torch
from torch.utils.data import DataLoader
from loss import *
from data import DataReader, BGCLabelsDataset
import argparse
import os
from tqdm import tqdm
import numpy as np
from utils import evaluate
from sklearn.metrics import roc_curve, auc, confusion_matrix

class evalAll:
    def __init__(self, args, data) -> None:
        self.args = args
        self.data = data
        self.models_folder = args.models_folder
        # self.save_dir = args.save_dir
        self.epochs = args.epochs
        self.lmdbPath = args.lmdbPath
        self.batch_size = args.batch_size
        # self.models = os.listdir(self.models)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # os.makedirs(self.save_dir, exist_ok=True)

    def eval_all(self):
        loss = trainLoss()
        self.dataset = BGCLabelsDataset(self.data, lmdbPath=self.lmdbPath, mode='eval')
        for i in range(self.epochs):
            modelPath = self.models_folder+f'transformerEncoder_Model_TD_{i}.pt'
            model = torch.load(modelPath)
            print(f"#Epoch: {i}")
            self.eval_one(model=model, loss=loss)    

    def eval_one(self, model, loss):
        model.to(self.device)
        self.dataLoader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=5)
        model.eval()
        total_TD_loss = 0
        distributions = []
        results = []
        total_TD_acc = 0
        with torch.no_grad():
            # print(len(self.dataLoader))
            # print(type(self.dataLoader))
            for i, data in tqdm(enumerate(self.dataLoader), desc='Evaluate', leave=True):
                sentence, labels, distribution = data[0], data[1], data[2]
                sentence = sentence.to(self.device)
                labels = labels.to(self.device)
                distribution = distribution.to(self.device)

                outputsTD = model(sentence) 
                TDLoss = loss(outputsTD, distribution)
                total_TD_loss += TDLoss

                # Label_correct = evaluate(outputsLabels.clone().detach(), labels)
                # Label_accuracy = Label_correct*100/torch.sum(labels).item()
                TD_correct = evaluate(outputsTD.clone().detach(), distribution)
                if torch.sum(distribution).item()!=0:
                    TD_accuracy = TD_correct*100/torch.sum(distribution).item()
                else:
                    predict = outputsTD.clone().detach()
                    predict[predict>0.5] = 1.
                    predict[predict<0.5] = 0.
                    TD_accuracy = torch.sum(torch.eq(predict, distribution)).item()/distribution.numel()


                # total_label_acc += Label_accuracy
                total_TD_acc += TD_accuracy
                if (i%10 == 0):
                    print(f'{i}/{len(self.dataLoader)} TD_accuracy: {TD_accuracy}')
                distributions.append(distribution.numpy(force=True))
                results.append(outputsTD.numpy(force=True)) 

            
        # self.total_label_acc = total_label_acc/len(self.dataLoader)
        total_TD_acc = total_TD_acc/len(self.dataLoader)
        distributions = np.vstack(distributions).flatten()
        results = np.vstack(results).flatten()
        print(f'distribution.shape: {distributions.shape}')
        print(f'result.shape: {results.shape}')

        # self.total_loss = total_loss/len(self.dataLoader)

        print('#TD_acc:%.3f' % total_TD_acc)
        fpr, tpr, _ = roc_curve(distributions.astype('int'), results)
        roc_auc = auc(fpr, tpr)
        print(f"AUC: {roc_auc}")
        predicts = results.copy()
        predicts[predicts>=0.5] = 1.
        predicts[predicts<0.5] = 0.
        # 计算混淆矩阵
        cm = confusion_matrix(distributions, predicts)
        print(f'confusion-matrix:\n{cm}')
        # 计算每个元素的百分比
        cm_percent = 100*cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        tn = cm_percent[0][0]
        fp = cm_percent[0][1]
        fn = cm_percent[1][0]
        tp = cm_percent[1][1]
        print('TP:\t{:.2f}%\tFN:\t{:.2f}%\nFP:\t{:.2f}%\tTN:\t{:.2f}%'.format(tp,fn,fp,tn))
        accuracy = 100*(tp+tn)/(tp+tn+fp+fn)
        recall = 100*tp/(tp+fn)
        precision = 100*tp/(tp+fp)
        F1_Score = 2*precision*recall/(precision+recall)
        print('Accuracy={:.2f}%\tRecall={:.2f}%\tprecision={:.2f}%\tF1-score={:2f}'.format(accuracy,recall,precision,F1_Score))


            


if __name__=='__main__':
    parser = argparse.ArgumentParser(
        prog = 'Evaluate_all',
        description='evaluate all trained transformer encoder model',
    )
    parser.add_argument('--models_folder', required=True, type=str)
    # parser.add_argument('--save_dir', required=True, type=str)
    parser.add_argument('--epochs', required=True, type=int)
    parser.add_argument('--lmdbPath', required=True, type=str)
    parser.add_argument('--datasetPath', required=True, type=str)
    parser.add_argument('--max_len', required=True, type=int)
    parser.add_argument('--batch_size', required=True, type=int)

    args = parser.parse_args()

    print('--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('--------args----------\n')

    data = DataReader(args.datasetPath, args.max_len)


    evaluater = evalAll(args, data=data)
    evaluater.eval_all()

