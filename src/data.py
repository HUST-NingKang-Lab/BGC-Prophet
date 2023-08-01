import torch
from torch.utils.data import Dataset
import lmdb
import pandas as pd
import numpy as np
import random
import pickle


class DataReader:

    def __init__(self, datasetPath, test_ratio = 0.2) -> None:
        self.datasetPath = datasetPath
        self.test_ratio = test_ratio
        self.embedding_dim = 320
        self.__getData()
        self.__train_test_split()

    def __getData(self):
        self.df = pd.read_csv(self.datasetPath)
        # 读取csv并展开为list便于处理
        self.df['labels'] = self.df['labels'].apply(lambda x:x.split())
        self.df['TDsentence'] = self.df['TDsentence'].apply(lambda x:x.split())
        self.df['TDlabels'] = self.df['TDlabels'].apply(lambda x:x.split()).apply(lambda x:[int(eval(i)) for i in x])
        self.df['labels_rep'] = self.df['labels'].apply(lambda x:x[0]) # 仅取第一个便于分层抽样

        self.labels_list = ['Alkaloid', 'Terpene', 'NRP', 'Polyketide', 'RiPP', 'Saccharide', 'Other']
        print(self.labels_list)
        self.labels_num = len(self.labels_list)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):

        sentence = self.df['TDsentence'][idx]
        labels = self.df['labels'][idx]
        TDlabels = self.df['TDlabels'][idx]

        if labels == ['Unknown']:
            labels = random.choice(self.labels_list)
        # 在labels未知时随机一个值以便后面eval时能计算
        labels_onehot = [1 if label in labels else 0 for label in self.labels_list]
        return sentence, labels_onehot, TDlabels
    
    def __train_test_split(self):
        groups = self.df.groupby('labels_rep')
        self.train = []
        self.test = []
        # 根据代表的类别进行分层抽样
        for k, v in groups.groups.items():
            # k为代表的标签
            # v为索引
            vv = list(v)
            # print(vv)
            random.shuffle(vv)
            self.train += vv[int(v.size*self.test_ratio):]
            self.test += vv[:int(v.size*self.test_ratio)]
        # print(len(self.train))

    

    
class BGCLabelsDataset(Dataset):
    def __init__(self, data, lmdbPath, mode='train') -> None:
        super().__init__()
        self.data = data
        self.mode = mode
        self.env = lmdb.open(lmdbPath, readonly=True)
        self.txn = self.env.begin()

    def __len__(self):
        if self.mode == 'train':
            # print("length of training dataset",len(self.data.train))
            return len(self.data.train)
        elif self.mode == 'test':
            # print("length of testing dataset",len(self.data.test))
            return len(self.data.test)
        elif self.mode == 'eval':
            return len(self.data)
        else:
            raise ValueError("No such mode: {}!".format(self.mode))
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            index = self.data.train[idx]
        elif self.mode == 'test':
            index = self.data.test[idx]
        elif self.mode == 'eval':
            index = idx
        else:
            raise ValueError("No such mode: {}!".format(self.mode))
        
        # print(index)
        sentence, labels_onehot, TDlabels = self.data[index]
        sentence_embedding = [pickle.loads(self.txn.get(str(word).encode('ascii')))['mean_representations'][6] for word in sentence]
        # print(sentence_embedding)
        sentence_embedding = torch.stack(sentence_embedding)
        labels_onehot = np.array(labels_onehot)
        TDlabels = np.array(TDlabels)
        # assert sentence_embedding.shape[0]==len(TDlabels), f'{idx} sample: length of sentence: {len(sentence_embedding)}, length of labels: {len(TDlabels)}'
        # print(index, self.data.df.iloc[index], sentence_embedding.shape, labels_onehot.shape, TDlabels.shape)
        return sentence_embedding, torch.tensor(labels_onehot, dtype=torch.float32), torch.tensor(TDlabels, dtype=torch.float32)
    
    
    def __del__(self):
        self.env.close()
        # super().__del__()
        # print('env closed')





        
            
            




