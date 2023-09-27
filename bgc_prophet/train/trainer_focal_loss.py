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
from timm.scheduler.cosine_lr import CosineLRScheduler

from data import DataReader, BGCLabelsDataset
from focal_loss import FocalLoss
from utils import evaluate
from model import transformerEncoderNet



# 预测每个基因是否属于BGC
class TransformerEncoderTrainer:
    def __init__(self, args, writer, data, model, TimeDistributedLoss) -> None:
        self.args = args
        self.writer = writer
        self.d_model = data.embedding_dim
        self.lmdbPath = args.lmdbPath
        self.batch_size = args.batch_size
        self.interval = args.interval
        self.ditribute_epochs = args.distribute_epochs
        self.warmup_epochs = args.warmup_epochs
        self.save_dir = args.save_dir
        self.learning_rate = args.learning_rate
        # self.savePath = args.save_dir

        self.data = data
        self.train_dataset = BGCLabelsDataset(self.data, self.lmdbPath, mode='train')
        print('Train Data length: ', len(self.train_dataset))
        self.test_dataset = BGCLabelsDataset(self.data, self.lmdbPath, mode='test')
        print('Train Data length: ', len(self.test_dataset))
        self.train_dataLoader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=5)
        self.test_dataLoader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=5)
        self.model = model
        self.TimeDistributedLoss = TimeDistributedLoss
        print(torch.cuda.is_available())
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        self.save_path = self.save_dir + \
        f'transformerEncoder_TD_focal/bS_{self.batch_size}_dE_{self.ditribute_epochs}_lR_{self.learning_rate}_mL_{args.max_len}_d_{self.d_model}_nH_{args.nhead}_nEL_{args.num_encoder_layers}_tdP_{args.transformer_dropout}_mdP_{args.mlp_dropout}_alpha_{args.alpha}_gamma_{args.gamma}_TD/'
        os.makedirs(self.save_path, exist_ok=True)
        # with open(self.save_path+'labels_list.pkl', 'wb') as fp:
        #     pickle.dump(self.data.labels_list, fp)
        #     print("Save labels_list")
        print(self.model)
        # self.writer.add_graph(model=self.model, input_to_model=torch.randn(self.batch_size, args.max_len, data.embedding_dim))

    def train_TD_step(self, epoch):
        self.model.train()
        total_TD_loss = 0
        total_TD_acc = 0
        for i, data in tqdm(enumerate(self.train_dataLoader), desc='Train TD', leave=True):
            sentence, labels, distribution = data[0], data[1], data[2]
            sentence = sentence.to(self.device)
            labels = labels.to(self.device)
            distribution = distribution.to(self.device)

            self.optimizer.zero_grad()
            outputsTD = self.model(sentence)
            TDLoss = self.TimeDistributedLoss(outputsTD, distribution)

            total_TD_loss += TDLoss

            # 计算准确度
            TD_correct = evaluate(outputsTD.clone().detach(), distribution)
            if torch.sum(distribution).item() <= 0.0:
                TD_accuracy = 100.0
            else:
                TD_accuracy = TD_correct*100/torch.sum(distribution).item()

            # Label_accuracy = Label_correct*100/labels.numel()
            # TD_accuracy = TD_correct*100/distribution.numel()

            total_TD_acc += TD_accuracy

            TDLoss.backward()
            self.optimizer.step()

            if (i%self.interval == 0):
                print('#Epoch:%d, %d/%d Loss:%.5f dacc:%.3f' % (epoch, i, len(self.train_dataLoader), 
                    total_TD_loss/self.interval, total_TD_acc/self.interval))
                self.train_total_TD_loss += total_TD_loss
                self.train_total_TD_acc += total_TD_acc
                total_TD_loss = 0
                total_TD_acc = 0
        self.train_total_TD_acc /= len(self.train_dataLoader)
        self.train_total_TD_loss /= len(self.train_dataLoader)
        torch.save(self.model, self.save_path + f'transformerEncoder_Model_TD_{epoch}.pt')

    def validate_step(self):
        self.model.eval()
        with torch.no_grad():
            total_TD_loss = 0
            total_TD_acc = 0
            for i, data in tqdm(enumerate(self.test_dataLoader), desc="Test", leave=True):
                # 数据预处理
                sentence, labels, distribution = data[0], data[1], data[2]
                sentence = sentence.to(self.device)
                labels = labels.to(self.device)
                distribution = distribution.to(self.device)

                # 模型推理，计算loss
                outputsTD = self.model(sentence)
                TDLoss = self.TimeDistributedLoss(outputsTD, distribution)
                total_TD_loss += TDLoss

                # 计算准确度
                TD_correct = evaluate(outputsTD.clone().detach(), distribution)
                # Label_accuracy = Label_correct*100/labels.numel()
                # TD_accuracy = TD_correct*100/distribution.numel()
                if torch.sum(distribution).item() <= 0.0:
                    TD_accuracy = 100.0
                else:
                    TD_accuracy = TD_correct*100/torch.sum(distribution).item()
                total_TD_acc += TD_accuracy

        print('#Loss:%.5f dacc:%.3f' % (total_TD_loss/len(self.test_dataLoader), total_TD_acc/len(self.test_dataLoader)))
        
        self.test_total_TD_loss = total_TD_loss / len(self.test_dataLoader)
        self.test_total_TD_acc = total_TD_acc / len(self.test_dataLoader)


    def train(self):

        # self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        # self.scheduler = CosineLRScheduler(optimizer=self.optimizer,
        #                                                     t_initial=self.ditribute_epochs,
        #                                                     lr_min=5e-6,
        #                                                     warmup_t=self.warmup_epochs,
        #                                                     warmup_lr_init=1e-4
        #                                                     )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95, last_epoch=-1, verbose=True)  

        for epoch in range(self.ditribute_epochs):
            self.train_total_TD_acc = 0
            self.train_total_TD_loss = 0
            self.train_TD_step(epoch=epoch)
            # tensorboard train
            self.writer.add_scalar('Loss/trainTDLoss', self.train_total_TD_loss, epoch)
            self.writer.add_scalar('Acc/trainTDAcc', self.train_total_TD_acc, epoch)
            if epoch>=5 and self.scheduler.get_last_lr()[0]>5e-6:
                # self.scheduler.step(epoch)
                self.scheduler.step()
            # print(f'Train Set: lloss:{self.train_total_label_loss}, lacc: {self.train_total_label_acc}')
            print(f'Train Set: tdloss:{self.train_total_TD_loss}, tdacc: {self.train_total_TD_acc}')
            self.test_total_TD_loss = 0
            self.test_total_TD_acc = 0
            self.validate_step()
            # tensorboard validate
            self.writer.add_scalar('Loss/testTDLoss', self.test_total_TD_loss, epoch)
            self.writer.add_scalar('Acc/testTDAcc', self.test_total_TD_acc, epoch)
            print(f'Test Set: tdloss:{self.test_total_TD_loss}, tdacc:{self.test_total_TD_acc}')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

            

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        prog = 'transformerEncoder',
        description='transformerEncoder model to every gene blong to BGC or not',
    )
    parser.add_argument('--datasetPath', required=True)
    parser.add_argument('--max_len', default=64, type=int)
    parser.add_argument('--lmdbPath', required=True)
    # parser.add_argument('--hidden_dim', required=True, type=int)
    parser.add_argument('--nhead', type=int, required=True)
    parser.add_argument('--num_encoder_layers', required=True, default=4, type=int)
    parser.add_argument('--transformer_dropout', default=0.1, type=float)
    parser.add_argument('--mlp_dropout', default=0.5, type=float)
    parser.add_argument('--batch_size', required=True, type=int)
    parser.add_argument('--learning_rate', required=True, type=float)
    parser.add_argument('--interval', required=False, default=10, type=int)
    parser.add_argument('--distribute_epochs', required=True, type=int)
    parser.add_argument('--warmup_epochs', default=200, type=int)
    parser.add_argument('--save_dir', default='./modelSave/')
    parser.add_argument('--load_label_model', action='store_true', required=False)
    parser.add_argument('--label_model_path', required=False)
    parser.add_argument('--two_gpu', required=False, action='store_true')
    parser.add_argument('--seed', required=False, default=42, type=int)
    parser.add_argument('--alpha', required=True, default=0.05, type=float)
    parser.add_argument('--gamma', required=True, default=1, type=float)
    # parser.add_argument()

    args = parser.parse_args()

    print('--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('--------args----------\n')
    
    setup_seed(args.seed)
    writer = SummaryWriter('./log/TransformerEncoder/')
    data = DataReader(args.datasetPath, test_ratio=0.2)
    embedding_dim = 320
    if args.load_label_model:
        model = torch.load(args.label_model_path)
    else:
        model = transformerEncoderNet(d_model=embedding_dim, nhead=args.nhead, num_encoder_layers=args.num_encoder_layers, max_len=args.max_len, 
                                      dim_feedforward=embedding_dim*4, transformer_dropout=args.transformer_dropout, mlp_dropout=args.mlp_dropout, batch_first=True)
        # model = transformerEncoderNet(embedding_dim=embedding_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, max_len=args.max_len, labels_num=data.labels_num, dropout=args.dropout)
    if args.two_gpu:
        model = torch.nn.DataParallel(model)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total Parameters: {total}\nTrainable Parameters: {trainable}\n')
    TimeDistributedLoss = FocalLoss(alpha=args.alpha, gamma=args.gamma, reduction='sum')
    # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1, verbose=True)  
    trainer = TransformerEncoderTrainer(args=args, writer=writer,data=data, model=model, TimeDistributedLoss=TimeDistributedLoss)
    trainer.train()