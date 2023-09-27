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
from focal_loss import FocalLoss
from utils import evaluate
# from model import transformerEncoderNet
# from classifyTrainer import TransformerClassifier
from classifier import transformerClassifier

class classifyTrainer():
    def __init__(self, args, writer, data, model, loss) -> None:
        self.args = args
        self.writer = writer
        self.data = data
        self.model = model
        self.loss = loss
        self.d_model = data.embedding_dim
        self.lmdbPath = args.lmdbPath
        self.batch_size = args.batch_size
        self.interval = args.interval
        self.epochs = args.epochs
        self.save_dir = args.save_dir
        self.learning_rate = args.learning_rate

        self.train_dataset = BGCLabelsDataset(self.data, self.lmdbPath, 'train')
        print('Train dataset size: ', len(self.train_dataset))
        self.test_dataset = BGCLabelsDataset(self.data, self.lmdbPath, 'test')
        self.train_dataLoader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=5)
        self.test_dataLoader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=5)

        print(torch.cuda.is_available())
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        self.save_path = self.save_dir + \
        f'transformerClassifier/transformerClassifier_{self.args.max_len}_{self.args.nhead}_{self.args.num_encoder_layers}_{self.args.mlp_dropout}_{self.args.transformer_dropout}_{self.args.learning_rate}_{self.args.epochs}_{self.args.alpha}_{self.args.gamma}/'
        # self.save_path = self.save_dir + \
        # f'transformerClassifier/transformerClassifier_{self.args.max_len}_{self.args.nhead}_ \
        #     {self.args.num_encoder_layers}_{self.args.mlp_dropout}_{self.args.transformer_dropout}_ \
        #     {self.args.learning_rate}_{self.args.epochs}/'
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        print('--------model----------')
        print(self.model)
        print('--------model----------\n')

    def __train_step(self, epoch):
        self.model.train()
        total_loss = 0
        total_acc = 0
        for i, data in tqdm(enumerate(self.train_dataLoader), total=len(self.train_dataLoader), desc='Training', leave=True):
            sentence, labels, distribution = data[0], data[1], data[2]
            sentence = sentence.to(self.device)
            labels = labels.to(self.device)
            distribution = distribution.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(src=sentence, src_key_padding_mask=distribution)
            # print(output.shape, labels.shape)
            loss = self.loss(output, labels)
            total_loss += loss.item()
            correct = evaluate(output.clone().detach(), labels)
            if torch.sum(labels) > 0:
                acc = correct*100/torch.sum(labels).item()
            else:
                acc = 0
            total_acc += acc
            loss.backward()
            self.optimizer.step()

            if i % self.interval == 0:
                print('Epoch: %d, %d/%d, Loss: %.3f, Acc: %.3f' % (epoch, i, len(self.train_dataLoader), total_loss/self.interval, total_acc/self.interval))
                self.train_total_loss += total_loss
                self.train_total_acc += total_acc
                total_loss = 0
                total_acc = 0  
        self.train_total_loss /= len(self.train_dataLoader)
        self.train_total_acc /= len(self.train_dataLoader)
        torch.save(self.model, self.save_path + f'transformerClassifier_{epoch}.pt')
        
    def __val_step(self,):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            total_acc = 0
            for i, data in tqdm(enumerate(self.test_dataLoader), total=len(self.test_dataLoader), desc='Testing', leave=True):
                sentence, labels, distribution = data[0], data[1], data[2]
                sentence = sentence.to(self.device)
                labels = labels.to(self.device)
                distribution = distribution.to(self.device)

                output = self.model(src=sentence, src_key_padding_mask=distribution)
                loss = self.loss(output, labels)
                total_loss += loss.item()
                correct = evaluate(output.clone().detach(), labels)

                if torch.sum(labels) > 0:
                    acc = correct*100/torch.sum(labels).item()
                else:
                    acc = 0
                total_acc += acc
            self.test_total_loss = total_loss/len(self.test_dataLoader)
            self.test_total_acc = total_acc/len(self.test_dataLoader)

    def train(self,):
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9, last_epoch=-1, verbose=True)

        for epoch in range(self.epochs):
            self.train_total_loss = 0
            self.train_total_acc = 0
            self.__train_step(epoch)
            self.writer.add_scalar('Loss/trainLoss', self.train_total_loss, epoch)
            self.writer.add_scalar('Acc/trainAcc', self.train_total_acc, epoch)
            if epoch>=5 and self.scheduler.get_last_lr()[0]>5e-6:
                self.scheduler.step()
            print(f'Train Set: Epoch: {epoch}, Loss: {self.train_total_loss}, Acc: {self.train_total_acc}')

            self.test_total_loss = 0
            self.test_total_acc = 0
            self.__val_step()
            self.writer.add_scalar('Loss/testLoss', self.test_total_loss, epoch)
            self.writer.add_scalar('Acc/testAcc', self.test_total_acc, epoch)
            print(f'Test Set: Epoch: {epoch}, Loss: {self.test_total_loss}, Acc: {self.test_total_acc}')


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        prog='transformerClassifier',
        description='Train a transformer classifier for BGCs',
    )
    parser.add_argument('--datasetPath', type=str, required=True, help='dataset path')
    parser.add_argument('--max_len', type=int, required=True, help='max gene numbers of the input sequence')
    parser.add_argument('--batch_size', type=int, required=True, help='batch size')
    parser.add_argument('--lmdbPath', type=str, required=True, help='training dataset lmdb path')
    parser.add_argument('--nhead', type=int, required=True, help='number of heads in transformer')
    parser.add_argument('--num_encoder_layers', type=int, required=True, help='number of encoder layers in transformer')
    parser.add_argument('--mlp_dropout', type=float, required=False, default=0.5, help='dropout rate of mlp')
    parser.add_argument('--transformer_dropout', type=float, required=False, default=0.1, help='dropout rate of transformer')
    parser.add_argument('--learning_rate', type=float, required=True, help='learning rate')
    parser.add_argument('--interval', type=int, required=False, default=10, help='interval of printing training loss')
    parser.add_argument('--epochs', type=int, required=True, help='epochs')
    parser.add_argument('--seed', type=int, required=False, default=42, help='random seed')
    parser.add_argument('--save_dir', type=str, required=False, default='./modelSave/', help='save dir')
    parser.add_argument('--alpha', type=float, required=False, default=0.25, help='alpha of focal loss')
    parser.add_argument('--gamma', type=float, required=False, default=2, help='gamma of focal loss')

    args = parser.parse_args()
    print('--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('--------args----------\n')

    setup_seed(args.seed)

    writer = SummaryWriter('./log/transformerClassifier/')
    data = DataReader(args.datasetPath, test_ratio=0.1)
    embedding_dim = 320
    model = transformerClassifier(d_model=embedding_dim, nhead=args.nhead, num_encoder_layers=args.num_encoder_layers, max_len=args.max_len,
                                   dim_feedforward=embedding_dim*4, labels_num=data.labels_num, transformer_dropout=args.transformer_dropout,
                                   mlp_dropout=args.mlp_dropout,)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params: {total_params}, Trainable params: {trainable_params}')
    # loss = trainLoss()
    loss = FocalLoss(alpha=args.alpha, gamma=args.gamma)
    trainer = classifyTrainer(args, writer, data, model, loss)
    trainer.train()

