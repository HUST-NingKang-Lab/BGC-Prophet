import torch
import math
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, Any, Union, Callable

from batch3Linear import batch3Linear
# from torch.nn.module import Transformer


class LSTMNet(nn.Module):
    def __init__(self, embedding_dim:int,  
                 hidden_dim:int, num_layers:int, 
                 dropout:int=0.1, initWeight:bool=False) -> None:
        super().__init__()

        self.LSTM = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        # 全连接层的权重相互独立，用于预测每个基因是否是BGC的组成部分
        self.TD = nn.Sequential(
            nn.LayerNorm(hidden_dim*2),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_dim*2, out_features=128),
            nn.GELU(),
            nn.Linear(in_features=128, out_features=32),
            nn.GELU(),
            nn.Linear(in_features=32, out_features=1),
            nn.Sigmoid()
        )
        # self.classifier = nn.Sequential(
        #     nn.LayerNorm(hidden_dim*2),
        #     nn.Dropout(dropout),
        #     nn.Linear(in_features=hidden_dim*2, out_features=64), 
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(in_features=64, out_features=labels_num),
        #     nn.Sigmoid()
        # )
        if initWeight:
            self.__initWeight()

    def forward(self, inputs):
        # x = self.MAblock(inputs)
        # x = self.attention(inputs)
        # print(x.shape)
        x, _ = self.LSTM(inputs, None)
        TD = self.TD(x)
        # memory = torch.sum(x, dim=1) / x.shape[1]
        # labels = self.classifier(memory)
        # 输出是否是BGC的组成部分，以及BGC种类两个信息
        return TD.squeeze()
    
    def __initWeight(self):
        pass



    
    