import math
from typing import Any

import torch 
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter

class batch3Linear(nn.Module):
    r'''
    batch linear layer
    '''
    __constants__ = ['batch_size', 'in_features', 'out_features']
    batch_size:int
    in_features:int
    out_features:int
    weight:Tensor

    def __init__(self, batch_size:int, in_features:int, out_features:int, bias:bool=True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device':device, 'dtype':dtype}
        super().__init__()
        self.batch_size = batch_size
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((batch_size, in_features, out_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty((batch_size, 1, out_features), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input:Tensor) -> Tensor:
        assert input.shape[0]==self.weight.shape[0], 'batch_size not equal!'
        # input: (max_len, batch_size in_features)
        # self.weight: (max_len, in_features, out_features)
        # return: (max_len, batch_size, out_features)
        return torch.bmm(input, self.weight) + self.bias
    
    def extra_repr(self) -> str:
        return 'batch_size={}, in_features={}, out_features={}, bias={}'.format(
            self.batch_size, self.in_features, self.out_features, self.bias is not None
        )
