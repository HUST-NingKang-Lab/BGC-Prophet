import torch
import math
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, Any, Union, Callable

from .model import PositionalEncoding

class transformerClassifier(nn.Module):
    def __init__(self, d_model: int=512, nhead: int=8, num_encoder_layers: int=6, max_len: int=128,
                 dim_feedforward: int=2048, labels_num: int=7, transformer_dropout: float=0.1, mlp_dropout: float=0.5,
                 activation: Union[str, Callable[[Tensor], Tensor]]=F.gelu,
                 layer_norm_eps: float=1e-5, batch_first: bool=True, norm_first: bool=True,
                 device=None, dtype=None) -> None:
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.pos_encoder = PositionalEncoding(num_hiddens=d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, 
                                                   transformer_dropout, activation, layer_norm_eps, 
                                                   batch_first, norm_first, **factory_kwargs)
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self.classifier = nn.Sequential(
            nn.Dropout(mlp_dropout),
            # nn.Linear(d_model, 128),
            # nn.GELU(),
            # nn.Linear(128, 32),
            # nn.GELU(),
            # nn.Linear(32, labels_num),
            nn.Linear(d_model, labels_num),
            nn.Sigmoid()
        )

        self.__reset_parameters()
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first

    def forward(self, src: Tensor, src_key_padding_mask: Tensor) -> Any:
        # src: (batch_size, max_len, embed_dim)
        memory = self.pos_encoder(src)
        memory = self.encoder(src, mask=None, src_key_padding_mask=src_key_padding_mask)
        # memory = memory*src_key_padding_mask.unsqueeze(-1)
        # memory: (batch_size, max_len, embed_dim), src_key_padding_mask: (batch_size, max_len, embed_dim)
        # memory = memory[:, -1, :].squeeze(1)
        memory = memory.mean(dim=1).squeeze(1)
        labels = self.classifier(memory)
        return labels

    def __reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

