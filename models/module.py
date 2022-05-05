import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class CrossModalEncoder(nn.Module):
    def __init__(self, hidden_size, n_head, dropout):
        super(CrossModalEncoder, self).__init__()
        self.MHA = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=n_head, batch_first=True)
        self.MHA_dropout = nn.Dropout(dropout)
        self.MHA_norm = nn.LayerNorm(hidden_size)

        # FFN
        self.linear1 = nn.Linear(hidden_size, hidden_size*4)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_size*4, hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.FFN_norm = nn.LayerNorm(hidden_size)
        # 
    def forward(self, query, memory):
        x = query
        x = self.MHA_norm(x + self.attn_block(x, memory))
        x = self.FFN_norm(x + self.ffn_block(x))    
        return x 

    def attn_block(self, query, memory):
        x = self.MHA(query, memory, memory, need_weights=False)[0]
        return self.MHA_dropout(x)
    
    def ffn_block(self, x):
        x = self.linear2(self.dropout1(self.activation(self.linear1(x))))
        return self.dropout2(x)