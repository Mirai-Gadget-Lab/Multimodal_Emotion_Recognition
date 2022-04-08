import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from utils import get_sinusoid_encoding_table
from models.modules import *

class Mel_Encoder(nn.Module):
    def __init__(self, train_config, n_mels):
        """
        
        Mel Encoder 
        
        """
        super(Mel_Encoder, self).__init__()

        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(2048,
                                                                                train_config.hidden_size, padding_idx=0),
                                                    freeze=True)
        self.mel_encoder_prenet = Mel_Encoder_Prenet(
            n_mels,
            train_config.decoder_prenet_hidden_size,
            train_config.hidden_size,
            dropout_p=0.5
        )
        
        self.blocks = clones(EncoderBlock(train_config.hidden_size, 
                                          train_config.n_head, 
                                          train_config.dropout_p), train_config.n_layers)

    def forward(self, x, pos, mask):
        x = self.mel_encoder_prenet(x)
        pos = self.pos_emb(pos)
        x = pos + x
        for block in self.blocks:
            x = block(x, mask=mask)
        return x
