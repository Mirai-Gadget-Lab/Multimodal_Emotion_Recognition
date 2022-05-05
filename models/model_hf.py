from transformers import AutoModel, Wav2Vec2ForCTC
from torch import nn
import torch 
from models.module import *
    
class Emotion_MultinomialModel(nn.Module):
    def __init__(self, config):
        super(Emotion_MultinomialModel, self).__init__()
        
        self.config = config
        
        if config.using_model == 'both': 
            self.text_encoder = AutoModel.from_pretrained(config.text_encoder)
            self.audio_encoder = Wav2Vec2ForCTC.from_pretrained(config.audio_processor)
            self.audio_encoder.lm_head = nn.Linear(1024, 768)
            self.audio_pool = nn.AdaptiveAvgPool2d((1, 768))
        
            self.emotion_out = nn.Linear(1536, 7)

        elif config.using_model == 'audio':
            self.audio_encoder =Wav2Vec2ForCTC.from_pretrained(config.audio_processor)
            self.audio_encoder.lm_head = nn.Linear(1024, 768)
            self.audio_pool = nn.AdaptiveAvgPool2d((1, 768))
            
            self.emotion_out = nn.Linear(768, 7)
            
        elif config.using_model =='text':
            self.text_encoder = AutoModel.from_pretrained(config.text_encoder)
            self.emotion_out = nn.Linear(768, 7)
        else:
            raise "WrongModelName"
            
    def forward(self, text_inputs, audio_inputs):
        
        if self.config.using_model == 'both': 
            text_feat = self.text_encoder(**text_inputs)['pooler_output']
            audio_feat = self.audio_encoder(**audio_inputs)[0]
            audio_feat = self.audio_pool(audio_feat).squeeze()
            
            feat = torch.cat([text_feat, audio_feat], dim=1)
            
        elif self.config.using_model == 'text':
            feat = self.text_encoder(**text_inputs)['pooler_output']
        
        elif self.config.using_model == 'audio':
            feat = self.audio_encoder(**audio_inputs)[0]
            feat = self.audio_pool(feat).squeeze()
            
        pred_emotion = self.emotion_out(feat)
        
        return pred_emotion
    
class Emotion_MMER(nn.Module):
    def __init__(self, config):
        super(Emotion_MMER, self).__init__()
        
        self.config = config
        
        # Encoder
        self.text_encoder = AutoModel.from_pretrained(config.text_encoder)
        self.audio_encoder = Wav2Vec2ForCTC.from_pretrained(config.audio_processor)
        self.audio_encoder.lm_head = nn.Linear(1024, 768)
        
        # MMER
        self.CMA_1 = CrossModalEncoder(768, 12, 0.3)
        self.CMA_2 = CrossModalEncoder(768, 12, 0.3)
        self.CMA_3 = CrossModalEncoder(768, 12, 0.3)
        
        self.gate_linear = nn.Linear(1536, 768, bias=True)
        self.gate_sigmoid = nn.Sigmoid()
        self.projection = nn.Linear(1536, 768, bias=False)
        
        # pooling
        self.pool_layer = nn.AdaptiveAvgPool2d((1, 768))
        
        self.emotion_out = nn.Linear(1536, 7)
            
    def forward(self, text_inputs, audio_inputs):
        
        text_feat = self.text_encoder(**text_inputs)['last_hidden_state']
        audio_feat = self.audio_encoder(**audio_inputs)[0]
        
        h = self.MMER(text_feat, audio_feat)
        
        pooled_audio = self.pool_layer(audio_feat)
        pooled_h = self.pool_layer(h)
        
        concated_feat = torch.cat([pooled_audio, pooled_h], dim=2).squeeze()
        
        return self.emotion_out(concated_feat)
    
    def MMER(self, text_feat, audio_feat):
        p = self.CMA_1(audio_feat, text_feat)
        r = self.CMA_2(text_feat, p)
        q = self.CMA_3(text_feat, audio_feat)
        b = self.gate_sigmoid(self.gate_linear(torch.cat([r, q],dim=2)))
        return self.projection(torch.cat([r, b], dim=2))