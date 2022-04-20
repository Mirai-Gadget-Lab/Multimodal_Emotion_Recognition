from transformers import AutoModel, Wav2Vec2ForCTC
from torch import nn
import torch 

class MultinomialModel(nn.Module):
    def __init__(self, config):
        super(MultinomialModel, self).__init__()
        
        self.config = config
        
        if config.using_model == 'both': 
            self.text_encoder = AutoModel.from_pretrained(config.text_encoder)
            self.audio_encoder = Wav2Vec2ForCTC.from_pretrained(config.audio_processor)
            self.audio_encoder.lm_head = nn.Linear(1024, 768)
            self.audio_pool = nn.AdaptiveAvgPool2d((1, 768))
        
            self.emotion_out = nn.Linear(1536, 7)
            self.valence_out = nn.Linear(1536, 5)
            self.arousal_out = nn.Linear(1536, 5)
        elif config.using_model == 'audio':
            self.audio_encoder =Wav2Vec2ForCTC.from_pretrained(config.audio_processor)
            self.audio_encoder.lm_head = nn.Linear(1024, 768)
            self.audio_pool = nn.AdaptiveAvgPool2d((1, 768))
            
            self.emotion_out = nn.Linear(768, 7)
            self.valence_out = nn.Linear(768, 5)
            self.arousal_out = nn.Linear(768, 5)
            
        elif config.using_model =='text':
            self.text_encoder = AutoModel.from_pretrained(config.text_encoder)
            
            self.emotion_out = nn.Linear(768, 7)
            self.valence_out = nn.Linear(768, 5)
            self.arousal_out = nn.Linear(768, 5)
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
        pred_valence = self.valence_out(feat)
        pred_arousal = self.arousal_out(feat)
        
        return {'pred_emotion': pred_emotion, 'pred_valence':pred_valence, 'pred_arousal': pred_arousal}
    
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