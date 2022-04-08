from transformers import AutoModel, Wav2Vec2ForSequenceClassification
from torch import nn
import torch 
from models.modules import LinearBlock

class MultinomialModel(nn.Module):
    def __init__(self, config):
        super(MultinomialModel, self).__init__()
        self.text_encoder = AutoModel.from_pretrained(config.text_encoder)
        self.audio_encoder = Wav2Vec2ForSequenceClassification.from_pretrained(config.audio_processor)
        self.audio_encoder.projector = nn.Linear(1024, 1024)
        self.audio_encoder.classifier = nn.Linear(1024, 768)
        self.projection = LinearBlock(1536, 768)
        
        self.emotion_out = nn.Linear(768, 7)
        self.valence_out = nn.Linear(768, 5)
        self.arousal_out = nn.Linear(768, 5)
        
    def forward(self, text_inputs, audio_inputs):
        text_feat = self.text_encoder(**text_inputs)['pooler_output']
        audio_feat = self.audio_encoder(**audio_inputs)[0]
        
        concated_feat = torch.cat([text_feat, audio_feat], dim=1)
        feat = self.projection(concated_feat)
        
        pred_emotion = self.emotion_out(feat)
        pred_valence = self.valence_out(feat)
        pred_arousal = self.arousal_out(feat)
        
        return {'pred_emotion': pred_emotion, 'pred_valence':pred_valence, 'pred_arousal': pred_arousal}
        