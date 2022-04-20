import torch
from torch import nn
import pytorch_lightning as pl
from models.model_hf import MultinomialModel
import torchmetrics

class PL_model(pl.LightningModule):
    def __init__(self, data_config, train_config):
        super().__init__()
        self.model = MultinomialModel(train_config)
        self.train_config = train_config
        
        # Define Accuracy
        self.train_accuracy_emotion = torchmetrics.Accuracy()
        self.train_accuracy_valence = torchmetrics.Accuracy()
        self.train_accuracy_arousal = torchmetrics.Accuracy()
        
        self.val_accuracy_emotion = torchmetrics.Accuracy()
        self.val_accuracy_valence = torchmetrics.Accuracy()
        self.val_accuracy_arousal = torchmetrics.Accuracy()
        
        
        # Define Loss
        self.emotion_loss = nn.CrossEntropyLoss()
        self.valence_loss = nn.CrossEntropyLoss()
        self.arousal_loss = nn.CrossEntropyLoss()
        
    def forward(self, text_inputs, audio_inputs):
        return self.model(text_inputs, audio_inputs)
    
    def training_step(self, batch, batch_idx):
        text_inputs, audio_inputs, labels = batch
        pred = self.forward(text_inputs, audio_inputs)
        emotion_loss, valence_loss, arousal_loss = self.custom_loss(**labels, **pred)
        train_loss = emotion_loss + valence_loss + arousal_loss
        
        self.train_accuracy_emotion(pred['pred_emotion'], labels['emotion'])
        self.train_accuracy_valence(pred['pred_valence'], labels['valence'])
        self.train_accuracy_arousal(pred['pred_arousal'], labels['arousal'])
        
        self.log("train_loss",  train_loss, on_epoch=True)
        self.log("train_emotion_loss", emotion_loss, on_epoch=True)
        self.log("train_valence_loss", valence_loss, on_epoch=True)
        self.log("train_arousal_loss", arousal_loss, on_epoch=True)
        self.log('train_accuracy_emotion', self.train_accuracy_emotion, on_step=True, on_epoch=False)
        self.log('train_accuracy_valence', self.train_accuracy_valence, on_step=True, on_epoch=False)
        self.log('train_accuracy_arousal', self.train_accuracy_arousal, on_step=True, on_epoch=False)
        
        return train_loss

    def validation_step(self, batch, batch_idx):
        text_inputs, audio_inputs, labels = batch
        pred = self.forward(text_inputs, audio_inputs)
        emotion_loss, valence_loss, arousal_loss = self.custom_loss(**labels, **pred)
        val_loss = emotion_loss + valence_loss + arousal_loss
        
        self.val_accuracy_emotion(pred['pred_emotion'], labels['emotion'])
        self.val_accuracy_valence(pred['pred_valence'], labels['valence'])
        self.val_accuracy_arousal(pred['pred_arousal'], labels['arousal'])
        
        self.log("val_loss", val_loss, on_epoch=True)
        self.log("val_emotion_loss", emotion_loss, on_epoch=True)
        self.log("val_valence_loss", valence_loss, on_epoch=True)
        self.log("val_arousal_loss", arousal_loss, on_epoch=True)
        self.log('val_accuracy_emotion', self.val_accuracy_emotion, on_step=True, on_epoch=False)
        self.log('val_accuracy_valence', self.val_accuracy_valence, on_step=True, on_epoch=False)
        self.log('val_accuracy_arousal', self.val_accuracy_arousal, on_step=True, on_epoch=False)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.train_config.lr)
            # scheduler = transformers.get_cosine_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=self.train_config.num_warmup_steps,
        #     num_training_steps=self.train_config.num_training_steps
        # )
        return optimizer

    def custom_loss(self, pred_emotion, pred_valence, pred_arousal, emotion, valence, arousal):
        emotion_loss = self.emotion_loss(pred_emotion, emotion)
        valence_loss = self.valence_loss(pred_valence, valence)
        arousal_loss = self.arousal_loss(pred_arousal, arousal)
        return emotion_loss, valence_loss, arousal_loss