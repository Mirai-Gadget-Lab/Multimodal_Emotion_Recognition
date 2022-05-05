import torch
from torch import nn
import pytorch_lightning as pl
from models.model_hf import *
import torchmetrics
import torch.nn.functional as F
class PL_model(pl.LightningModule):
    def __init__(self, train_config):
        super().__init__()
        self.save_hyperparameters()
        self.model = Emotion_MultinomialModel(train_config)
        self.train_config = train_config
        
        # Define Accuracy
        self.train_accuracy = torchmetrics.Accuracy()
        self.valid_accuracy = torchmetrics.Accuracy()
        
        # Define Loss
        self.ce = nn.CrossEntropyLoss()
        self.cs = nn.CosineEmbeddingLoss()
        self.y = torch.Tensor([1])
        
    def forward(self, text_inputs, audio_inputs):
        return self.model(text_inputs, audio_inputs)
    
    def training_step(self, batch, batch_idx):
        text_inputs, audio_inputs, labels = batch
        pred = self.forward(text_inputs, audio_inputs)
        loss = self.cal_loss(pred, labels[self.train_config.label_name])
        
        self.train_accuracy(pred, labels[self.train_config.label_name])
        
        self.log("train_loss",  loss, on_epoch=True)
        self.log('train_accuracy', self.train_accuracy, on_step=True, on_epoch=False)
        
        return loss

    def validation_step(self, batch, batch_idx):
        text_inputs, audio_inputs, labels = batch
        pred = self.forward(text_inputs, audio_inputs)
        loss = self.cal_loss(pred, labels[self.train_config.label_name])
        
        self.valid_accuracy(pred, labels[self.train_config.label_name])
        
        self.log("val_loss", loss, on_epoch=True)
        self.log('val_accuracy', self.valid_accuracy, on_step=True, on_epoch=False)
                 
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.train_config.lr)

        return optimizer
    
    def cal_loss(self, pred, label):
        ce_loss = self.ce(pred, label)
        cs_loss = self.cs(pred, F.one_hot(label, num_classes=7), self.y.type_as(pred).long())
        return 0.1*ce_loss + cs_loss

class PL_model_ce(pl.LightningModule):
    def __init__(self, train_config):
        super().__init__()
        self.save_hyperparameters()
        self.model = Emotion_MultinomialModel(train_config)
        self.train_config = train_config
        
        # Define Accuracy
        self.train_accuracy = torchmetrics.Accuracy()
        self.valid_accuracy = torchmetrics.Accuracy()
        
        # Define Loss
        self.ce = nn.CrossEntropyLoss()

        
    def forward(self, text_inputs, audio_inputs):
        return self.model(text_inputs, audio_inputs)
    
    def training_step(self, batch, batch_idx):
        text_inputs, audio_inputs, labels = batch
        pred = self.forward(text_inputs, audio_inputs)
        loss = self.ce(pred, labels[self.train_config.label_name])
        
        self.train_accuracy(pred, labels[self.train_config.label_name])
        
        self.log("train_loss",  loss, on_epoch=True)
        self.log('train_accuracy', self.train_accuracy, on_step=True, on_epoch=False)
        
        return loss

    def validation_step(self, batch, batch_idx):
        text_inputs, audio_inputs, labels = batch
        pred = self.forward(text_inputs, audio_inputs)
        loss = self.ce(pred, labels[self.train_config.label_name])
        
        self.valid_accuracy(pred, labels[self.train_config.label_name])
        
        self.log("val_loss", loss, on_epoch=True)
        self.log('val_accuracy', self.valid_accuracy, on_step=True, on_epoch=False)
                 
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.train_config.lr)

        return optimizer
    
class PL_model_MMER(pl.LightningModule):
    def __init__(self, train_config):
        super().__init__()
        self.save_hyperparameters()
        self.model = Emotion_MMER(train_config)
        self.train_config = train_config
        
        # Define Accuracy
        self.train_accuracy = torchmetrics.Accuracy()
        self.valid_accuracy = torchmetrics.Accuracy()
        
        # Define Loss
        self.ce = nn.CrossEntropyLoss()

        
    def forward(self, text_inputs, audio_inputs):
        return self.model(text_inputs, audio_inputs)
    
    def training_step(self, batch, batch_idx):
        text_inputs, audio_inputs, labels = batch
        pred = self.forward(text_inputs, audio_inputs)
        loss = self.ce(pred, labels[self.train_config.label_name])
        
        self.train_accuracy(pred, labels[self.train_config.label_name])
        
        self.log("train_loss",  loss, on_epoch=True)
        self.log('train_accuracy', self.train_accuracy, on_step=True, on_epoch=False)
        
        return loss

    def validation_step(self, batch, batch_idx):
        text_inputs, audio_inputs, labels = batch
        pred = self.forward(text_inputs, audio_inputs)
        loss = self.ce(pred, labels[self.train_config.label_name])
        
        self.valid_accuracy(pred, labels[self.train_config.label_name])
        
        self.log("val_loss", loss, on_epoch=True)
        self.log('val_accuracy', self.valid_accuracy, on_step=True, on_epoch=False)
                 
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.train_config.lr)

        return optimizer
    

class PL_model_MMER_multiloss(pl.LightningModule):
    def __init__(self, train_config):
        super().__init__()
        self.save_hyperparameters()
        self.model = Emotion_MMER(train_config)
        self.train_config = train_config
        
        # Define Accuracy
        self.train_accuracy = torchmetrics.Accuracy()
        self.valid_accuracy = torchmetrics.Accuracy()
        
        # Define Loss
        self.ce = nn.CrossEntropyLoss()
        self.cs = nn.CosineEmbeddingLoss()
        self.y = torch.Tensor([1])
        
    def forward(self, text_inputs, audio_inputs):
        return self.model(text_inputs, audio_inputs)
    
    def training_step(self, batch, batch_idx):
        text_inputs, audio_inputs, labels = batch
        pred = self.forward(text_inputs, audio_inputs)
        loss = self.cal_loss(pred, labels[self.train_config.label_name])
        
        self.train_accuracy(pred, labels[self.train_config.label_name])
        
        self.log("train_loss",  loss, on_epoch=True)
        self.log('train_accuracy', self.train_accuracy, on_step=True, on_epoch=False)
        
        return loss

    def validation_step(self, batch, batch_idx):
        text_inputs, audio_inputs, labels = batch
        pred = self.forward(text_inputs, audio_inputs)
        loss = self.cal_loss(pred, labels[self.train_config.label_name])
        
        self.valid_accuracy(pred, labels[self.train_config.label_name])
        
        self.log("val_loss", loss, on_epoch=True)
        self.log('val_accuracy', self.valid_accuracy, on_step=True, on_epoch=False)
                 
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.train_config.lr)

        return optimizer
    
    def cal_loss(self, pred, label):
        ce_loss = self.ce(pred, label)
        cs_loss = self.cs(pred, F.one_hot(label, num_classes=7), self.y.type_as(pred).long())
        return 0.1*ce_loss + cs_loss