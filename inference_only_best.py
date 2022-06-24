# %%
import pandas as pd
import os
from dataset_hf import multimodal_dataset, multimodal_collator
from torch.utils.data import DataLoader
from config import *
from models.pl_model_hf import *
from transformers import AutoTokenizer, Wav2Vec2Processor
from sklearn.model_selection import train_test_split
from pytorch_lightning import Trainer
from sklearn.metrics import f1_score, accuracy_score
import argparse
from torch import nn
from glob import glob

def predict_step(self, batch, batch_idx=0, dataloader_idx=0):
    text_inputs, audio_inputs, labels = batch
    pred = self.forward(text_inputs, audio_inputs)
    pred = nn.functional.softmax(pred, dim=1)
    return pred, labels

def predict_MMI(trainer, loader, train_config, ckp_path, loss=None):
    if loss == 'ce':
        model = PL_model_MMER(train_config).load_from_checkpoint(ckp_path)
    else:
        model = PL_model_MMER_multiloss(train_config).load_from_checkpoint(ckp_path)
    model.predict_step = predict_step.__get__(model)
    predictions = trainer.predict(model, loader)
    preds = [i[0] for i in predictions]
    labels = [i[1]['emotion'] for i in predictions]
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    return preds, labels
# %%
data_config = HF_DataConfig()
train_config = HF_TrainConfig(
    batch_size=3
)
# %%
csv = pd.read_csv(data_config.csv_path)
csv = csv.drop_duplicates(subset=['segment_id'], ignore_index=True)

csv['wav_length'] = csv['wav_end'] - csv['wav_start']
csv = csv.query("wav_length <= %d"%25)
_, test = train_test_split(csv, test_size=0.2, random_state=1004)
# %%
text_tokenizer = AutoTokenizer.from_pretrained(train_config.text_encoder)
audio_processor = Wav2Vec2Processor.from_pretrained(train_config.audio_processor)
# %%
test_dataset = multimodal_dataset(test, data_config)
test_loader = DataLoader(test_dataset, 3, num_workers=8,
                            collate_fn=multimodal_collator(text_tokenizer, audio_processor), pin_memory=True,
                            shuffle=False, drop_last=True)
trainer = Trainer(gpus=1,
                logger=False)
# %%
root_path = './models_zoo/checkpoint/both_multiloss_MMI/epoch=05-val_loss=0.30955.ckpt'
preds, label = predict_MMI(trainer, test_loader, train_config, root_path)
# %%
emotion2int = {
    'neutral': 0, 
    'angry' : 1,
    'happy' : 2, 
    'surprise' : 3, 
    'sad' : 4, 
    'fear' : 5, 
    'disgust' : 6
}

# %%
int2emotion = {}
for k, v in emotion2int.items():
    int2emotion[v]= k
# %%
test['preds'] = np.argmax(preds, axis=1)
# %%
test['preds'] = test['preds'].replace(int2emotion)
# %%
test[test['preds'] == test['emotion']]
# %%
test.to_csv('./data/result.csv', index=False)
# %%
# %%
