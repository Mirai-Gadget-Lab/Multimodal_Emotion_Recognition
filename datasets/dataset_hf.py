import pandas as pd
from torch.utils.data import Dataset
import torchaudio
import torch
import os, re, librosa
from utils import *
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional
from transformers import AutoTokenizer, Wav2Vec2Processor

class multimodal_dataset(Dataset):
    
    def __init__(self, csv, config):
        self.csv = csv
        self.root_path = config.root_path
        self.remove_non_text = config.remove_non_text
        
    def __len__(self):
        return len(self.csv)
    
    def _load_wav(self, wav_path):
        wav, _ = librosa.load(wav_path, sr=16000)
        return wav
    
    def _load_txt(self, txt_path):
        with open(txt_path, 'r') as f:
            txt = f.readlines()
        assert len(txt) == 1,  'Text line Must be 1'
        
        txt = txt[0][:-1]
        if self.remove_non_text:
            txt = re.sub('[a-zA-Z]/', '', txt).strip()
        return txt
    
    def _load_data(self, idx):
        wav_path = os.path.join(self.root_path, self.csv['segment_id'].iloc[idx]+'.wav')
        txt_path = os.path.join(self.root_path, self.csv['segment_id'].iloc[idx]+'.txt')
        
        wav = self._load_wav(wav_path)
        txt = self._load_txt(txt_path)
        
        emotion = self.csv['emotion'].iloc[idx]
        valence = self.csv['valence'].iloc[idx]
        arousal = self.csv['arousal'].iloc[idx]
        
        sample = {
            'text' : txt,
            'wav' : wav,
            'emotion': emotion2int[emotion],
            'valence': int(valence)-1,
            'arousal': round(arousal)-1
        }
        return sample
    
    def __getitem__(self, idx):
        sample = self._load_data(idx)
        return sample

class multimodal_collator():
    
    def __init__(self, text_tokenizer, audio_processor, return_text=False, max_length=512):
        self.text_tokenizer = text_tokenizer
        self.audio_processor = audio_processor
        self.return_text = return_text
        self.max_length = max_length
        
    def __call__(self, batch):
        text = [d['text'] for d in batch]
        wav = [d['wav'] for d in batch]
        emotion = [d['emotion'] for d in batch]
        valence = [d['valence'] for d in batch]
        arousal = [d['arousal'] for d in batch]
        text_length = [len(d['text']) for d in batch]
        
        text = [i for i, _ in sorted(
            zip(text, text_length), key=lambda x: x[1], reverse=True)]
        wav = [i for i, _ in sorted(
            zip(wav, text_length), key=lambda x: x[1], reverse=True)]
        emotion = [i for i, _ in sorted(
            zip(emotion, text_length), key=lambda x: x[1], reverse=True)]
        valence = [i for i, _ in sorted(
            zip(valence, text_length), key=lambda x: x[1], reverse=True)]
        arousal = [i for i, _ in sorted(
            zip(arousal, text_length), key=lambda x: x[1], reverse=True)]
        
        text_inputs = self.text_tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
            max_length=self.max_length
        )
        
        audio_inputs = self.audio_processor(
            wav,
            sampling_rate=16000, 
            padding=True, 
            return_tensors='pt'
        )
        
        labels = {
            "emotion" : torch.LongTensor(emotion),
            "valence" : torch.LongTensor(valence),
            "arousal" : torch.LongTensor(arousal)
        }
        if self.return_text:
            labels['text'] = text
        return text_inputs, audio_inputs, labels