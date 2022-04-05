import pandas as pd
from torch.utils.data import Dataset
import torchaudio
import torch
import os
from utils import *

class multimodal_dataset(Dataset):
    
    def __init__(self, config):
        self.csv = pd.read_csv(config.csv_path)
        self.root_path = config.root_path
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            power=config.power,
            normalized=config.normalized
        )
    def __len__(self):
        return len(self.csv)
    
    def _load_wav(self, wav_path):
        wav, _ = torchaudio.load(wav_path)
        mel = self.mel_transform(wav)
        return mel.squeeze().permute(1, 0)
    
    def _load_txt(self, txt_path):
        with open(txt_path, 'r') as f:
            txt = f.readlines()
        assert len(txt) == 1,  'Text line Must be 1'
        return txt[0][:-1]
    def _load_data(self, idx):
        wav_path = os.path.join(self.root_path, self.csv['segment_id'].iloc[idx]+'.wav')
        txt_path = os.path.join(self.root_path, self.csv['segment_id'].iloc[idx]+'.txt')
        
        mel = self._load_wav(wav_path)
        txt = self._load_txt(txt_path)
        
        emotion = self.csv['emotion'].iloc[idx]
        valence = self.csv['valence'].iloc[idx]
        arousal = self.csv['arousal'].iloc[idx]
        
        sample = {
            'text' : txt,
            'mel' : mel,
            'emotion': emotion2int[emotion],
            'valence': round(valence),
            'arousal': round(arousal)
        }
        return sample
    
    def __getitem__(self, idx):
        sample = self._load_data(idx)
        return sample

class multimodal_collator():
    
    def __init__(self):
        pass
    def __call__(self, batch):
        text = [d['text'] for d in batch]
        mel = [d['mel'] for d in batch]
        emotion = [d['emotion'] for d in batch]
        valence = [d['valence'] for d in batch]
        arousal = [d['arousal'] for d in batch]
        
        mel = pad_mel(mel)
        
        inputs = {
            'text': text,
            'mel' : torch.FloatTensor(mel).permute(0, 2, 1),
        }
        labels = {
            "emotion" : torch.LongTensor(emotion),
            "valence" : torch.LongTensor(valence),
            "arousal" : torch.LongTensor(arousal)
        }
        return inputs, labels