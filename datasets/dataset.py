import pandas as pd
from torch.utils.data import Dataset
import torchaudio
import torch
import os
from utils import *
import re
from torch.nn.utils.rnn import pad_sequence

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
        self.remove_non_text = config.remove_non_text
        
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
        
        txt = txt[0][:-1]
        if self.remove_non_text:
            txt = re.sub('[a-zA-Z]/', '', txt).strip()
        return txt
    
    def _load_data(self, idx):
        wav_path = os.path.join(self.root_path, self.csv['segment_id'].iloc[idx]+'.wav')
        txt_path = os.path.join(self.root_path, self.csv['segment_id'].iloc[idx]+'.txt')
        
        mel = self._load_wav(wav_path)
        txt = self._load_txt(txt_path)
        pos_mel = np.arange(1, mel.shape[0] + 1)
        
        emotion = self.csv['emotion'].iloc[idx]
        valence = self.csv['valence'].iloc[idx]
        arousal = self.csv['arousal'].iloc[idx]
        
        sample = {
            'text' : txt,
            'mel' : mel,
            'pos_mel': pos_mel,
            'emotion': emotion2int[emotion],
            'valence': round(valence),
            'arousal': round(arousal)
        }
        return sample
    
    def __getitem__(self, idx):
        sample = self._load_data(idx)
        return sample

class multimodal_collator():
    
    def __init__(self, tokenizer, return_text=False, max_length=512):
        self.tokenizer = tokenizer
        self.return_text = return_text
        self.max_length = max_length
        
    def __call__(self, batch):
        text = [d['text'] for d in batch]
        mel = [d['mel'] for d in batch]
        pos_mel = [d['pos_mel'] for d in batch]
        emotion = [d['emotion'] for d in batch]
        valence = [d['valence'] for d in batch]
        arousal = [d['arousal'] for d in batch]
        text_length = [len(d['text']) for d in batch]
        
        text = [i for i, _ in sorted(
            zip(text, text_length), key=lambda x: x[1], reverse=True)]
        mel = [i for i, _ in sorted(
            zip(mel, text_length), key=lambda x: x[1], reverse=True)]
        pos_mel = [i for i, _ in sorted(
            zip(pos_mel, text_length), key=lambda x: x[1], reverse=True)]
        emotion = [i for i, _ in sorted(
            zip(emotion, text_length), key=lambda x: x[1], reverse=True)]
        valence = [i for i, _ in sorted(
            zip(valence, text_length), key=lambda x: x[1], reverse=True)]
        arousal = [i for i, _ in sorted(
            zip(arousal, text_length), key=lambda x: x[1], reverse=True)]
        
        mel = pad_mel(mel)
        pos_mel = prepare_data(pos_mel).astype(np.int32)
        
        text_inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
            max_length=self.max_length
        )
        mel_inputs = {
            'mel' : torch.FloatTensor(mel).permute(0, 2, 1),
            'pos_mel': torch.LongTensor(pos_mel)
        }
        labels = {
            "emotion" : torch.LongTensor(emotion),
            "valence" : torch.LongTensor(valence),
            "arousal" : torch.LongTensor(arousal)
        }
        if self.return_text:
            labels['text'] = text
        return text_inputs, mel_inputs, labels