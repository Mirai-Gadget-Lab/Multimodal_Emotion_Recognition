# %%
import pandas as pd
from glob import glob
import os
import shutil
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import librosa
import matplotlib.pyplot as plt
# %%
pd.set_option('display.max_columns', None)
# %%
annotations = sorted(glob('/home/spow12/data/Korean_SER/KEMD_19/annotation/*.csv'))
# %%
# 모든 Annotation 읽은이후 하나의 DataFrame으로 변환
save_path = "/home/spow12/data/Korean_SER/KEMD_19/annotation/"
df_ls = []
for path in annotations:
    df = pd.read_csv(path)
    df_ls.append(df.loc[1:, "Wav":"Unnamed: 12"])
    
dfs = pd.concat(df_ls)
# %%
dfs['Total Evaluation'].value_counts()

# %%
# 중복되는 Label(즉 헷갈리는 데이터는 제거)
dfs = dfs[dfs['Total Evaluation'].map(lambda x: ';' not in x)]
# %%
dfs.columns = ['wav_start', 'wav_end', 'ecg_start',
               'ecg_end', 'eda_start', 'eda_end', 
               'temp_start', 'temp_end', 'segment_id',
               'emotion', 'valence', 'arousal']
# %%
dfs = dfs[['segment_id', 'emotion', 'valence', 'arousal',
    'wav_start', 'wav_end', 'ecg_start',
    'ecg_end', 'eda_start', 'eda_end', 
    'temp_start', 'temp_end']]
# %%
dfs.to_csv('./data/annotation.csv', index=False)
# %%
# %%
# segment_id로 충분히 핸들링 할수 있으므로, 굳이 여러 디렉토리에 나눠져있을 필요가없음
# wav, txt를 하나의 디렉토리에 놓음. 
datas = glob('/home/spow12/data/Korean_SER/KEMD_19/wav/*/*/*')
for path in datas:
    try:
        shutil.copy(path, './data/txt_wav/%s'%os.path.basename(path))
    except IsADirectoryError:
        pass
# %%
dfs = pd.read_csv('./data/annotation.csv')
root_path = './data/txt_wav/'
wavfrom, sr = torchaudio.load(os.path.join(root_path,dfs['segment_id'].iloc[1] +'.wav'))
# %%
with open(os.path.join(root_path, dfs['segment_id'].iloc[0] + '.txt'), 'r') as f:
    t = f.readlines()
assert len(t) == 1,  'Fuck'
# %%
wavfrom.size(1)/ 16000
# %%
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    win_length=1024,
    hop_length=256,
    n_mels=80,
    power=2,
    normalized=True,
    norm='slaney',
    onesided=True,
    mel_scale="htk",
)
# %%
t_2 = mel_transform(wavfrom)
# %%
pad_sequence([t.squeeze(), t.squeeze(), t_2.squeeze()], batch_first=True, padding_value=0)
# %%
dfs[dfs['segment_id'] == 'Sess01_script01_F002']
# %%
torch.cat([t, t, t_2])
# %%
def _pad_mel(inputs):
    _pad = 0

    def _pad_one(x, max_len):
        mel_len = x.shape[0]
        return np.pad(x, [[0, max_len - mel_len], [0, 0]], mode='constant', constant_values=_pad)

    max_len = max((x.shape[0] for x in inputs))
    return np.stack([_pad_one(x, max_len) for x in inputs])

# %%
_pad_mel([t, t, t_2])
# %%
t.size()
# %%
temp = _pad_mel([t.squeeze().permute(1, 0), t_2.squeeze().permute(1, 0)])
# %%
torch.FloatTensor(temp)

# %%
def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)
# %%
plot_spectrogram(t_2[0])