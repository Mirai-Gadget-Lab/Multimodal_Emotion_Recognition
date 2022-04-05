import numpy as np

emotion2int = {
    'neutral': 0, 
    'angry' : 1,
    'happy' : 2, 
    'surprise' : 3, 
    'sad' : 4, 
    'fear' : 5, 
    'disgust' : 6
}

def pad_mel(inputs):
    _pad = 0

    def _pad_one(x, max_len):
        mel_len = x.shape[0]
        return np.pad(x, [[0, max_len - mel_len], [0, 0]], mode='constant', constant_values=_pad)

    max_len = max((x.shape[0] for x in inputs))
    return np.stack([_pad_one(x, max_len) for x in inputs])