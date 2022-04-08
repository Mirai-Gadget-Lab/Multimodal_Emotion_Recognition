from dataclasses import dataclass


@dataclass
class DataConfig():
    """
    Data Settings
    """
    root_path: str = "./data/txt_wav"
    csv_path: str = "./data/annotation.csv"
    sample_rate: int = 16000
    n_fft: int = 1024
    win_length: int = 1024
    hop_length: int = 256
    n_mels: int = 80
    power: int = 2
    normalized: bool = True
    remove_non_text: bool = True
    
@dataclass
class TrainConfig():
    hidden_size: int = 256
    n_head: int = 4
    n_layers: int = 6
    lr: float = 1e-3
    decoder_prenet_hidden_size: int = 32

@dataclass
class HF_DataConfig():
    """
    Data Settings
    """
    root_path: str = "./data/txt_wav"
    csv_path: str = "./data/annotation.csv"
    normalized: bool = True
    remove_non_text: bool = True
    # return_text: bool = False
    
@dataclass
class HF_TrainConfig():
    num_warmup_steps: int = 1000
    num_training_steps: int = 10000
    lr: float = 1e-3
    checkpoint_path: str = './models/checkpoint/'
    log_dir: str = './models/tensorboard/'
    batch_size: int = 2
    text_encoder: str = "klue/roberta-base"
    audio_processor: str = "kresnik/wav2vec2-large-xlsr-korean"