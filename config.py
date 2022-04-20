from dataclasses import dataclass


@dataclass
class DataConfig():
    """
    Data Settings
    """
    root_path: str = "./data/txt_wav"
    csv_path: str = "./data/annotation_for_dev.csv"
    sample_rate: int = 16000
    n_fft: int = 2048
    win_length: int = 2048
    hop_length: int = 512
    n_mels: int = 80
    power: int = 2
    normalized: bool = True
    remove_non_text: bool = True
    
@dataclass
class TrainConfig():
    hidden_size: int = 384
    n_head: int = 12
    n_layers: int = 6
    lr: float = 5e-4
    decoder_prenet_hidden_size: int = 32
    dropout_p: float = 0.1
    batch_size: int = 4
    num_warmup_steps: int = 1500
    num_training_steps: int = 15000
    
    using_model: str = 'both'
    audio_model: str = 'transformer'
    label_name: str = 'emotion'
    out_dim: int = 7
    text_encoder: str = "klue/roberta-base"
    checkpoint_path: str = './models_zoo/checkpoint/'
    log_dir: str = './models_zoo/tensorboard/'
    
@dataclass
class HF_DataConfig():
    """
    Data Settings
    """
    root_path: str = "./data/txt_wav"
    csv_path: str = "./data/annotation_for_dev.csv"
    normalized: bool = True
    remove_non_text: bool = True
    # return_text: bool = False
    
@dataclass
class HF_TrainConfig():
    num_warmup_steps: int = 1000
    num_training_steps: int = 10000
    lr: float = 5e-5
    label_name: str = 'emotion'
    checkpoint_path: str = './models_zoo/checkpoint/'
    log_dir: str = './models_zoo/tensorboard/'
    using_model: str = 'both'
    batch_size: int = 4
    text_encoder: str = "klue/roberta-base"
    audio_processor: str = "w11wo/wav2vec2-xls-r-300m-korean"