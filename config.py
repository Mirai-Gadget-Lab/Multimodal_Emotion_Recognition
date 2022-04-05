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