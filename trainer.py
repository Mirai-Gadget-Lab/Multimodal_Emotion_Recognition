from config import HF_DataConfig, HF_TrainConfig
import os
from models.pl_model import PL_model
import pytorch_lightning as pl
from datasets.dataset_hf import *
import pytorch_lightning.callbacks as plc
import torch
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
from sklearn.model_selection import train_test_split

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_name', required=True, type=str)
    p.add_argument('--model_save_path', required=True, type=str)
    config = p.parse_args()

    return config

def main(config):
    pl.seed_everything(42)
    gpus = torch.cuda.device_count()
    # Define Configs 
    data_config = HF_DataConfig()
    train_config = HF_TrainConfig(
        checkpoint_path=os.path.join(config.model_save_path, 'checkpoint'),
        log_dir=os.path.join(config.model_save_path, 'tensorboard'),
    )
    model = PL_model(train_config)

    csv = pd.read_csv(data_config.csv_path)
    train, val = train_test_split(csv, test_size=0.2, random_state=1004)
    # data = PartitionPerEpochDataModule(train, val, train_config.batch_size, data_config, num_workers=8)
    
    text_tokenizer = AutoTokenizer.from_pretrained(train_config.text_encoder)
    audio_processor = Wav2Vec2Processor.from_pretrained(train_config.audio_processor)
    
    val_dataset = multimodal_dataset(val, data_config)
    train_dataset = multimodal_dataset(train, data_config)
    
    train_loader = DataLoader(train_dataset, train_config.batch_size, num_workers=8, 
                              collate_fn=multimodal_collator(text_tokenizer, audio_processor), shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, train_config.batch_size, num_workers=8,
                            collate_fn=multimodal_collator(text_tokenizer, audio_processor), pin_memory=True)
        
    checkpoint_callback = plc.ModelCheckpoint(
        monitor="val_loss",
        dirpath=train_config.checkpoint_path,
        filename="{epoch:02d}-{val_loss:.5f}",
        save_top_k=2,
        mode="min",
    )

    logger = TensorBoardLogger(train_config.log_dir, name=config.exp_name)

    trainer = pl.Trainer(
        gpus=gpus,
        strategy="ddp_spawn",
        max_epochs=100,
        checkpoint_callback=True,
        callbacks=[checkpoint_callback],
        precision=16,
        amp_backend="native",
        profiler="simple",
        logger=logger,
        num_nodes=1
        )
    trainer.fit(model, train_loader, val_loader)
    
if __name__ == '__main__':
    config = define_argparser()
    main(config)