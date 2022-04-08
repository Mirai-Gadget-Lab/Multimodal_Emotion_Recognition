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
import horovod.torch as hvd
import torch.multiprocessing as mp
import tempfile

os.environ['CUDA_VISIBLE_DEVICES']="0,1"
def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_name', required=True, type=str)
    p.add_argument('--model_save_path', required=True, type=str)
    config = p.parse_args()

    return config

def main(args):
    kwargs = {'num_workers': 8}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'
    data_config = HF_DataConfig()
    train_config = HF_TrainConfig(
        checkpoint_path=os.path.join(args.model_save_path, 'checkpoint'),
        log_dir=os.path.join(args.model_save_path, 'tensorboard'),
    )

    # Load train and validation data
    csv = pd.read_csv(data_config.csv_path)
    csv = csv.drop_duplicates(subset=['segment_id'], ignore_index=True)
    train, val = train_test_split(csv, test_size=0.2, random_state=1004)
    
    text_tokenizer = AutoTokenizer.from_pretrained(train_config.text_encoder)
    audio_processor = Wav2Vec2Processor.from_pretrained(train_config.audio_processor)
    
    train_dataset = multimodal_dataset(train, data_config)
    val_dataset = multimodal_dataset(val, data_config)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    
    train_loader = DataLoader(train_dataset, train_config.batch_size, 
                              **kwargs, sampler=train_sampler,
                              collate_fn=multimodal_collator(text_tokenizer, audio_processor))
    
    val_loader = DataLoader(val_dataset, train_config.batch_size, 
                            **kwargs, sampler=val_sampler, shuffle=False,
                            collate_fn=multimodal_collator(text_tokenizer, audio_processor))
        
    # Load model and configuration.
    
    model = PL_model(train_config)
    setattr(model, 'train_dataloader', lambda: train_loader)
    setattr(model, 'val_dataloader', lambda: val_loader)
        
    checkpoint_callback = plc.ModelCheckpoint(
        monitor="val_loss",
        dirpath=train_config.checkpoint_path,
        filename="{epoch:02d}-{val_loss:.5f}",
        save_top_k=2,
        mode="min",
    )

    logger = TensorBoardLogger(train_config.log_dir, name=args.exp_name)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        strategy="horovod",
        max_steps=train_config.num_training_steps,
        checkpoint_callback=True,
        callbacks=[checkpoint_callback],
        precision=16,
        amp_backend="native",
        profiler="simple",
        accumulate_grad_batches=8,
        logger=logger,
        )
    
    trainer.fit(model)
    
if __name__ == '__main__':
    args = define_argparser()
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(42)
    
    main(args)