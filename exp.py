from torch.utils.data import DataLoader
from dataset import multimodal_dataset, multimodal_collator
from config import *
# %%
data_config = DataConfig()
dataset = multimodal_dataset(data_config)
loader = DataLoader(dataset, batch_size=16, collate_fn=multimodal_collator())
# %%
for i in loader:
    inputs_, label = i
    break
# %%