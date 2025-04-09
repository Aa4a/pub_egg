#%%
import os
from torcheeg.datasets import DEAPDataset
from torcheeg import transforms

from torcheeg.datasets.constants import \
    DEAP_CHANNEL_LOCATION_DICT


# 设置 HTTP 代理
os.environ['http_proxy'] = 'http://192.168.32.28:18000'
# 设置 HTTPS 代理
os.environ['https_proxy'] = 'http://192.168.32.28:18000'


dataset = DEAPDataset(
    io_path=".torcheeg/datasets_1744101603306_r9ceZ",
    root_path='deap_set/data_preprocessed_python',
    offline_transform=transforms.Compose([
        transforms.BandDifferentialEntropy(apply_to_baseline=True),
        transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT, apply_to_baseline=True)
    ]),
    online_transform=transforms.Compose(
        [transforms.BaselineRemoval(),
         transforms.ToTensor()]),
    label_transform=transforms.Compose([
        transforms.Select('valence'),
        transforms.Binary(5.0),
    ]),
    num_worker=8)

#%%
import torch
from torcheeg.utils import plot_3d_tensor

img = plot_3d_tensor(torch.tensor(dataset[0][0]))


#%%
import numpy as np
from torcheeg import transforms

from torcheeg.datasets.constants import DEAP_CHANNEL_LOCATION_DICT

t = transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
eeg = t(eeg=np.random.randn(32, 128))['eeg']
print(eeg.shape)

eeg = t(eeg=np.random.randn(32, 128), baseline=np.random.randn(32, 128))['eeg']
print(eeg.shape)

#%%
t = transforms.BaselineRemoval()

eeg = t(eeg=np.random.randn(32, 128), baseline=np.random.randn(32, 128))['eeg']
print(eeg.shape)

#%%
eeg = np.random.randn(32, 128)
transformed_eeg = transforms.BandDifferentialEntropy()(eeg=eeg)['eeg']
