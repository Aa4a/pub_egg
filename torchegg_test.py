#%%
from torcheeg.datasets import DEAPDataset
from torcheeg import transforms
from torcheeg.datasets.constants import DEAP_CHANNEL_LIST
from torcheeg.models.cnn import TSCeption
import torch


dataset = DEAPDataset(
    io_path='/pub_egg/dateset/datasets_1744101603306_r123',
    root_path='/pub_egg/dateset/deap_set/data_preprocessed_python',
    chunk_size=512,
    num_baseline=1,
    baseline_chunk_size=512,
    offline_transform=transforms.Compose([
        transforms.PickElectrode(transforms.PickElectrode.to_index_list(
            ['FP1', 'AF3', 'F3', 'F7',
             'FC5', 'FC1', 'C3', 'T7',
             'CP5', 'CP1', 'P3', 'P7',
             'PO3','O1', 'FP2', 'AF4',
             'F4', 'F8', 'FC6', 'FC2',
             'C4', 'T8', 'CP6', 'CP2',
             'P4', 'P8', 'PO4', 'O2'], DEAP_CHANNEL_LIST)),
        # 不要 To2d()
    ]),
    online_transform=transforms.ToTensor(),
    label_transform=transforms.Compose([
        transforms.Select('valence'),
        transforms.Binary(5.0),
    ])
)


model = TSCeption(num_classes=2,
                  num_electrodes=28,
                  sampling_rate=128,
                  num_T=15,
                  num_S=15,
                  hid_channels=32,
                  dropout=0.5)

#%%
x = dataset[0][0]         # [28, 512]
print(x.shape)
x = torch.unsqueeze(x, 0) # [1, 28, 512]
x = torch.unsqueeze(x, 1) # [1, 1, 28, 512]
print(x.shape)

pred = model(x)           # OK
print(pred.shape)         # [1, 2]



#%%
from torcheeg.models import GRU

model = GRU(num_electrodes=32, hid_channels=64, num_classes=2)

eeg = torch.randn(2, 32, 128)
pred = model(eeg)


#%%
from torcheeg.models import DGCNN

eeg = torch.randn(1, 62, 200)
model = DGCNN(in_channels=200, num_electrodes=62, hid_channels=32, num_layers=2,num_classes=2)
pred = model(eeg)


#%%
from torcheeg.models import SimpleViT

eeg = torch.randn(1, 128, 9, 9)
model = SimpleViT(chunk_size=128, t_patch_size=32, s_patch_size=(3, 3), num_classes=2)
pred = model(eeg)


#%%
import torch
from torcheeg.models import ATCNet
from torcheeg.datasets import BCICIV2aDataset
from torcheeg import transforms

dataset = BCICIV2aDataset(io_path=f'/pub_egg/dateset/bciciv_2a',
                              root_path='/pub_egg/dateset/downloads/card_1',
                              online_transform=transforms.Compose([
                                  transforms.To2d(),
                                  transforms.ToTensor()
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select('label'),
                                  transforms.Lambda(lambda x: x - 1)
                              ]))
model = ATCNet(num_classes=4,
               num_windows=3,
               num_electrodes=22,
               chunk_size=1750)
x = dataset[0][0]
x = torch.unsqueeze(x,dim=0)
pred = model(x)


#%%
from torcheeg.models import BCGenerator, BCDiscriminator

g_model = BCGenerator(in_channels=128, num_classes=3)
d_model = BCDiscriminator(in_channels=4, num_classes=3)
z = torch.normal(mean=0, std=1, size=(1, 128))
y = torch.randint(low=0, high=3, size=(1, ))
fake_X = g_model(z, y)
disc_X = d_model(fake_X, y)


#%%
from torcheeg.models import BCEncoder, BCDecoder

encoder = BCEncoder(in_channels=4, num_classes=3)
decoder = BCDecoder(in_channels=64, out_channels=4, num_classes=3)
y = torch.randint(low=0, high=3, size=(1, ))
mock_eeg = torch.randn(1, 4, 9, 9)
mu, logvar = encoder(mock_eeg, y)
std = torch.exp(0.5 * logvar)
eps = torch.randn_like(std)
z = eps * std + mu
fake_X = decoder(z, y)


#%%
import torch.nn.functional as F
import torch
from torcheeg.models import BCGlow

model = BCGlow(num_classes=2)

# forward to calculate loss function
mock_eeg = torch.randn(2, 4, 32, 32)
y = torch.randint(0, 2, (2, ))  # 默认 LongTensor，不要转 float


nll_loss, y_logits, z_outs = model(mock_eeg, y)
# y_logits shape: (batch,), y shape: (batch,)
bce_loss = F.binary_cross_entropy_with_logits(y_logits, y.float())  # y 转 float
loss = nll_loss.mean() + bce_loss

# sample a generated result
fake_X = model.sample(y, temperature=1.0)
