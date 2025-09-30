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



#%%
from torcheeg.models import BCUNet

unet = BCUNet(num_classes=2)
mock_eeg = torch.randn(2, 4, 9, 9)
t = torch.randint(low=1, high=1000, size=(2, ))
y = torch.randint(low=0, high=2, size=(1, ))
fake_X = unet(mock_eeg, t, y)


#%%
from torcheeg.models import EEGfuseNet,EFDiscriminator

fusenet = EEGfuseNet(in_channels=1,
                     num_electrodes=32,
                     hid_channels_gru=16,
                     num_layers_gru= 1,
                     hid_channels_cnn=1,
                     chunk_size=384)
eeg = torch.randn(2,1, 32, 384)
# simply input the EEG signal to output generated samples and deep fusion codes
fake_X,deep_code = fusenet(eeg)

discriminator = EFDiscriminator(in_channels=1,
                                num_electrodes=32,
                                hid_channels_cnn=1,
                                chunk_size=384)
p_real = discriminator(eeg)
p_fake = discriminator(fake_X)



#%%
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
from torcheeg.datasets import DEAPDataset
from torcheeg import transforms
from torcheeg.datasets.constants import DEAP_CHANNEL_LOCATION_DICT
import torch.backends.cudnn as cudnn
from torcheeg.model_selection import KFoldGroupbyTrial
from torcheeg.models import CCNN
import torch.multiprocessing as mp
import torch
import gc
import os

# ================== 系统优化 ==================
cudnn.benchmark = True   # 对固定输入大小的 CNN 加速

# ================== 强制使用 spawn 启动方式（关键修复）==================
mp.set_start_method('spawn', force=True)

# ================== 数据集 ==================
dataset = DEAPDataset(
    io_path='/pub_egg/dateset/examples_trainers_1/deap',
    root_path='/pub_egg/dateset/deap_set/data_preprocessed_python',
    offline_transform=transforms.Compose([
        transforms.BandDifferentialEntropy(apply_to_baseline=True),
        transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT, apply_to_baseline=True)
    ]),
    online_transform=transforms.Compose([
        transforms.BaselineRemoval(),
        transforms.ToTensor()
    ]),
    label_transform=transforms.Compose([
        transforms.Select('valence'),
        transforms.Binary(5.0),
    ]),
    num_worker=0  # 注意：这里设为 0，避免预处理时多进程崩溃
)

# ================== KFold ==================
k_fold = KFoldGroupbyTrial(
    n_splits=18,
    split_path='/pub_egg/dateset/examples_trainers_1/split',
    shuffle=True,
    random_state=42
)

# ================== Accelerator ==================
accelerator = Accelerator(mixed_precision="fp16")   # 开启混合精度

# ================== 自动寻找最大 batch size ==================
def find_max_batch_size(train_dataset, val_dataset, start_bs=64, max_bs=1024):
    device = accelerator.device
    bs = start_bs
    last_success_bs = bs
    while bs <= max_bs:
        try:
            train_loader = DataLoader(
                train_dataset,
                batch_size=bs,  # ⚠️ 关键：不是 max_bs，而是当前测试的 bs
                shuffle=True,
                num_workers=0,  # 必须为 0，避免多进程崩溃
                pin_memory=True,
                persistent_workers=False
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=bs,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                persistent_workers=False
            )

            # 创建并移动模型到设备
            model = CCNN(num_classes=2, in_channels=4, grid_size=(9, 9)).to(device)
            x, y = next(iter(train_loader))
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                _ = model(x)  # forward 一次看看是否 OOM 或崩溃

            last_success_bs = bs
            bs *= 2  # 二分增长

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                break
            else:
                print(f"[Batch Size Test] Error at bs={bs}: {e}")
                break
        except Exception as e:
            print(f"[Batch Size Test] Unexpected error at bs={bs}: {e}")
            break

    return last_success_bs

# ================== 训练循环 ==================
def train_one_fold(train_loader, val_loader, fold_idx, accelerator):
    model = CCNN(num_classes=2, in_channels=4, grid_size=(9, 9))
    model = torch.compile(model)   # PyTorch 2.x 编译模式

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # ✅ 让 Accelerator 自动准备模型、优化器、数据加载器、并管理 scaler
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    best_acc = 0.0
    for epoch in range(10):
        # ---- 训练 ----
        model.train()
        total_loss = 0.0
        for x, y in tqdm(train_loader, disable=not accelerator.is_local_main_process):
            optimizer.zero_grad()

            # ✅ 使用 accelerator.autocast() 自动管理混合精度
            with accelerator.autocast():
                outputs = model(x)
                loss = criterion(outputs, y)

            # ✅ Accelerator 自动处理 scale + backward + step
            accelerator.backward(loss)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ---- 验证 ----
        model.eval()
        correct, total = 0, 0
        with torch.no_grad(), accelerator.autocast():
            for x, y in val_loader:
                outputs = model(x)
                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = correct / total if total > 0 else 0
        accelerator.print(f"[Fold {fold_idx}] Epoch {epoch+1} | Loss={avg_loss:.4f} | Val Acc={acc:.4f}")

        if acc > best_acc and accelerator.is_local_main_process:
            save_path = f'/pub_egg/dateset/examples_trainers_1/model/fold_{fold_idx}_best.pt'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            best_acc = acc

    return best_acc

# ================== 主训练流程 ==================
all_scores = []
for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):
    # 自动找最大 batch size
    max_bs = find_max_batch_size(train_dataset, val_dataset, start_bs=64, max_bs=1024)
    accelerator.print(f"[Fold {i}] Using batch size = {max_bs}")

    # 使用 num_workers=0 的 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=max_bs, shuffle=True,
                              num_workers=0, pin_memory=True, persistent_workers=False)
    val_loader = DataLoader(val_dataset, batch_size=max_bs, shuffle=False,
                            num_workers=0, pin_memory=True, persistent_workers=False)

    score = train_one_fold(train_loader, val_loader, i, accelerator)
    accelerator.print(f"Fold {i} Best Val Accuracy: {score:.4f}")
    all_scores.append(score)

    # 清理显存和缓存，防止累积
    torch.cuda.empty_cache()
    gc.collect()

accelerator.print(f"Average Accuracy over {len(all_scores)} folds: {sum(all_scores)/len(all_scores):.4f}")



#%%
import torch
from torch.utils.data import DataLoader
from torcheeg.models import CCNN
from torcheeg.datasets import DEAPDataset
from torcheeg import transforms
from torcheeg.datasets.constants import DEAP_CHANNEL_LOCATION_DICT


def load_model(model_path, device):
    # 定义模型结构（和训练时保持一致）
    model = CCNN(num_classes=2, in_channels=4, grid_size=(9, 9))
    # 加载权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_dataset():
    # 使用和训练时相同的 transform
    dataset = DEAPDataset(
        io_path='/pub_egg/dateset/examples_trainers_1/deap',
        root_path='/pub_egg/dateset/deap_set/data_preprocessed_python',
        offline_transform=transforms.Compose([
            transforms.BandDifferentialEntropy(apply_to_baseline=True),
            transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT, apply_to_baseline=True)
        ]),
        online_transform=transforms.Compose([
            transforms.BaselineRemoval(),
            transforms.ToTensor()
        ]),
        label_transform=transforms.Compose([
            transforms.Select('valence'),
            transforms.Binary(5.0),
        ]),
        num_worker=0  # 推理时不需要太多进程
    )
    return dataset


def evaluate(model, dataloader, device):
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0


if __name__ == "__main__":
    # ========== 配置 ==========
    model_path = "/pub_egg/dateset/examples_trainers_1/model/fold_3_best.pt"
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # ========== 加载模型和数据 ==========
    model = load_model(model_path, device)
    dataset = load_dataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # ========== 评估 ==========
    acc = evaluate(model, dataloader, device)
    print(f"Accuracy on dataset: {acc:.4f}")

    # ========== 单样本推理（例子） ==========
    x, y = dataset[0]
    x = x.unsqueeze(0).to(device)  # 增加 batch 维度
    with torch.no_grad():
        output = model(x)
        pred = output.argmax(dim=1).item()
    print(f"Sample true label: {y}, predicted: {pred}")



#%%
from torcheeg.datasets import DEAPDataset
from torcheeg import transforms

from torcheeg.model_selection import LeaveOneSubjectOut
from torcheeg.datasets.constants import \
    DEAP_CHANNEL_LOCATION_DICT
from torch.utils.data import DataLoader
from torcheeg.models import CCNN

from torcheeg.trainers import CORALTrainer

import pytorch_lightning as pl
import ipdb

dataset = DEAPDataset(
    io_path=f'/pub_egg/dateset/deap_set/examples_trainers_2/deap',
    root_path='/pub_egg/dateset/deap_set/data_preprocessed_python',
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


k_fold = LeaveOneSubjectOut(split_path='/pub_egg/dateset/deap_set/examples_trainers_2/split')


class Extractor(CCNN):
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.flatten(start_dim=1)
        return x


class Classifier(CCNN):
    def forward(self, x):
        x = self.lin1(x)
        x = self.lin2(x)
        return x

#%%
for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):
    ipdb.set_trace()
    source_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    target_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    extractor = Extractor(in_channels=4, num_classes=2)
    classifier = Classifier(in_channels=4, num_classes=2)

    trainer = CORALTrainer(extractor=extractor,
                                classifier=classifier,
                                num_classes=2,
                                lr=1e-4,
                                weight_decay=0.0,
                                accelerator='gpu')
    trainer.fit(source_loader,
                target_loader,
                target_loader,
                max_epochs=10,
                default_root_dir=f'/pub_egg/examples_trainers_2/model/{i}',
                callbacks=[pl.callbacks.ModelCheckpoint(save_last=True)],
                enable_progress_bar=True,
                enable_model_summary=True,
                limit_val_batches=0.0)
    score = trainer.test(target_loader,
                         enable_progress_bar=True,
                         enable_model_summary=True)[0]
    print(f'Fold {i} test accuracy: {score["test_accuracy"]:.4f}')

