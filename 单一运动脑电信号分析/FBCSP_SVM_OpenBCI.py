#单一运动信号的脑电分析

'''
pyriemann 的核心是利用黎曼几何学来分析和处理协方差矩阵（covariance matrices），这在脑电图（EEG）等生物医学信号处理中非常有效。
协方差矩阵在黎曼流形上具有特殊的结构，传统的欧几里得空间方法（如线性代数操作）不适合处理这些矩阵。而 pyriemann 提供了在黎曼空间中计算平均、距离、内积等操作的工具。
'''

import time
import scipy.io
from scipy import signal

import mne
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import cohen_kappa_score

from matplotlib import pyplot as plt
from pyriemann.utils.viz import plot_confusion_matrix

from Module_FBCSP_OpenBCI import fbcsp
from module.pre_function import mean_0, fft_data, notch_filter

# 程序运行计时
time_start = time.time()

# 数据标签导入
datapath = './data/'
labelpath = './label/'

# 数据循环预处理
all_data = []
all_label = []
for i in range(1, 4):
    # 数据导入
    datafile = datapath + '2023-05-05-ct-data-E%s.mat' % i
    labelfile = labelpath + '2023-05-05-ct-label-E%s.mat' % i
    olddata = scipy.io.loadmat(datafile)
    oldlabel = scipy.io.loadmat(labelfile)
    rawdata = list(olddata['Data'])
    rawlabel = oldlabel['Labels']
    # 数据预处理
    n_data = {}
    for j in range(3):
        ch_data = rawdata[j]  # 通道数据导入
        # 滑动平均去基线漂移
        mean_data = mean_0(ch_data)
        # 50Hz陷波滤波器
        notch_data = notch_filter(mean_data, 50)
        # 0.5-100Hz的带通滤波
        # 带通滤波器换算公式为（滤波器阶数*截止频率/采样频率）
        b, a = signal.butter(2, [0.004, 0.8], 'bandpass')
        band_data = signal.filtfilt(b, a, notch_data)
        n_data[j] = band_data  # 通道数据整合
    # 数据拼接
    n_data = list(n_data.values())
    all_data = np.append(all_data, n_data)
    # 标签拼接
    all_label = np.append(all_label, rawlabel)

# 将数据格式转化为（trials，channels，samples）
trials, chans, samples = 120, 3, 2000
data = all_data.reshape(trials, chans, samples)

# 标签预处理
replace = {'left ': '1', 'right': '2'}
label = [replace[i] if i in replace else i for i in all_label]
label = np.array([int(x) for x in label])

# 用MNE重新创建Raw
'''
构建一个Raw对象时,需要准备两种数据,一种是data数据,一种是Info数据,
data数据是一个二维数据,形状为(n_channels,n_times)
info数据是一个三维数据，包括（通道名，采样频率，通道数据类型）
'''
ch_names = ['C3', 'Cz', 'C4']  # 通道名称
ch_types = ['eeg', 'eeg', 'eeg']  # 通道类型
sfreq = 250  # 采样率

# 设置info信息
info = mne.create_info(ch_names, sfreq, ch_types)  # 创建信号的信息
info.set_montage('standard_1020')

# 设置data信息，将三维信号转化为二维信号
raw_0 = data[0, :, :]
for i in range(1, trials):
    raw_i = data[i, :, :]
    raw_0 = np.concatenate((raw_0, raw_i), axis=1)
raw_data = raw_0
raw = mne.io.RawArray(raw_data, info)

# FIR带通滤波（8-30Hz）
# raw.filter(8., 30., fir_design='firwin', skip_by_annotation='edge')
'''
在创建Epochs对象时,必须提供一个"events"数组,
事件(event)描述的是某一种波形(症状)的起始点,其为一个三元组,形状为(n_events,3):
第一列元素以整数来描述的事件起始采样点;
第二列元素对应的是当前事件来源的刺激通道(stimulus channel)的先前值(previous value),该值大多数情况是0;
第三列元素表示的是该event的id.
'''
# 创建 events & event_id（截取3-7s的数据）
events = np.zeros((trials, chans), dtype='int')
k = sfreq * 3
for i in range(trials):
    events[i, 0] = k
    k += sfreq * 7
    events[i, 2] = label[i]
event_id = dict(left_hand=1, right_hand=2)

# 创建epochs
# 记录点的前1秒后4秒用于生成epoch数据
tmin, tmax = -1., 4.
epochs = mne.Epochs(raw, events, event_id
                    , tmin, tmax
                    , proj=True
                    , baseline=(None, 0)
                    , preload=True)
labels = epochs.events[:, -1]

# 划分测试集合训练集（划分比例为8:2）
epochs_data = epochs.get_data()
X_train, X_test, y_train, y_test = \
    train_test_split(epochs_data, labels, test_size=0.2)

# fbcsp
X_train_feature, X_test_feature = \
    fbcsp(X_train, y_train, X_test, y_test, filter_order=2)

# 设置svm模型参数
model = SVC(C=1.0, kernel='linear', gamma='auto')

# 10倍交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)
results = cross_val_score(model, X_train_feature, y_train, cv=kf)
print('---------------------------------')
print('KFold score:%f' % results.mean())

# 训练模型
model.fit(X_train_feature, y_train)

# 输出分类结果
score = model.score(X_test_feature, y_test)
y_predict = model.predict(X_test_feature)
kappa = cohen_kappa_score(y_test, y_predict)
time_end = time.time()

print('True label:%s' % y_test)
print('Predict label:%s' % y_predict)
print('Accuracy: %f' % (score*100) + '%')
print("Kappa: %f" % kappa)
print('Time cost', time_end - time_start, 's')

# 设置混淆矩阵
names = ['Left', 'Right']
plt.figure()
plot_confusion_matrix(y_predict, y_test, names, title='FBCSP-SVM_OpenBCI')
print('---------------------------------')
plt.show()


