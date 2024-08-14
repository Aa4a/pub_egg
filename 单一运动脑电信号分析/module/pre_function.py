import numpy as np
from scipy import signal
from scipy.fftpack import fft


# 滑动平均法
def mean_0(data):
    # 设置滑动窗大小N、步进P和数据长度
    N, P, L = 10, 1, len(data)
    k, m = 0, 0
    T1 = [0] * L
    W = {}
    for i in range(0, (L - N) // P + 2):
        if i + N - 1 > L:
            break
        else:
            for j in range(i, N + i):
                k = k + 1
                W[k - 1] = data[j - 1]
            W = np.array(list(W))
            T1[m - 1] = np.mean(W)
            k = 0
        m = m + 1
    a = [T1[m - 2]]
    T1[(L - N) // P + 1:] = a * len(T1[(L - N) // P + 1:])
    data = np.array(data)
    T1 = np.array(T1)
    new_data = data - T1
    return new_data


# 快速傅里叶变换
def fft_data(data):
    N = len(data)
    fft_y = fft(data)  # 变换进行FFT
    abs_y = np.abs(fft_y) / N  # 取复数的绝对值，即复数的模，获得振幅值,归一化处理
    abs_y_half = abs_y[range(int(N / 2))]  # 获得单边频谱

    # 确定频率。
    Fs = 250  # 采样率为250
    T = N / Fs  # 用采样率算出段数据中一共有多少个周期
    K = np.arange(N)  # 把采样点数的等差数列k除以周期T，就是频率 frq = k/T
    freq = K / T  # 计算每个点的频率值
    freq_half = freq[range(int(N / 2))]  # 由于对称性，取一半即可
    return freq_half, abs_y_half


# 50Hz陷波滤波器
def notch_filter(data, f0):
    fs = 250.0  # Sample frequency (Hz)
    Q = 30.0  # Quality factor
    # f0 = Frequency to be removed from signal (Hz)
    w0 = f0 / (fs / 2)  # Normalized Frequency
    # Design notch filter
    b, a = signal.iirnotch(w0, Q)  # 陷波滤波器signal.iirnotch
    # b, a = signal.iirpeak(w0, Q)   # 峰值滤波器signal.iirpeak
    result = signal.filtfilt(b, a, data)
    return result