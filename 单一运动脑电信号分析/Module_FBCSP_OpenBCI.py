import numpy as np
from mne.decoding import CSP
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from scipy.signal import butter, filtfilt
from sklearn.metrics import confusion_matrix


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


def sen_spec(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    total=sum(sum(cm))
    accuracy=(cm[0,0]+cm[1,1])/total
    sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
    specificity = cm[1,1]/(cm[1,0]+cm[1,1])
    return sensitivity, specificity


def fbcsp(X_train, y_train, X_test, y_test, filter_order=2):
    # 设置滤波器组信息（带宽和个数）
    filters = [[4, 8], [8, 12], [12, 16],
               [16, 20], [20, 24], [24, 28],
               [28, 32], [32, 36], [36, 40]]
    # 设置分类个数、特征个数和滤波器组长度
    n_components, n_features, n_fbank = 2, 9, len(filters)
    # csp = CSP(n_components=n_components, norm_trace=False)
    X_train_fbcsp = np.zeros([X_train.shape[0], n_fbank, n_components])
    X_test_fbcsp = np.zeros((X_test.shape[0], n_fbank, n_components))

    fbcsp = {}  # dict
    for idx, (f1, f2) in enumerate(filters, start=0):
        X_train_fb = butter_bandpass_filter(X_train, f1, f2, fs=250, order=filter_order)
        X_test_fb = butter_bandpass_filter(X_test, f1, f2, fs=250, order=filter_order)
        csp = CSP(n_components=2
                  , reg=None
                  , log=False
                  , norm_trace=False)
        X_train_fbcsp[:, idx, :] = csp.fit_transform(X_train_fb, y_train)
        fbcsp[(f1, f2)] = csp
        for n_sample in range(X_test_fb.shape[0]):
            csp_test = X_test_fb[n_sample, :, :].reshape(1, X_test_fb.shape[1], X_test_fb.shape[2])
            X_test_fbcsp[n_sample, idx, :] = csp.transform(csp_test)

    nsamples, nx, ny = X_train_fbcsp.shape
    X_train_fbcsp = X_train_fbcsp.reshape((nsamples, nx * ny))
    nsamples, nx, ny = X_test_fbcsp.shape
    X_test_fbcsp = X_test_fbcsp.reshape((nsamples, nx * ny))

    selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
    X_train_feature = selector.fit_transform(X_train_fbcsp, y_train)
    X_test_feature = selector.transform(X_test_fbcsp)

    return X_train_feature, X_test_feature
