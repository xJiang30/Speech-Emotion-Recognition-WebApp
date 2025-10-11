import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler

def extract_features_single(y, sr, target_frames=495):
    """
    提取特征，并自动调整到指定帧数
    """
    # 提取特征
    rms = librosa.feature.rms(y=y).T
    mel = librosa.feature.melspectrogram(y=y, sr=sr).T
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T

    scaler = StandardScaler()
    f_rms = scaler.fit_transform(rms)
    f_mel = scaler.fit_transform(mel)
    f_mfcc = scaler.fit_transform(mfcc)

    features = np.concatenate((f_rms, f_mel, f_mfcc), axis=1)

    # 调整到模型所需帧数
    if features.shape[0] < target_frames:
        pad_width = target_frames - features.shape[0]
        features = np.pad(features, ((0, pad_width), (0, 0)), mode="constant")
    else:
        features = features[:target_frames, :]

    features = np.expand_dims(features, axis=0)
    return features



