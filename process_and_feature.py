from sklearn.preprocessing import StandardScaler, OneHotEncoder
from data_utils import *
import config


#load config
config = config.load_config("config.yaml")


# Determine maximum audio length dynamically
def compute_max_length(dataset):
    return max([len(librosa.load(path, sr=None)[0]) for path in dataset['Path']])


# data augmentation
def augment_audio(audio, sr):
    if config["df_path"] == "data/crema_df.csv":
        return [audio, noise(audio)]
    if config["df_path"] == "data/ravdess_df.csv":
        return [audio, noise(audio), shift(audio)]
    if config["df_path"] == "data/savee_df.csv":
        return [audio, noise(audio), stretch(audio),shift(audio),pitch(audio,sr)]
    if config["df_path"] == "data/tess_df.csv":
        return [audio]


# feature extraction: RMS、Mel Spectrogram、MFCC
def extract_features(audio, sr, scaler, max_length):
    if len(audio) < max_length:
        final_audio = np.pad(audio, (0, max_length - len(audio)), 'constant')
    else:
        final_audio = audio[:max_length]

    f1 = librosa.feature.rms(y=final_audio).T
    f2 = librosa.feature.melspectrogram(y=final_audio, sr=sr).T
    f3 = librosa.feature.mfcc(y=final_audio, sr=sr, n_mfcc=40).T

    return (
        scaler.fit_transform(f1),
        scaler.fit_transform(f2),
        scaler.fit_transform(f3)
    )


# main function
def build_feature_set(dataset):
    max_length = compute_max_length(dataset)
    scaler = StandardScaler()

    rms, mfcc, mel, emotions = [], [], [], []

    for _, row in dataset.iterrows():
        path, emotion = row['Path'], row['Emotions']
        rawsound, sr = librosa.load(path)
        normalizedsound = librosa.util.normalize(rawsound)
        trimmed_audio, _ = librosa.effects.trim(y=normalizedsound, top_db=30)

        augmented_audios = augment_audio(trimmed_audio, sr)

        for audio in augmented_audios:
            f1, f2, f3 = extract_features(audio, sr, scaler, max_length)
            rms.append(f1)
            mel.append(f2)
            mfcc.append(f3)
            emotions.append(emotion)

    f_rms = np.asarray(rms)
    f_mel = np.asarray(mel)
    f_mfccs = np.asarray(mfcc)

    X = np.concatenate((f_rms, f_mel, f_mfccs), axis=2)
    Y = np.asarray(emotions).reshape(-1, 1)
    enc = OneHotEncoder()
    Y = enc.fit_transform(Y).toarray()

    return X, Y, enc

