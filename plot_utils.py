import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_emotion_distribution(df, title):
    font1 = {'color':'black','size':30}
    sns.set(style="whitegrid", color_codes=True)
    plt.figure(figsize=(20,10),dpi=200)
    count = df['Emotions'].value_counts()
    ax = sns.barplot(x=count.values, y=count.index, palette="mako")
    ax.bar_label(ax.containers[0], fontsize=16)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("Count of Emotions", fontdict=font1)
    plt.ylabel("Name of Emotions", fontdict=font1)
    plt.title(title, fontdict=font1)
    plt.show()

def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss Curve')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['categorical_accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')

    plt.show()


'''
               FONT
'''
font2 = {'color':'black','size':20}


def create_waveplot(df, n):
    e = df.iloc[n-1,0]
    path = df.iloc[n-1,1]
    data, sr = librosa.load(path)
    plt.figure(figsize=(12, 6), dpi=200)
    plt.title('{}'.format(e), fontdict=font2)
    librosa.display.waveshow(data, sr=sr, color='#2E40CB')
    plt.show()


def create_rmse(df, n):
    e = df.iloc[n-1,0]
    path = df.iloc[n-1,1]
    data, sr = librosa.load(path)
    rmse = librosa.feature.rms(y=data, frame_length=2048, hop_length=512)
    rmse = rmse[0]
    plt.figure(figsize=(12, 6), dpi=200)
    plt.title('Root Mean Square Energy - {}'.format(e), fontdict=font2)

    energy = np.array(
        [sum(abs(data[i:i + 2048] ** 2)) for i in range(0, len(data), 512)])

    frames = range(len(energy))
    t = librosa.frames_to_time(frames, sr=sr, hop_length=512)

    librosa.display.waveshow(data, sr=sr, color='#2E40CB')
    plt.plot(t, energy / energy.max(), 'r--', label='Energy')  # normalized
    plt.plot(t[:len(rmse)], rmse / rmse.max(), color='g', label='RMSE')  # normalized
    plt.legend()
    plt.show()

    plt.show()

def create_melspectrogram(df, n):
    e = df.iloc[n-1,0]
    path = df.iloc[n-1,1]
    data, sr = librosa.load(path)
    X = librosa.feature.melspectrogram(y=data, sr=sr)

    plt.figure(figsize=(12, 6), dpi=200)
    plt.title('MelSpectrogram - {}'.format(e), fontdict= font2)
    librosa.display.specshow(X, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.show()

def create_mfcc(df,n):
    e = df.iloc[n-1,0]
    path = df.iloc[n-1,1]
    data, sr = librosa.load(path)
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
    plt.figure(figsize=(12, 6), dpi=200)
    plt.title('Mel-frequency cepstral coefficient - {}'.format(e))
    librosa.display.specshow(mfcc, x_axis='time')
    plt.ylabel('MFCC')
    plt.colorbar()
    plt.show()
