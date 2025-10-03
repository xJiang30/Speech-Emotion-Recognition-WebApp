import librosa
import numpy as np
import config


#load config
config = config.load_config("../config/config.yaml")


def noise(data, noise_rate=config["noise_rate"]):
    noise_amp = noise_rate * np.random.uniform() * np.amax(data)
    data = data + noise_amp * np.random.normal(size=data.shape[0])
    return data


def stretch(data, rate=config["stretch_rate"]):
    return librosa.effects.time_stretch(y=data, rate=rate)


def shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
    return np.roll(data, shift_range)


def pitch(data, sampling_rate, pitch_factor=config["pitch_factor"]):
    return librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps=pitch_factor)