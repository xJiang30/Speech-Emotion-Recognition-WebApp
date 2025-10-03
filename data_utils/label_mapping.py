import os
import pandas as pd

def ensure_data_dir():
    os.makedirs('../data', exist_ok=True)


def parse_crema(path):
    ensure_data_dir()
    emotions = []
    paths = []
    for file in os.listdir(path):
        paths.append(os.path.join(path, file))
        label = file.split('_')[2]
        mapping = {
            'SAD': 'sad', 'ANG': 'angry', 'DIS': 'disgust',
            'FEA': 'fearful', 'HAP': 'happy', 'NEU': 'neutral'
        }
        emotions.append(mapping.get(label, 'Unknown'))
    df = pd.DataFrame({'Emotions': emotions, 'Path': paths})
    df.to_csv('data/crema_df.csv', index=False)
    return df

def parse_ravdess(path):
    emotions = []
    paths = []
    for actor in os.listdir(path):
        actor_dir = os.path.join(path, actor)
        for file in os.listdir(actor_dir):
            parts = file.split('-')
            emotion = int(parts[2])
            emotions.append({
                1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
                5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprise'
            }.get(emotion, 'Unknown'))
            paths.append(os.path.join(actor_dir, file))
    df = pd.DataFrame({'Emotions': emotions, 'Path': paths})
    df.to_csv('data/ravdess_df.csv', index=False)
    return df

def parse_savee(path):
    emotions = []
    paths = []
    for file in os.listdir(path):
        part = file.split('_')[1][:-6]
        mapping = {
            'a': 'angry', 'd': 'disgust', 'f': 'fearful',
            'h': 'happy', 'n': 'neutral', 'sa': 'sad', 'su': 'surprise'
        }
        emotions.append(mapping.get(part, 'Unknown'))
        paths.append(os.path.join(path, file))
    df = pd.DataFrame({'Emotions': emotions, 'Path': paths})
    df.to_csv('data/savee_df.csv', index=False)
    return df

def parse_tess(path):
    emotions = []
    paths = []
    for subfolder in os.listdir(path):
        for file in os.listdir(os.path.join(path, subfolder)):
            part = file.split('_')[2].split('.')[0]
            if part == 'ps':
                emotions.append('surprise')
            elif part == 'fear':
                emotions.append('fearful')
            else:
                emotions.append(part)

            paths.append(os.path.join(path, subfolder, file))
    df = pd.DataFrame({'Emotions': emotions, 'Path': paths})
    df.to_csv('data/tess_df.csv', index=False)
    return df
