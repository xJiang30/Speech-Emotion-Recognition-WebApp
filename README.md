# Speech Emotion Recognition using Long Short-Term Model

Speech emotion recognition base on Long Short-Term Model (LSTM), implemented in tensorflow.

The system improve the accuracy of emotion recognition while maintaining lightweight through feature selection and data enhancement optimization strategies.

Details are shown in the report.
&nbsp;

## Environments

- Python 3.9
- CUDA 11.5.2
- cudnn 8.9.6
- tensorFlow-gpu 2.10.1 


&nbsp;

## Structure

```
├── project/               
│   ├── config                 // experiment config
│   │   ├── config.py          // load config
│   │   └── config.yaml 
│   ├── data/                  // dataframes of four databases (output of load_database.py)
│   ├── data_utils             // data processing tools
│   │   ├── data_augmentation.py         // four data augmentation methods
│   │   └── label_mapping.py        // standardizing label mappings of four datasets
│   ├── kaggle/             // orignial databases
│   ├── logs/       // traning logs
│   ├── Models/           // save best weight of model
│   ├── load_database.py            // setup dataset
│   ├── lstm.py           // LSTM model, train function, evaluation function
│   ├── main.py           // main file of system
│   ├── plot_utils.py           // plot graphs
│   └── process_and_feature.py           // preprocessing and feature extraction
```


&nbsp;

## Requirments

### Python

- tensorflow-gpu: LSTM model
- scikit-learn: divide training sets and test sets, label One-Hot encoding, model evaluation
- librosa: audio data preprocessing, feature extraction 
- pandas: organizing data table structures
- Matplotlib: plot graphs
- NumPy: numerical calculation, audio feature matrix processing


&nbsp;



## Usage

### Prepare

Install dependencies:

```python
pip install -r requirements.txt
```

&nbsp;

### Configuration

Parameters can be configured in the config files (YAML) under `configs/`.


&nbsp;

### Data preparation 

First, you should load the database correctly, complete the label standardization mapping, and store the mapped dataframe in the data/ folder. (If it has been completed, please ignore it)
```python
python load_database.py
```

&nbsp;

### main.py
Then, you can directly start running the entire system according to your configuration by running `main.py` and reproduce the methods in the report.


```python
python main.py 
```
It includes the following steps:
1. load config
2. read dataset
3. preprocessing and extracting feature
4. train model
5. model evaluation
6. visualization of the training process


&nbsp;

### Other Functions

#### Chart Tool

All chart functions of the system are encapsulated in plot_utils.py, so you can use them very easily.

&nbsp;

#### Emotion distribution histogram of the dataset

```python
import plot_utils

"""
Args:
    df: xxx_df.csv file under the data/ folder (such as 'crema_df') 
    title: database name (such as 'Crema')
"""
plot_utils.plot_emotion_distribution(df, title)
```

&nbsp;

#### Waveplot of audio

```python
import plot_utils

"""
Args:
    df: xxx_df.csv file under the data/ folder (such as 'crema_df') 
    n: the n(th) audio file
"""

plot_utils.create_waveplot(df, n)
```

&nbsp;

#### RMS Energy Curve


```python
import plot_utils

plot_utils.create_rmse(df, n)
```

&nbsp;

#### Mel frequency spectrogram

```python
import plot_utils

plot_utils.create_melspectrogram(df, n)
```

&nbsp;

####  Distribution of MFCC

```python
import plot_utils

plot_utils.create_mfcc(df, n)
```

## Datasets

### [Kaggle](https://www.kaggle.com/datasets/dmitrybabko/speech-emotion-recognition-en): Speech Emotion Recognition (en)

### Crema

The third component is responsible for the emotion label:
- SAD - sadness;
- ANG - angry;
- DIS - disgust;
- FEA - fear;
- HAP - happy;
- NEU - neutral.

### Ravdess

Here is the filename identifiers as per the official RAVDESS website:

- Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
- Vocal channel (01 = speech, 02 = song).
- Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
- Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
- Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
- Repetition (01 = 1st repetition, 02 = 2nd repetition).
- Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

### Savee

The audio files in this dataset are named in such a way that the prefix letters describes the emotion classes as follows:

- 'a' = 'anger'
- 'd' = 'disgust'
- 'f' = 'fear'
- 'h' = 'happiness'
- 'n' = 'neutral'
- 'sa' = 'sadness'
- 'su' = 'surprise'

### TESS

The suffix after the underscore in the file name indicates the emotion.

&nbsp;


&nbsp;

## Author

#### Xin Jiang
- u01xj21@abdn.ac.uk (University of Aberdeen)
- x.jiang@m.scnu.edu.cn (South China Normal University)
