# Speech Emotion Recognition Web App

An interactive web application that performs real-time speech emotion recognition using LSTM-based deep learning models, supporting both file upload and live microphone recording.
Built with FastAPI, TensorFlow, and Librosa, deployed on AWS/Render-compatible backend.
&nbsp;

## Features

- Live Recording – Record your voice directly from browser for instant prediction
- File Upload – Upload .wav files for batch emotion analysis
- Multi-Model Support – Switch between datasets: CREMA-D, RAVDESS, SAVEE, TESS
- Deep Learning Powered – LSTM models trained for speech emotion classification
- Dynamic UI – Animated results, emoji mapping by emotion, progress confidence bar
- Deployable – Fully functional FastAPI backend, compatible with AWS EC2 or Render


&nbsp;

## Structure

```
├── SER-deplop/               
│   ├── project/                // Trained models
│   │   ├── Models/          
│   │       ├── best_weights_crema.hdf5
│   │       ├── best_weights_ravdess.hdf5
│   │       ├── best_weights_savee.hdf5 
│   │       └── best_weights_tess.hdf5 
│   │ 
│   ├── app/                  
│   │   ├── app.py              // FastAPI entrypoint (UI + routes)
│   │   ├── predict.py          // Model loading and inference logic
│   │   ├── feature_utils.py    // Audio feature extraction
│   │   ├── mode_config.json    // Model paths and label maps
│   │   ├── requirement.txt     // Dependencies
│   │   └── templates
│   │       └── index.html      // Home page
│   │ 
│   └── README.md               // Project documentatio
```


&nbsp;

## Installation

### Clone the repository

```python
git clone https://github.com/xjiang30/Speech-Emotion-Recognition-using-LSTM.git
cd Speech-Emotion-Recognition-using-LSTM/app
```

### Install dependencies

```python
pip install -r requirements.txt
```

### Install FFmpeg (for audio conversion)

Download from ffmpeg.org/download, extract it, and add C:\ffmpeg\bin to your PATH environment variable.
Then verify:
```python 
ffmpeg -version
```
&nbsp;



## Run Locally

Start the FastAPI server:
```python 
python -m uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload

```

Access in browser: http://127.0.0.1:8000

## Usage

### Record Live Audio
Click Start Recording, speak for a few seconds, then stop and the app automatically uploads your audio and predicts emotion.

### Upload a File
Upload any .wav file and select a dataset (CREMA, RAVDESS, etc).

You’ll see:

Detected Emotion 😄😠😢

Confidence percentage

Audio playback of your sample

&nbsp;

## Author

#### Xin Jiang
- jiangx15@uci.edu (University of California, Irvine)
- linkedin.con/in/xin-jiang12 (LinkedIn)
