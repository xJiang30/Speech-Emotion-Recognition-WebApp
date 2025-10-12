# Speech Emotion Recognition Web App

An interactive web application that performs real-time speech emotion recognition using LSTM-based deep learning models, supporting both file upload and live microphone recording.
powered by **TensorFlow + FastAPI + Docker + AWS EC2**.

## Demo Preview

🌐 **Live Demo:** [http://51.20.65.148](http://51.20.65.148)

&nbsp;

## Features

- Record your voice directly in the browser
- Upload `.wav` audio for prediction
- Multi-Model Support – Switch between datasets: CREMA-D, RAVDESS, SAVEE, TESS
- Real-time emotion visualization with emojis
- Confidence score animation bar
- Deployed on AWS EC2 via Docker container


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

## 🧩 Tech Stack

### AI / ML
- TensorFlow 2.10 (LSTM model)
- Librosa (audio feature extraction: MFCC, RMS, Mel-spectrogram)
- Scikit-learn (feature normalization, encoding)

### Backend
- FastAPI (Python async web framework)
- Uvicorn (ASGI server)
- Pydub + FFmpeg (audio format conversion)

### Frontend
- HTML + Bootstrap 5
- JavaScript (fetch API + dynamic rendering)
- Emoji-based emotion UI with animated transitions

### Deployment
- Docker containerized application
- AWS EC2 Ubuntu instance
- Configurable port access (`8080` by default)

&nbsp;


## Local Development Setup

### Clone the repository

```python
git clone https://github.com/xjiang30/Speech-Emotion-Recognition-using-LSTM.git
cd Speech-Emotion-Recognition-using-LSTM/app
```

### Install dependencies

```python
pip install -r requirements.txt
pip install pydub ffmpeg-python
```

### Install FFmpeg (for audio conversion)

Download from ffmpeg.org/download, extract it, and add C:\ffmpeg\bin to your PATH environment variable.
Then verify:
```python 
ffmpeg -version
```

### Run Locally

Start the FastAPI server:
```python 
python -m uvicorn app.app:app --port 8080

```

Access in browser: http://127.0.0.1:8080

&nbsp;

## Docker Deployment

### build Docker Image
docker build -t ser-app .

### Run Container
docker run -p 8080:8080 ser-app

## Cloud Deployment (AWS EC2)
- Launch an Ubuntu 22.04 EC2 instance
- SSH into the instance:
```bash
ssh -i your-key.pem ec2-user@<EC2-IP>
```
- Install Docker:
```bash
sudo apt update && sudo apt install -y docker.io
```
- Clone the repo & build:
```bash
git clone https://github.com/xJiang30/Speech-Emotion-WebApp.git
cd Speech-Emotion-Recognition-using-LSTM
sudo docker build -t ser-app .
sudo docker run -d -p 8080:8080 ser-app
```
- Access via: http://<your-EC2-IP>:8080 (In my case: http://51.20.65.148/8080)

## System Architecture

          ┌────────────────────────────────────────────┐
          │                Frontend UI                 │
          │  - HTML / Bootstrap                        │
          │  - JS + Fetch API                          │
          │  - Live Audio Recorder                     │
          └────────────────────────────────────────────┘
                              │
                              ▼
          ┌────────────────────────────────────────────┐
          │               FastAPI Backend              │
          │  - Audio Upload / Recording Endpoint       │
          │  - Dynamic Model Loader (RAVDESS, CREMA…)  │
          │  - TensorFlow LSTM Prediction              │
          └────────────────────────────────────────────┘
                              │
                              ▼
          ┌────────────────────────────────────────────┐
          │           Deployed via Docker              │
          │  - Runs on AWS EC2 (Ubuntu 22.04)          │
          │  - FFmpeg + TensorFlow Runtime             │
          └────────────────────────────────────────────┘

&nbsp;

## Author

#### Xin Jiang
- Master of Computer Science @ UC Irvine
- linkedin.con/in/xin-jiang12 (LinkedIn)
