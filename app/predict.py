import json
import tensorflow as tf
import numpy as np
import librosa
import os
from app.feature_utils import extract_features_single
import io
from pydub import AudioSegment

# 获取当前文件所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 加载模型注册表
with open(os.path.join(BASE_DIR, "model_config.json"), "r") as f:
    MODEL_REGISTRY = json.load(f)

MODEL_CACHE = {}

def load_model(dataset):
    """懒加载模型并构造绝对路径"""
    if dataset not in MODEL_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset}")
    if dataset not in MODEL_CACHE:
        relative_path = MODEL_REGISTRY[dataset]["path"]
        model_path = os.path.normpath(os.path.join(BASE_DIR, relative_path))
        print(f"✅ Loading model from: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        MODEL_CACHE[dataset] = tf.keras.models.load_model(model_path)
    return MODEL_CACHE[dataset], MODEL_REGISTRY[dataset]["labels"]

def predict_emotion(audio_bytes, dataset="ravdess"):
    model, labels = load_model(dataset)

    # 保存临时文件
    temp_path = os.path.join(BASE_DIR, "temp.wav")

    with open(temp_path, "wb") as f:
        f.write(audio_bytes)

    # 尝试读取音频文件
    try:
        # librosa 直接读取（WAV 文件）
        y, sr = librosa.load(temp_path, sr=None)
    except Exception:
        print("⚠️ Librosa failed to load file, trying to convert from WebM to WAV...")
        try:
            # 使用 pydub 转换为 WAV 格式
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")
            wav_bytes = io.BytesIO()
            audio.export(wav_bytes, format="wav")
            wav_bytes.seek(0)
            y, sr = librosa.load(wav_bytes, sr=None)
        except Exception as e:
            print("❌ Failed to convert audio:", e)
            raise e

    # 自动获取模型输入帧数
    target_frames = model.input_shape[1]
    print(f"🎯 Target frame length for {dataset}: {target_frames}")

    # 提取特征并调整长度
    features = extract_features_single(y, sr, target_frames=target_frames)

    preds = model.predict(features)

    idx = np.argmax(preds)
    confidence = float(np.max(preds))

    return {
        "dataset": dataset.upper(),
        "emotion": labels[idx],
        "confidence": round(confidence, 3)
    }
