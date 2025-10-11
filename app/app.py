from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.predict import predict_emotion

app = FastAPI(title="Speech Emotion Recognition Web App")

# # 挂载静态文件（CSS/JS等）
# app.mount("/static", StaticFiles(directory="app/static"), name="static")

# 模板目录
templates = Jinja2Templates(directory="app/templates")

# 首页
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 预测接口
@app.post("/predict")
async def predict(dataset: str = Form(...), file: UploadFile = File(...)):
    print("🔥 predict() called!")
    print("📦 Received dataset:", dataset)

    audio_bytes = await file.read()
    result = predict_emotion(audio_bytes, dataset=dataset)

    print("✅ Prediction complete:", result)
    return result
