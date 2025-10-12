FROM python:3.9-slim

WORKDIR /app

COPY ./app/requirements.txt /app/requirements.txt
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    pip install pydub ffmpeg-python jinja2

COPY . /app

# FastAPI port
EXPOSE 8080

# start
CMD ["python", "-m", "uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8080"]

