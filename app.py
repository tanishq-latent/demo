import os
import pickle
import numpy as np
import tensorflow as tf
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Pydantic Request & Response Schemas
class TextRequest(BaseModel):
    text: str

class EmotionResponse(BaseModel):
    text: str
    emotion: str
    confidence: float

# 2. Artifact Paths & Constants
MODEL_PATH = "emotion_model_artifacts/best_emotion_model.keras"
TOKENIZER_PATH = "emotion_model_artifacts/tokenizer.pickle"
MAX_LEN = 50
LABEL_NAMES = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

model = None
tokenizer = None

def load_artifacts():
    global model, tokenizer
    if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)
        print("Model and Tokenizer loaded successfully.")

# Load artifacts upon module import
load_artifacts()

@asynccontextmanager
async def lifespan(app: FastAPI):
    if model is None or tokenizer is None:
        load_artifacts()
    yield

# 3. Initialize FastAPI App
app = FastAPI(
    title="MindPulse AI — Emotion Classification API",
    description="FastAPI service serving Bidirectional GRU model",
    lifespan=lifespan
)

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Endpoints
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    index_path = "index.html"
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>MindPulse AI API</h1><p>Visit <a href='/docs'>/docs</a> for API documentation.</p>"

@app.post("/predict", response_model=EmotionResponse)
def predict(request: TextRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model artifacts are not loaded.")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    # Preprocess text sequence
    sequence = tokenizer.texts_to_sequences([request.text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')

    # Predict emotion probabilities
    probabilities = model.predict(padded, verbose=0)[0]
    top_index = int(np.argmax(probabilities))
    confidence = float(probabilities[top_index])

    return EmotionResponse(
        text=request.text,
        emotion=LABEL_NAMES[top_index],
        confidence=round(confidence, 4)
    )
