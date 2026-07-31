"""
Emotion Classifier API
======================
A simple, beginner-friendly FastAPI web service that uses a pre-trained
Bidirectional GRU (BiGRU) neural network to detect emotions in text.

Emotions recognized:
    sadness, joy, love, anger, fear, surprise
"""

import pickle
import re
from contextlib import asynccontextmanager
from typing import Dict

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


# ---------------------------------------------------------------------------
# 1. CONFIGURATION & CONSTANTS
# ---------------------------------------------------------------------------

MODEL_PATH = "Artifacts/BiGRU_Modle.h5"
TOKENIZER_PATH = "Artifacts/tokenizer.pkl"

# The sequence length expected by the model during training
MAX_SEQUENCE_LENGTH = 50

# Labels in the exact order output by the neural network (indices 0 to 5)
EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# Emojis for display in the UI
EMOTION_EMOJIS = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}


# ---------------------------------------------------------------------------
# 2. TEXT PREPROCESSING HELPER
# ---------------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    """
    Cleans raw text so it matches the format used when training the model:
    - Converts text to lowercase
    - Removes apostrophes (e.g. "can't" -> "cant")
    - Removes special characters & punctuation
    - Removes extra spaces
    """
    text = text.lower()
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# 3. REQUEST & RESPONSE SCHEMAS
# ---------------------------------------------------------------------------

class TextInput(BaseModel):
    """Input schema: The text sent by the user."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The sentence to analyze.",
        json_schema_extra={"example": "I feel so happy and excited today!"},
    )


class PredictionResponse(BaseModel):
    """Output schema: The emotion prediction result."""
    text: str
    predicted_emotion: str
    confidence: float
    all_probabilities: Dict[str, float]


class HealthResponse(BaseModel):
    """Output schema: Server health check status."""
    status: str
    model_loaded: bool


# ---------------------------------------------------------------------------
# 4. MODEL LOADING & LIFESPAN MANAGEMENT
# ---------------------------------------------------------------------------

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the model and tokenizer once when the server starts up."""
    print("Loading model and tokenizer...")
    ml_models["model"] = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        ml_models["tokenizer"] = pickle.load(f)
    print("Model and tokenizer loaded successfully!")

    yield

    ml_models.clear()
    print("Cleaned up model resources.")


# ---------------------------------------------------------------------------
# 5. FASTAPI APP SETUP
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Emotion Classifier API",
    description="Detects emotions in text using a BiGRU deep learning model.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS so browser apps can interact with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static folder containing HTML/CSS/JS frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# 6. API ENDPOINTS
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def serve_ui():
    """Serves the web application homepage."""
    return FileResponse("static/index.html")


@app.get("/health", response_model=HealthResponse, tags=["Utility"])
def health_check():
    """Health check endpoint to confirm API & model status."""
    return HealthResponse(status="ok", model_loaded="model" in ml_models)


@app.post("/predict", response_model=PredictionResponse, tags=["Emotion Detection"])
def predict_emotion(payload: TextInput):
    """
    Main prediction endpoint:
    1. Cleans the input sentence.
    2. Converts words to numbers using tokenizer.
    3. Pads the sequence to length 50.
    4. Runs prediction through the BiGRU model.
    5. Returns the top emotion and full probability breakdown.
    """
    model = ml_models.get("model")
    tokenizer = ml_models.get("tokenizer")

    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not ready yet.")

    # 1. Clean input text
    cleaned_text = preprocess_text(payload.text)

    # 2 & 3. Tokenize & Pad sequence
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(
        sequence, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post"
    )

    # 4. Model prediction
    probabilities = model.predict(padded_sequence, verbose=0)[0]

    # 5. Format results
    top_index = int(np.argmax(probabilities))
    all_probabilities = {
        label: float(prob) for label, prob in zip(EMOTION_LABELS, probabilities)
    }

    return PredictionResponse(
        text=payload.text,
        predicted_emotion=EMOTION_LABELS[top_index],
        confidence=float(probabilities[top_index]),
        all_probabilities=all_probabilities,
    )
