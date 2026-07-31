"""
Emotion Classifier API
======================

This is a simple FastAPI web service that wraps a pre-trained
BiGRU (Bidirectional GRU) deep learning model. You send it a piece
of text and it tells you which emotion the text expresses.

The model was trained to recognize 6 emotions:
    sadness, joy, love, anger, fear, surprise

HOW THIS FILE IS ORGANIZED (read top to bottom, it tells a story):
    1. Imports
    2. Settings / constants  (things you might want to tweak)
    3. Pydantic models       (the "shape" of requests & responses)
    4. Loading the ML model  (done once, when the server starts)
    5. The actual API endpoints (the URLs people can call)

To run this API:
    1. pip install -r requirements.txt
    2. uvicorn main:app --reload
    3. Open http://127.0.0.1:8000 in your browser for the interactive UI
       or http://127.0.0.1:8000/docs for the auto-generated API docs
"""

import pickle
from contextlib import asynccontextmanager
from typing import Dict

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ---------------------------------------------------------------------------
# 2. SETTINGS
# These are the only values you are likely to need to change.
# ---------------------------------------------------------------------------

MODEL_PATH = "Artifacts/BiGRU_Modle.keras"
TOKENIZER_PATH = "Artifacts/tokenizer.pickle"

# The model expects every sentence to be turned into a sequence of exactly
# this many numbers (shorter sentences are padded with zeros, longer ones
# are cut off). This MUST match the value used while training the model.
MAX_SEQUENCE_LENGTH = 50

# The order of these labels must match the order the model was trained on.
# This model outputs 6 numbers (probabilities) and index 0 corresponds to
# the first label below, index 1 to the second, and so on.
EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# A little emoji per emotion, purely to make the UI nicer.
EMOTION_EMOJIS = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}


# ---------------------------------------------------------------------------
# 3. PYDANTIC MODELS
# Pydantic models describe exactly what data goes IN and OUT of our API.
# FastAPI uses them to validate requests automatically and to generate
# the interactive docs at /docs.
# ---------------------------------------------------------------------------

class TextInput(BaseModel):
    """What the client must send us: just a piece of text."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The sentence you want to analyze.",
        json_schema_extra={"example": "I am so excited about my trip tomorrow!"},
    )


class PredictionResponse(BaseModel):
    """What we send back: the predicted emotion plus the full breakdown."""

    text: str = Field(..., description="The original text that was analyzed.")
    predicted_emotion: str = Field(..., description="The single most likely emotion.")
    confidence: float = Field(..., description="Confidence score for the top emotion (0 to 1).")
    all_probabilities: Dict[str, float] = Field(
        ..., description="Probability for every emotion the model knows about."
    )


class HealthResponse(BaseModel):
    """Simple response for the health-check endpoint."""

    status: str
    model_loaded: bool


# ---------------------------------------------------------------------------
# 4. LOADING THE MACHINE LEARNING MODEL
# Loading a deep learning model is slow, so we do it ONCE when the server
# starts up, and then reuse it for every request. We store it in a small
# dictionary called `ml_models` so the endpoints below can access it.
# ---------------------------------------------------------------------------

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup: runs once, before the server accepts any requests ---
    print("Loading model and tokenizer, please wait...")
    ml_models["model"] = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        ml_models["tokenizer"] = pickle.load(f)
    print("Model and tokenizer loaded successfully!")

    yield  # <-- the app runs while we're paused here

    # --- Shutdown: runs once, when the server is stopping ---
    ml_models.clear()
    print("Cleaned up model resources. Goodbye!")


# ---------------------------------------------------------------------------
# 5. THE FASTAPI APP AND ITS ENDPOINTS
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Emotion Classifier API",
    description="A simple API that detects emotion (sadness, joy, love, anger, fear, surprise) in text using a BiGRU deep learning model.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the browser-based UI (or any other website) to call this API.
# In a real production app you would restrict allow_origins to your
# actual frontend's domain instead of "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the "static" folder (our HTML/CSS/JS UI) so it's reachable in a browser.
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
def serve_ui():
    """Show the interactive web UI when someone visits the homepage."""
    return FileResponse("static/index.html")


@app.get("/health", response_model=HealthResponse, tags=["Utility"])
def health_check():
    """
    Quick check to confirm the API is running and the model loaded fine.
    Handy for uptime monitors or for debugging deployment issues.
    """
    return HealthResponse(status="ok", model_loaded="model" in ml_models)


import re

def preprocess_text(text: str) -> str:
    """Clean and normalize input text to match the format used during model training."""
    text = text.lower()
    text = re.sub(r"'", "", text)  # remove apostrophes (e.g. can't -> cant)
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    return text


@app.post("/predict", response_model=PredictionResponse, tags=["Emotion Detection"])
def predict_emotion(payload: TextInput):
    """
    Analyze a piece of text and return the emotion the model detects.

    Steps performed under the hood:
        1. Clean and normalize input text (lowercase, remove punctuation).
        2. Turn text into a sequence of numbers (tokenize).
        3. Pad/truncate sequence to max length.
        4. Ask the model for a prediction.
        5. Package the result into a friendly response.
    """
    model = ml_models.get("model")
    tokenizer = ml_models.get("tokenizer")

    if model is None or tokenizer is None:
        # This should only happen if something went wrong on startup.
        raise HTTPException(status_code=503, detail="Model is not ready yet. Please try again shortly.")

    # Clean text to match tokenizer vocabulary format
    cleaned_text = preprocess_text(payload.text)

    # Step 1 & 2: text -> numbers -> fixed-length sequence
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(
        sequence, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post"
    )

    # Step 3: run the model. `verbose=0` just silences the progress bar.
    probabilities = model.predict(padded_sequence, verbose=0)[0]

    # Step 4: build a nice response
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
