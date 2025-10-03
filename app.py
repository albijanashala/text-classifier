from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import numpy as np

# === ripërkufizo funksionin clean_text (nga notebook) ===
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # lowercase
    text = text.lower()
    # hiq simbolet dhe pikësimin
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # mund të shtosh stopwords nëse ke përdorur
    return " ".join(text.split())

# ngarko modelin & vectorizer-in e ruajtur
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = FastAPI(title="Text Classifier API", version="1.0")

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    prediction: int
    proba: dict | None = None

@app.get("/")
def root():
    return {"message": "Text Classifier API is running. Go to /docs for Swagger UI."}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    text = (req.text or "").strip()
    if not text:
        return PredictResponse(prediction=-1, proba=None)

    # përdor clean_text si në train
    text = clean_text(text)

    # transformo tekstin
    X_vec = vectorizer.transform([text])
    pred = int(model.predict(X_vec)[0])

    # probabilitetet (nëse suportohen)
    proba = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_vec)[0]
        if hasattr(model, "classes_"):
            proba = {int(c): float(p) for c, p in zip(model.classes_, probs)}
        else:
            proba = {i: float(p) for i, p in enumerate(probs)}

    return PredictResponse(prediction=pred, proba=proba)
