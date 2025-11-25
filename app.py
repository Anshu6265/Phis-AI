from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import uvicorn

# -----------------------------
# Load existing model & vectors
# -----------------------------
with open("vector.pkl", "rb") as f:
    VECTOR = pickle.load(f)

with open("model.pkl", "rb") as f:
    MODEL = pickle.load(f)

# -----------------------------
# FastAPI setup
# -----------------------------
app = FastAPI(title="Spam/Phishing Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend origin in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Request / Response models
# -----------------------------
class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    confidence: float

# -----------------------------
# Prediction logic
# -----------------------------
def classify_text(text: str):
    vect = VECTOR.transform([text])
    proba = MODEL.predict_proba(vect)[0]  # [ham_prob, spam_prob]
    spam_prob = float(proba[1])
    label = "spam" if spam_prob >= 0.5 else "ham"
    return label, spam_prob

# -----------------------------
# API endpoint
# -----------------------------
@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    try:
        label, score = classify_text(req.text)
        return PredictResponse(label=label, confidence=score)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"status": "running", "model": "loaded"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
