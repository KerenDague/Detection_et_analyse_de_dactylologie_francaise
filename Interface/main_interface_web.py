"""
FastAPI + ASGI (Uvicorn)

Sert à la fois l'interface utilisateur (interface_web.html) et une API REST

Installation :
    pip install fastapi uvicorn opencv-python numpy
Lancement :
    uvicorn main_interface_web:app --reload --host 0.0.0.0 --port 8000

Puis ouvrir  →  http://localhost:8000
Swagger UI   →  http://localhost:8000/docs


"""

import base64
import time
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


# Application
app = FastAPI(
    title="API Dactylologie française",
    description="Interface REST pour la détection de dactylologie française.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_ASSET_PATH = 'hand_landmarker.task'
base_options = python.BaseOptions(model_asset_path=MODEL_ASSET_PATH)
options_image = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
)
landmarker = vision.HandLandmarker.create_from_options(options_image)

# Chargement du modèle

LETTERS = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

MAX_FRAMES = 150
INPUT_SIZE = 63

class LSFTranslator(nn.Module):
    def __init__(self, input_size=63, hidden_size=256, num_classes=26):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            batch_first=True, num_layers=2,
                            dropout=0.3, bidirectional=True)
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, num_classes),
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.classifier(self.dropout(out.mean(dim=1)))


app = FastAPI(title="API LSF - HandLandmarker")

# Charge le modele sauvegardé
MODEL_PATH = Path(__file__).parent / "lsf_model.pt"
model = None
X_mean = None
X_std = None

if MODEL_PATH.exists():
    checkpoint = torch.load(str(MODEL_PATH), map_location="cpu")
    model= LSFTranslator()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    X_mean = checkpoint["X_mean"]
    X_std = checkpoint["X_std"]
    print("Modèle chargé.")
else:
    print("lsf_model.pt introuvable")

# Données
LETTERS_DB = {
    "A": {"type": "voyelle",  "description": "Poing fermé, pouce sur le côté.", "difficulté": "facile"},
    "B": {"type": "consonne", "description": "Main ouverte, doigts joints vers le haut, pouce replié.","difficulté": "facile"},
    "C": {"type": "consonne", "description": "Main en forme de C, doigts arrondis.","difficulté": "facile"},
    "D": {"type": "consonne", "description": "Index pointé, autres doigts et pouce formant un cercle.", "difficulté": "moyen"},
    "E": {"type": "voyelle",  "description": "Doigts repliés sur la paume, pouce sous les doigts.","difficulté": "moyen"},
    "F": {"type": "consonne", "description": "Pouce et index formant un cercle, autres doigts écartés.","difficulté": "moyen"},
    "G": {"type": "consonne", "description": "Index et pouce pointés horizontalement.", "difficulté": "moyen"},
    "H": {"type": "consonne", "description": "Index et majeur pointés horizontalement.","difficulté": "moyen"},
    "I": {"type": "voyelle",  "description": "Auriculaire levé, autres doigts repliés.", "difficulté": "facile"},
    "J": {"type": "consonne", "description": "Auriculaire levé avec mouvement en J.","difficulté": "difficile"},
    "K": {"type": "consonne", "description": "Index et majeur en V, pouce entre les deux.","difficulté": "difficile"},
    "L": {"type": "consonne", "description": "Index levé, pouce écarté formant un L.","difficulté": "facile"},
    "M": {"type": "consonne", "description": "Trois doigts repliés sur le pouce.", "difficulté": "moyen"},
    "N": {"type": "consonne", "description": "Deux doigts repliés sur le pouce.", "difficulté": "moyen"},
    "O": {"type": "voyelle",  "description": "Tous les doigts forment un cercle avec le pouce.", "difficulté": "facile"},
    "P": {"type": "consonne", "description": "Comme K mais orienté vers le bas.", "difficulté": "difficile"},
    "Q": {"type": "consonne", "description": "Comme G mais vers le bas.", "difficulté": "difficile"},
    "R": {"type": "consonne", "description": "Index et majeur croisés.",  "difficulté": "moyen"},
    "S": {"type": "consonne", "description": "Poing fermé, pouce sur les doigts.","difficulté": "facile"},
    "T": {"type": "consonne", "description": "Pouce entre index et majeur repliés.","difficulté": "moyen"},
    "U": {"type": "voyelle",  "description": "Index et majeur joints et levés.","difficulté": "facile"},
    "V": {"type": "consonne", "description": "Index et majeur en V.","difficulté": "facile"},
    "W": {"type": "consonne", "description": "Index, majeur et annulaire en W.","difficulté": "moyen"},
    "X": {"type": "consonne", "description": "Index recourbé en crochet.", "difficulté": "moyen"},
    "Y": {"type": "voyelle", "description": "Pouce et auriculaire écartés.","difficulté": "facile"},
    "Z": {"type": "consonne", "description": "Index traçant un Z dans l'air.","difficulté": "difficile"},
}

session_stats = {
    "total_predictions": 0,
    "predictions_by_letter": defaultdict(int),
    "high_confidence_count": 0,
    "session_start": time.time(),
}


# Schémas Pydantic
class ImagePayload(BaseModel):
    image: str
    format: str = "jpeg"

class PredictionResult(BaseModel):
    letter: str
    confidence: float
    top3: list[dict]
    processing_time_ms: float

class LetterInfo(BaseModel):
    letter: str
    type: str
    description: str
    difficulty: str

class ApiStatus(BaseModel):
    status: str
    version: str
    letters_supported: int
    uptime_seconds: float

class PreviewResult(BaseModel):
    image: str
    hand_detected: bool


# Inférence
def decode_image(b64: str) -> np.ndarray:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    try:
        arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Décodage impossible.")
        return frame
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image invalide : {e}")


def run_inference(frame: np.ndarray) -> dict:
    """
    Extrait les landmarks MediaPipe de la frame et prédit la lettre
    avec le modèle LSTM.
    """

    # Extraction des landmarks MediaPipe

    if model is not None:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = landmarker.detect(mp_image)

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            landmarks = []
            for lm in hand_landmarks:
                landmarks.extend([lm.x, lm.y, lm.z])

            landmarks_array = np.array(landmarks, dtype=np.float32)

            # Accumulation de frames dans un buffer par requete
            sequence = np.tile(landmarks_array, (MAX_FRAMES, 1))  # (150, 63)

            # Normalisation
            x = torch.FloatTensor(sequence).unsqueeze(0)    # (1, 150, 63)
            x = (x - X_mean) / X_std

            # Prédiction
            with torch.no_grad():
                logits = model(x)
                probs  = torch.softmax(logits, dim=1)[0]

            top3_idx = probs.topk(3).indices.tolist()
            return {
    "letter": LETTERS[top3_idx[0]],
    "confidence": round(probs[top3_idx[0]].item(), 4),
    "top3": [{"letter": LETTERS[i], "confidence": round(probs[i].item(), 4)} for i in top3_idx],
}

    # Aucune main détectée ou modèle absent
    return {
        "letter": "?",
        "confidence": 0.0,
        "top3": [{"letter": "?", "confidence": 0.0}] * 3,
    }

# Routes
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_root():
    """Lance interface_web.html (doit etre dans le meme dossier)."""
    html_path = Path(__file__).parent / "interface_web.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="interface_web.html introuvable.")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def serve_ui():
    return serve_root()


@app.get("/status", response_model=ApiStatus, summary="Statut de l'API")
def status():
    """Vérifie que l'API est disponible."""
    return ApiStatus(
        status="ok", version="1.0.0",
        letters_supported=len(LETTERS_DB),
        uptime_seconds=round(time.time() - session_stats["session_start"], 1),
    )


@app.get("/letters/", summary="Liste des lettres reconnues")
def get_letters(type: Optional[str] = None):
    """
    Retourne toutes les lettres avec leurs métadonnées.
    Chaque lettre inclut gif_url indiquant l'URL du GIF correspondant.
    """
    result = []
    for letter, info in LETTERS_DB.items():
        if type and info["type"] != type:
            continue
        result.append({
            "letter": letter,
            **info,
            "gif_url": f"https://raw.githubusercontent.com/KerenDague/Detection_et_analyse_de_dactylologie_francaise/main/Interface/GIF/{letter.lower()}.gif",
        })
    return {"count": len(result), "letters": result}


@app.get("/letters/{letter}", response_model=LetterInfo, summary="Détails d'une lettre")
def get_letter(letter: str):
    """Retourne les information détaillées pour une lettre donnée."""
    letter = letter.upper()
    if letter not in LETTERS_DB:
        raise HTTPException(status_code=404, detail=f"Lettre '{letter}' non trouvée.")
    return LetterInfo(letter=letter, **LETTERS_DB[letter])


@app.post("/predict", response_model=PredictionResult, summary="Prédire une lettre")
def predict(payload: ImagePayload):
    """Reçoit une image base64, retourne la lettre prédite + confiance + top-3."""
    t0 = time.perf_counter()
    result = run_inference(decode_image(payload.image))
    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    session_stats["total_predictions"] += 1
    session_stats["predictions_by_letter"][result["letter"]] += 1
    if result["confidence"] >= 0.8:
        session_stats["high_confidence_count"] += 1
    return PredictionResult(processing_time_ms=elapsed, **result)

@app.post("/preview")
def preview(payload: ImagePayload):
    frame = decode_image(payload.image)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    result = landmarker.detect(mp_image)
    hand_detected = False

    if result.hand_landmarks:
        hand_detected = True
        h, w, _ = frame.shape
        # Dessin manuel (plus flexible avec la nouvelle API)
        for hand_landmarks in result.hand_landmarks:
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

    _, buffer = cv2.imencode('.jpg', frame)
    b64 = base64.b64encode(buffer).decode('utf-8')
    return {"image": f"data:image/jpeg;base64,{b64}", "hand_detected": hand_detected}

@app.get("/stats", summary="Statistiques de session")
def get_stats():
    """Retourne les statistiques d'utilisation."""
    total = session_stats["total_predictions"]
    uptime = time.time() - session_stats["session_start"]
    return {
        "total_predictions": total,
        "high_confidence_rate": round(session_stats["high_confidence_count"] / total, 3) if total else 0,
        "predictions_per_minute": round(total / (uptime / 60), 2) if uptime > 0 else 0,
        "uptime_seconds": round(uptime, 1),
        "most_predicted": sorted(session_stats["predictions_by_letter"].items(), key=lambda x: -x[1])[:5],
    }


@app.delete("/stats", summary="Réinitialise les statistiques")
def reset_stats():
    """Remet à zéro les statistiques de session."""
    session_stats["total_predictions"] = 0
    session_stats["predictions_by_letter"] = defaultdict(int)
    session_stats["high_confidence_count"] = 0
    session_stats["session_start"] = time.time()
    return {"message": "Statistiques réinitialisées."}
