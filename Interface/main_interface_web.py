"""
FastAPI + ASGI (Uvicorn)

Sert à la fois l'interface utilisateur (interface_web.html) et une API REST

Installation :
    pip install fastapi uvicorn opencv-python numpy mediapipe torch
Lancement :
    uvicorn main_interface_web:app --reload --host 0.0.0.0 --port 8000

Puis ouvrir  →  http://localhost:8000
Swagger UI   →  http://localhost:8000/docs
"""

import base64
import time
import urllib.request
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


# Modèle LSTM
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


# Chargement du modèle LSTM
MODEL_PATH = Path(__file__).parent / "corpus_augmente_model.pt"
MODEL_URL  = ("https://raw.githubusercontent.com/KerenDague/Detection_et_analyse_de_dactylologie_francaise/main/Interface/corpus_augmente_model.pt")

model  = None
X_mean = None
X_std  = None


def load_model():
    """Charge le modèle depuis le disque local ou le télécharge depuis GitHub si absent."""
    global model, X_mean, X_std

    if MODEL_PATH.exists():
        print("Chargement du modèle local…")
    else:
        print("Modèle local introuvable, téléchargement depuis GitHub…")
        try:
            urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
            print("Trouvé sur le git")
        except Exception as e:
            print(f" Impossible de télécharger le modèle : {e}")
            print("L'API démarrera sans modèle (les prédictions retourneront '?').")
            return

    try:
        checkpoint = torch.load(str(MODEL_PATH), map_location="cpu")
        model = LSFTranslator()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        X_mean = checkpoint["X_mean"]
        X_std  = checkpoint["X_std"]
        print("Modèle chargé avec succès.")
    except Exception as e:
        print(f"Chargement du modèle échoué : {e}")
        model = None


load_model()


# Chargement du HandLandmarker
LANDMARKER_PATH = Path(__file__).parent / "hand_landmarker.task"
LANDMARKER_URL  = ("https://raw.githubusercontent.com/KerenDague/Detection_et_analyse_de_dactylologie_francaise/main/RNN/hand_landmarker.task")

landmarker = None


def load_landmarker():
    """Charge le HandLandmarker en local ; le télécharge depuis GitHub si absent."""
    global landmarker

    if LANDMARKER_PATH.exists():
        print("Chargement du hand_landmarker local…")
    else:
        print("hand_landmarker introuvable en local, téléchargement depuis GitHub…")
        try:
            urllib.request.urlretrieve(LANDMARKER_URL, str(LANDMARKER_PATH))
            print("Trouvé sur le git")
        except Exception as e:
            print(f"Impossible de télécharger hand_landmarker.task : {e}")
            landmarker = None
            return

    try:
        base_options = python.BaseOptions(model_asset_path=str(LANDMARKER_PATH))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        landmarker = vision.HandLandmarker.create_from_options(options)
        print("HandLandmarker chargé avec succès.")
    except Exception as e:
        print(f" Chargement du HandLandmarker échoué : {e}")
        landmarker = None


load_landmarker()


# ─── Données lettres ─────────────────────────────────────────────────────────────
LETTERS_DB = {
    "A": {"type": "voyelle",  "description": "Poing fermé, pouce sur le côté.",                          "difficulté": "facile"},
    "B": {"type": "consonne", "description": "Main ouverte, doigts joints vers le haut, pouce replié.",  "difficulté": "facile"},
    "C": {"type": "consonne", "description": "Main en forme de C, doigts arrondis.",                     "difficulté": "facile"},
    "D": {"type": "consonne", "description": "Index pointé, autres doigts et pouce formant un cercle.",  "difficulté": "moyen"},
    "E": {"type": "voyelle",  "description": "Doigts repliés sur la paume, pouce sous les doigts.",      "difficulté": "moyen"},
    "F": {"type": "consonne", "description": "Pouce et index formant un cercle, autres doigts écartés.", "difficulté": "moyen"},
    "G": {"type": "consonne", "description": "Index et pouce pointés horizontalement.",                  "difficulté": "moyen"},
    "H": {"type": "consonne", "description": "Index et majeur pointés horizontalement.",                 "difficulté": "moyen"},
    "I": {"type": "voyelle",  "description": "Auriculaire levé, autres doigts repliés.",                 "difficulté": "facile"},
    "J": {"type": "consonne", "description": "Auriculaire levé avec mouvement en J.",                    "difficulté": "difficile"},
    "K": {"type": "consonne", "description": "Index et majeur en V, pouce entre les deux.",              "difficulté": "difficile"},
    "L": {"type": "consonne", "description": "Index levé, pouce écarté formant un L.",                   "difficulté": "facile"},
    "M": {"type": "consonne", "description": "Trois doigts repliés sur le pouce.",                       "difficulté": "moyen"},
    "N": {"type": "consonne", "description": "Deux doigts repliés sur le pouce.",                        "difficulté": "moyen"},
    "O": {"type": "voyelle",  "description": "Tous les doigts forment un cercle avec le pouce.",         "difficulté": "facile"},
    "P": {"type": "consonne", "description": "Comme K mais orienté vers le bas.",                        "difficulté": "difficile"},
    "Q": {"type": "consonne", "description": "Comme G mais vers le bas.",                                "difficulté": "difficile"},
    "R": {"type": "consonne", "description": "Index et majeur croisés.",                                 "difficulté": "moyen"},
    "S": {"type": "consonne", "description": "Poing fermé, pouce sur les doigts.",                       "difficulté": "facile"},
    "T": {"type": "consonne", "description": "Pouce entre index et majeur repliés.",                     "difficulté": "moyen"},
    "U": {"type": "voyelle",  "description": "Index et majeur joints et levés.",                         "difficulté": "facile"},
    "V": {"type": "consonne", "description": "Index et majeur en V.",                                    "difficulté": "facile"},
    "W": {"type": "consonne", "description": "Index, majeur et annulaire en W.",                         "difficulté": "moyen"},
    "X": {"type": "consonne", "description": "Index recourbé en crochet.",                               "difficulté": "moyen"},
    "Y": {"type": "voyelle",  "description": "Pouce et auriculaire écartés.",                            "difficulté": "facile"},
    "Z": {"type": "consonne", "description": "Index traçant un Z dans l'air.",                           "difficulté": "difficile"},
}

session_stats = {
    "total_predictions":     0,
    "predictions_by_letter": defaultdict(int),
    "high_confidence_count": 0,
    "session_start":         time.time(),
}


# ─── Schémas Pydantic ────────────────────────────────────────────────────────────
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
    difficulté: str

class ApiStatus(BaseModel):
    status: str
    version: str
    letters_supported: int
    uptime_seconds: float

class PreviewResult(BaseModel):
    image: str
    hand_detected: bool


# ─── Utilitaires ─────────────────────────────────────────────────────────────────
def decode_image(b64: str) -> np.ndarray:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    try:
        arr   = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Décodage impossible.")
        return frame
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image invalide : {e}")


def run_inference(frame: np.ndarray) -> dict:
    """Extrait les landmarks via HandLandmarker et prédit la lettre avec le modèle LSTM."""
    if model is not None and landmarker is not None:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result    = landmarker.detect(mp_image)

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            landmarks = []
            for lm in hand_landmarks:
                landmarks.extend([lm.x, lm.y, lm.z])

            landmarks_array = np.array(landmarks, dtype=np.float32)
            sequence        = np.tile(landmarks_array, (MAX_FRAMES, 1))  # (150, 63)

            x = torch.FloatTensor(sequence).unsqueeze(0)                 # (1, 150, 63)
            x = (x - X_mean) / X_std

            with torch.no_grad():
                logits = model(x)
                probs  = torch.softmax(logits, dim=1)[0]

            top3_idx = probs.topk(3).indices.tolist()
            return {
                "letter":     LETTERS[top3_idx[0]],
                "confidence": round(probs[top3_idx[0]].item(), 4),
                "top3": [
                    {"letter": LETTERS[i], "confidence": round(probs[i].item(), 4)}
                    for i in top3_idx
                ],
            }

    return {
        "letter":     "?",
        "confidence": 0.0,
        "top3":       [{"letter": "?", "confidence": 0.0}] * 3,
    }


# ─── Routes ──────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_root():
    """Sert interface_web.html (doit être dans le même dossier)."""
    html_path = Path(__file__).parent / "interface_web.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="interface_web.html introuvable.")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def serve_ui():
    return serve_root()


@app.get("/status", response_model=ApiStatus, summary="Statut de l'API")
def status():
    return ApiStatus(
        status="ok",
        version="1.0.0",
        letters_supported=len(LETTERS_DB),
        uptime_seconds=round(time.time() - session_stats["session_start"], 1),
    )


@app.get("/letters/", summary="Liste des lettres reconnues")
def get_letters(type: Optional[str] = None):
    """Retourne toutes les lettres avec leurs métadonnées et l'URL du GIF."""
    result = []
    for letter, info in LETTERS_DB.items():
        if type and info["type"] != type:
            continue
        result.append({
            "letter": letter,
            **info,
            "gif_url": (
                "https://raw.githubusercontent.com/KerenDague/"
                "Detection_et_analyse_de_dactylologie_francaise/main/Interface/GIF/"
                f"{letter.lower()}.gif"
            ),
        })
    return {"count": len(result), "letters": result}


@app.get("/letters/{letter}", response_model=LetterInfo, summary="Détails d'une lettre")
def get_letter(letter: str):
    letter = letter.upper()
    if letter not in LETTERS_DB:
        raise HTTPException(status_code=404, detail=f"Lettre '{letter}' non trouvée.")
    return LetterInfo(letter=letter, **LETTERS_DB[letter])


@app.post("/predict", response_model=PredictionResult, summary="Prédire une lettre")
def predict(payload: ImagePayload):
    """Reçoit une image base64, retourne la lettre prédite + confiance + top-3."""
    t0      = time.perf_counter()
    result  = run_inference(decode_image(payload.image))
    elapsed = round((time.perf_counter() - t0) * 1000, 2)

    session_stats["total_predictions"] += 1
    session_stats["predictions_by_letter"][result["letter"]] += 1
    if result["confidence"] >= 0.8:
        session_stats["high_confidence_count"] += 1

    return PredictionResult(processing_time_ms=elapsed, **result)


@app.post("/preview", response_model=PreviewResult, summary="Frame annotée avec landmarks")
def preview(payload: ImagePayload):
    print(">>> /preview appelé")
    frame     = decode_image(payload.image)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result    = landmarker.detect(mp_image)
    hand_detected = bool(result.hand_landmarks)
    print(f">>> Main détectée : {hand_detected}")

    if hand_detected:
        h, w = frame.shape[:2]
        for hand_landmarks in result.hand_landmarks:
            # Points
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
            # Connexions définies manuellement (sans mp.solutions)
            connections = [
                (0,1),(1,2),(2,3),(3,4),           # pouce
                (0,5),(5,6),(6,7),(7,8),            # index
                (5,9),(9,10),(10,11),(11,12),        # majeur
                (9,13),(13,14),(14,15),(15,16),      # annulaire
                (13,17),(17,18),(18,19),(19,20),     # auriculaire
                (0,17),                              # paume
            ]
            for start_idx, end_idx in connections:
                s = hand_landmarks[start_idx]
                e = hand_landmarks[end_idx]
                cv2.line(
                    frame,
                    (int(s.x * w), int(s.y * h)),
                    (int(e.x * w), int(e.y * h)),
                    (139, 34, 64), 2,
                )

    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    b64 = base64.b64encode(buffer).decode('utf-8')
    return PreviewResult(image=f"data:image/jpeg;base64,{b64}", hand_detected=hand_detected)


@app.get("/stats", summary="Statistiques de session")
def get_stats():
    total  = session_stats["total_predictions"]
    uptime = time.time() - session_stats["session_start"]
    return {
        "total_predictions":      total,
        "high_confidence_rate":   round(session_stats["high_confidence_count"] / total, 3) if total else 0,
        "predictions_per_minute": round(total / (uptime / 60), 2) if uptime > 0 else 0,
        "uptime_seconds":         round(uptime, 1),
        "most_predicted":         sorted(
            session_stats["predictions_by_letter"].items(), key=lambda x: -x[1]
        )[:5],
    }


@app.delete("/stats", summary="Réinitialise les statistiques")
def reset_stats():
    session_stats["total_predictions"]     = 0
    session_stats["predictions_by_letter"] = defaultdict(int)
    session_stats["high_confidence_count"] = 0
    session_stats["session_start"]         = time.time()
    return {"message": "Statistiques réinitialisées."}
