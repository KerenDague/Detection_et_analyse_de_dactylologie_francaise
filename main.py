"""
FastAPI + ASGI (Uvicorn)

Sert à la fois l'interface utilisateur (index.html) et une API REST

Installation : 
    pip install fastapi uvicorn opencv-python numpy
Lancement :
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

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
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
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

# Dossier gifs
GIFS_DIR = Path(__file__).parent / "gifs"
GIFS_DIR.mkdir(exist_ok=True)
app.mount("/gifs", StaticFiles(directory=str(GIFS_DIR)), name="gifs")


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


# Inférence
def decode_image(b64: str) -> np.ndarray:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    try:
        arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Décodage impossible.")
        return frame
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image invalide : {e}")


def run_inference(frame: np.ndarray) -> dict:
    """
    Metter notre modèle ici
    Retourne : {"letter": str, "confidence": float, "top3": list[dict]}
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]
    mask = cv2.inRange(hsv, np.array([0, 20, 70], np.uint8), np.array([20, 255, 255], np.uint8))
    skin_ratio = mask.sum() / (255 * h * w)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = max((cv2.contourArea(c) for c in contours), default=0) / (h * w)

    rng = random.Random(int(skin_ratio * 1000 + area * 500) % (2 ** 31))
    letters = list(LETTERS_DB.keys())
    rng.shuffle(letters)
    base = rng.uniform(0.5, 0.95)
    scores = {
        letters[0]: round(base, 3),
        letters[1]: round(base * rng.uniform(0.3, 0.7), 3),
        letters[2]: round(base * rng.uniform(0.05, 0.3), 3),
    }
    for l in letters[3:]:
        scores[l] = round(rng.uniform(0.001, 0.05), 3)
    total = sum(scores.values())
    scores = {k: round(v / total, 4) for k, v in scores.items()}
    s = sorted(scores.items(), key=lambda x: -x[1])
    return {
        "letter": s[0][0],
        "confidence": s[0][1],
        "top3": [{"letter": l, "confidence": c} for l, c in s[:3]],
    }


# Routes
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_root():
    """Lance index.html (doit etre dans le meme dossier."""
    html_path = Path(__file__).parent / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html introuvable.")
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


@app.get("/lettders", summary="Liste des lettres reconnues")
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
            "gif_url": f"https://raw.githubusercontent.com/KerenDague/Detection_et_analyse_de_dactylologie_francaise/main/GIF/{letter}.gif",
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
