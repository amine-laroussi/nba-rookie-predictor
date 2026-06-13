import logging
from typing import Optional

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

try:
    pipeline = joblib.load("model_pipeline.joblib")
except FileNotFoundError:
    raise RuntimeError("model_pipeline.joblib introuvable")

try:
    SEUIL = joblib.load("seuil_optimal.joblib")["seuil"]
except FileNotFoundError:
    SEUIL = 0.5

try:
    _wb = joblib.load("winsor_bounds.joblib")
    WINSOR_LOWER = _wb["lower"]
    WINSOR_UPPER = _wb["upper"]
except FileNotFoundError:
    WINSOR_LOWER = None
    WINSOR_UPPER = None

FEATURES_FINALES = [
    "GP", "MIN", "FGA", "FG%", "3PA", "FTA", "FT%",
    "REB", "AST", "STL", "BLK", "TOV",
    "efficiency", "court_impact", "shooting_vol", "true_shooting"
]


class PlayerStats(BaseModel):
    name:       Optional[str] = Field(None)
    GP:         float = Field(..., ge=1,  le=82,  description="Games Played")
    MIN:        float = Field(..., ge=0,  le=48,  description="Minutes par match")
    PTS:        float = Field(..., ge=0,  le=50,  description="Points par match")
    FGA:        float = Field(..., ge=0,          description="Field Goal Attempts")
    FG_pct:     float = Field(..., ge=0,  le=100, description="Field Goal %", alias="FG%")
    threePA:    float = Field(..., ge=0,          description="3-Point Attempts", alias="3PA")
    FTA:        float = Field(..., ge=0,          description="Free Throw Attempts")
    FT_pct:     float = Field(..., ge=0,  le=100, description="Free Throw %", alias="FT%")
    REB:        float = Field(..., ge=0,          description="Rebounds par match")
    AST:        float = Field(..., ge=0,          description="Assists par match")
    STL:        float = Field(..., ge=0,          description="Steals par match")
    BLK:        float = Field(..., ge=0,          description="Blocks par match")
    TOV:        float = Field(..., ge=0,          description="Turnovers par match")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "name": "LeBron James",
                "GP": 79, "MIN": 39.5, "PTS": 20.9,
                "FGA": 18.9, "FG%": 47.2,
                "3PA": 4.0, "FTA": 8.9, "FT%": 75.4,
                "REB": 6.9, "AST": 7.2, "STL": 1.7, "BLK": 0.7, "TOV": 3.5
            }
        }
    }


class PredictionResponse(BaseModel):
    name:               Optional[str]
    prediction:         int
    probabilite:        float
    confiance:          str
    conseil:            str
    interpretation:     str
    features_utilisees: dict


app = FastAPI(title="NBA Rookie Longevity Predictor", version="3.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess(player: PlayerStats) -> tuple:
    pts = float(np.clip(player.PTS, 0.0, 50.0))

    raw = {
        "GP": player.GP, "MIN": player.MIN,
        "FGA": player.FGA, "FG%": player.FG_pct,
        "3PA": player.threePA, "FTA": player.FTA, "FT%": player.FT_pct,
        "REB": player.REB, "AST": player.AST,
        "STL": player.STL, "BLK": player.BLK, "TOV": player.TOV,
    }

    raw["efficiency"]    = pts / (raw["MIN"] + 1e-5)
    raw["court_impact"]  = raw["REB"] + raw["AST"] + raw["STL"] + raw["BLK"]
    raw["shooting_vol"]  = raw["FGA"] + raw["FTA"]
    raw["true_shooting"] = pts / (2 * (raw["FGA"] + 0.44 * raw["FTA"]) + 1)

    if WINSOR_LOWER is not None:
        for feat in FEATURES_FINALES:
            if feat in WINSOR_LOWER:
                raw[feat] = float(np.clip(raw[feat], WINSOR_LOWER[feat], WINSOR_UPPER[feat]))

    vector = np.array([[raw[f] for f in FEATURES_FINALES]])
    features_log = {f: round(raw[f], 4) for f in FEATURES_FINALES}

    return vector, features_log


def get_interpretation(probabilite: float, seuil: float) -> str:
    high = seuil + (1 - seuil) * 0.6
    if probabilite >= high:
        return "Profil très prometteur — forte probabilité de longévité"
    elif probabilite >= seuil:
        return "Profil intéressant — au-dessus du seuil, à surveiller"
    elif probabilite >= seuil * 0.75:
        return "Profil limite — légèrement sous le seuil"
    else:
        return "Profil risqué — probabilité de longévité faible"


@app.get("/", tags=["Health"])
def root():
    return {
        "status": "running",
        "version": "3.4.0",
        "features_entree": ["GP", "MIN", "PTS", "FGA", "FG%", "3PA", "FTA", "FT%", "REB", "AST", "STL", "BLK", "TOV"],
        "features_modele": FEATURES_FINALES,
        "seuil_decision": SEUIL,
        "winsorization": "active" if WINSOR_LOWER is not None else "inactive",
    }


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "pipeline_loaded": pipeline is not None,
        "seuil": SEUIL,
        "winsor_loaded": WINSOR_LOWER is not None,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(player: PlayerStats):
    try:
        X_input, features_log = preprocess(player)
        probabilite = float(round(pipeline.predict_proba(X_input)[0][1], 4))
        decision    = int(probabilite >= SEUIL)

        return PredictionResponse(
            name               = player.name,
            prediction         = decision,
            probabilite        = probabilite,
            confiance          = f"{probabilite * 100:.1f}%",
            conseil            = "Investir " if decision else "Ne pas investir ",
            interpretation     = get_interpretation(probabilite, SEUIL),
            features_utilisees = features_log,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
