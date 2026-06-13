# NBA Rookie Longevity Predictor

Prédit si un rookie NBA durera **plus de 5 ans** en ligue à partir de ses statistiques de première saison — exposé via une API REST FastAPI et déployable en un seul script.

---

## Table des matières

- [🎬 Démo](#-démo)
- [Aperçu du projet](#aperçu-du-projet)
- [Structure du projet](#structure-du-projet)
- [Installation](#installation)
- [Lancer le projet](#lancer-le-projet)
- [API — Endpoints](#api--endpoints)
- [Exemple de requête](#exemple-de-requête)
- [Features](#features)
- [Pipeline de modélisation](#pipeline-de-modélisation)
- [Résultats](#résultats)
- [Fichiers de modèle](#fichiers-de-modèle)

---

## 🎬 Démo

https://github.com/user-attachments/assets/68c63445-2376-473c-b8a6-3e9de0fb9330

---

## Aperçu du projet

À partir des statistiques d'un rookie sur sa première saison NBA, le modèle prédit si ce joueur restera en ligue plus de 5 ans (`TARGET_5Yrs = 1`) ou non (`TARGET_5Yrs = 0`).

Le projet couvre l'intégralité du cycle de vie ML :

- Exploration et nettoyage des données (EDA, doublons, outliers)
- Feature engineering métier (4 nouvelles features)
- Comparaison de modèles (Logistic Regression, Random Forest, XGBoost)
- Optimisation bayésienne des hyperparamètres via Optuna (100 trials)
- Déploiement en API REST FastAPI avec Docker

---

## Structure du projet

```
├── main.py                  # API FastAPI (prétraitement + prédiction)
├── version_final.ipynb      # Notebook complet d'entraînement
├── model_pipeline.joblib    # Pipeline sklearn (MinMaxScaler + RandomForest)
├── seuil_optimal.joblib     # Seuil de décision (défaut : 0.50)
├── winsor_bounds.joblib     # Bornes de winsorisation (calculées sur X_train)
├── Dockerfile               # Image Docker (python:3.11-slim)
├── docker-compose.yml       # Orchestration du container (port 8000)
├── Script.sh                # Script de démarrage automatique
└── requirements.txt         # Dépendances Python
```

---

## Installation

**Prérequis :** Docker Desktop (recommandé) ou Python 3.11+

### Sans Docker

```bash
# Créer et activer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# Installer les dépendances
pip install -r requirements.txt
```

**Dépendances principales :**

```
# API
fastapi==0.115.0
uvicorn==0.30.6
pydantic==2.8.2

# ML
scikit-learn==1.7.2
numpy==1.26.4
joblib==1.4.2

# Notebook / exploration
pandas==2.2.2
matplotlib==3.9.0
seaborn==0.13.2
xgboost==2.1.1
optuna==3.6.1
```

---

## Lancer le projet

### Option 1 — Script automatique (recommandé)

Vérifie que Docker est installé et démarré, puis lance l'API en arrière-plan.

```bash
chmod +x Script.sh
./Script.sh
```

### Option 2 — Docker Compose

```bash
# Construire l'image et démarrer le container
docker-compose up --build -d

# Vérifier que le container tourne
docker ps

# Arrêter le container
docker-compose down
```

Les fichiers `.joblib` sont montés en volume : vous pouvez remplacer les modèles **sans rebuilder l'image**.

### Option 3 — Lancement manuel

```bash
uvicorn main:app --reload
```

---

## API — Endpoints

| Méthode | URL | Description |
|---|---|---|
| GET | `http://localhost:8000/` | Statut, configuration et seuil actif |
| GET | `http://localhost:8000/health` | Vérification du pipeline chargé |
| GET | `http://localhost:8000/docs` | Documentation Swagger interactive |
| POST | `http://localhost:8000/predict` | Prédiction pour un joueur |

---

## Exemple de requête

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "LeBron James",
    "GP": 79, "MIN": 39.5, "PTS": 20.9,
    "FGA": 18.9, "FG%": 47.2,
    "3PA": 4.0, "FTA": 8.9, "FT%": 75.4,
    "REB": 6.9, "AST": 7.2, "STL": 1.7, "BLK": 0.7, "TOV": 3.5
  }'
```

Réponse :

```json
{
  "name": "LeBron James",
  "prediction": 1,
  "probabilite": 0.87,
  "confiance": "87.0%",
  "conseil": "Investir",
  "interpretation": "Profil très prometteur — forte probabilité de longévité",
  "features_utilisees": { "GP": 79.0, "MIN": 39.5, "efficiency": 0.529, ... }
}
```

---

## Features

### Features en entrée (13)

| Feature | Description | Contraintes |
|---|---|---|
| GP | Matchs joués | 1 – 82 |
| MIN | Minutes par match | 0 – 48 |
| PTS | Points par match | 0 – 50 |
| FGA | Tentatives de tir | ≥ 0 |
| FG% | Pourcentage au tir | 0 – 100 |
| 3PA | Tentatives à 3 points | ≥ 0 |
| FTA | Tentatives de lancer franc | ≥ 0 |
| FT% | Pourcentage aux lancers | 0 – 100 |
| REB | Rebonds par match | ≥ 0 |
| AST | Passes décisives par match | ≥ 0 |
| STL | Interceptions par match | ≥ 0 |
| BLK | Contres par match | ≥ 0 |
| TOV | Pertes de balle par match | ≥ 0 |

### Features engineerées (4, calculées automatiquement)

| Feature | Formule | Interprétation |
|---|---|---|
| efficiency | PTS / (MIN + 1e-5) | Points par minute (indépendant du volume) |
| court_impact | REB + AST + STL + BLK | Contribution globale hors scoring |
| shooting_vol | FGA + FTA | Volume offensif total |
| true_shooting | PTS / (2 × (FGA + 0.44 × FTA) + 1) | Efficacité réelle au tir (métrique NBA officielle) |

> `PTS` est utilisé pour calculer les features engineerées mais n'est pas passé directement au modèle.

---

## Pipeline de modélisation

### 1. EDA

- Histogrammes comparatifs par classe (joueurs durables vs partants)
- Matrice de corrélation — identification des features redondantes
- PCA 2D — visualisation de la séparabilité des classes (chevauchement → justifie un modèle non-linéaire)

### 2. Nettoyage

- Suppression des lignes contradictoires (mêmes stats, target différente) : 1340 → 1273 lignes
- Doublons de noms : conservation de la saison avec le plus de `GP`

### 3. Feature Engineering

4 features métier ajoutées (voir tableau ci-dessus).

### 4. Feature Selection

Suppression des features redondantes (corrélation inter-features > 0.95) :

- `PTS`, `FGM` → redondantes avec `FGA`
- `FTM` → redondante avec `FTA`
- `OREB`, `DREB` → redondantes avec `REB`
- `3P Made` → redondante avec `3PA`
- `3P%` → corrélation quasi-nulle avec la cible + contient des NaN

**16 features conservées** pour le modèle.

### 5. Traitement des outliers

Winsorisation IQR ×1.5 calculée **uniquement sur X_train** pour éviter tout data leakage. Les bornes sont sauvegardées dans `winsor_bounds.joblib` et appliquées à l'inférence.

### 6. Modélisation

- Split train/test 80/20 stratifié (hold-out)
- Comparaison en CV 5-fold : Logistic Regression, Random Forest, XGBoost
- **Random Forest** retenu (meilleur Recall en CV)
- Optimisation bayésienne via **Optuna** (100 trials, objectif : maximiser le Recall)
- Pipeline sklearn `MinMaxScaler + RandomForestClassifier` pour éviter tout data leakage

### 7. Seuil de décision

Score métier = **0.7 × Recall + 0.3 × Precision**

> Rater un talent (faux négatif) coûte plus cher qu'un mauvais investissement (faux positif).

Seuil retenu : **0.50** — validé empiriquement sur prédictions out-of-fold (CV train). `class_weight='balanced'` calibre déjà bien les probabilités, aucun ajustement supplémentaire nécessaire.

---

## Résultats

Évaluation finale sur le **test set hold-out** (20% des données, jamais utilisé pendant l'entraînement) :

| Métrique | Valeur | Interprétation |
|---|---|---|
| Recall | **0.748** | 74.8% des vrais talents détectés |
| Spécificité | **0.521** | 52.1% des flops évités |
| AUC-ROC | **0.734** | Bonne capacité de discrimination globale |

### Recommandations métier

Un rookie mérite attention si :
- `GP > 60` → l'équipe lui fait confiance sur la durée de la saison
- `MIN > 18` → il occupe un rôle significatif dans la rotation
- `court_impact > 5` → il contribue au-delà du scoring (rebonds, passes, défense)

---

## Fichiers de modèle

| Fichier | Contenu |
|---|---|
| `model_pipeline.joblib` | Pipeline sklearn complet (MinMaxScaler + RandomForest optimisé) |
| `seuil_optimal.joblib` | Dictionnaire `{"seuil": 0.5}` — seuil de décision |
| `winsor_bounds.joblib` | Dictionnaire `{"lower": {...}, "upper": {...}}` — bornes de winsorisation par feature |

Ces fichiers sont montés en volume Docker et peuvent être mis à jour sans rebuilder l'image.
