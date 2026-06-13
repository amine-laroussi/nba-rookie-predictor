# NBA Rookie Longevity Predictor

Prédit si un rookie NBA durera plus de 5 ans en ligue à partir
de ses statistiques de première saison.

---

## Structure du projet

    ├── main.py                  # API FastAPI
    ├── Dockerfile               # Image Docker
    ├── docker-compose.yml       # Orchestration du container
    ├── script.sh                # Script de démarrage rapide
    ├── requirements.txt         # Dépendances Python
    ├── model_pipeline.joblib    # Pipeline entraîné (MinMaxScaler + RandomForest)
    ├── seuil_optimal.joblib     # Seuil de décision optimal
    ├── winsor_bounds.joblib     # Bornes de winsorisation calculées sur X_train
    ├── nba_logreg.csv           # Dataset source
    └── version_final.ipynb      # Notebook d'entraînement

---

## Lancer le projet

### Option 1 - Script automatique (recommandé)

La façon la plus simple de démarrer le projet.

    chmod +x script.sh
    ./script.sh

### Option 2 - Docker Compose

    # Construire l'image et démarrer le container
    docker-compose up --build -d

    # Vérifier que le container tourne
    docker ps

    # Arrêter le container
    docker-compose down

### Option 3 - Installation manuelle sans Docker

    # Créer et activer un environnement virtuel
    python -m venv venv
    source venv/bin/activate        # Linux / macOS
    venv\Scripts\activate           # Windows

    # Installer les dépendances
    pip install -r requirements.txt

    # Lancer l'API
    uvicorn main:app --reload

---

## Accès à l'API

| URL | Description |
|---|---|
| http://localhost:8000 | Statut et configuration |
| http://localhost:8000/docs | Documentation Swagger interactive |
| http://localhost:8000/health | Vérification du pipeline |
| http://localhost:8000/predict | Prédiction (POST) |

---

## Exemple de requête

    curl -X POST "http://localhost:8000/predict" \
      -H "Content-Type: application/json" \
      -d '{
        "name": "LeBron James",
        "GP": 79, "MIN": 39.5, "PTS": 20.9,
        "FGA": 18.9, "FG%": 47.2,
        "3PA": 4.0, "FTA": 8.9, "FT%": 75.4,
        "REB": 6.9, "AST": 7.2, "STL": 1.7, "BLK": 0.7, "TOV": 3.5
      }'

Réponse attendue :

    {
      "name": "LeBron James",
      "prediction": 1,
      "probabilite": 0.87,
      "confiance": "87.0%",
      "conseil": "Investir",
      "interpretation": "Profil très prometteur — forte probabilité de longévité"
    }

---

## Features

### Features en entrée (13)

| Feature | Description |
|---|---|
| GP | Matchs joués (1-82) |
| MIN | Minutes par match (0-48) |
| PTS | Points par match (0-50) |
| FGA | Tentatives de tir |
| FG% | Pourcentage au tir (0-100) |
| 3PA | Tentatives à 3 points |
| FTA | Tentatives de lancer franc |
| FT% | Pourcentage aux lancers (0-100) |
| REB | Rebonds par match |
| AST | Passes décisives par match |
| STL | Interceptions par match |
| BLK | Contres par match |
| TOV | Pertes de balle par match |

### Features engineerées (4)

| Feature | Formule |
|---|---|
| efficiency | PTS / (MIN + 1e-5) |
| court_impact | REB + AST + STL + BLK |
| shooting_vol | FGA + FTA |
| true_shooting | PTS / (2 x (FGA + 0.44 x FTA) + 1) |

---

## Pipeline de modélisation

1. EDA — analyse visuelle, corrélations, PCA 2D
2. Nettoyage — suppression des doublons contradictoires (1340 à 1273 lignes)
3. Feature engineering — 4 features métier ajoutées
4. Feature selection — 7 features redondantes supprimées (corrélation > 0.95)
5. Winsorisation — IQR x1.5 calculée sur X_train uniquement
6. Train/Test split — 80/20 stratifié
7. Comparaison — Régression Logistique, Random Forest, XGBoost en CV 5-fold
8. Optimisation — Optuna 100 trials sur le Recall
9. Seuil — 0.5 validé empiriquement avec class_weight='balanced'

---

## Résultats sur le test set

| Métrique | Valeur |
|---|---|
| Recall | 0.748 |
| Spécificité | 0.521 |
| AUC-ROC | 0.734 |