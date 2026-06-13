#!/bin/bash

# Script de démarrage — NBA Rookie Longevity Predictor
# Lancer ce script suffit pour démarrer l'API avec Docker

set -e

echo "NBA Rookie Longevity Predictor - Démarrage"
echo "-------------------------------------------"

# On vérifie que Docker est bien installé sur la machine
if ! command -v docker &> /dev/null; then
    echo ""
    echo "Docker n'est pas installé sur cette machine."
    echo ""
    echo "Installez Docker Desktop selon votre système :"
    echo "  Windows / macOS  : https://www.docker.com/products/docker-desktop"
    echo "  Linux            : https://docs.docker.com/engine/install"
    echo ""
    exit 1
fi

# Même vérification pour docker-compose
if ! command -v docker-compose &> /dev/null; then
    echo ""
    echo "docker-compose n'est pas installé."
    echo "Il est inclus dans Docker Desktop, ou installez-le via :"
    echo "  https://docs.docker.com/compose/install"
    echo ""
    exit 1
fi

# On vérifie que le daemon Docker est bien démarré
# docker info échoue silencieusement si Docker n'est pas lancé
if ! docker info &> /dev/null; then
    echo ""
    echo "Docker est installé mais n'est pas démarré."
    echo ""
    echo "Démarrez-le selon votre système :"
    echo "  Windows / macOS  : ouvrez Docker Desktop et attendez qu'il soit prêt"
    echo "  Linux (systemd)  : sudo systemctl start docker"
    echo "  macOS (Colima)   : colima start"
    echo ""
    echo "Une fois Docker démarré, relancez ce script."
    exit 1
fi

echo "Docker détecté et démarré, lancement en cours..."
echo ""

# Construction de l'image et démarrage en arrière-plan
docker-compose up --build -d

echo ""
echo "L'API est disponible sur      : http://localhost:8000"
echo "Documentation Swagger         : http://localhost:8000/docs"
echo "Health check                  : http://localhost:8000/health"
echo ""
echo "Pour arrêter le projet        : docker-compose down"