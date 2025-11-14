#  API RESTful de Détection d'Émotions (FastAPI + ML + PostgreSQL)

## Introduction

Ce projet implémente une API RESTful moderne à l'aide de **FastAPI** pour effectuer la détection d'émotions sur des images de visage. L'API utilise une combinaison de l'apprentissage profond (TensorFlow/Keras) et de la vision par ordinateur (OpenCV), et persiste les résultats de prédiction dans une base de données **PostgreSQL** via **SQLAlchemy**.

##  Fonctionnalités Clés

* **API Haute Performance** : Construite avec FastAPI pour une vitesse et une documentation automatique optimale (via `/docs`).
* **Modèle ML/CV** : Détection de visage (OpenCV Haar Cascade) suivie de la classification d'émotions (TensorFlow/Keras).
* **Persistance des Données** : Enregistrement de chaque prédiction (émotion, confiance, horodatage) dans PostgreSQL.
* **Tests Unitaires et Intégration** : Utilisation de **Pytest** pour garantir la fiabilité des composants.
* **Intégration Continue (CI)** : Automatisation des tests via **GitHub Actions** après chaque mise à jour du code.

##  Structure du Projet

```
├── .github/workflows/
│   └── python-tests.yml        #  Workflow d'intégration continue (CI)
├── src/
│   ├── main.py                 #  Point d'entrée de l'API et endpoints
│   ├── detect_and_predict.py   #  Logique ML/CV (chargement du modèle, détection, prédiction)
│   ├── database.py             # Configuration de la connexion PostgreSQL (SQLAlchemy)
│   ├── models.py               #  Modèle ORM pour la table emotion_records
│   └── schemas.py              #  Schémas Pydantic pour la validation des données
├── models/
│   └── emotion_model.h5        # Modèle CNN entraîné (fichier binaire)
├── haarcascade_frontalface_default.xml  # Fichier Haar Cascade pour la détection de visage
├── requirement.txt             # Liste des dépendances Python
└── .env                        # Fichier de variables d'environnement (IGNORÉ par Git)
```

## Installation et Configuration

### 1. Prérequis

* **Python 3.10+**
* **PostgreSQL** (le serveur doit être installé et accessible)

### 2. Cloner le dépôt et installer les dépendances

```bash
# Cloner le dépôt
git clone https://github.com/AyoubMotei/emotion-detection-api.git
cd emotion-detection-api

# Créer et activer l'environnement virtuel
python -m venv venv
source venv\Scripts\activate  # Sur Windows: 

# Installer toutes les dépendances
pip install -r requirement.txt
```

### 3. Fichier de Configuration `.env`

Créez un fichier racine nommé `.env` (il est ignoré par Git pour des raisons de sécurité) pour stocker les informations de connexion à votre base de données :

```env
# Contenu du fichier .env
user=votre_utilisateur_postgres
password=votre_mot_de_passe_db
host=localhost
port=5432
database=emotion_db
```

### 4. Artefacts de Machine Learning

Les fichiers suivants sont obligatoires pour que la logique de prédiction fonctionne :

**Modèle CNN (`models/emotion_model.h5`)** :
- Placez votre modèle entraîné à cet emplacement
- Il est supposé être le résultat de l'exécution de votre `train_cnn.ipynb`

**Classifieur de Visage (`haarcascade_frontalface_default.xml`)** :
- Téléchargez ce fichier Haar Cascade d'OpenCV à la racine de votre projet :

```bash
curl -O https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml
```

## Exécution de l'API

Démarrez l'API en utilisant le serveur Uvicorn :

```bash
uvicorn src.main:app --reload
```

L'API est accessible à **http://127.0.0.1:8000**

**Documentation interactive (Swagger UI)** : http://127.0.0.1:8000/docs

### Endpoint Principal

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| POST | `/predict_emotion` | Reçoit une image de visage, prédit l'émotion et enregistre le résultat dans PostgreSQL |

### Schéma de Réponse (`EmotionResponse`)

| Champ | Type | Exemple | Description |
|-------|------|---------|-------------|
| `id` | integer | `1` | ID de l'enregistrement dans la base de données |
| `emotion` | string | `"happy"` | Émotion prédite (e.g., happy, sad, neutral...) |
| `confidence` | float | `0.985` | Niveau de confiance de la prédiction (entre 0 et 1) |
| `created_at` | string | `"2023-11-14T10:30:00.123456"` | Horodatage de l'enregistrement |

##  Tests Unitaires et Intégration

Les tests sont configurés pour s'exécuter avec Pytest et sont essentiels pour valider la logique du modèle ML et le fonctionnement de l'API (y compris l'insertion simulée en DB).

### Exécution locale

```bash
# Assurer que Python peut trouver les modules de 'src'
export PYTHONPATH="$PYTHONPATH:$(pwd):$(pwd)/src"

# Lancer Pytest
python -m pytest -v
```

### Intégration Continue (GitHub Actions)

Le workflow `./.github/workflows/python-tests.yml` automatise l'exécution des tests.

**Déclencheurs :**
- `push` sur la branche `main`
- `pull_request` sur la branche `main`

**Étapes clés du Workflow :**
1. **Setup ML/CV Artifacts** : Télécharge le fichier `haarcascade_frontalface_default.xml` nécessaire, garantissant que les tests peuvent s'exécuter même si le fichier n'est pas versionné
2. **Run Pytest tests** : Exécute tous les tests Pytest

##  Auteur

AYOUB MOTEI
