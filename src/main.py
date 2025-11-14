from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from sqlalchemy.orm import Session
from tensorflow.keras.models import load_model 
import cv2 
import numpy as np 
from typing import List
import os

# Importations locales des fichiers de la structure src/
from src.database import Base, engin, get_db
from src.models import EmotionRecord
from src.schemas import EmotionResponse
from src.detect_and_predict import predict_emotion, CLASS_NAMES 


# --- CHARGEMENT GLOBAL DES MODÈLES  ---

MODEL_PATH = 'models/emotion_model.h5' 
CASCADE_PATH = 'haarcascade_frontalface_default.xml'

GLOBAL_CNN_MODEL = None
GLOBAL_FACE_CASCADE = None

# Tente de charger les modèles au démarrage de l'API
try:
    if os.path.exists(MODEL_PATH):
        GLOBAL_CNN_MODEL = load_model(MODEL_PATH)
        print(f"Modèle CNN chargé depuis : {MODEL_PATH}")
    else:
        print(f"ATTENTION: Modèle CNN non trouvé à {MODEL_PATH}. Le service IA sera indisponible.")

    if os.path.exists(CASCADE_PATH):
        GLOBAL_FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)
        print(f"Haar Cascade chargé depuis : {CASCADE_PATH}")
    else:
        print(f"ATTENTION: Haar Cascade non trouvé à {CASCADE_PATH}. Le service IA sera indisponible.")

except Exception as e:
    print(f"ERREUR lors du chargement des modèles. Le service IA est indisponible: {e}")
    GLOBAL_CNN_MODEL = None
    GLOBAL_FACE_CASCADE = None

# --- INITIALISATION DE FASTAPI ---
app = FastAPI(title="Emotion Detection API")
# Création des tables si elles n'existent pas
Base.metadata.create_all(bind=engin) 


@app.get("/")
async def root():
    # Affiche l'état du service IA au démarrage
    model_status = "Opérationnel" if GLOBAL_CNN_MODEL and GLOBAL_FACE_CASCADE else "Indisponible"
    return {"status": "Service en cours", "IA_status": model_status}


# --- ROUTE DE PRÉDICTION (POST /predict_emotion) ---
@app.post("/predict_emotion", response_model=EmotionResponse, status_code=status.HTTP_201_CREATED)
async def predict_emotion_endpoint(
    file: UploadFile = File(..., description="Image d'un visage au format JPG ou PNG."), 
    db: Session = Depends(get_db) # Injection de dépendance pour la session DB 
):
    # Vérification de l'état du service IA
    if GLOBAL_CNN_MODEL is None or GLOBAL_FACE_CASCADE is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Le service de détection d'émotion est actuellement indisponible (modèles non chargés)."
        )

    # Lecture de l'image en mémoire 
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    
    # Décodage de l'image en tableau Numpy 
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR) 

    if img_np is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Fichier image non valide ou illisible.")

    #  Détection et Prédiction
    emotion, confidence = predict_emotion(img_np, GLOBAL_CNN_MODEL, GLOBAL_FACE_CASCADE)
    
    if emotion is None:
        # Aucun visage détecté
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Aucun visage détecté dans l'image fournie. Assurez-vous que l'image contient un visage clair."
        )

    # Sauvegarde dans la base de données
    emotion_record = EmotionRecord(
        emotion=emotion,
        confidence=confidence
        # created_at est géré automatiquement par 'default=datetime.utcnow' dans models.py
    )
    db.add(emotion_record)
    db.commit()
    db.refresh(emotion_record)

    return emotion_record


# --- ROUTE D'HISTORIQUE (GET /history)  ---
@app.get("/history", response_model=List[EmotionResponse])
def get_history_route(db: Session = Depends(get_db)):
    """Récupère les 100 dernières prédictions enregistrées, triées par date récente."""
    
    history = (
        db.query(EmotionRecord)
        .order_by(EmotionRecord.created_at.desc())
        .limit(100)
        .all()
    )
                
    return history

