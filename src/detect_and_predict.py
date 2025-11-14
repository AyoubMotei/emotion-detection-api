import cv2
import numpy as np
import tensorflow as tf

# --- CONSTANTES ---
IMG_SIZE = 48 
CLASS_NAMES = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised'] 

# --- FONCTION DE PRÉTRAITEMENT OPTIMISÉE ---
def preprocess_face_for_cnn(face_roi):
    """
    Prétraite la région du visage (ROI) pour l'entrée du CNN.
    Utilise np.reshape() pour définir la forme finale (1, 48, 48, 1).
    """
    
    #  Redimensionnement à 48x48
    face = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    
    # Normalisation (0-255 -> 0-1)
    face = face.astype('float32') / 255.0
    
    
    # Forme finale : (Batch=1, Hauteur=48, Largeur=48, Canaux=1)
    face = face.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    
    return face

# --- FONCTION PRINCIPALE DE PRÉDICTION ---
def predict_emotion(image_np, model, face_cascade):
    """
    Détecte les visages, prédit l'émotion du visage le plus grand et retourne le résultat.
    Retourne (None, None) si aucun visage n'est détecté.
    """
    
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    
    # Détection des visages
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30)
    )
    
    if len(faces) == 0:
        return None, None # Aucun visage détecté
    
    # Prendre le plus grand visage
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    
    # Extraire le ROI
    face_roi = gray[y:y + h, x:x + w]
    
    # Prétraitement 
    input_tensor = preprocess_face_for_cnn(face_roi)
    
    # Prédiction du CNN
    pred = model.predict(input_tensor, verbose=0)
    
    # Interprétation du résultat
    emotion_idx = np.argmax(pred)
    confidence = pred[0][emotion_idx]
    emotion = CLASS_NAMES[emotion_idx]
    
    
    
    return emotion, float(confidence)