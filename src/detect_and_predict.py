import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# === VÉRIFICATION DES FICHIERS ===
model_path = 'models/emotion_model.h5'
cascade_path = 'haarcascade_frontalface_default.xml'

if not os.path.exists(model_path):
    print(f"ERREUR : Modèle non trouvé → {model_path}")
    print("   → Retourne dans le notebook et exécute la cellule model.save(...)")
    exit()

if not os.path.exists(cascade_path):
    print(f"ERREUR : Fichier Haar Cascade non trouvé → {cascade_path}")
    print("   → Télécharge-le avec la commande curl fournie")
    exit()

# === CHARGEMENT ===
print("Chargement du modèle...")
model = load_model(model_path)
print("Modèle chargé avec succès !")

print("Chargement du détecteur de visages...")
face_cascade = cv2.CascadeClassifier(cascade_path)
print("Détecteur chargé !")

class_names = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

def predict_emotion(image_path):
    if not os.path.exists(image_path):
        print(f"Image non trouvée : {image_path}")
        return None

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    
    if len(faces) == 0:
        print("Aucun visage détecté")
        return None
    
    # Prendre le plus grand visage
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (48, 48))
    face = face.astype('float32') / 255.0
    face = face.reshape(1, 48, 48, 1)
    
    pred = model.predict(face, verbose=0)
    emotion_idx = np.argmax(pred)
    confidence = pred[0][emotion_idx]
    emotion = class_names[emotion_idx]
    
    # Dessin
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
    cv2.putText(img, f"{emotion} ({confidence:.2f})", (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    output_path = image_path.replace('.jpg', '_detected.jpg').replace('.png', '_detected.jpg')
    cv2.imwrite(output_path, img)
    print(f"Résultat → {output_path}")
    cv2.imshow('Résultat', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return emotion, confidence

# === TEST ===
if __name__ == "__main__":
    predict_emotion("test_image.jpg")  


