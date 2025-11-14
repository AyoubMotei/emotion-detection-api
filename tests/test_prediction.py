from fastapi.testclient import TestClient
from src.main import app    

client=TestClient(app)

def test_prediction_format():
    files={'file':('happy_face.jpg',open('tests/test_images/happy_face.jpg','rb'),'image/jpeg')}
    api=client.post("/predict_emotion",files=files)
    assert api.status_code==201, "La requête de prédiction a échoué."
    resultat=api.json()  
    
    assert isinstance(resultat,dict), "La réponse doit être un dictionnaire."
    assert isinstance(resultat["emotion"], str ), "L'émotion doit être une chaîne de caractères."
    assert isinstance(resultat["confidence"], float), "La confiance doit être un nombre à virgule flottante."
    assert 0.0 <= resultat["confidence"] <= 1.0, "La confiance doit être entre 0 et 1."