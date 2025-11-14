import pytest
import os
import tensorflow as tf

def test_model_existance():
    """Teste que le modèle CNN existe."""
    model_path = './models/emotion_model.h5'
    assert os.path.exists(model_path), "Le modèle CNN n'est pas trouvé."    
    
def test_model_loading():
    """Teste le chargement du modèle CNN."""
    model_path = './models/emotion_model.h5'
    model = tf.keras.models.load_model(model_path)
    assert model is not None, "Le modèle CNN n'a pas pu être chargé."