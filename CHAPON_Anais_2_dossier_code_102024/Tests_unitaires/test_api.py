import os
import sys
import joblib
import pandas as pd
import pytest
from flask import Flask, jsonify, request

# Ajouter le chemin relatif du fichier api.py au sys.path pour pouvoir l'importer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Scripts')))

# Importer les éléments nécessaires du fichier api.py
from api import app, current_directory, model, predict

# Créer un client de test pour l'application Flask
@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# Teste le chargement du modèle de prédiction
def test_model_loading():
    # Détermine le chemin du fichier contenant le modèle entraîné
    model_path = os.path.join(current_directory, "..", "Notebooks", "mlflow_model_complet_light", "model.pkl")
    # Charge le modèle à partir du fichier
    model = joblib.load(model_path)
    # Vérifie que le modèle a été chargé correctement
    assert model is not None, "Erreur dans le chargement du modèle."

# Teste le chargement du fichier CSV contenant les données de train
def test_csv_loading():
    # Détermine le chemin du fichier CSV
    csv_path = os.path.join(current_directory, "..", "Dataset", "application_train_light.csv")
    # Charge le fichier CSV dans un DataFrame pandas
    df = pd.read_csv(csv_path)
    # Vérifie que le DataFrame n'est pas vide
    assert not df.empty, "Erreur dans le chargement du CSV."

# Teste la fonction de prédiction de l'API
def test_prediction():
    import os
    import pandas as pd
    from flask import json
    # Détermine le chemin du répertoire courant
    current_directory = os.path.dirname(os.path.abspath(__file__))
    # Détermine le chemin du fichier CSV contenant les données de test
    csv_path = os.path.join(current_directory, "..", "Dataset", "application_train_light.csv")
    # Charge le fichier CSV dans un DataFrame pandas
    df = pd.read_csv(csv_path)
    # Prend un échantillon pour la prédiction
    sk_id_curr = df.iloc[0]['SK_ID_CURR']
    # Crée une requête de test pour la prédiction en utilisant l'échantillon sélectionné
    with app.test_client() as client:
        response = client.post('/predict', json={'SK_ID_CURR': sk_id_curr})
        data = json.loads(response.data)
        prediction = data['probability']
        # Vérifie que la prédiction a été effectuée correctement
        assert prediction is not None, "La prédiction a échoué."
