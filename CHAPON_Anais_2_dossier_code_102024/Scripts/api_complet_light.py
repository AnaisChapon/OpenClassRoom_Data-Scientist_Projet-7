from flask import Flask, jsonify, request
import pandas as pd
import shap
import joblib
import os

app = Flask(__name__)

# Charger le modèle
model_path = '/home/achapon/OpenClassRoom_Data-Scientist_Projet-7/CHAPON_Anais_2_dossier_code_102024/Notebooks/mlflow_model_complet_light/model.pkl'
model = joblib.load(model_path)

@app.route("/predict", methods=['POST'])
def predict():
    try:
        data = request.json
        sk_id_curr = data['SK_ID_CURR']
        print("Received SK_ID_CURR:", sk_id_curr)

        # Charger le dataset complet pour obtenir les données de fond
        csv_path = '/home/achapon/OpenClassRoom_Data-Scientist_Projet-7/CHAPON_Anais_2_dossier_code_102024/application_train_light.csv'
        df = pd.read_csv(csv_path)
        print("DataFrame loaded successfully.")

        # Extraire l'échantillon correspondant à SK_ID_CURR
        sample = df[df['SK_ID_CURR'] == sk_id_curr]
        if sample.empty:
            raise ValueError("No data found for given ID")

        print("Sample extracted:", sample)

        # Supprimer les colonnes non nécessaires
        sample = sample.drop(columns=['SK_ID_CURR', 'TARGET'], errors='ignore')
        print("Sample before scaling:", sample)

        # Mise à l'échelle des données du client en utilisant le scaler du pipeline 
        sample_scaled = model.named_steps['scaler'].transform(sample)
        print("Sample after scaling:", sample_scaled)

        # Utiliser le pipeline pour prédire la probabilité de défaut
        prediction = model.named_steps['classifier'].predict_proba(sample_scaled)
        proba = prediction[0][1]  # Probabilité de la seconde classe

        # Initialiser l'explainer SHAP avec les données mises à l'échelle
        explainer = shap.TreeExplainer(model.named_steps['classifier'])
    
        # Calculer les valeurs SHAP locales pour ce client
        shap_values = explainer.shap_values(sample_scaled)
        
        # Vérifier la structure de shap_values et récupérer les valeurs pour la classe positive
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values_for_client = shap_values[1][0]  # Extraire les valeurs SHAP pour la classe positive
        else:
            shap_values_for_client = shap_values[0]  # Si une seule classe est présente, utiliser les valeurs disponibles

        # Créer un DataFrame avec les valeurs SHAP et les noms de colonnes correspondants
        shap_local_df = pd.DataFrame([shap_values_for_client], columns=sample.columns)

        # Trier les valeurs SHAP et les features par leur impact dans l'ordre croissant
        sorted_shap_local_importance = shap_local_df.T.sort_values(by=0, ascending=True)
        sorted_feature_names = sorted_shap_local_importance.index.tolist()
        sorted_shap_values = sorted_shap_local_importance[0].values.tolist()

        # Imprimer les valeurs SHAP triées dans la console pour débogage
        print("Valeurs SHAP triées pour le client :", sorted_shap_values)

        # Retourner les résultats sous forme de JSON pour l'API
        return jsonify({
            'success': True,
            'probability': proba * 100,
            'shap_values': sorted_shap_values,  # valeurs SHAP triées
            'feature_names': sorted_feature_names,  # noms des features triés
            'feature_values': sample.values[0].tolist(),
            'top_features': dict(zip(sorted_feature_names, sorted_shap_values))  # dictionnaire des features et leurs valeurs SHAP
        })

    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify(success=False, message=str(e)), 500