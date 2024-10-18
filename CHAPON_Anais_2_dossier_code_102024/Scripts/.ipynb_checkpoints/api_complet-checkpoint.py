from flask import Flask, jsonify, request
import pandas as pd
import os
import shap
import joblib
import numpy as np

app = Flask(__name__)

# Charger le modèle en dehors de la clause if __name__ == "__main__":
model_path = r"C:\Users\anais\Documents\Data Science\Projets\Projet 7\Données\mlflow_model_complet\model.pkl"
model = joblib.load(model_path)

# Charger le scaler si nécessaire
scaler_path = r"C:\Users\anais\Documents\Data Science\Projets\Projet 7\Données\StandardScaler.pkl"
scaler = joblib.load(scaler_path)

@app.route("/predict", methods=['POST'])
def predict():
    try:
        data = request.json
        sk_id_curr = data['SK_ID_CURR']
        print("Received SK_ID_CURR:", sk_id_curr)

        csv_path = r"C:\Users\anais\Documents\Data Science\Projets\Projet 7\Données\Implementez un modèle de scoring_CHAPON_Anais\application_train_encoded.csv"
        df = pd.read_csv(csv_path)
        print("DataFrame loaded successfully.")

        sample = df[df['SK_ID_CURR'] == sk_id_curr]
        if sample.empty:
            raise ValueError("No data found for given ID")

        print("Sample extracted:", sample)

        sample = sample.drop(columns=['SK_ID_CURR'])
        print("SK_ID_CURR column dropped.")

        # Supposons que vous avez un scaler à appliquer
        sample_scaled = scaler.transform(sample)

        # Utiliser le modèle pour prédire
        prediction = model.predict_proba(sample_scaled)
        proba = prediction[0][1]  # Probabilité de la seconde classe

        # Utiliser KernelExplainer pour des modèles génériques
        background = np.zeros((1, sample_scaled.shape[1]))  # Vous pouvez choisir d'autres données de référence
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(sample_scaled)

        return jsonify({
            'success': True,
            'probability': proba * 100,
            'shap_values': shap_values[1][0].tolist(),
            'feature_names': sample.columns.tolist(),
            'feature_values': sample.values[0].tolist()
        })

    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify(success=False, message=str(e)), 500

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)