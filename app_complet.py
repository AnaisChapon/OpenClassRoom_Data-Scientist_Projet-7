from flask import Flask, jsonify, request
import pandas as pd
import os
import shap
import joblib

app = Flask(__name__)

# Charger le modèle en dehors de la clause if __name__ == "__main__":
model_path = r"C:\Users\anais\Documents\Data Science\Projets\Projet 7\Données\pipeline_home_credit_complet.joblib"
model = joblib.load(model_path)

# Charger le scaler si nécessaire
# scaler_path = os.path.join('chemin_vers_le_scaler')
# scaler = joblib.load(scaler_path)

@app.route("/predict", methods=['POST'])
def predict():
    try:
        data = request.json
        sk_id_curr = data['SK_ID_CURR']
        print("Received SK_ID_CURR:", sk_id_curr)

        csv_path = os.path.join("df_train_encoded.csv")
        df = pd.read_csv(csv_path)
        print("DataFrame loaded successfully.")

        sample = df[df['SK_ID_CURR'] == sk_id_curr]
        if sample.empty:
            raise ValueError("No data found for given ID")

        print("Sample extracted:", sample)

        sample = sample.drop(columns=['SK_ID_CURR'])
        print("SK_ID_CURR column dropped.")

        # Supposons que vous avez un scaler à appliquer
        # sample_scaled = scaler.transform(sample)

        # Utiliser le modèle pour prédire
        prediction = model.predict_proba(sample)
        proba = prediction[0][1]  # Probabilité de la seconde classe

        # Calculer les valeurs SHAP pour l'échantillon donné
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        
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
    app.run(debug=True, host="0.0.0.0", port=5000)