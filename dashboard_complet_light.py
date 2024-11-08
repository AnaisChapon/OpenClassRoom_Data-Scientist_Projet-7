import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# Obtenez le répertoire courant du script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Chargement des fichiers
path_df_train_url = "https://github.com/AnaisChapon/OpenClassRoom_Data-Scientist_Projet-7/blob/main/CHAPON_Anais_2_dossier_code_102024/Dataset/application_test_light.csv"
path_definition_features_df_url = "https://github.com/AnaisChapon/OpenClassRoom_Data-Scientist_Projet-7/blob/main/CHAPON_Anais_2_dossier_code_102024/Dataset/definition_features.csv"

definition_features_df = pd.read_csv(path_definition_features_df_url)
df_train = pd.read_csv(path_df_train_url, low_memory=True)

st.set_page_config(layout="wide")

# Fonction pour ajuster la taille de la police en fonction de la hauteur du graphique
def get_title_font_size(height):
    base_size = 12
    scale_factor = height / 600.0
    return base_size * scale_factor

# Génération des figures pour les graphiques
def generate_figure(df, title_text, x_anchor, yaxis_categoryorder, yaxis_side):
    fig = go.Figure(data=[go.Bar(y=df["Feature"], x=df["SHAP Value"], orientation="h")])
    annotations = generate_annotations(df, x_anchor)
    title_font_size = get_title_font_size(600)
    fig.update_layout(
        annotations=annotations,
        title_text=title_text,
        title_x=0.25,
        title_y=0.88,
        title_font=dict(size=title_font_size),
        yaxis=dict(
            categoryorder=yaxis_categoryorder, side=yaxis_side, tickfont=dict(size=14)
        ),
        height=600,
    )
    fig.update_xaxes(title_text="Impact of features")
    return fig

# Génération des annotations pour les graphiques
def generate_annotations(df, x_anchor):
    annotations = []
    for y_val, x_val, feat_val in zip(df["Feature"], df["SHAP Value"], df["Feature Value"]):
        formatted_feat_val = (
            feat_val
            if pd.isna(feat_val)
            else (int(feat_val) if feat_val == int(feat_val) else feat_val)
        )
        annotations.append(
            dict(
                x=x_val,
                y=y_val,
                text=f"<b>{formatted_feat_val}</b>",
                showarrow=False,
                xanchor=x_anchor,
                yanchor="middle",
                font=dict(color="white"),
            )
        )
    return annotations

# Fonction pour afficher la jauge de probabilité
def plot_gauge(proba, threshold=52):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=proba,
        delta={
            'reference': threshold, 
            'position': "top",
            'increasing': {'color': '#D55E00'},  # orange
            'decreasing': {'color': '#009E73'}    # turquoise
        },
        title={'text': "Non-reimbursement probability", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, threshold], 'color': '#56B4E9'},  # bleu clair
                {'range': [threshold, 100], 'color': '#F0E442'}  # jaune
            ],
            'threshold': {
                'line': {'color': "#D55E00", 'width': 4},
                'thickness': 0.75,
                'value': threshold
            }
        }
    ))

    fig.update_layout(height=400, width=600, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# Fonction pour déterminer la couleur selon la probabilité
def compute_color(value):
    if 0 <= value < 52:
        return "#009E73"  # turquoise pour une probabilité faible
    elif 52 <= value <= 100:
        return "#D55E00"  # orange pour une probabilité élevée

# Fonction pour formater les valeurs
def format_value(val):
    if pd.isna(val):
        return val
    if isinstance(val, (float, int)):
        if val == int(val):
            return int(val)
        return round(val, 2)
    return val

# Trouver la description d'une feature dans le fichier des définitions
def find_closest_description(feature_name, definitions_df):
    for index, row in definitions_df.iterrows():
        if row["Row"] in feature_name:
            return row["Description"]
    return None

# Affichage de la distribution d'une feature choisie
def plot_distribution(selected_feature, col):
    if selected_feature:
        data = df_train[selected_feature]

        # Trouver la valeur de la fonctionnalité pour le client actuel :
        client_feature_value = feature_values[feature_names.index(selected_feature)]

        fig = go.Figure()

        # Vérifier si la fonctionnalité est catégorielle :
        unique_values = sorted(data.dropna().unique())
        if set(unique_values) <= {0, 1, 2, 3, 4, 5, 6, 7}:
            # Compter les occurrences de chaque valeur :
            counts = data.value_counts().sort_index()

            # Assurez-vous que les longueurs correspondent
            assert len(unique_values) == len(counts)

            # Modifier la déclaration de la liste de couleurs pour correspondre à la taille de unique_values
            colors = ["#0072B2"] * len(unique_values)  # Bleu adapté

            # Mettre à jour client_value
            client_value = (
                unique_values.index(client_feature_value)
                if client_feature_value in unique_values
                else None
            )

            # Mettre à jour la couleur correspondante si client_value n'est pas None
            if client_value is not None:
                colors[client_value] = "#D55E00"  # Orange adapté

            # Modifier le tracé pour utiliser unique_values
            fig.add_trace(go.Bar(x=unique_values, y=counts.values, marker_color=colors))

        else:
            # Calculer les bins pour l'histogramme :
            hist_data, bins = np.histogram(data.dropna(), bins=20)

            # Trouvez le bin pour client_feature_value :
            client_bin_index = np.digitize(client_feature_value, bins) - 1

            # Créer une liste de couleurs pour les bins :
            colors = ["#0072B2"] * len(hist_data)  # Bleu adapté
            if 0 <= client_bin_index < len(hist_data):  # Vérifiez que l'index est valide
                colors[client_bin_index] = "#D55E00"  # Orange adapté

            # Tracer la distribution pour les variables continues :
            fig.add_trace(
                go.Histogram(
                    x=data,
                    marker=dict(color=colors, opacity=0.7),
                    name="Distribution",
                    xbins=dict(start=bins[0], end=bins[-1], size=bins[1] - bins[0]),
                )
            )

            # Utiliser une échelle logarithmique si la distribution est fortement asymétrique :
            mean_val = np.mean(hist_data)
            std_val = np.std(hist_data)
            if std_val > 3 * mean_val:  # Ce seuil peut être ajusté
                fig.update_layout(yaxis_type="log")

        height = 600  # Ajustez cette valeur selon la hauteur par défaut de votre figure
        title_font_size = get_title_font_size(height)

        fig.update_layout(
            title_text=f"Distribution pour {selected_feature}",
            title_font=dict(size=title_font_size),
            xaxis_title=selected_feature,
            yaxis_title="Nombre de clients",
            title_x=0.3,
        )

        col.plotly_chart(fig, use_container_width=True)

        # Afficher la définition de la feature choisie :
        description = find_closest_description(selected_feature, definition_features_df)
        if description:
            col.write(f"**Definition:** {description}")

# Fonction pour gérer l'état de session
def get_state():
    if "state" not in st.session_state:
        st.session_state["state"] = {"data_received": False, "data": None, "last_sk_id_curr": None}
    elif "last_sk_id_curr" not in st.session_state["state"]:
        st.session_state["state"]["last_sk_id_curr"] = None
    return st.session_state["state"]

state = get_state()

# Interface utilisateur
st.markdown("<h1 style='text-align: center; color: black;'>Estimated risk of non-reimbursement</h1>", unsafe_allow_html=True)
sk_id_curr = st.text_input("Enter customer number :", on_change=lambda: state.update(run=True))
col1, col2 = st.columns([1, 20])

st.markdown("""
    <style>
        button {
            width: 60px !important;
            white-space: nowrap !important;
        }
    </style>
    """, unsafe_allow_html=True)

if col1.button("Run") or state["data_received"]:
    if state["last_sk_id_curr"] != sk_id_curr:
        state["data_received"] = False
        state["last_sk_id_curr"] = sk_id_curr

    if not state["data_received"]:
        response = requests.post("https://achapon.pythonanywhere.com/predict", json={"SK_ID_CURR": int(sk_id_curr)})
        if response.status_code != 200:
            st.error(f"Erreur lors de l'appel à l'API: {response.status_code}")
            st.stop()

        state["data"] = response.json()
        state["data_received"] = True

    data = state["data"]
    proba = data["probability"]
    feature_names = data["feature_names"]
    shap_values = data["shap_values"]
    feature_values = data["feature_values"]
    shap_values = [val[0] if isinstance(val, list) else val for val in shap_values]
    shap_df = pd.DataFrame(list(zip(feature_names, shap_values, [format_value(val) for val in feature_values])), columns=["Feature", "SHAP Value", "Feature Value"])

    color = compute_color(proba)
    col2.markdown(f"<p style='margin: 10px;'>The probability that this customer will not be able to repay his loan is <span style='color:{color}; font-weight:bold;'>{proba:.2f}%</span> (max tolerance: <strong>52%</strong>)</p>", unsafe_allow_html=True)

    decision_message = "The loan will be granted." if proba < 52 else "The loan will not be granted."
    st.markdown(f"<div style='text-align: center; color:{color}; font-size:30px; border:2px solid {color}; padding:10px;'>{decision_message}</div>", unsafe_allow_html=True)
    
    gauge_fig = plot_gauge(proba)
    col2.plotly_chart(gauge_fig, use_container_width=True)

    top_positive_shap = shap_df.sort_values(by="SHAP Value", ascending=False).head(3)
    top_negative_shap = shap_df.sort_values(by="SHAP Value").head(3)

    fig_positive = generate_figure(top_positive_shap, "Top 3 features increasing risk", "right", "total ascending", "left")
    fig_negative = generate_figure(top_negative_shap, "Top 3 features that reduce risk", "left", "total descending", "right")
    

    col_left, col_right = st.columns(2)
    col_left.plotly_chart(fig_positive, use_container_width=True)
    col_right.plotly_chart(fig_negative, use_container_width=True)

    selected_feature = st.selectbox("Select a feature to see its distribution", shap_df["Feature"].tolist())
    plot_distribution(selected_feature, st)

    if st.button("Reload"):
        st.experimental_rerun()
