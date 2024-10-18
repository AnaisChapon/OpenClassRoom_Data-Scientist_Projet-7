import pytest
from unittest.mock import patch, Mock
import os
import sys

# Obtenir le répertoire du fichier actuel 
current_file_directory = os.path.dirname(__file__)

# Créer un chemin relatif vers le dossier 'Scripts'
scripts_directory = os.path.abspath(os.path.join(current_file_directory, '..', 'Scripts'))

# Insérer ce chemin au début de sys.path
sys.path.insert(0, scripts_directory)

# Simulation d'un état pour st.session_state
mocked_session_state = {'state': {'data_received': False, 'data': None, 'last_sk_id_curr': None}}

# Définir le décorateur pytest pour simuler st.session_state
@pytest.fixture
def mocked_st(monkeypatch):
    # Utilisez monkeypatch pour remplacer l'attribut session_state de streamlit par notre état simulé
    monkeypatch.setattr("streamlit.session_state", mocked_session_state)
    return mocked_session_state


#TESTS

# Utilise pytest.mark pour appliquer le mock à des tests spécifiques
@pytest.mark.parametrize('mocked_st', [mocked_session_state], indirect=True)
def test_compute_color(mocked_st):
    # Importez la fonction compute_color à partir du module dashboard
    from dashboard import compute_color
    # Teste la fonction compute_color pour différentes valeurs
    assert compute_color(30) == "green", "Erreur dans la fonction compute_color."
    assert compute_color(50) == "red", "Erreur dans la fonction compute_color."

@pytest.mark.parametrize('mocked_st', [mocked_session_state], indirect=True)
def test_format_value(mocked_st):
    # Importez la fonction format_value à partir du module dashboard
    from dashboard import format_value
    # Teste la fonction format_value pour différentes valeurs
    assert format_value(5.67) == 5.67, "Erreur dans la fonction format_value."
    assert format_value(5.00) == 5, "Erreur dans la fonction format_value."

@pytest.mark.parametrize('mocked_st', [mocked_session_state], indirect=True)
def test_find_closest_description(mocked_st):
    # Importez les fonctions find_closest_description et definition_features_df à partir du module dashboard
    from dashboard import find_closest_description, definition_features_df
    # Teste la fonction pour trouver la description la plus proche d'un terme donné
    description = find_closest_description("AMT_INCOME_TOTAL", definition_features_df)
    assert description is not None, "Erreur dans la fonction find_closest_description."

def test_get_state(mocked_st):	
    # Importez la fonction get_state à partir du module dashboard
    from dashboard import get_state
    state = get_state()
    assert isinstance(state, dict), "La fonction get_state doit renvoyer un dictionnaire."
    assert "data_received" in state, "Le dictionnaire renvoyé par get_state doit contenir la clé 'data_received'."
    assert "data" in state, "Le dictionnaire renvoyé par get_state doit contenir la clé 'data'."
    assert "last_sk_id_curr" in state, "Le dictionnaire renvoyé par get_state doit contenir la clé 'last_sk_id_curr'."
