import subprocess
import os

# Chemin relatif pour accéder à api.py et dashboard.py
scripts_directory = "./CHAPON_Anais_2_dossier_code_102024/Scripts"

# Définir le port pour l'API Flask et le dashboard Streamlit
api_port = "5000"
dashboard_port = "8000"

# Lancer l'API Flask
api_process = subprocess.Popen(["python", os.path.join(scripts_directory, "api_complet_light.py")], env=dict(os.environ, PORT=api_port))
print(f"API Flask lancée sur le port {api_port}")

# Lancer le dashboard Streamlit
dashboard_process = subprocess.Popen(["streamlit", "run", os.path.join(scripts_directory, "dashboard_complet_light.py"), "--server.port", dashboard_port])
print(f"Dashboard Streamlit lancé sur le port {dashboard_port}")
