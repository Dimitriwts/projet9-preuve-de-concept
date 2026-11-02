"""
Monte l'archive à déposer sur OpenClassrooms, avec les noms imposés par l'énoncé :
    Nom_Prénom_n°_nom du livrable_date de démarrage du projet (102026)
et le zip nommé « Titre_du_projet_nom_prénom ».

Le livrable 2 contient le notebook ET le jeu de données (le zip Flipkart d'origine),
comme le demande l'énoncé. Le livrable 4 contient le code du dashboard et un fichier
texte avec l'adresse du dashboard déployé.

Lancement, depuis la racine du projet :
    .venv\\Scripts\\python docs\\build_archive.py
"""

import glob
import shutil
import zipfile
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
LIVRABLES = RACINE / "livrables"
SORTIE = LIVRABLES / "Developpez_une_preuve_de_concept_Wyts_Dimitri.zip"
URL_DASHBOARD = "https://flipkart-dinov2-poc.streamlit.app/"
URL_DEPOT = "https://github.com/Dimitriwts/projet9-preuve-de-concept"

# Les fichiers du dashboard à livrer : le code, les classifieurs exportés, les miniatures.
# On écarte les caches Python, qui ne sont pas du code.
EXCLURE = {"__pycache__"}

contenu = {
    "Wyts_Dimitri_1_plan_travail_102026.pdf": LIVRABLES / "Wyts_Dimitri_1_plan_travail_102026.pdf",
    "Wyts_Dimitri_2_notebook_102026/Wyts_Dimitri_2_notebook_102026.ipynb":
        RACINE / "notebooks" / "Wyts_Dimitri_2_notebook_102026.ipynb",
    "Wyts_Dimitri_2_notebook_102026/donnees_Flipkart.zip": Path(glob.glob(str(RACINE / "data" / "Dataset*.zip"))[0]),
    "Wyts_Dimitri_3_note_methodo_102026.pdf": LIVRABLES / "Wyts_Dimitri_3_note_methodo_102026.pdf",
    "Wyts_Dimitri_5_presentation_102026.pdf": LIVRABLES / "Wyts_Dimitri_5_presentation_102026.pdf",
}
for fichier in (RACINE / "dashboard").rglob("*"):
    if fichier.is_file() and not EXCLURE & set(fichier.parts):
        contenu[f"Wyts_Dimitri_4_dashboard_102026/{fichier.relative_to(RACINE / 'dashboard').as_posix()}"] = fichier
# Le thème accessible du dashboard est à la racine du dépôt, où Streamlit le cherche
contenu["Wyts_Dimitri_4_dashboard_102026/.streamlit/config.toml"] = RACINE / ".streamlit" / "config.toml"

manquants = [nom for nom, chemin in contenu.items() if not chemin.exists()]
if manquants:
    raise SystemExit(f"Fichiers manquants, archive non créée : {manquants}")

adresse = (f"Dashboard déployé (Streamlit Community Cloud) : {URL_DASHBOARD}\n"
           f"Code source versionné (GitHub) : {URL_DEPOT}\n"
           "Lancement en local : pip install -r requirements.txt, puis streamlit run app.py\n")

# ZIP_STORED pour le zip Flipkart (déjà compressé), compression normale pour le reste
with zipfile.ZipFile(SORTIE, "w", zipfile.ZIP_DEFLATED) as archive:
    for nom, chemin in contenu.items():
        methode = zipfile.ZIP_STORED if chemin.suffix == ".zip" else zipfile.ZIP_DEFLATED
        archive.write(chemin, nom, compress_type=methode)
    archive.writestr("Wyts_Dimitri_4_dashboard_102026/adresse_du_dashboard.txt", adresse)

print(f"{len(contenu) + 1} fichiers, {SORTIE.stat().st_size / 1e6:.0f} Mo : {SORTIE}")
