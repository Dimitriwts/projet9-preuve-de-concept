"""
Transforme source_notebook.py (format « percent » : cellules séparées par
`# %%` et `# %% [markdown]`) en notebook Jupyter.

Le code est écrit dans un fichier .py pour être relu et versionné facilement,
puis converti et exécuté :
    .venv\\Scripts\\python notebooks\\construire_notebook.py
    .venv\\Scripts\\jupyter nbconvert --to notebook --execute --inplace notebooks\\<nom>.ipynb
"""

import re
import sys
from pathlib import Path

import nbformat

ICI = Path(__file__).resolve().parent
SOURCE = ICI / "source_notebook.py"
CIBLE = ICI / (sys.argv[1] if len(sys.argv) > 1 else "Wyts_Dimitri_2_notebook_102026.ipynb")

texte = SOURCE.read_text(encoding="utf-8")
blocs = re.split(r"^# %%(.*)$", texte, flags=re.M)

cellules = []
# blocs = [avant, entete1, contenu1, entete2, contenu2, ...]
for entete, contenu in zip(blocs[1::2], blocs[2::2]):
    contenu = contenu.strip("\n")
    if "[markdown]" in entete:
        lignes = [re.sub(r"^# ?", "", ligne) for ligne in contenu.splitlines()]
        cellules.append(nbformat.v4.new_markdown_cell("\n".join(lignes)))
    else:
        cellules.append(nbformat.v4.new_code_cell(contenu))

notebook = nbformat.v4.new_notebook(cells=cellules)
notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbformat.write(notebook, CIBLE)
print(f"{CIBLE.name} : {len(cellules)} cellules")
