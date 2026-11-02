"""
Crée les miniatures des 1 050 images pour le dashboard.

Les photos d'origine pèsent 345 Mo (certaines dépassent 11 000 pixels de côté),
trop lourd pour un hébergement gratuit et inutile pour l'affichage. Les
miniatures font au plus 600 pixels de côté. Les modèles travaillent de toute
façon en 224 × 224 : la prédiction sur une miniature est quasi identique à
celle sur l'original (vérifié par test_modele.py).

Lancement, depuis la racine du projet :
    .venv\\Scripts\\python dashboard\\preparer_miniatures.py
"""

from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

RACINE = Path(__file__).resolve().parent.parent
SOURCE = RACINE / "data" / "raw" / "Flipkart" / "Images"
CIBLE = Path(__file__).resolve().parent / "donnees" / "miniatures"
COTE_MAX = 600

CIBLE.mkdir(parents=True, exist_ok=True)
total = 0
for chemin in sorted(SOURCE.glob("*.jpg")):
    with Image.open(chemin) as image:
        image = image.convert("RGB")
        image.thumbnail((COTE_MAX, COTE_MAX), Image.LANCZOS)
        image.save(CIBLE / chemin.name, quality=88, optimize=True)
    total += (CIBLE / chemin.name).stat().st_size
print(f"{len(list(CIBLE.glob('*.jpg')))} miniatures, {total / 1e6:.1f} Mo")
