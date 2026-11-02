"""
Tests du dashboard. Lancement, depuis le dossier dashboard :
    python -m pytest test_modele.py -v
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import accessibilite as acc
from modele import CATEGORIES, Predicteur, carre_blanc, vers_tenseur

ICI = Path(__file__).resolve().parent
ORIGINALES = ICI.parent / "data" / "raw" / "Flipkart" / "Images"
CATALOGUE = pd.read_csv(ICI / "modele" / "predictions_hors_pli.csv")


@pytest.fixture(scope="module")
def predicteur():
    return Predicteur()


@pytest.mark.skipif(not ORIGINALES.exists(), reason="images d'origine absentes (serveur)")
def test_meme_representation_que_le_notebook(predicteur):
    """Le dashboard doit préparer les images exactement comme le notebook."""
    attendu = np.load(ICI / "modele" / "controle_representations.npy")
    for k, nom in enumerate(CATALOGUE["image"].head(len(attendu))):
        lot = vers_tenseur(carre_blanc(Image.open(ORIGINALES / nom)))[None]
        obtenu, _ = predicteur.representation_dinov2(lot)
        np.testing.assert_allclose(obtenu[0], attendu[k], rtol=1e-3, atol=1e-3)


def test_miniature_donne_la_meme_prediction(predicteur):
    """Prédire sur la miniature du dashboard ou sur l'original donne la même catégorie."""
    if not ORIGINALES.exists():
        pytest.skip("images d'origine absentes (serveur)")
    identiques = 0
    echantillon = CATALOGUE.sample(30, random_state=0)["image"]
    for nom in echantillon:
        a = predicteur.predire(Image.open(ORIGINALES / nom))["dinov2"]
        b = predicteur.predire(Image.open(ICI / "donnees" / "miniatures" / nom))["dinov2"]
        identiques += max(a, key=a.get) == max(b, key=b.get)
    assert identiques >= 29


def test_prediction_bien_formee(predicteur):
    image = Image.open(ICI / "donnees" / "miniatures" / CATALOGUE["image"].iloc[0])
    resultat = predicteur.predire(image)
    for modele in ("dinov2", "mobilenet"):
        assert list(resultat[modele]) == CATEGORIES
        assert sum(resultat[modele].values()) == pytest.approx(1.0, abs=1e-6)
    assert resultat["attention"].shape == (16, 16)
    assert resultat["attention"].max() == pytest.approx(1.0)


def test_image_en_niveaux_de_gris_ou_transparente(predicteur):
    """Une photo envoyée peut être en niveaux de gris ou avoir de la transparence."""
    for mode in ("L", "RGBA", "P"):
        image = Image.new(mode, (300, 120))
        assert set(predicteur.predire(image)["dinov2"]) == set(CATEGORIES)


def contraste(couleur):
    def luminance(c):
        r, g, b = (int(c[i:i + 2], 16) / 255 for i in (1, 3, 5))
        f = lambda v: v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
        return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)
    return 1.05 / (luminance(couleur) + 0.05)


def test_contrastes_wcag():
    """WCAG 1.4.11 : éléments graphiques à 3:1 au moins ; 1.4.3 : couleurs de texte à 4,5:1."""
    assert all(contraste(c) >= 3.0 for c in acc.PALETTE_CATEGORIES)
    assert contraste(acc.BLEU_FONCE) >= 4.5 and contraste(acc.GRIS_FONCE) >= 4.5
    assert contraste(acc.POLICE["color"]) >= 4.5
    assert len(set(acc.MOTIFS)) == len(acc.MOTIFS) == len(CATEGORIES)
