"""
Le moteur de prédiction du dashboard.

La préparation des images reprend exactement celle du notebook (section 2) :
compléter en carré avec du blanc, réduire en 224 × 224, normaliser avec les
statistiques d'ImageNet. Le test test_modele.py vérifie que les représentations
calculées ici sont identiques à celles du notebook.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from torchvision import models as tv_models
from transformers import AutoModel

Image.MAX_IMAGE_PIXELS = None

DOSSIER_MODELE = Path(__file__).resolve().parent / "modele"
CONFIGURATION = json.loads((DOSSIER_MODELE / "configuration.json").read_text(encoding="utf-8"))
CATEGORIES = CONFIGURATION["categories"]
TAILLE = CONFIGURATION["preparation"]["taille"]
MOYENNE = np.array(CONFIGURATION["preparation"]["moyenne"], dtype=np.float32)
ECART_TYPE = np.array(CONFIGURATION["preparation"]["ecart_type"], dtype=np.float32)


# --- La préparation ----------------------------------------------------------

def carre_blanc(image: Image.Image, taille: int = TAILLE) -> Image.Image:
    """Complète l'image en carré avec du blanc, sans la déformer, puis la réduit."""
    image = image.convert("RGB")
    cote = max(image.size)
    fond = Image.new("RGB", (cote, cote), (255, 255, 255))
    fond.paste(image, ((cote - image.width) // 2, (cote - image.height) // 2))
    return fond.resize((taille, taille), Image.BICUBIC)


def vers_tenseur(image: Image.Image) -> torch.Tensor:
    tableau = np.asarray(image, dtype=np.float32) / 255.0
    tableau = (tableau - MOYENNE) / ECART_TYPE
    return torch.from_numpy(tableau.transpose(2, 0, 1))


# --- Les deux modèles --------------------------------------------------------

class Predicteur:
    """DINOv2 gelé + régression logistique, et la baseline MobileNetV2 pour comparer."""

    # Piège rencontré au premier test du dashboard : désactiver les gradients une seule fois
    # dans __init__ (torch.set_grad_enabled(False)) ne vaut que pour le fil d'exécution en
    # cours. Streamlit réexécute la page dans un autre fil à chaque interaction : PyTorch y
    # recalculait les gradients, et .numpy() échouait dès la deuxième prédiction
    # (« Can't call numpy() on Tensor that requires grad »). On désactive donc les gradients
    # à chaque appel, avec le décorateur @torch.no_grad().

    def __init__(self):
        self.dinov2 = AutoModel.from_pretrained(
            CONFIGURATION["nouveau_modele"]["hugging_face"], attn_implementation="eager").eval()
        self.mobilenet = tv_models.mobilenet_v2(
            weights=tv_models.MobileNet_V2_Weights.IMAGENET1K_V1).features.eval()
        # joblib repose sur pickle, qui peut exécuter du code au chargement : on ne
        # charge que les fichiers produits par notre notebook et livrés avec l'application.
        self.classifieur_dinov2 = joblib.load(DOSSIER_MODELE / "classifieur_dinov2.joblib")
        self.classifieur_mobilenet = joblib.load(DOSSIER_MODELE / "classifieur_mobilenetv2.joblib")

    @torch.no_grad()
    def representation_dinov2(self, lot):
        sortie = self.dinov2(pixel_values=lot, output_attentions=True)
        jetons = sortie.last_hidden_state
        X = torch.cat([jetons[:, 0], jetons[:, 1:].mean(dim=1)], dim=1).numpy()
        return X, sortie.attentions[-1]

    @torch.no_grad()
    def predire(self, image: Image.Image) -> dict:
        """Renvoie les probabilités des deux modèles et la carte d'attention de DINOv2."""
        carree = carre_blanc(image)
        lot = vers_tenseur(carree)[None]

        X, attentions = self.representation_dinov2(lot)
        probas_dinov2 = self.classifieur_dinov2.predict_proba(X)[0]

        X_mobile = self.mobilenet(lot).mean(dim=(2, 3)).numpy()
        probas_mobilenet = self.classifieur_mobilenet.predict_proba(X_mobile)[0]

        attention = attentions[0, :, 0, 1:].mean(dim=0)
        cote = int(attention.numel() ** 0.5)
        carte = attention.reshape(cote, cote).numpy()
        carte = carte / carte.max()

        return {
            "image_preparee": carree,
            "dinov2": dict(zip(CATEGORIES, probas_dinov2.tolist())),
            "mobilenet": dict(zip(CATEGORIES, probas_mobilenet.tolist())),
            "attention": carte,
        }


def superposer_attention(image: Image.Image, carte: np.ndarray, opacite: float = 0.55) -> Image.Image:
    """Superpose la carte d'attention (palette « inferno », lisible par les daltoniens) sur l'image."""
    from matplotlib import colormaps
    carte_image = Image.fromarray(np.uint8(carte * 255)).resize(image.size, Image.BILINEAR)
    couleurs = colormaps["inferno"](np.asarray(carte_image) / 255.0)[..., :3]
    couleurs = Image.fromarray(np.uint8(couleurs * 255))
    return Image.blend(image.convert("RGB"), couleurs, opacite)
