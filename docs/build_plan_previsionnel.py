"""
Génère le livrable 1, le plan prévisionnel, à partir du modèle Word fourni par
OpenClassrooms (docs/modeles/Modèle_Plan_prévisionnel.docx).

On garde l'en-tête du modèle (le logo DataSpace) et ses styles de titres, on
remplace les consignes par le contenu, puis Word exporte le PDF.

Lancement, depuis la racine du projet :
    .venv\\Scripts\\python docs\\build_plan_previsionnel.py
"""

import glob
from pathlib import Path

from docx import Document
from docx.shared import Pt, Cm

RACINE = Path(__file__).resolve().parent.parent
MODELE = glob.glob(str(RACINE / "docs" / "modeles" / "*Plan_pr*.docx"))[0]
SORTIE_DOCX = RACINE / "livrables" / "Wyts_Dimitri_1_plan_travail_102026.docx"

TAILLE_TEXTE = Pt(9.25)


# --- Le contenu --------------------------------------------------------------
# Chaque section est une liste de paragraphes. Un paragraphe est une liste de
# morceaux (texte, gras ?), pour pouvoir mettre quelques mots en évidence.
# Un paragraphe qui commence par "• " est rendu comme une puce.

SECTIONS = {
    "Dataset retenu": [
        [("Le jeu Flipkart, déjà exploré au projet 6 : ", False),
         ("1 050 produits", True),
         (" d'une place de marché indienne, chacun avec une photo, un nom et une "
          "description. La cible est le premier niveau de l'arborescence produit, soit ", False),
         ("7 catégories parfaitement équilibrées", True),
         (" (150 images chacune) : Baby Care, Beauty and Personal Care, Computers, "
          "Home Decor & Festive Needs, Home Furnishing, Kitchen & Dining, Watches. "
          "Je n'utilise que les images, très hétérogènes (de 145 à plus de 11 000 pixels "
          "de côté). Enjeu métier : ranger "
          "automatiquement un nouvel article à partir de sa seule photo.", False)],
    ],
    "Modèle envisagé": [
        [("DINOv2", True),
         (" (Meta AI, avril 2023) est un modèle de fondation pour la vision : un "
          "Vision Transformer pré-entraîné ", False),
         ("sans aucune étiquette", True),
         (", par auto-distillation, sur 142 millions d'images sélectionnées "
          "automatiquement. Il produit une représentation générale de l'image, "
          "utilisable telle quelle (modèle gelé) pour classer, rechercher des images "
          "similaires ou segmenter. Il vise les cas où les données étiquetées sont "
          "rares, comme ici avec quelques centaines d'images par entraînement. J'utilise sa "
          "variante avec registres (même équipe, septembre 2023), qui corrige des artefacts "
          "dans ses cartes d'attention.", False)],
        [("• Plus performant : ", True), ("sur ImageNet, une simple couche linéaire posée sur "
          "DINOv2 ViT-B/14 avec registres gelé atteint 84,6 % de bonnes réponses, contre 71,9 % "
          "pour MobileNetV2 pourtant entraîné de bout en bout (chiffres publiés par "
          "Meta et par torchvision pour les poids utilisés ici). Les auteurs montrent "
          "aussi qu'il dépasse OpenCLIP, la meilleure représentation généraliste "
          "jusque-là, sur la plupart des tests, au niveau de l'image comme du pixel.", False)],
        [("• Adapté au projet : ", True), ("pas de réentraînement du réseau, donc compatible avec "
          "un petit jeu de données et une machine modeste ; poids libres (licence "
          "Apache 2.0) sur Hugging Face.", False)],
        [("• Interprétable : ", True), ("ses cartes d'attention montrent où le modèle regarde, "
          "ce qui servira à expliquer chaque prédiction.", False)],
    ],
    "Références bibliographiques": [
        [("1. Oquab M. et al. (2023), ", False),
         ("DINOv2: Learning Robust Visual Features without Supervision", True),
         (", arXiv:2304.07193 ; illustré par Rafaat Y. (24 mai 2023), « DINOv2: "
          "Self-Supervised Computer Vision Models by Meta AI », KDnuggets.", False)],
        [("2. Caron M. et al. (2021), ", False),
         ("Emerging Properties in Self-Supervised Vision Transformers", True),
         (" (DINO), arXiv:2104.14294 : le principe de l'auto-distillation et des "
          "cartes d'attention.", False)],
        [("3. Darcet T. et al. (2023), ", False),
         ("Vision Transformers Need Registers", True),
         (", arXiv:2309.16588 : l'origine des artefacts d'attention et leur correction "
          "par des registres.", False)],
        [("Code réutilisé : l'exemple d'extraction de la fiche Hugging Face "
          "facebook/dinov2-with-registers-base et le chargement des données de mon projet 6. "
          "L'application, elle, est originale (jeu Flipkart).", False)],
    ],
    "Démarche de test du nouvel algorithme (preuve de concept)": [
        [("• Baseline : ", True),
         ("MobileNetV2 pré-entraîné sur ImageNet, gelé, surmonté d'un classifieur. "
          "C'est l'approche retenue au projet 6 (82,4 % de bonnes réponses), "
          "reproduite dans le même notebook que le nouveau modèle. "
          "J'ajoute ResNet50 (23,5 M de paramètres) et DINOv2 ViT-S/14 avec registres (22,1 M) : à taille "
          "comparable, ils montrent si le gain vient de la méthode et pas seulement de la "
          "taille du modèle.", False)],
        [("• Nouvelle méthode : ", True),
         ("DINOv2 ViT-B/14 avec registres, gelé, surmonté du même classifieur (régression "
          "logistique). Mêmes images, même préparation : seule la représentation "
          "de l'image change, la comparaison est donc équitable.", False)],
        [("• Évaluation : ", True),
         ("validation croisée stratifiée et groupée en 5 plis (les quasi-doublons restent ensemble), les réglages étant choisis à "
          "l'intérieur de chaque pli. Chaque image est prédite une fois par un "
          "modèle qui ne l'a jamais vue, ce qui permet un test de McNemar sur les "
          "1 050 images. Métrique principale : l'exactitude (classes équilibrées), "
          "complétée par le F1 macro, la matrice de confusion, le temps de "
          "prédiction et la taille du modèle.", False)],
        [("• Interprétabilité : ", True),
         ("cartes d'attention et occlusion pour expliquer une image (locale), "
          "moyennes par catégorie (globale).", False)],
        [("• Démonstration : ", True),
         ("dashboard Streamlit déployé sur Streamlit Community Cloud : analyse exploratoire, "
          "choix ou envoi d'une image, prédiction des deux modèles, critères "
          "d'accessibilité WCAG.", False)],
    ],
}


# --- La fabrication du document ----------------------------------------------

def remplir(document):
    """Vide le corps du modèle sous chaque titre et y écrit le contenu."""
    corps = document.element.body
    titre_courant = None
    a_supprimer = []

    for paragraphe in list(document.paragraphs):
        style = paragraphe.style.name
        if style.startswith("Heading"):
            titre_courant = paragraphe.text.strip()
            # Le dernier titre du modèle est un peu long : on le raccourcit
            # (on modifie le texte du premier morceau pour garder sa mise en forme)
            if titre_courant.startswith("Explication"):
                paragraphe.runs[0].text = "Démarche de test du nouvel algorithme (preuve de concept)"
                for run in paragraphe.runs[1:]:
                    run.text = ""
            continue
        if style == "Title":
            continue
        # Les consignes du modèle disparaissent
        a_supprimer.append(paragraphe)

    for paragraphe in a_supprimer:
        corps.remove(paragraphe._element)

    # On insère le contenu juste après chaque titre, dans l'ordre
    for paragraphe in list(document.paragraphs):
        if not paragraphe.style.name.startswith("Heading"):
            continue
        cle = paragraphe.text.strip()
        ancre = paragraphe
        for morceaux in SECTIONS[cle]:
            nouveau = document.add_paragraph()
            ancre._element.addnext(nouveau._element)
            ancre = nouveau
            format_ = nouveau.paragraph_format
            format_.space_after = Pt(2)
            format_.space_before = Pt(0)
            format_.line_spacing = 1.0
            if morceaux[0][0].startswith("• "):
                format_.left_indent = Cm(0.4)
                format_.first_line_indent = Cm(-0.3)
            for texte, gras in morceaux:
                run = nouveau.add_run(texte)
                run.bold = gras
                run.font.size = TAILLE_TEXTE

        # Titres plus compacts pour tenir sur une page
        paragraphe.paragraph_format.space_before = Pt(6)
        paragraphe.paragraph_format.space_after = Pt(2)


def main():
    document = Document(MODELE)

    # Marges un peu réduites (2 cm au lieu de 2,54 cm) pour tenir sur une page
    for section in document.sections:
        section.top_margin = section.bottom_margin = Cm(2)
        section.left_margin = section.right_margin = Cm(2)

    remplir(document)

    SORTIE_DOCX.parent.mkdir(exist_ok=True)
    document.save(SORTIE_DOCX)
    print("Word :", SORTIE_DOCX)

    # Export PDF par Word lui-même, pour un rendu identique au .docx
    from docx2pdf import convert
    sortie_pdf = SORTIE_DOCX.with_suffix(".pdf")
    convert(str(SORTIE_DOCX), str(sortie_pdf))
    print("PDF  :", sortie_pdf)


if __name__ == "__main__":
    main()
