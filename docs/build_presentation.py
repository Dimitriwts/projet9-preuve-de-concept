"""
Génère le livrable 5, le support de soutenance (PowerPoint), puis l'exporte en PDF.

Comme pour la note, aucun chiffre n'est recopié à la main : ils sont lus dans les
fichiers exportés par le notebook (dashboard/modele/). Le .pptx peut être importé
dans Google Slides pour être retouché (Fichier > Importer des diapositives).

Déroulé calé sur la soutenance (20 minutes) : plan prévisionnel (5 min),
démarche et résultats (10 min), dashboard (5 min).

Lancement, depuis la racine du projet :
    .venv\\Scripts\\python docs\\build_presentation.py
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Cm, Pt

RACINE = Path(__file__).resolve().parent.parent
FIGURES = RACINE / "docs" / "figures"
EXPORT = RACINE / "dashboard" / "modele"
SORTIE = RACINE / "livrables" / "Wyts_Dimitri_5_presentation_102026.pptx"
URL_DASHBOARD = "https://flipkart-dinov2-poc.streamlit.app/"
URL_DEPOT = "https://github.com/Dimitriwts/projet9-preuve-de-concept"

# --- Les chiffres, lus dans les sorties du notebook ---------------------------
CONFIG = json.loads((EXPORT / "configuration.json").read_text(encoding="utf-8"))
TABLEAU = pd.read_csv(EXPORT / "tableau_resultats.csv").set_index("modèle")
TESTS = pd.read_csv(EXPORT / "tests_mcnemar.csv").set_index("comparaison")
PREDICTIONS = pd.read_csv(EXPORT / "predictions_hors_pli.csv")
NOUVEAU = CONFIG["nouveau_modele"]["nom"]
BASE = CONFIG["baseline"]["nom"]
PETIT = [n for n in TABLEAU.index if n.startswith("DINOv2") and n != NOUVEAU][0]


def pct(v, d=1):
    return f"{100 * v:.{d}f} %".replace(".", ",")


def virgule(v, d=1):
    return f"{v:.{d}f}".replace(".", ",")


def test(a, b):
    """Ligne du test de McNemar « b vs a » (b est le modèle censé gagner)."""
    return TESTS.loc[f"{b}  vs  {a}"]


# --- La mise en forme ---------------------------------------------------------
BLEU = RGBColor(0x0B, 0x4F, 0x8A)
NOIR = RGBColor(0x1A, 0x1A, 0x1A)
GRIS = RGBColor(0x5A, 0x5A, 0x5A)
POLICE = "Arial"          # disponible partout, y compris dans Google Slides

prs = Presentation()
prs.slide_width, prs.slide_height = Cm(33.867), Cm(19.05)       # format 16:9
VIDE = prs.slide_layouts[6]
numero = 0


def texte(slide, x, y, l, h, lignes, taille=18, couleur=NOIR, gras=False, align=PP_ALIGN.LEFT):
    """lignes : liste de chaînes ; une ligne qui commence par '• ' devient une puce."""
    cadre = slide.shapes.add_textbox(Cm(x), Cm(y), Cm(l), Cm(h)).text_frame
    cadre.word_wrap = True
    for k, ligne in enumerate(lignes):
        p = cadre.paragraphs[0] if k == 0 else cadre.add_paragraph()
        p.alignment = align
        p.space_after = Pt(6)
        run = p.add_run()
        run.text = ligne
        run.font.size, run.font.name, run.font.bold = Pt(taille), POLICE, gras
        run.font.color.rgb = couleur
    return cadre


def nouvelle(titre, sous_titre=None):
    global numero
    numero += 1
    slide = prs.slides.add_slide(VIDE)
    texte(slide, 1.5, 0.8, 30, 1.8, [titre], taille=30, couleur=BLEU, gras=True)
    if sous_titre:
        texte(slide, 1.5, 2.5, 30, 1, [sous_titre], taille=16, couleur=GRIS)
    texte(slide, 30.5, 17.8, 2.5, 0.8, [str(numero)], taille=11, couleur=GRIS, align=PP_ALIGN.RIGHT)
    return slide


def image(slide, fichier, x, y, largeur=None, hauteur=None):
    kwargs = {}
    if largeur:
        kwargs["width"] = Cm(largeur)
    if hauteur:
        kwargs["height"] = Cm(hauteur)
    slide.shapes.add_picture(str(FIGURES / fichier), Cm(x), Cm(y), **kwargs)


def tableau(slide, df, x, y, largeur, taille=14):
    lignes, colonnes = df.shape[0] + 1, df.shape[1] + 1
    forme = slide.shapes.add_table(lignes, colonnes, Cm(x), Cm(y), Cm(largeur), Cm(0.9 * lignes)).table
    valeurs = [[df.index.name or ""] + list(df.columns)] + [[i] + list(r) for i, r in zip(df.index, df.values)]
    for i, rangee in enumerate(valeurs):
        for j, v in enumerate(rangee):
            cellule = forme.cell(i, j)
            cellule.text = str(v)
            for p in cellule.text_frame.paragraphs:
                for run in p.runs:
                    run.font.size, run.font.name = Pt(taille), POLICE
                    run.font.bold = i == 0


# =============================================================================
# Partie 1 : le plan prévisionnel (5 minutes)
# =============================================================================

s = nouvelle("Classer un produit à partir de sa photo")
texte(s, 1.5, 4.5, 30, 6, [
    "Preuve de concept : DINOv2 (Meta AI, 2023) contre la baseline du projet 6",
    "",
    "Dimitri Wyts, projet 9 « Développez une preuve de concept », OpenClassrooms",
    f"Dashboard : {URL_DASHBOARD}",
    f"Code : {URL_DEPOT}",
], taille=20)

s = nouvelle("La mission", "Test technique pour DataSpace, demandé par Karim")
texte(s, 1.5, 4.2, 30, 12, [
    "• Faire une veille et trouver une méthode de moins de 5 ans, publiée sur arXiv ou un site reconnu",
    "• Montrer sur un jeu de données qu'elle fait mieux qu'une méthode plus ancienne",
    "• Livrer : plan prévisionnel, notebook, note méthodologique, dashboard déployé, cette présentation",
    "",
    "Mon choix : reprendre le classement de produits Flipkart du projet 6",
    "• Enjeu métier : ranger automatiquement un nouvel article dans sa catégorie, à partir de sa photo",
    "• Métrique métier : l'exactitude, c'est-à-dire la part d'articles bien rangés",
], taille=18)

d = CONFIG["doublons"]
s = nouvelle("Le jeu de données : Flipkart", "1 050 produits, 7 catégories de 150 images, seules les photos sont utilisées")
image(s, "eda_exemples.png", 1.5, 4.0, largeur=30.8)
texte(s, 1.5, 9.8, 30, 8, [
    "• Classes parfaitement équilibrées : répondre au hasard donne 1 chance sur 7 (14 %)",
    "• Photos très hétérogènes : de 145 à plus de 11 000 pixels de côté, souvent pas carrées",
    f"• {d['paires']} paires de quasi-doublons, dont {d['paires_categories_differentes']} rangées dans deux "
    "catégories différentes : la même photo porte deux étiquettes",
], taille=18)

s = nouvelle("La veille : pourquoi DINOv2", "Trois articles de recherche, le premier illustré par un article de blog")
texte(s, 1.5, 4.2, 30, 13, [
    "• Oquab et al. (2023), DINOv2, arXiv:2304.07193, illustré par KDnuggets (mai 2023)",
    "• Caron et al. (2021), DINO, arXiv:2104.14294 : l'auto-distillation et les cartes d'attention",
    "• Darcet et al. (2023), Vision Transformers Need Registers, arXiv:2309.16588",
    "",
    "Pourquoi ce modèle :",
    "• Appris sans étiquettes sur 142 millions d'images : une représentation générale, utilisable gelée",
    "• Sur ImageNet, un classifieur linéaire posé dessus atteint 84,6 %, contre 71,9 % pour MobileNetV2",
    "• Pas de réentraînement : adapté à 1 050 images et à un ordinateur sans carte graphique",
], taille=18)

s = nouvelle("La démarche de test", "Une seule chose change d'un modèle à l'autre : le réseau qui décrit l'image")
texte(s, 1.5, 4.2, 30, 13, [
    "• Mêmes images, même préparation, même classifieur (régression logistique), mêmes plis",
    f"• Baselines : {BASE} (celle du projet 6) et ResNet50, pré-entraînés de façon supervisée sur ImageNet",
    f"• Nouvelle méthode : {NOUVEAU}, plus {PETIT} pour comparer à taille égale avec ResNet50",
    "• Validation croisée en 5 plis, stratifiée, groupée (les doublons restent ensemble) et imbriquée "
    "(le réglage est choisi sans jamais regarder les images de test)",
    "• Chaque image est prédite une fois par un modèle qui ne l'a jamais vue : test de McNemar sur 1 050 images",
], taille=18)

# =============================================================================
# Partie 2 : la démarche et les résultats (10 minutes)
# =============================================================================

s = nouvelle("Préparer les images sans les déformer")
image(s, "preparation_image.png", 1.5, 3.6, largeur=19)
texte(s, 21.5, 4.5, 11, 12, [
    "Au projet 6, les images étaient étirées en 224 × 224 : les produits allongés étaient déformés.",
    "",
    "Ici, l'image est complétée en carré avec du blanc, la couleur du fond des photos de catalogue, "
    "puis réduite.",
    "",
    "Cette préparation est identique pour les quatre modèles et dans le dashboard.",
], taille=16)

s = nouvelle("Comment fonctionne DINOv2", "Un Vision Transformer appris sans étiquettes")
texte(s, 1.5, 4.2, 30, 13, [
    "• L'image est découpée en 256 patchs de 14 × 14 pixels, chacun devient un jeton, plus un jeton [CLS] "
    "qui résume l'image",
    "• L'attention permet à chaque jeton de regarder tous les autres, même à l'autre bout de l'image",
    "• Auto-distillation : un élève doit donner la même réponse qu'un professeur sur deux vues différentes "
    "de la même photo. Il apprend ainsi le contenu sans étiquette",
    "• iBOT : l'élève retrouve des patchs masqués, ce qui lui donne une compréhension locale",
    "• Données : 142 millions d'images triées à partir de 1,2 milliard",
    "• Le modèle utilisé est une version distillée (86,6 millions de paramètres) d'un modèle d'un milliard",
], taille=18)

registres = {r["registres"]: r for r in CONFIG["registres"]}
sans, avec = registres[0], registres[max(registres)]
s = nouvelle("Un piège trouvé en route : les artefacts d'attention")
texte(s, 1.5, 3.6, 30, 4, [
    "Avec la version d'origine, les cartes d'attention montraient surtout le coin de l'image et des points du "
    "fond. C'est un défaut connu de DINOv2 (Darcet et al., 2023), corrigé par 4 jetons « registres ».",
], taille=17)
tableau(s, pd.DataFrame({
    "sans registres": [pct(sans["attention sur le produit"]), pct(sans["attention sur le coin haut-gauche"]),
                       pct(sans["attention sur les 5 patchs les plus regardés"])],
    "avec registres": [pct(avec["attention sur le produit"]), pct(avec["attention sur le coin haut-gauche"]),
                       pct(avec["attention sur les 5 patchs les plus regardés"])],
}, index=pd.Index([f"Attention sur le produit (surface : {pct(sans['surface du produit'], 0)})",
                   "Attention sur le seul patch du coin", "Attention sur les 5 patchs les plus regardés"],
                  name="Mesure sur 42 images")), 1.5, 7.8, 30, taille=16)
texte(s, 1.5, 14.0, 30, 2, ["Décision : utiliser DINOv2 avec registres dans tout le projet."], taille=18, gras=True)

t = TABLEAU
s = nouvelle("Les résultats", "Exactitude en validation croisée, sur les 1 050 images")
image(s, "comparaison_exactitude.png", 1.5, 3.6, largeur=18)
texte(s, 20.5, 4.0, 12, 13, [
    f"{NOUVEAU} : {pct(t.loc[NOUVEAU, 'exactitude'])}",
    f"{BASE} : {pct(t.loc[BASE, 'exactitude'])}",
    "",
    f"Soit {int(t.loc[BASE, 'erreurs / 1 050'] - t.loc[NOUVEAU, 'erreurs / 1 050'])} erreurs de moins "
    f"sur 1 050 images ({int(t.loc[BASE, 'erreurs / 1 050'])} contre {int(t.loc[NOUVEAU, 'erreurs / 1 050'])}).",
    "",
    "Les barres noires montrent le pli le plus faible et le plus fort : l'écart dépasse largement les "
    "variations d'un pli à l'autre.",
], taille=17)

s = nouvelle("L'écart est-il dû au hasard ?", "Test de McNemar, corrigé pour 5 comparaisons (Bonferroni)")
lignes = []
for a, b in ((BASE, NOUVEAU), ("ResNet50", NOUVEAU), ("ResNet50", PETIT), (PETIT, NOUVEAU)):
    r = test(a, b)
    p = r["p-valeur corrigée (Bonferroni)"]
    p_texte = virgule(p, 2) if p >= 0.01 else f"{p:.0e}".replace("e-0", "e-").replace("e-", " × 10^-")
    court = lambda nom: nom.replace("DINOv2-reg ", "DINOv2 ")
    lignes.append([f"{court(b)} contre {court(a)}", f"{virgule(r['écart (points)'])} pts",
                   r["IC 95 % de l'écart"].replace(".", ","), p_texte, r["significatif au seuil de 5 %"]])
tableau(s, pd.DataFrame([l[1:] for l in lignes], index=pd.Index([l[0] for l in lignes], name="Comparaison"),
                        columns=["écart", "IC 95 %", "p corrigée", "significatif"]), 1.5, 3.6, 30.5, taille=14)
r = test("ResNet50", PETIT)
texte(s, 1.5, 11.6, 30, 7, [
    f"• Le gain sur la baseline du projet 6 est net et certain.",
    f"• À taille comparable ({PETIT} contre ResNet50, environ 22 et 23 millions de paramètres), DINOv2 est "
    f"devant de {virgule(r['écart (points)'])} point, mais l'écart n'est {'pas ' if r['significatif au seuil de 5 %'] == 'non' else ''}"
    f"distinguable du bruit avec 1 050 images.",
    "• Je ne peux donc pas affirmer que la méthode seule explique le gain : le grand modèle y contribue.",
], taille=17)

par_classe = pd.DataFrame({
    BASE: f1_score(PREDICTIONS["categorie"], PREDICTIONS[f"prediction {BASE}"], average=None,
                   labels=CONFIG["categories"]),
    NOUVEAU: f1_score(PREDICTIONS["categorie"], PREDICTIONS[f"prediction {NOUVEAU}"], average=None,
                      labels=CONFIG["categories"]),
}, index=pd.Index(CONFIG["categories"], name="F1 par catégorie"))
s = nouvelle("Catégorie par catégorie")
image(s, "matrices_confusion.png", 1.5, 2.9, largeur=19.5)
tableau(s, par_classe.map(lambda v: virgule(v, 2)), 21.5, 3.2, 11, taille=12)
gains = (par_classe[NOUVEAU] - par_classe[BASE])
texte(s, 21.5, 14.2, 11, 4, [
    f"Gain le plus fort : {gains.idxmax()} (+{virgule(100 * gains.max())} points de F1).",
    f"Gain le plus faible : {gains.idxmin()} (+{virgule(100 * gains.min())} points), "
    f"où la baseline avait déjà un F1 de {virgule(par_classe.loc[gains.idxmin(), BASE], 2)}.",
], taille=14)

cont = CONFIG["continuite_projet6"]
s = nouvelle("Pourquoi pas un simple découpage 80/20 ?", "Le découpage du projet 6, reproduit à l'identique")
tableau(s, pd.DataFrame({
    "découpage unique du projet 6 (210 images)": [pct(cont[n]) for n in TABLEAU.index],
    "validation croisée groupée (1 050 images)": [pct(TABLEAU.loc[n, "exactitude"]) for n in TABLEAU.index],
}, index=pd.Index(TABLEAU.index, name="Modèle")), 1.5, 3.8, 30, taille=16)
texte(s, 1.5, 9.5, 30, 8, [
    "• Sur 210 images, le hasard du découpage pèse lourd, et les doublons peuvent tomber des deux côtés.",
    "• La validation croisée groupée prédit chaque image une fois, sans doublon à cheval : c'est la mesure "
    "retenue pour tous les chiffres de cette présentation.",
], taille=17)

s = nouvelle("Ce qu'il coûte", "Le gain se paie en calcul")
tableau(s, pd.DataFrame({
    "paramètres (millions)": [virgule(TABLEAU.loc[n, "paramètres (M)"]) for n in TABLEAU.index],
    "temps par image sur processeur (ms)": [virgule(TABLEAU.loc[n, "temps par image, CPU (ms)"], 0) for n in TABLEAU.index],
    "fois plus lent que MobileNetV2": [virgule(TABLEAU.loc[n, "fois plus lent que MobileNetV2"]) for n in TABLEAU.index],
    "exactitude": [pct(TABLEAU.loc[n, "exactitude"]) for n in TABLEAU.index],
}, index=pd.Index(TABLEAU.index, name="Modèle")), 1.5, 3.8, 30, taille=16)
texte(s, 1.5, 9.5, 30, 8, [
    "• Pour un catalogue qui reçoit des articles au fil de l'eau, quelques centaines de millisecondes par photo "
    "restent acceptables, sur un serveur sans carte graphique.",
    f"• {NOUVEAU} est environ {TABLEAU.loc[NOUVEAU, 'fois plus lent que MobileNetV2']:.0f} fois plus lent que "
    f"{BASE}. Temps médian de 40 mesures sur mon portable, 6 cœurs, modèles mesurés en alternance : "
    "les valeurs absolues dépendent de la machine, le rapport est stable.",
], taille=17)

s = nouvelle("Où regarde le modèle ? Une image à la fois", "Importance locale : attention et occlusion")
image(s, "importance_locale.png", 1.5, 3.4, hauteur=15.2)
texte(s, 11.5, 4.0, 21, 13, [
    "• Attention : le poids que le jeton [CLS] donne à chaque patch.",
    "• Occlusion : on masque une zone et on mesure la chute de la probabilité de la catégorie prédite.",
    "• L'occlusion s'applique aussi à la baseline : on compare les deux modèles sur la même image.",
], taille=16)

ig = CONFIG["interpretabilite"]
s = nouvelle("Où regarde le modèle ? En moyenne", "Importance globale")
image(s, "importance_globale_occlusion.png", 1.5, 3.6, largeur=15)
image(s, "importance_globale_attention.png", 1.5, 11.0, largeur=30)
texte(s, 17.5, 4.0, 15, 7, [
    f"• Le produit occupe {pct(ig['surface_produit'], 0)} de l'image et reçoit "
    f"{pct(ig['attention_sur_produit'], 0)} de l'attention de DINOv2.",
    "• Masquer une zone fait beaucoup moins chuter la confiance de DINOv2 que celle de MobileNetV2 : "
    "sa décision repose sur l'ensemble de l'image.",
], taille=15)

s = nouvelle("Les erreurs restantes", f"{int(TABLEAU.loc[NOUVEAU, 'erreurs / 1 050'])} erreurs sur 1 050")
image(s, "erreurs_dinov2.png", 1.5, 3.4, largeur=22)
confusion = pd.crosstab(PREDICTIONS["categorie"], PREDICTIONS[f"prediction {NOUVEAU}"])
for c in confusion.columns:
    if c in confusion.index:
        confusion.loc[c, c] = 0
vraie, predite = confusion.stack().idxmax()
texte(s, 24.5, 4.0, 8.5, 13, [
    f"Confusion la plus fréquente : {vraie} prise pour {predite} ({int(confusion.stack().max())} fois sur 150).",
    "",
    f"Une partie vient des étiquettes elles-mêmes : {CONFIG['doublons']['paires_categories_differentes']} "
    "paires de photos quasi identiques sont rangées dans deux catégories différentes.",
], taille=15)

# =============================================================================
# Partie 3 : le dashboard (5 minutes)
# =============================================================================

s = nouvelle("Le dashboard", URL_DASHBOARD)
texte(s, 1.5, 4.2, 30, 13, [
    "• Page 1, le jeu de données : comptage par catégorie, statistiques de taille, exemples, "
    "transformations (égalisation, floutage)",
    "• Page 2, la prédiction : une image du catalogue ou une photo envoyée, probabilités des deux modèles, "
    "carte d'attention",
    "• Page 3, les performances : exactitude, test de McNemar, matrices de confusion",
    "",
    "• Streamlit, déployé sur Streamlit Community Cloud depuis le dépôt GitHub",
    "• La préparation des images est vérifiée identique à celle du notebook (même prédiction sur 30 images)",
], taille=18)

s = nouvelle("Accessibilité : les critères WCAG couverts")
texte(s, 1.5, 4.2, 30, 13, [
    "• 1.1.1 : chaque graphique a une phrase « À retenir » et son tableau de données",
    "• 1.4.1 : la couleur n'est jamais seule (motifs sur les barres, valeurs écrites, résultats en toutes lettres)",
    "• 1.4.3 et 1.4.11 : contrastes vérifiés par un test, palette lisible par les daltoniens",
    "• 2.1.1 : uniquement des composants natifs, utilisables au clavier",
    "• 3.1.1 : la langue de la page est déclarée en français pour les lecteurs d'écran",
], taille=18)

s = nouvelle("Démonstration")
texte(s, 1.5, 6.5, 30, 4, [URL_DASHBOARD], taille=28, couleur=BLEU, align=PP_ALIGN.CENTER)

# =============================================================================
# Conclusion
# =============================================================================

s = nouvelle("Les limites")
texte(s, 1.5, 4.2, 30, 13, [
    "• 1 050 images : assez pour un écart de 10 points, pas pour départager deux modèles proches",
    "• Des étiquettes imparfaites : la même photo rangée dans deux catégories plafonne tout modèle",
    "• Un coût de calcul plus élevé que la baseline",
    "• Les modèles du dashboard sont entraînés sur tout le catalogue : pour une image du catalogue, la "
    "prédiction honnête est celle de la validation croisée, affichée à côté",
], taille=18)

s = nouvelle("Conclusion et suites possibles")
texte(s, 1.5, 4.2, 30, 13, [
    f"• {NOUVEAU} classe mieux les produits que la baseline du projet 6 : "
    f"{pct(TABLEAU.loc[NOUVEAU, 'exactitude'])} contre {pct(TABLEAU.loc[BASE, 'exactitude'])}, écart certain",
    "• Suites : affiner les dernières couches du réseau sur une carte graphique, combiner la photo et le "
    "texte de la description, corriger les étiquettes en double",
], taille=18)

SORTIE.parent.mkdir(exist_ok=True)
prs.save(SORTIE)
print(f"{numero} slides :", SORTIE)

# Export PDF par PowerPoint lui-même (pywin32 est installé avec docx2pdf). 32 = format PDF.
import win32com.client

powerpoint = win32com.client.Dispatch("PowerPoint.Application")
document = powerpoint.Presentations.Open(str(SORTIE), WithWindow=False)
document.SaveAs(str(SORTIE.with_suffix(".pdf")), 32)
document.Close()
print("PDF :", SORTIE.with_suffix(".pdf"))
