"""
Génère le livrable 3, la note méthodologique, à partir du modèle Word fourni par
OpenClassrooms (docs/modeles/Modèle_Note_Méthodologique.docx), puis l'exporte en PDF.

Les chiffres des sections de résultats ne sont pas recopiés à la main : ils sont
lus dans les fichiers exportés par le notebook (dashboard/modele/). Relancer le
notebook puis ce script suffit à mettre la note à jour, sans risque d'écart.

Lancement, depuis la racine du projet :
    .venv\\Scripts\\python docs\\build_note_methodologique.py
"""

import glob
import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.shared import Pt, Cm

RACINE = Path(__file__).resolve().parent.parent
MODELE = glob.glob(str(RACINE / "docs" / "modeles" / "*Note_M*.docx"))[0]
FIGURES = RACINE / "docs" / "figures"
EXPORT = RACINE / "dashboard" / "modele"
SORTIE_DOCX = RACINE / "livrables" / "Wyts_Dimitri_3_note_methodo_102026.docx"
TAILLE = Pt(10.5)

CONFIG = json.loads((EXPORT / "configuration.json").read_text(encoding="utf-8"))
NOUVEAU = CONFIG["nouveau_modele"]["nom"]


def pct(valeur, decimales=1):
    """Pourcentage à la française : 92,6 %."""
    return f"{100 * valeur:.{decimales}f} %".replace(".", ",")


def virgule(valeur, decimales=1):
    """Nombre décimal à la française : 0,7."""
    return f"{valeur:.{decimales}f}".replace(".", ",")


# --- Section 1 : le jeu de données -------------------------------------------

SECTION_1 = [
    "Le jeu de données est celui du projet 6 : 1 050 produits vendus sur Flipkart, une grande place de "
    "marché en ligne indienne. Chaque produit a une photo, un nom, une description et une arborescence de "
    "catégories du type « Home Furnishing >> Curtains & Accessories >> Curtains ». La catégorie à prédire "
    "est le premier niveau de cette arborescence.",
    "Il y a 7 catégories, avec exactement 150 produits chacune : Baby Care, Beauty and Personal Care, "
    "Computers, Home Decor & Festive Needs, Home Furnishing, Kitchen & Dining, Watches. Comme les classes "
    "sont parfaitement équilibrées, un modèle qui répondrait au hasard aurait 1 chance sur 7 (14 %) d'avoir "
    "juste, et la part de bonnes réponses (l'exactitude) est une mesure fiable de la performance.",
    "Je n'utilise que les photos. Le texte du projet 6 n'est pas repris, pour que la comparaison porte sur "
    "une seule question : quel réseau transforme le mieux une image en description exploitable. Les photos "
    "sont très hétérogènes : de 145 à plus de 11 000 pixels de côté, souvent pas carrées, sur fond blanc de "
    "studio ou dans une mise en scène.",
    "Deux choix de préparation découlent de cette exploration. D'abord, les images sont complétées en carré "
    "avec du blanc avant d'être réduites à 224 × 224 pixels, la taille attendue par les réseaux : le projet 6 "
    "les étirait directement, ce qui déformait les produits allongés. Ensuite, j'ai cherché les photos en "
    "double grâce à une empreinte de l'image réduite à 8 × 8 pixels. Il y a 258 paires de quasi-doublons, "
    "des variantes d'un même article, dont 13 rangées dans deux catégories différentes. Les quasi-doublons "
    "sont gardés ensemble lors du découpage en plis de validation (933 groupes pour 1 050 images), pour "
    "qu'un modèle ne soit jamais testé sur une copie d'une image qu'il a apprise.",
    "L'enjeu métier : ranger automatiquement un nouvel article dans la bonne catégorie à partir de sa photo, "
    "pour réduire la saisie manuelle des vendeurs. Chaque erreur se traduit par un produit mal rangé, donc "
    "moins visible pour les acheteurs qui le cherchent : l'exactitude mesure directement la part d'articles "
    "bien rangés.",
]

# --- Section 2 : les concepts de DINOv2 --------------------------------------
# Un tuple (texte,) est un sous-titre en gras.

SECTION_2 = [
    ("Le problème que DINOv2 cherche à résoudre",),
    "Pour classer une image, on passe d'abord par une représentation : un réseau pré-entraîné transforme "
    "l'image en un vecteur de quelques centaines de nombres qui la décrit. Au projet 6, ce réseau était "
    "MobileNetV2, entraîné de façon supervisée sur ImageNet : il a appris à reconnaître 1 000 catégories "
    "d'images à partir de 1,3 million de photos étiquetées à la main. Sa représentation est donc taillée "
    "pour ces 1 000 catégories-là, et étiqueter beaucoup plus d'images coûterait très cher.",
    "DINOv2 (Meta AI, 2023) apprend sa représentation sans aucune étiquette, sur 142 millions d'images. "
    "L'objectif des auteurs est d'obtenir une représentation générale, utilisable telle quelle sur n'importe "
    "quel jeu d'images, sans réentraîner le réseau.",
    ("Un Vision Transformer : l'image découpée en morceaux",),
    "DINOv2 n'est pas un réseau convolutif mais un Vision Transformer (ViT), l'architecture des modèles de "
    "langage adaptée aux images. L'image de 224 × 224 pixels est découpée en une grille de 16 × 16 = 256 "
    "carrés de 14 × 14 pixels, appelés patchs (d'où le nom ViT-B/14). Chaque patch devient un vecteur, "
    "appelé jeton, comme un mot dans une phrase. On ajoute un jeton spécial, [CLS], qui ne correspond à "
    "aucun morceau de l'image et dont le rôle est d'en construire le résumé.",
    "Ces jetons traversent 12 couches. Dans chaque couche, le mécanisme d'attention permet à chaque jeton "
    "de regarder tous les autres et de décider lesquels comptent pour lui : un patch de bracelet peut "
    "s'appuyer sur un patch de cadran situé à l'autre bout de l'image. Un réseau convolutif, lui, ne combine "
    "au départ que des pixels voisins. Le modèle utilisé ici (ViT-B/14) a 12 « têtes » d'attention par "
    "couche, qui regardent chacune l'image à leur façon, et des jetons de 768 nombres.",
    ("Apprendre sans étiquettes : l'auto-distillation",),
    "La méthode reprend DINO (Caron et al., 2021). Deux copies du même réseau travaillent ensemble : un élève "
    "et un professeur. On découpe dans une même photo plusieurs vues : de grands recadrages et de petits "
    "morceaux. Le professeur voit les grandes vues, l'élève voit toutes les vues, et l'élève doit produire "
    "pour son jeton [CLS] la même réponse que le professeur. Pour y arriver, il doit comprendre qu'un petit "
    "morceau de cadran et la montre entière montrent le même objet : il apprend le contenu de l'image, sans "
    "qu'on lui dise jamais ce que c'est.",
    "Le professeur n'est pas entraîné directement : ses poids sont une moyenne glissante des poids de "
    "l'élève au fil de l'entraînement, ce qui le rend plus stable. Le danger de ce type d'apprentissage est "
    "que les deux réseaux s'entendent sur une réponse constante, identique pour toutes les images. DINOv2 "
    "l'empêche en rééquilibrant les réponses du professeur sur chaque lot d'images (une normalisation dite "
    "de Sinkhorn-Knopp), pour qu'elles restent variées.",
    ("Les ajouts de DINOv2",),
    "Au niveau des patchs, DINOv2 ajoute l'objectif d'iBOT : on masque une partie des patchs dans l'image "
    "de l'élève, et il doit retrouver ce que le professeur, qui voit l'image entière, produit pour ces patchs "
    "masqués. C'est ce qui donne au modèle une compréhension locale, patch par patch, et pas seulement "
    "globale. Un terme supplémentaire, appelé régularisation KoLeo, pousse les représentations des images "
    "d'un même lot à s'écarter les unes des autres, pour mieux distinguer des images proches.",
    "Les données comptent autant que la méthode. Les auteurs partent de 1,2 milliard d'images du web, "
    "retirent les doublons, puis ne gardent que les images proches de jeux de données de référence : on "
    "obtient 142 millions d'images variées et de bonne qualité (le jeu LVD-142M). Enfin, ils entraînent un "
    "très grand modèle d'un milliard de paramètres, puis transmettent ses connaissances à des modèles plus "
    "petits par distillation : le petit modèle apprend à imiter le grand. Le ViT-B/14 utilisé ici a été "
    "obtenu ainsi.",
    ("Les registres : corriger les artefacts d'attention",),
    "En analysant DINOv2, ses auteurs ont découvert un défaut (Darcet et al., 2023, « Vision Transformers "
    "Need Registers ») : le modèle détourne quelques patchs de fond peu informatifs pour y stocker des "
    "informations sur l'image entière. Ces patchs reçoivent alors une attention anormale, et les cartes "
    "d'attention ne montrent plus ce que le modèle regarde vraiment. La correction consiste à ajouter quatre "
    "jetons vides, les registres, qui servent de zone de stockage : les patchs de fond retrouvent leur rôle "
    "normal. Je l'ai constaté sur nos propres images : avec la version d'origine, un seul patch de fond, celui "
    "du coin en haut à gauche, captait 15,1 % de l'attention (mesure sur 42 images). Avec les registres, il "
    "n'en reçoit plus que 0,1 %. J'utilise donc DINOv2 avec registres dans tout le projet.",
    ("Ce que cela apporte à ce projet",),
    "Le modèle est utilisé gelé : on ne modifie aucun de ses poids, on calcule seulement la représentation "
    "de chaque image, et on entraîne par-dessus un classifieur simple (une régression logistique). C'est "
    "exactement le cas d'usage visé par les auteurs, et c'est adapté à un petit jeu de 1 050 images qu'un "
    "réentraînement complet risquerait d'apprendre par cœur. Sur ImageNet, un simple classifieur linéaire "
    "posé sur DINOv2 ViT-B/14 avec registres gelé atteint 84,6 % de bonnes réponses, contre 71,9 % pour "
    "MobileNetV2 entraîné de bout en bout.",
    "Le mécanisme d'attention apporte aussi un moyen d'explication : le poids que le jeton [CLS] accorde à "
    "chaque patch montre où le modèle a regardé. Je m'en sers pour analyser les décisions du modèle, en le "
    "complétant par une méthode qui mesure l'effet réel de chaque zone sur la décision (section 5), et le "
    "dashboard affiche cette carte pour chaque prédiction.",
]


# --- Les sections de résultats, écrites à partir des sorties du notebook -----

def sections_resultats():
    import re
    from sklearn.metrics import f1_score

    tableau = pd.read_csv(EXPORT / "tableau_resultats.csv").set_index("modèle")
    tests = pd.read_csv(EXPORT / "tests_mcnemar.csv").set_index("comparaison")
    predictions = pd.read_csv(EXPORT / "predictions_hors_pli.csv")
    base = CONFIG["baseline"]["nom"]
    petit = [n for n in tableau.index if n.startswith("DINOv2") and n != NOUVEAU][0]
    t = lambda a, b: tests.loc[f"{b}  vs  {a}"]
    e = lambda n, col="exactitude": tableau.loc[n, col]

    # Les chiffres de l'occlusion globale sont affichés par une cellule du notebook (section 6) :
    # on les relit dans la sortie de cette cellule plutôt que de les recopier.
    cahier = json.loads((RACINE / "notebooks" / "Wyts_Dimitri_2_notebook_102026.ipynb").read_text(encoding="utf-8"))
    sorties = "\n".join("".join(o.get("text", "")) for c in cahier["cells"] if c["cell_type"] == "code"
                        for o in c.get("outputs", []) if o["output_type"] == "stream")
    occ = {m: tuple(map(float, re.search(rf"{m}\s+importance moyenne au centre ([\d.]+) \| sur les bords ([\d.]+)",
                                         sorties).groups())) for m in ("DINOv2", "MobileNetV2")}

    c_base, c_nouveau = CONFIG["c_choisis"][base], CONFIG["c_choisis"][NOUVEAU]
    section_3 = [
        ("Le principe : une seule chose change",),
        "La comparaison est construite pour qu'une seule chose diffère d'un modèle à l'autre : le réseau qui "
        "transforme l'image en vecteur. Les images, leur préparation, le classifieur posé au-dessus, le "
        "découpage en plis et la recherche du réglage sont identiques. Je compare quatre réseaux gelés : deux "
        f"baselines pré-entraînées de façon supervisée sur ImageNet, {base} (celle du projet 6) et ResNet50, "
        f"et deux tailles de DINOv2 avec registres, {petit} et {NOUVEAU}. ResNet50 et {petit} ont presque la "
        f"même taille ({virgule(e('ResNet50', 'paramètres (M)'))} et {virgule(e(petit, 'paramètres (M)'))} "
        "millions de paramètres) : leur comparaison sert à savoir si le gain vient de la méthode ou seulement "
        "d'un modèle plus gros. Le nouveau modèle retenu, {NOUVEAU}, est fixé à l'avance dans le plan "
        "prévisionnel, et non choisi après coup sur les scores, qui sont mes scores de test.".replace("{NOUVEAU}", NOUVEAU),
        ("Le classifieur et son réglage",),
        "Sur chaque représentation, j'entraîne une régression logistique précédée d'une standardisation des "
        "variables. C'est mathématiquement la même chose que la couche « softmax » du projet 6 : une "
        "combinaison linéaire des variables, transformée en probabilités. Son seul réglage est C, l'inverse de "
        "la force de régularisation : un C petit empêche le modèle d'apprendre par cœur les quelques centaines "
        "d'images d'entraînement. C est choisi parmi 7 valeurs, de 0,001 à 1.",
        ("La validation croisée imbriquée, groupée et stratifiée",),
        "Avec 1 050 images, un seul découpage entraînement / test laisse trop de place au hasard. J'utilise "
        "une validation croisée en 5 plis de 210 images : chaque pli sert de test à son tour, le modèle étant "
        "entraîné sur les 4 autres. Elle est stratifiée (30 images de chaque catégorie par pli) et groupée "
        "(les quasi-doublons restent dans le même pli). Elle est surtout imbriquée : à l'intérieur des 4 plis "
        "d'entraînement, une seconde validation croisée choisit C, et le pli de test n'intervient jamais dans "
        "ce choix. À la fin, chaque image a été prédite une fois, par un modèle qui ne l'avait jamais vue. "
        f"Les valeurs de C retenues d'un pli à l'autre vont de {virgule(min(c_nouveau), 3)} à "
        f"{virgule(max(c_nouveau), 3)} pour {NOUVEAU} : toujours une régularisation forte, cohérente avec le "
        "petit nombre d'images. Toutes les graines aléatoires sont fixées.",
        ("Les métriques",),
        "La métrique principale est l'exactitude, la part d'images bien classées. C'est la métrique métier : "
        "la part d'articles rangés dans la bonne catégorie. Comme les catégories sont équilibrées, elle n'est "
        "pas faussée par une catégorie majoritaire. Je la complète par le F1 macro, qui moyenne la qualité de "
        "chaque catégorie et révèlerait une catégorie sacrifiée, par l'intervalle de confiance à 95 % "
        "(obtenu en retirant 2 000 fois au hasard 1 050 prédictions), et par le temps de prédiction.",
        "Pour savoir si un écart entre deux modèles est réel, j'utilise le test de McNemar. Il ne regarde que "
        "les images sur lesquelles les deux modèles sont en désaccord, et vérifie si l'un réussit nettement "
        "plus souvent là où l'autre échoue. Comme je fais 5 comparaisons, je corrige les p-valeurs par la "
        "méthode de Bonferroni (multiplication par 5), pour ne pas trouver un écart « significatif » par "
        "simple effet du nombre de tests.",
    ]

    resume = [f"{n} : {pct(e(n))} {e(n, 'IC 95 %').replace('.', ',')}, F1 macro {virgule(e(n, 'F1 macro'), 3)}, "
              f"{int(e(n, 'erreurs / 1 050'))} erreurs, {virgule(e(n, 'paramètres (M)'))} M de paramètres, "
              f"{virgule(e(n, 'temps par image, CPU (ms)'), 0)} ms par image"
              for n in tableau.index]
    r1, r2, r3 = t(base, NOUVEAU), t("ResNet50", NOUVEAU), t("ResNet50", petit)
    f1 = pd.DataFrame({m: f1_score(predictions["categorie"], predictions[f"prediction {m}"], average=None,
                                   labels=CONFIG["categories"]) for m in (base, NOUVEAU)}, index=CONFIG["categories"])
    gain = f1[NOUVEAU] - f1[base]
    ic1 = r1["IC 95 % de l'écart"].replace(".", ",")
    # Confusion la plus fréquente du nouveau modèle, calculée sur les prédictions hors pli
    confusion = pd.crosstab(predictions["categorie"], predictions[f"prediction {NOUVEAU}"])
    for c in confusion.columns:
        if c in confusion.index:
            confusion.loc[c, c] = 0
    vraie, predite = confusion.stack().idxmax()
    nb_confusion = int(confusion.stack().max())
    cont = CONFIG["continuite_projet6"]
    section_4 = [
        ("Le tableau de comparaison",),
        *[f"• {ligne}" for ligne in resume],
        ("figure", "comparaison_exactitude.png", "Figure 2 : exactitude en validation croisée ; les barres "
         "noires vont du pli le plus faible au plus fort (notebook, section 5).", 14),
        f"{NOUVEAU} atteint {pct(e(NOUVEAU))} de bonnes réponses, contre {pct(e(base))} pour la baseline du "
        f"projet 6, soit {int(e(base, 'erreurs / 1 050') - e(NOUVEAU, 'erreurs / 1 050'))} erreurs de moins "
        f"sur 1 050 images. L'écart est de {virgule(r1['écart (points)'])} points, avec un intervalle de "
        f"confiance de {ic1} points : il est "
        f"certain (p-valeur corrigée de {r1['p-valeur corrigée (Bonferroni)']:.0e}). {NOUVEAU} bat aussi "
        f"ResNet50, de {virgule(r2['écart (points)'])} points.",
        f"À taille comparable, en revanche, le résultat est plus nuancé : {petit} devance ResNet50 de "
        f"{virgule(r3['écart (points)'])} point, mais avec une p-valeur corrigée de "
        f"{virgule(r3['p-valeur corrigée (Bonferroni)'], 2)}"
        + (", donc l'écart n'est pas distinguable du bruit avec 1 050 images. Je ne peux pas affirmer que la "
           "méthode seule explique le gain : la taille du grand modèle y contribue."
           if r3["significatif au seuil de 5 %"] == "non" else ", donc l'écart est significatif."),
        f"Catégorie par catégorie, le gain de F1 va de +{virgule(100 * gain.min())} points ({gain.idxmin()}, où "
        f"la baseline avait déjà {virgule(f1.loc[gain.idxmin(), base], 2)}) à +{virgule(100 * gain.max())} "
        f"points ({gain.idxmax()}). Aucune catégorie n'est sacrifiée. La confusion la plus fréquente de "
        f"{NOUVEAU} : {vraie} prise pour {predite}, {nb_confusion} fois sur 150.",
        f"Enfin, le découpage unique du projet 6 (210 images de test) donne {pct(cont[base])} à {base} et "
        f"{pct(cont['ResNet50'])} à ResNet50, alors que la validation croisée les classe dans l'ordre inverse. "
        "Un seul découpage ne suffit pas pour comparer des modèles proches : c'est ce qui justifie la "
        "validation croisée groupée.",
        ("Conclusion",),
        f"La preuve de concept est concluante : {NOUVEAU} classe nettement mieux les produits que la baseline "
        "du projet 6, sans réentraîner le réseau et avec un classifieur très simple. Le gain se paie en "
        f"calcul : {virgule(e(NOUVEAU, 'temps par image, CPU (ms)'), 0)} ms par image contre "
        f"{virgule(e(base, 'temps par image, CPU (ms)'), 0)} ms pour {base} sur mon ordinateur portable, ce qui "
        "reste acceptable pour classer des articles au fil de leur mise en ligne.",
    ]

    ig = CONFIG["interpretabilite"]
    section_5 = [
        "Pour une image, les « features » ne sont pas des colonnes comme dans un tableau : ce sont des zones "
        "de l'image. J'utilise deux méthodes complémentaires.",
        ("Importance locale : une image à la fois",),
        "La première est l'attention du jeton [CLS] dans la dernière couche de DINOv2 : le poids qu'il accorde "
        "à chaque patch indique où le modèle a regardé pour construire son résumé de l'image. Elle ne coûte "
        "rien à calculer, mais elle ne dit pas si la zone a poussé la décision vers la bonne catégorie. La "
        "seconde est l'occlusion : je masque successivement chaque zone de l'image avec un carré gris de 56 "
        "pixels et je mesure la chute de la probabilité de la catégorie prédite. Elle mesure directement "
        "l'effet sur la décision du modèle complet, et s'applique aussi à la baseline.",
        ("figure", "importance_locale.png", "Figure 3 : pour une image par catégorie, l'attention de DINOv2 puis "
         "l'occlusion pour DINOv2 et pour MobileNetV2 ; rouge : zone qui soutient la décision (notebook, "
         "section 6).", 10),
        "Avec les registres, l'attention se pose sur les objets eux-mêmes. Un exemple parlant : pour un petit "
        "ventilateur classé dans Computers, l'attention se concentre sur sa prise USB, la seule partie de "
        "l'objet qui le relie à un ordinateur.",
        ("Importance globale : en moyenne",),
        f"Sur les 1 050 images, le produit occupe {pct(ig['surface_produit'], 0)} de la surface et reçoit "
        f"{pct(ig['attention_sur_produit'], 0)} de l'attention de DINOv2 : le modèle regarde le produit "
        "plus que le fond. L'occlusion moyenne sur 35 images montre une différence nette entre les deux "
        f"modèles : masquer le centre de l'image fait chuter la confiance de MobileNetV2 de "
        f"{virgule(100 * occ['MobileNetV2'][0])} points en moyenne, contre {virgule(100 * occ['DINOv2'][0])} "
        "pour DINOv2. La décision de MobileNetV2 dépend fortement de la zone centrale ; celle de DINOv2 "
        "s'appuie sur l'ensemble de l'image, ce qui la rend plus robuste à un masquage partiel.",
        ("figure", "importance_globale_occlusion.png", "Figure 4 : chute moyenne de la probabilité quand on "
         "masque chaque zone, sur 35 images (notebook, section 6).", 11),
    ]

    section_6 = [
        ("Les limites",),
        "• Le jeu est petit : 1 050 images suffisent pour un écart de 10 points, pas pour départager deux "
        f"modèles proches, comme {petit} et ResNet50.",
        f"• Les étiquettes sont imparfaites : {CONFIG['doublons']['paires_categories_differentes']} paires de "
        "photos quasi identiques sont rangées dans deux catégories différentes, ce qui plafonne tout modèle.",
        "• Le coût de calcul est plus élevé que celui de la baseline, et les temps ont été mesurés sur un "
        "seul ordinateur portable, sensibles à sa charge.",
        "• L'occlusion explique les décisions zone par zone, avec des carrés de 56 pixels : elle ne descend "
        "pas au niveau des détails fins, et l'analyse globale porte sur 35 images seulement.",
        ("Les améliorations possibles",),
        "• Performance : affiner les dernières couches de DINOv2 sur les images Flipkart, avec une carte "
        "graphique et de l'augmentation de données ; combiner la photo et le texte de la description, que le "
        "projet 6 avait montré informatif ; corriger les étiquettes en double.",
        "• Interprétabilité : remplacer l'occlusion par des méthodes plus fines comme les valeurs de Shapley "
        "sur des groupes de patchs ; montrer dans le dashboard les images du catalogue les plus proches de "
        "l'image analysée, une explication par l'exemple que la représentation de DINOv2 rend possible.",
    ]
    return section_3, section_4, section_5, section_6


# --- La fabrication du document ----------------------------------------------

def ajouter(document, ancre, texte, gras=False, taille=TAILLE, avant=0, apres=4):
    paragraphe = document.add_paragraph()
    ancre._element.addnext(paragraphe._element)
    run = paragraphe.add_run(texte)
    run.bold = gras
    run.font.size = taille
    paragraphe.paragraph_format.space_before = Pt(avant)
    paragraphe.paragraph_format.space_after = Pt(apres)
    return paragraphe


def ajouter_image(document, ancre, fichier, legende, largeur=16):
    paragraphe = document.add_paragraph()
    ancre._element.addnext(paragraphe._element)
    paragraphe.add_run().add_picture(str(FIGURES / fichier), width=Cm(largeur))
    return ajouter(document, paragraphe, legende, taille=Pt(9))


def remplir_section(document, ancre, elements):
    """Un tuple d'un élément est un sous-titre ; ("figure", fichier, légende, largeur) une image."""
    for element in elements:
        if isinstance(element, tuple) and len(element) == 1:
            ancre = ajouter(document, ancre, element[0], gras=True, avant=6, apres=2)
        elif isinstance(element, tuple) and element[0] == "figure":
            ancre = ajouter_image(document, ancre, *element[1:])
        else:
            ancre = ajouter(document, ancre, element)
    return ancre


def construire(sections):
    """sections : liste de listes d'éléments, une par titre du modèle, dans l'ordre."""
    document = Document(MODELE)
    corps = document.element.body
    paragraphes = document.paragraphs
    titres = [p for p in paragraphes if p.style.name.startswith("Heading")
              or (p.text.strip() and p.style.name not in ("Title",) and not p.text.startswith("Présentez"))]
    # Le modèle mélange de vrais titres (Heading2) et des titres en texte simple : on garde
    # les 6 intitulés de section et on supprime les consignes (« Présentez ... ») et les lignes vides.
    intitules = [p for p in titres if not p.text.startswith("Présentez")][:6]
    garder = {id(p._element) for p in paragraphes if p.style.name == "Title"} | {id(p._element) for p in intitules}
    for p in paragraphes:
        if id(p._element) not in garder:
            corps.remove(p._element)
    for titre in intitules:
        titre.style = document.styles["Heading 2"]
    for titre, elements in zip(intitules, sections):
        remplir_section(document, titre, elements)
    return document


if __name__ == "__main__":
    import sys
    sections = [SECTION_1 + [("figure", "eda_exemples.png", "Figure 1 : une photo par catégorie (notebook, "
                              "section 1).", 16)],
                SECTION_2, *sections_resultats()]
    sortie = Path(sys.argv[1]) if len(sys.argv) > 1 else SORTIE_DOCX
    document = construire(sections)
    document.save(sortie)
    from docx2pdf import convert
    convert(str(sortie), str(sortie.with_suffix(".pdf")))
    print("PDF :", sortie.with_suffix(".pdf"))
