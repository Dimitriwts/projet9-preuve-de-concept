"""
Dashboard de la preuve de concept : classer un produit Flipkart à partir de sa photo.

Trois pages :
1. Le jeu de données : l'analyse exploratoire des images.
2. Prédire une catégorie : choisir ou envoyer une image, voir la prédiction.
3. Les performances : la comparaison entre la baseline et DINOv2.

Lancement en local, depuis la racine du projet (pour que le thème accessible de .streamlit/config.toml soit lu) :
    streamlit run dashboard/app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageFilter, ImageOps

import accessibilite as acc


def pct(valeur, decimales=1):
    """Pourcentage écrit à la française : virgule décimale et espace avant le signe (92,6 %)."""
    return f"{100 * valeur:.{decimales}f} %".replace(".", ",")


def nombre(valeur, decimales=1):
    """Nombre décimal écrit à la française (0,7 et non 0.7)."""
    return f"{valeur:.{decimales}f}".replace(".", ",")

ICI = Path(__file__).resolve().parent
DOSSIER_MINIATURES = ICI / "donnees" / "miniatures"
DOSSIER_MODELE = ICI / "modele"

st.set_page_config(page_title="Flipkart : classer un produit par sa photo", layout="wide")
acc.declarer_langue()


# --- Les données, chargées une seule fois -------------------------------------

@st.cache_data
def charger_catalogue():
    catalogue = pd.read_csv(DOSSIER_MODELE / "predictions_hors_pli.csv")
    catalogue["ratio"] = (catalogue["largeur"] / catalogue["hauteur"]).round(2)
    return catalogue


@st.cache_resource(show_spinner="Chargement des modèles (DINOv2 et MobileNetV2)…")
def charger_predicteur():
    from modele import Predicteur
    return Predicteur()


def miniature(nom_image: str) -> Image.Image:
    return Image.open(DOSSIER_MINIATURES / nom_image).convert("RGB")


catalogue = charger_catalogue()
CATEGORIES = sorted(catalogue["categorie"].unique())
COULEUR = dict(zip(CATEGORIES, acc.PALETTE_CATEGORIES))
MOTIF = dict(zip(CATEGORIES, acc.MOTIFS))
COL_NOUVEAU = [c for c in catalogue.columns if c.startswith("prediction DINOv2")][0]
NOM_NOUVEAU = COL_NOUVEAU.removeprefix("prediction ")
NOM_BASELINE = "MobileNetV2"


# =============================================================================
# Page 1 : le jeu de données
# =============================================================================

def page_donnees():
    st.title("Le jeu de données Flipkart")
    st.markdown(
        "1 050 produits d'une place de marché indienne, chacun avec sa photo. "
        "La catégorie à prédire est le premier niveau de l'arborescence produit.")

    colonnes = st.columns(4)
    colonnes[0].metric("Images", f"{len(catalogue):,}".replace(",", " "))
    colonnes[1].metric("Catégories", len(CATEGORIES))
    colonnes[2].metric("Images par catégorie", int(catalogue["categorie"].value_counts().median()))
    colonnes[3].metric("Taille médiane", f"{int(catalogue['largeur'].median())} px")

    # --- Comptage par catégorie ---
    st.header("Combien d'images par catégorie ?")
    comptage = catalogue["categorie"].value_counts().rename_axis("catégorie").reset_index(name="images")
    figure = px.bar(comptage, x="images", y="catégorie", orientation="h", text="images",
                    color="catégorie", color_discrete_map=COULEUR,
                    pattern_shape="catégorie", pattern_shape_map=MOTIF)
    figure.update_traces(textposition="outside", textfont_size=15)
    figure.update_layout(showlegend=False, xaxis_range=[0, 175])
    acc.graphique(acc.mise_en_page(figure, "Nombre d'images par catégorie", 380),
                  "les 7 catégories ont exactement 150 images chacune. Le jeu est parfaitement équilibré : "
                  "un modèle qui répondrait au hasard aurait 1 chance sur 7 (14 %) d'avoir juste.",
                  comptage)

    # --- Statistiques de taille ---
    st.header("Quelle taille font les images ?")
    st.markdown("Statistiques descriptives des dimensions, en pixels :")
    stats = catalogue[["largeur", "hauteur", "ratio"]].describe().round(1)
    stats.index = ["nombre", "moyenne", "écart-type", "minimum", "25 %", "médiane", "75 %", "maximum"]
    st.dataframe(stats.rename(columns={"ratio": "largeur / hauteur"}), use_container_width=True)

    dimension = st.radio("Dimension à afficher", ["largeur", "hauteur", "rapport largeur / hauteur"],
                         horizontal=True)
    colonne = {"rapport largeur / hauteur": "ratio"}.get(dimension, dimension)
    figure = px.box(catalogue, x=colonne, y="categorie", color="categorie", color_discrete_map=COULEUR,
                    points="outliers", labels={colonne: dimension, "categorie": "catégorie"},
                    hover_data=["product_name"])
    figure.update_layout(showlegend=False)
    if colonne != "ratio":
        figure.update_xaxes(type="log", title=f"{dimension} (pixels, échelle logarithmique)")
    mediane = catalogue.groupby("categorie")[colonne].median().round(2)
    acc.graphique(acc.mise_en_page(figure, f"Distribution de la {dimension} par catégorie", 460),
                  f"les tailles sont très hétérogènes, de moins de 200 à plus de 11 000 pixels de côté. "
                  f"La médiane la plus élevée est celle de « {mediane.idxmax()} » ({mediane.max():g}), "
                  f"la plus faible celle de « {mediane.idxmin()} » ({mediane.min():g}). "
                  "Beaucoup d'images ne sont pas carrées : c'est pourquoi la préparation les complète "
                  "en carré au lieu de les déformer.",
                  mediane.rename(f"{dimension} médiane").to_frame())

    # --- Exemples ---
    st.header("Des exemples d'images par catégorie")
    categorie = st.selectbox("Catégorie", CATEGORIES, key="exemples")
    graine = st.number_input("Tirage (changez le nombre pour voir d'autres images)", 0, 999, 0, step=1)
    exemples = catalogue[catalogue["categorie"] == categorie].sample(8, random_state=int(graine))
    colonnes = st.columns(4)
    for k, (_, produit) in enumerate(exemples.iterrows()):
        with colonnes[k % 4]:
            st.image(miniature(produit["image"]), caption=produit["product_name"][:70],
                     use_container_width=True)

    # --- Transformations ---
    st.header("Les transformations d'une image")
    st.markdown(
        "La **préparation** (carré blanc puis 224 × 224 pixels) est celle qu'utilisent réellement les "
        "modèles. **L'égalisation d'histogramme** étale les niveaux de luminosité pour renforcer le "
        "contraste, et le **floutage** retire les détails fins : deux transformations classiques pour "
        "tester la robustesse d'un modèle.")
    produit = catalogue[catalogue["categorie"] == categorie].iloc[int(graine) % 150]
    rayon = st.slider("Intensité du floutage (rayon en pixels)", 0, 12, 4)
    image = miniature(produit["image"])

    from modele import carre_blanc
    preparee = carre_blanc(image)
    egalisee = ImageOps.equalize(preparee)
    floutee = preparee.filter(ImageFilter.GaussianBlur(rayon))
    colonnes = st.columns(4)
    for colonne, (titre, img) in zip(colonnes, [("Originale", image), ("Préparée 224 × 224", preparee),
                                               ("Égalisée", egalisee), (f"Floutée (rayon {rayon})", floutee)]):
        colonne.image(img, caption=f"{titre} : {produit['product_name'][:40]}", use_container_width=True)

    # histogramme de luminosité avant / après égalisation
    def histogramme(img):
        valeurs = np.asarray(img.convert("L")).ravel()
        return np.bincount(valeurs // 8, minlength=32)

    luminosite = pd.DataFrame({
        "luminosité (0 = noir, 255 = blanc)": np.arange(32) * 8 + 4,
        "préparée": histogramme(preparee),
        "égalisée": histogramme(egalisee),
    })
    longue = luminosite.melt(id_vars=luminosite.columns[0], var_name="version", value_name="pixels")
    figure = px.line(longue, x=luminosite.columns[0], y="pixels", color="version", line_dash="version",
                     markers=True, symbol="version",
                     color_discrete_map={"préparée": acc.GRIS_FONCE, "égalisée": acc.BLEU_FONCE})
    figure.update_yaxes(type="log", title="nombre de pixels (échelle logarithmique)")
    acc.graphique(acc.mise_en_page(figure, "Histogramme de luminosité avant et après égalisation", 380),
                  "avant égalisation, les pixels se concentrent sur quelques niveaux (en particulier le blanc "
                  "du fond, à droite). Après égalisation, ils sont répartis sur toute l'échelle : le "
                  "contraste de l'image est renforcé.",
                  luminosite)


# =============================================================================
# Page 2 : prédire une catégorie
# =============================================================================

def barres_probabilites(resultat, vraie_categorie=None):
    """Probabilités des deux modèles, côte à côte, avec les valeurs écrites."""
    tableau = pd.DataFrame({
        "catégorie": CATEGORIES,
        NOM_NOUVEAU: [resultat["dinov2"][c] for c in CATEGORIES],
        NOM_BASELINE: [resultat["mobilenet"][c] for c in CATEGORIES],
    }).sort_values(NOM_NOUVEAU)
    figure = go.Figure()
    for nom, couleur, motif in ((NOM_BASELINE, acc.GRIS_FONCE, "/"), (NOM_NOUVEAU, acc.BLEU_FONCE, "")):
        figure.add_bar(y=tableau["catégorie"], x=tableau[nom], name=nom, orientation="h",
                       marker=dict(color=couleur, pattern_shape=motif),
                       text=[f"{pct(v, 0)}" for v in tableau[nom]], textposition="outside")
    figure.update_layout(barmode="group", xaxis=dict(range=[0, 1.15], tickformat=".0%", title="probabilité"),
                         legend=dict(orientation="h", y=-0.2))
    return acc.mise_en_page(figure, "Probabilité de chaque catégorie selon les deux modèles", 480), tableau


def page_prediction():
    st.title("Prédire la catégorie d'un produit")
    source = st.radio("D'où vient l'image ?", ["Une image du catalogue Flipkart", "Ma propre photo"],
                      horizontal=True)

    vraie_categorie, produit = None, None
    if source == "Une image du catalogue Flipkart":
        colonnes = st.columns([1, 2])
        filtre = colonnes[0].selectbox("Filtrer par catégorie", ["Toutes"] + CATEGORIES)
        choix = catalogue if filtre == "Toutes" else catalogue[catalogue["categorie"] == filtre]
        etiquettes = choix["product_name"].str.slice(0, 90) + "  (" + choix["image"].str.slice(0, 8) + ")"
        selection = colonnes[1].selectbox("Produit", etiquettes.tolist())
        produit = choix.iloc[etiquettes.tolist().index(selection)]
        image = miniature(produit["image"])
        vraie_categorie = produit["categorie"]
    else:
        fichier = st.file_uploader("Choisissez une photo de produit (JPEG ou PNG)", type=["jpg", "jpeg", "png"])
        if fichier is None:
            st.info("Envoyez une photo pour lancer la prédiction.")
            return
        image = Image.open(fichier).convert("RGB")

    with st.spinner("Prédiction en cours…"):
        resultat = charger_predicteur().predire(image)

    prediction = max(resultat["dinov2"], key=resultat["dinov2"].get)
    confiance = resultat["dinov2"][prediction]
    prediction_base = max(resultat["mobilenet"], key=resultat["mobilenet"].get)

    gauche, droite = st.columns([1, 1.4])
    with gauche:
        st.image(image, caption="Image analysée" + (f" : {produit['product_name'][:60]}" if produit is not None else ""),
                 use_container_width=True)
    with droite:
        st.subheader("Le résultat")
        st.markdown(f"### {NOM_NOUVEAU} prédit : **{prediction}** ({pct(confiance, 0)} de confiance)")
        st.markdown(f"La baseline {NOM_BASELINE} prédit : **{prediction_base}** "
                    f"({pct(resultat['mobilenet'][prediction_base], 0)} de confiance)")
        if vraie_categorie is not None:
            for nom, predite in ((NOM_NOUVEAU, prediction), (NOM_BASELINE, prediction_base)):
                if predite == vraie_categorie:
                    st.success(f"Correct : {nom} a trouvé la vraie catégorie, « {vraie_categorie} ».")
                else:
                    st.error(f"Erreur : {nom} se trompe, la vraie catégorie est « {vraie_categorie} ».")
            st.caption(
                "Attention : les modèles affichés ici ont été entraînés sur tout le catalogue, y compris "
                "cette image. La prédiction honnête, faite en validation croisée par un modèle qui n'avait "
                f"jamais vu l'image, était : {NOM_NOUVEAU} → « {produit[COL_NOUVEAU]} », "
                f"{NOM_BASELINE} → « {produit['prediction ' + NOM_BASELINE]} ».")

    figure, tableau = barres_probabilites(resultat, vraie_categorie)
    acc.graphique(figure,
                  f"{NOM_NOUVEAU} attribue {pct(confiance, 0)} à « {prediction} ». "
                  f"La baseline {NOM_BASELINE} attribue {pct(resultat['mobilenet'][prediction_base], 0)} "
                  f"à « {prediction_base} ».",
                  tableau.set_index("catégorie").map(lambda v: f"{pct(v)}"))

    st.header("Où le modèle a-t-il regardé ?")
    from modele import superposer_attention
    colonnes = st.columns(2)
    colonnes[0].image(resultat["image_preparee"], caption="Image préparée (224 × 224)", use_container_width=True)
    colonnes[1].image(superposer_attention(resultat["image_preparee"], resultat["attention"]),
                      caption="Carte d'attention de DINOv2 : les zones claires (jaune) sont celles que le "
                              "modèle a le plus regardées, les zones sombres (violet) les moins regardées.",
                      use_container_width=True)
    carte = resultat["attention"]
    centre = carte[4:12, 4:12].mean() / carte.mean()
    # Les chiffres du catalogue viennent du notebook (section 6), via configuration.json
    from modele import CONFIGURATION
    globale = CONFIGURATION["interpretabilite"]
    st.markdown(f"**À retenir :** sur cette image, l'attention au centre est {nombre(centre)} fois "
                f"l'attention moyenne. Sur l'ensemble du catalogue, le produit occupe "
                f"{pct(globale['surface_produit'], 0)} de l'image et reçoit "
                f"{pct(globale['attention_sur_produit'], 0)} de l'attention de DINOv2.")


# =============================================================================
# Page 3 : les performances
# =============================================================================

def page_performances():
    st.title("Baseline contre DINOv2 : les performances")
    st.markdown(
        "Chaque modèle a été évalué en **validation croisée en 5 plis** : chaque image a été prédite une "
        "fois, par un modèle qui ne l'avait jamais vue. Les quatre réseaux sont gelés et surmontés du même "
        "classifieur (régression logistique) : seule la représentation de l'image change.")

    tableau = pd.read_csv(DOSSIER_MODELE / "tableau_resultats.csv")
    figure = px.bar(tableau, x="exactitude", y="modèle", orientation="h", color="rôle",
                    pattern_shape="rôle", text=tableau["exactitude"].map(lambda v: f"{pct(v)}"),
                    color_discrete_map={"baseline": acc.GRIS_FONCE, "nouveau": acc.BLEU_FONCE},
                    pattern_shape_map={"baseline": "/", "nouveau": ""})
    figure.update_traces(textposition="outside")
    figure.update_layout(xaxis=dict(range=[0.5, 1.05], tickformat=".0%"), yaxis=dict(autorange="reversed"))
    # Le nouveau modèle est celui fixé à l'avance dans le notebook, pas le meilleur score du tableau
    meilleur = tableau.set_index("modèle").loc[NOM_NOUVEAU].copy()
    meilleur["modèle"] = NOM_NOUVEAU
    base = tableau.set_index("modèle").loc[NOM_BASELINE]
    acc.graphique(acc.mise_en_page(figure, "Exactitude en validation croisée (1 050 images)", 380),
                  f"{meilleur['modèle']} atteint {pct(meilleur['exactitude'])} de bonnes réponses, contre "
                  f"{pct(base['exactitude'])} pour la baseline {NOM_BASELINE} du projet 6, soit "
                  f"{base['erreurs / 1 050'] - meilleur['erreurs / 1 050']:.0f} erreurs de moins sur 1 050 images.",
                  tableau)

    st.header("L'écart est-il dû au hasard ?")
    tests = pd.read_csv(DOSSIER_MODELE / "tests_mcnemar.csv")
    # La significativité vient du notebook, après correction de Bonferroni (5 tests à la fois)
    for colonne in ("p-valeur McNemar", "p-valeur corrigée (Bonferroni)"):
        tests[colonne] = tests[colonne].map(lambda p: f"{p:.1e}")
    st.markdown(
        "Le **test de McNemar** compare deux modèles sur les images où ils ne sont pas d'accord : "
        "si l'un réussit nettement plus souvent là où l'autre échoue, l'écart n'est pas dû au hasard.")
    st.dataframe(tests, use_container_width=True, hide_index=True)

    st.header("Les erreurs, catégorie par catégorie")
    modele = st.radio("Modèle", [NOM_NOUVEAU, NOM_BASELINE], horizontal=True)
    matrice = pd.crosstab(catalogue["categorie"], catalogue[f"prediction {modele}"]).reindex(
        index=CATEGORIES, columns=CATEGORIES, fill_value=0)
    figure = px.imshow(matrice, text_auto=True, color_continuous_scale="Blues", aspect="auto",
                       labels=dict(x="catégorie prédite", y="vraie catégorie", color="images"))
    figure.update_traces(textfont_size=15)
    diagonale = np.diag(matrice.to_numpy())
    hors_diagonale = matrice.to_numpy().copy(); np.fill_diagonal(hors_diagonale, 0)
    i, j = np.unravel_index(hors_diagonale.argmax(), hors_diagonale.shape)
    acc.graphique(acc.mise_en_page(figure, f"Matrice de confusion de {modele}", 560),
                  f"{diagonale.sum()} images sur {matrice.to_numpy().sum()} sont bien classées (la diagonale). "
                  f"La catégorie la mieux reconnue est « {CATEGORIES[diagonale.argmax()]} » "
                  f"({diagonale.max()}/150), la moins bien « {CATEGORIES[diagonale.argmin()]} » "
                  f"({diagonale.min()}/150). La confusion la plus fréquente : « {CATEGORIES[i]} » prise pour "
                  f"« {CATEGORIES[j]} » ({hors_diagonale.max()} fois).",
                  matrice)


# --- La navigation ------------------------------------------------------------

navigation = st.navigation([
    st.Page(page_donnees, title="Le jeu de données", icon=":material/bar_chart:", url_path="donnees", default=True),
    st.Page(page_prediction, title="Prédire une catégorie", icon=":material/search:", url_path="prediction"),
    st.Page(page_performances, title="Les performances", icon=":material/leaderboard:", url_path="performances"),
])
st.sidebar.markdown(
    "**Preuve de concept DataSpace**  \n"
    "Classer un produit Flipkart à partir de sa photo : baseline MobileNetV2 (2018) contre DINOv2 (Meta, 2023).")
navigation.run()
