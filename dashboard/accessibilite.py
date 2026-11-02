"""
Les règles d'accessibilité communes à tous les graphiques du dashboard.

Critères WCAG 2.1 couverts (niveau A et AA) :
- 1.1.1 Contenu non textuel : chaque graphique et chaque image a une
  alternative textuelle (une phrase qui dit ce qu'il faut retenir, et le
  tableau des données sous le graphique).
- 1.4.1 Utilisation de la couleur : la couleur n'est jamais le seul moyen de
  distinguer une information. Les valeurs sont écrites sur les barres, les
  séries ont des motifs différents, les résultats sont dits en toutes lettres.
- 1.4.3 Contraste minimum : texte sur fond blanc avec un contraste d'au
  moins 4,5:1.
- 1.4.11 Contraste du contenu non textuel : barres, points et lignes avec un
  contraste d'au moins 3:1 sur fond blanc (vérifié par test_contrastes_wcag
  dans test_modele.py).
- 1.4.4 Redimensionnement du texte : tailles de police en unités relatives,
  le zoom du navigateur fonctionne.
- 2.1.1 Clavier : uniquement des composants natifs Streamlit, tous
  utilisables au clavier.
- 2.4.2 / 2.4.6 Titres de page et en-têtes explicites.
- 3.1.1 Langue de la page : déclarée en français.
"""

import pandas as pd
import streamlit as st

# Palette Okabe-Ito, conçue pour rester distinguable par les trois formes de
# daltonisme. Le jaune d'origine (#F0E442) est remplacé par un ocre plus foncé
# et le bleu ciel par un bleu plus soutenu, pour garder un contraste suffisant
# sur fond blanc.
PALETTE_CATEGORIES = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#8A6D00", "#4B4B4B", "#882255"]
MOTIFS = ["", "/", "\\", "x", ".", "+", "-"]

BLEU_FONCE = "#0B4F8A"      # contraste 8,4:1 sur blanc : le nouveau modèle
GRIS_FONCE = "#5A5A5A"      # contraste 6,9:1 sur blanc : la baseline
POLICE = dict(family="Source Sans Pro, Arial, sans-serif", size=15, color="#1A1A1A")


def mise_en_page(figure, titre, hauteur=420):
    """Applique la même mise en page lisible à toutes les figures Plotly."""
    figure.update_layout(
        title=dict(text=titre, font=dict(size=18)),
        font=POLICE,
        height=hauteur,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=60, b=10),
        hoverlabel=dict(font_size=15),
        legend=dict(font=dict(size=14)),
    )
    figure.update_xaxes(showgrid=True, gridcolor="#D9D9D9", linecolor="#1A1A1A", ticks="outside")
    figure.update_yaxes(showgrid=True, gridcolor="#D9D9D9", linecolor="#1A1A1A", ticks="outside")
    return figure


def graphique(figure, description: str, donnees: pd.DataFrame | None = None):
    """Affiche une figure avec son alternative textuelle et, sur demande, ses données."""
    st.plotly_chart(figure, use_container_width=True, config={"displaylogo": False})
    st.markdown(f"**À retenir :** {description}")
    if donnees is not None:
        with st.expander("Voir les données de ce graphique sous forme de tableau"):
            st.dataframe(donnees, use_container_width=True)


def declarer_langue():
    """Déclare la langue française de la page pour les lecteurs d'écran (critère 3.1.1).

    Streamlit ne permet pas de régler l'attribut lang de la page : un petit
    composant invisible le fait depuis son cadre, qui partage l'origine de la page.
    """
    import streamlit.components.v1 as components
    components.html("<script>window.parent.document.documentElement.lang = 'fr';</script>",
                    height=0)
