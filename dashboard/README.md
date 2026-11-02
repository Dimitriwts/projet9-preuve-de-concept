# Classer un produit Flipkart à partir de sa photo

Dashboard de la preuve de concept du projet 9 OpenClassrooms (Dimitri Wyts) :
la baseline MobileNetV2 (2018) contre DINOv2 (Meta AI, 2023), à protocole identique.

- **Le jeu de données** : analyse exploratoire des 1 050 images (répartition, tailles, exemples, transformations).
- **Prédire une catégorie** : choisir une image du catalogue ou envoyer sa propre photo ; prédiction des deux
  modèles et carte d'attention de DINOv2.
- **Les performances** : exactitude en validation croisée, test de McNemar, matrices de confusion.

Déployé sur Streamlit Community Cloud. Lancement en local, depuis la racine du projet :
`streamlit run dashboard/app.py`.

Les poids de DINOv2 (licence Apache 2.0) sont téléchargés depuis Hugging Face, ceux de MobileNetV2 depuis
torchvision. Le dossier `modele/` contient les classifieurs entraînés dans le notebook du projet.
