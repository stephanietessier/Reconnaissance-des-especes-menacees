# Reconnaissance automatique des espèces menacées — AI for Good

Projet éducatif de machine learning utilisant des **données imaginées** pour reconnaître automatiquement des espèces menacées à partir de caractéristiques observables.

> ⚠️ Ce projet utilise des données fictives générées artificiellement. Il est conçu pour apprendre, prototyper et être publié sur GitHub/Colab.

## Objectif

Créer un modèle capable de prédire l'espèce probable à partir de mesures simples : taille, poids, région, habitat, motif du pelage/plumage, activité, altitude et niveau de menace.

## Espèces fictives incluses

- Panda roux
- Tigre de Sumatra
- Gorille des montagnes
- Tortue imbriquée
- Vaquita
- Rhinocéros de Java
- Léopard des neiges
- Orang-outan de Bornéo

## Structure

```text
endangered_species_recognition_project/
├── data/
│   └── imagined_endangered_species.csv
├── src/
│   ├── generate_data.py
│   ├── train_model.py
│   ├── predict.py
│   └── visualize.py
├── notebooks/
│   └── endangered_species_colab_demo.ipynb
├── models/
├── outputs/
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation locale

```bash
pip install -r requirements.txt
python src/generate_data.py
python src/train_model.py
python src/predict.py
python src/visualize.py
```

## Utilisation sur Google Colab

1. Ouvre `notebooks/endangered_species_colab_demo.ipynb` dans Google Colab.
2. Lance les cellules une par une.
3. Le notebook génère les données, entraîne le modèle et teste des prédictions.

## Méthode ML

Le modèle utilise :

- `RandomForestClassifier`
- encodage One-Hot des variables catégorielles
- normalisation des variables numériques
- matrice de confusion
- rapport de classification

## Exemple de prédiction

```bash
python src/predict.py
```

Exemple de sortie :

```text
Espèce prédite : Léopard des neiges
Confiance : 86.2%
```

## Idées d'amélioration

- Utiliser de vraies images avec un CNN.
- Ajouter une carte interactive des observations.
- Utiliser des données publiques GBIF, iNaturalist ou IUCN.
- Créer une application Streamlit ou Hugging Face Spaces.
- Ajouter une détection d'espèces par photo.

## Éthique

Ce type d'outil doit être utilisé pour aider la conservation, pas pour faciliter le braconnage. Les localisations sensibles doivent être floutées ou protégées.
