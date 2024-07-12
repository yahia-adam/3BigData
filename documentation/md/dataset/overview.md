## Dataset

### Description

Notre Dataset Finale est utilisé pour entraîner et tester le modèle de classification des déchets. Il est organisé en trois catégories principales.

### Structure du Dataset


Le dataset est constitué d'une combinaison d'images téléchargées depuis Bing et de divers ensembles de données récupérés sur Kaggle.

1. **Paper :** (10 161 images)

2. **Metal :** (10 190 images)

3. **Plastic :** (10 140 images)

### Nomenclature des Fichiers

Les fichiers d'images suivent une nomenclature spécifique : `type_count.extension` (par exemple, `metal_1.jpg`).

### Résolution des Images

Toutes les images du dataset ont une résolution de 32 x 32 pixels.

### Répartition des Données

- 20% des images sont utilisées pour le jeu de test.
- Le reste des images est utilisé pour l'entraînement.

### Hiérarchie du Dataset

Voici une illustration de la hiérarchie du dataset :

![Hiérarchie du Dataset](img.png)

### Exemple de Structure de Répertoire

```plaintext
dataset/
├── train/
│ ├── paper/
│ │ ├── paper_1.jpg
│ │ ├── paper_2.jpg
│ │ └── ...
│ ├── metal/
│ │ ├── metal_1.jpg
│ │ ├── metal_2.jpg
│ │ └── ...
│ └── plastic/
│ ├── plastic_1.jpg
│ ├── plastic_2.jpg
│ └── ...
└── test/
  ├── paper/
  │ ├── paper_1.jpg
  │ ├── paper_2.jpg
  │ └── ...
  ├── metal/
  │ ├── metal_1.jpg
  │ ├── metal_2.jpg
  │ └── ...
  └── plastic/
  ├── plastic_1.jpg
  ├── plastic_2.jpg
  └── ...
```