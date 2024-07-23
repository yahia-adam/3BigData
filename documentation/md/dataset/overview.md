## Dataset

### Description

Notre Dataset Finale est utilisé pour entraîner et tester le modèle de classification des déchets. Il est organisé en trois catégories principales.

### Structure du Dataset


Le dataset est constitué d'une combinaison d'images téléchargées depuis Bing et de divers ensembles de données récupérés sur Kaggle 
ainsi quelques photos prises avec nos propres moyens.

Totalité : 39 046 images

1. **Paper :** -> Train : 10154 images, Test : 2159 images

2. **Metal :** -> Train : 11 877 images, Test : 2833 images

3. **Plastic :** -> Train : 10 114 images, Test : 1858 images 

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