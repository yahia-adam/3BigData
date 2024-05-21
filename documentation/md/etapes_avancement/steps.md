# Étape 1 du projet

### Problématiques Applicatives Choisies

Nous avons identifié plusieurs problématiques applicatives pertinentes pour notre projet de classification des déchets. Ces problématiques incluent :

- Détection et classification correcte des différents types de déchets.
- Répartition des déchets dans les catégories appropriées (jaune, rouge, verte).
- Optimisation de la précision du modèle de classification pour chaque catégorie de déchets.

### Création du Repository Git

Pour organiser et suivre notre travail de développement, nous avons créé un repository Git dédié à ce projet. Ce repository nous permet de gérer efficacement le code source, de collaborer en équipe et de suivre l'historique des modifications.

Lien vers le repository Git : [3BigData](https://github.com/yahia-adam/3BigData)

### Pistes de Constitution du Dataset

Pour constituer notre dataset, nous avons suivi les étapes suivantes :

1. **Collecte des Données** :
    - Nous avons développé un script Python pour importer des images de différents types de déchets depuis Bing.
    - Nous avons également intégré et mélangé plusieurs datasets existants disponibles sur Kaggle pour compléter notre dataset.

2. **Classification des Images** : Tri des images en trois catégories principales basées sur la couleur des poubelles (jaune, rouge, verte) et création de sous-catégories spécifiques.

3. **Prétraitement des Images** : Redimensionnement des images à une résolution uniforme de 64 x 64 pixels et renommage des fichiers selon la nomenclature `type_count.extension`.

4. **Répartition des Données** : Division du dataset en ensembles d'entraînement et de test, avec 10% des images dédiées au test et le reste à l'entraînement.

### Résolution des Images

Toutes les images du dataset ont une résolution de 64 x 64 pixels.

### Nomenclature des Fichiers

Les fichiers d'images suivent une nomenclature spécifique : `type_count.extension` (par exemple, `metal_1.jpg`).

---

En suivant ces étapes, nous avons constitué un dataset robuste et varié, prêt à être utilisé pour entraîner et tester notre modèle de classification des déchets.
