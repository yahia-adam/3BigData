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

2. **Classification des images** : Tri des images en trois catégories principales basées sur la couleur des poubelles (jaune, rouge, verte) et création de sous-catégories spécifiques.

3. **Prétraitement des images** : Redimensionnement des images à une résolution uniforme de 64 x 64 pixels et renommage des fichiers selon la nomenclature `type_count.extension`.

4. **Répartition des Données** : Division du dataset en ensembles d'entraînement et de test, avec 10% des images dédiées au test et le reste à l'entraînement.

### Résolution des images

Toutes les images du dataset ont une résolution de 32 x 32 pixels.

### Nomenclature des Fichiers

Les fichiers d'images suivent une nomenclature spécifique : `type_count.extension` (par exemple, `metal_1.jpg`).

# Étape 2 du projet

### Développement et Test des Modèles

Nous avons développé, implémenté et testé plusieurs modèles de classification pour notre projet. Voici les détails de chaque modèle :

1. **Modèle Linéaire** :
   - Développement d'un classificateur linéaire simple (un seul neurone).

2. **PMC (Perceptron Multicouche)** :
   - Création d'un réseau de neurones avec plusieurs couches cachées.

3. **Modèle RBF (Radial Basis Function)** :
   - Implémentation d'un modèle à noyau RBF pour capturer des relations non linéaires.

4. **SVM (Support Vector Machine)** :
   - Mise en place d'un modèle SVM avec différents noyaux.

### Phase de Test

Pour chaque modèle, nous avons suivi une procédure de test rigoureuse :

1. **Tests sur Cas Prédéfinis** :
   - Création d'un ensemble de cas de test couvrant diverses situations.
   - Vérification de la réponse de chaque modèle sur ces cas.
   - Validation que les sorties correspondent aux résultats attendus.

2. **Tests sur le Dataset** :
   - Utilisation de l'ensemble de test extrait de notre dataset principal.
   - Évaluation des performances de chaque modèle sur des données réelles.

### Résultats et Validation

- Tous les modèles ont passé avec succès les tests sur les cas prédéfinis, démontrant leur capacité à classifier correctement dans des scénarios contrôlés.
- Les tests sur notre dataset ont fourni des résultats prometteurs, avec des variations de performance entre les différents modèles.

---

En suivant ces étapes, nous avons constitué un dataset robuste et varié, prêt à être utilisé pour entraîner et tester notre modèle de classification des déchets.
