## Introduction

Nous avons implémenté et testé plusieurs cas de tests en utilisant des noyaux linéaires pour séparer les classes. 
Les sections suivantes décrivent les méthodes utilisées et présentent les différents cas de tests réalisés.

## Méthode Utilisée

### Support Vector Machine (SVM)

Pour les tâches de classification, nous avons utilisé le SVM avec des noyaux linéaires pour séparer les données :

- **SVM avec Noyau Linéaire** :
    - Utilisé pour maximiser la marge entre les classes en utilisant un hyperplan linéaire.
    - L'optimisation est réalisée pour trouver l'hyperplan qui maximise la distance aux points les plus proches de chaque classe.

## Cas de Tests

Nous avons évalué notre modèle SVM sur les cas de tests suivants :

1. **Classification Linéaire Simple** (`svm_classification_linear_simple`)
2. **Classification Linéaire Multiple** (`svm_classification_linear_multiple`)
3. **Classification Multi-Linéaire 3 Classes** (`svm_classification_multi_linear_3_classes`)
4. **Classification CRO** (`svm_classification_cross`)
5. **Classification Multi-CRO** (`svm_classification_multi_cross`)
6. **Classification XOR** (`svm_classification_xor`)

## Résultats et Visualisations

Pour chaque cas de test, des visualisations ont été générées pour illustrer la capacité du SVM à séparer les classes avec un noyau linéaire :

- **svm_classification_linear_simple** :

  ![Classification Linéaire Simple](images/svm_classification_linear_simple.png)


- **svm_classification_linear_multiple** :

  ![Classification Linéaire Multiple](images/svm_classification_linear_multiple.png)


- **svm_classification_multi_linear_3_classes** :

  ![Classification Linéaire Multiple](images/svm_classification_multi_linear_3_classes.png)


- **svm_classification_cross** :

  ![Classification CROSS](images/svm_classification_cros.png)


- **svm_classification_multi_cross** :

  ![Classification MULTI CROSS](images/svm_classification_multi_cros.png)


- **svm_classification_xor** :

  ![Classification XOR](images/svm_classification_xor.png)



Pour plus de détails sur l'implémentation et les résultats, veuillez consulter le code source et les commentaires associés dans les fichiers du projet.

---

N'hésitez pas à explorer les fichiers du projet pour mieux comprendre les implémentations et les résultats obtenus !
