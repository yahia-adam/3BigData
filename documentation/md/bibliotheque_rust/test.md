# Test du bien functionement des differents modèls

Dans cette section on retrouve un ensemble de test des differents model pour s'assurer leur bon implementation.

**voici la liste des tests disponible:**

- ml_classification_linear_simple
- ml_classification_linear_multiple
- ml_classification_cros
- ml_classification_xor
- ml_resgression_linear_simple_2D
- ml_resgression_linear_simple_3D
- ml_resgression_linear_tricky_3D
- ml_resgression_non_linear_simple_2D
- ml_resgression_non_linear_simple_3D

voici un example de comment on execute les tests:

```bash
# Commands a executer dans le dossier mylib
cargo run --example ml_classification_linear_simple
```
```sh
# le resultat de la command
Linear Simple : Linear Model : OK

    X:[1.0, 1.0], Y:1.0 ---> mon model: 1.0
    X:[2.0, 3.0], Y:-1.0 ---> mon model: -1.0
    X:[3.0, 3.0], Y:-1.0 ---> mon model: -1.0

#`X` c'est un example d'input `Y` c'est le result exact, et enfin `mon model` c'est le result que le model a renvoyé
```

## Model linear

Voici un ensemble de tests pour le model.
- Pour résoudre les problems de classification on utilise la Règle de Rosenblatt avec comme function d'activation la function signe.

- Et pour résoudre les problems de régression on utilise pseudo inverse de moore penrose.

### Classification

#### Model linear linear simple

```bash
cargo run --example ml_classification_linear_simple
```
```bash
Linear Simple : Linear Model : OK
X:[1.0, 1.0], Y:1.0 ---> mon model: 1.0
X:[2.0, 3.0], Y:-1.0 ---> mon model: -1.0
X:[3.0, 3.0], Y:-1.0 ---> mon model: -1.0
```
Le Model linear est cencé etre capable de résoudre un problém de classification linear, et comme observé il le reussi assez bien. 

#### Model linear linear multiple

```bash
cargo run --example ml_classification_linear_multiple
```
```bash
Linear Multiple : Linear Model : OK
X:[1.7688197, 1.4136543], Y:1.0 ---> mon model: 1.0
X:[1.3559669, 1.7645961], Y:1.0 ---> mon model: 1.0
X:[1.4687953, 1.5109376], Y:1.0 ---> mon model: 1.0
X:[2.5459971, 2.2028604], Y:-1.0 ---> mon model: -1.0
X:[2.5020037, 2.2388425], Y:-1.0 ---> mon model: -1.0
X:[2.7420998, 2.8660269], Y:-1.0 ---> mon model: -1.0
```
Notre modél functionne également très bien pour résoudre des probleme de classification multi-lineaire 

#### Model linear cros
```bash
cargo run --example ml_classification_cros
```
```bash
Cross : Linear Model    : KO
X:[0.5618494, -0.96608216], Y:[-1.0] ---> mon model: 1.0
X:[-0.05380274, 0.5139979], Y:[1.0] ---> mon model: 1.0
X:[-0.97588307, 0.6509608], Y:[-1.0] ---> mon model: -1.0
```

Comme observé le model produit des résult incorcts, et ceci est absolument fondé car le model linear et incapable de séparé une croix avec une ligne.

#### Model linear Xor

```bash
cargo run --example ml_classification_xor
```
```bash
XOR : Linear Model    : KO
X:[1.0, 0.0], Y:1.0 ---> mon model: 1.0
X:[0.0, 1.0], Y:1.0 ---> mon model: 1.0
X:[0.0, 0.0], Y:-1.0 ---> mon model: -1.0
X:[1.0, 1.0], Y:-1.0 ---> mon model: 1.0
```
Come pour le test précedant un model lineaire est incapable de resoudre ce problém comlex.

## Régression

#### model linear, linear simple 2D

```bash
cargo run --example ml_resgression_linear_simple_2D
```
```bash
Linear Simple 2D : Linear Model : OK
X:[1.0], Y:2.0 ---> mon model: 2.0
X:[2.0], Y:3.0 ---> mon model: 3.0
```
Résult parfait.




#### model linear, linear simple 2D

```bash
cargo run --example ml_resgression_linear_simple_3D
```
```bash
Linear Simple 3D : Linear Model    : OK
X:[1.0, 1.0], Y:2.0 ---> mon model: 2.0
X:[2.0, 2.0], Y:3.0 ---> mon model: 3.0
X:[3.0, 1.0], Y:2.5 ---> mon model: 2.5
```
Résult parfait.

#### model linear, No linear simple 2D

```bash
cargo run --example ml_resgression_non_linear_simple_2D
```
```bash
Non Linear Simple 2D : Linear Model    : OK
X:[1.0], Y:2.0 ---> mon model: 2.25
X:[2.0], Y:3.0 ---> mon model: 2.5
X:[3.0], Y:2.5 ---> mon model: 2.75
```

Les résult de ce test ne sont pas exactement ce qu'on attendant, le test etais cencé produire des results parfait or il ya quelque imperfection. 

Normalement le pseudo inverse est formule or une formule si les données sont exacte les result sont exact a tous les coup, en plus il ya pas de parametre extern en jeux tel que le learning rate, et le nombre d'iteration qui peuvent influencer le result.

Donc je ne vois qu'une seul hypothese qui tien la route: les donnée contiens du bruit, (pas tous corrects)

#### model linear, No linear simple 3D
```bash
cargo run --example ml_resgression_non_linear_simple_3D
```
```bash
Non Linear Simple 3D : Linear Model       : KO
X:[1.0, 0.0], Y:2.0 ---> mon model: 0.5
X:[0.0, 1.0], Y:1.0 ---> mon model: -0.5
X:[1.0, 1.0], Y:-2.0 ---> mon model: 0.5
X:[0.0, 0.0], Y:-1.0 ---> mon model: -0.5
```
Ce problem de peux pas etre resolut par un model linear.


## Perceptron Multi-couche
