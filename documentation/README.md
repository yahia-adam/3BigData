# README

This project is for :

>3BigData | Documentation

**Ce projet génère des fichiers HTML statiques dans le dossier `doc` à partir des fichiers Markdown dans le répertoire `md`.**.

- Le répertoire `priv/assets/` contient les fichiers CSS et `3BigData-logo.png`.

## Prérequis pour construire la documentation

- git
- erlang
- elixir

Vous pouvez utiliser le script suivant pour installer ces outils via le gestionnaire `asdf` :
```bash
# vous devez avoir un utilisateur sudo
chmod +x install-requirements.sh && ./install-requirements.sh
```

```bash
# puis redémarrez votre shell et entrez la commande suivante pour installer erlang et elixir via asdf
asdf plugin add erlang
asdf plugin add elixir
asdf install erlang latest
asdf install elixir latest
# asdf global elixir 1.15.7-otp-26
# asdf global erlang 26.1.2 
```

## Comment construire la documentation HTML

- Ouvrez un terminal et entrez : `mix docs`
- puis exécutez :

```bash
firefox doc/documentation.html`
```

## Comment modifier la documentation

- Modifiez un fichier `.md` dans le répertoire `md`.
- Enregistrez
- Tapez dans le terminal à la racine de ce projet : `mix docs`
- La documentation HTML est générée dans le dossier `doc`.

## Comment ajouter un nouveau fichier à la documentation et générer la documentation HTML

- Créez un fichier Markdown dans le dossier `md/[category]`.
.
- Il y a 11 dossiers `[category]`:
    - Introduction
    - Installation et Configuration
    - Application Web
    - Bibliothèque en Rust
    - Algorithmes et Modèles
    - Dataset
    - Étapes d'Avancement
    - Remarques et Observations
    - Exemples d'Utilisation
    - Annexes
    - Contributions

- Éditez votre texte avec la syntaxe `markdown` [Plus d'infos](https://guides.github.com/features/mastering-markdown/).
- Enregistrez votre fichier `.md`.
- Ajoutez une entrée pour votre fichier dans `mix.exs` dans la section `extras` à la fin comme ceci :  

```bash
"md/[category]/[nom-de-votre-ficher].md": [title: "Titre que vous voulez dans le menu"]
```

- Enregistrez et exécutez `mix docs`
- Les fichiers **HTML** sont générés/mis à jour dans le dossier `doc`. Copiez le contenu de ce dossier vers le site web/documentation.
- L'index est appelé `documentation.md` dans le répertoire `md`.


votre_projet/
├── md/
│   ├── introduction/
│   │   ├── overview.md
│   │   ├── objectives.md
│   │   └── summary.md
│   ├── installation_et_configuration/
│   │   ├── prerequisites.md
│   │   ├── cloning_repo.md
│   │   ├── installing_dependencies.md
│   │   └── running_application.md
│   ├── application_web/
│   │   ├── overview.md
│   │   ├── setup.md
│   │   ├── features.md
│   │   └── deployment.md
│   ├── bibliotheque_rust/
│   │   ├── overview.md
│   │   ├── linear_model.md
│   │   ├── pmc.md
│   │   ├── rbf.md
│   │   └── svm.md
│   ├── algorithmes_modeles/
│   │   ├── overview.md
│   │   ├── linear_models.md
│   │   ├── neural_networks.md
│   │   └── advanced_models.md
│   ├── dataset/
│   │   ├── overview.md
│   │   ├── data_collection.md
│   │   ├── data_preprocessing.md
│   │   └── data_usage.md
│   ├── etapes_avancement/
│   │   ├── step1.md
│   │   ├── step2.md
│   │   ├── step3.md
│   │   └── step4.md
│   ├── remarques_observations/
│   │   ├── overview.md
│   │   ├── analysis.md
│   │   ├── issues.md
│   │   └── future_work.md
│   ├── exemples_utilisation/
│   │   ├── overview.md
│   │   ├── python_examples.md
│   │   ├── cpp_examples.md
│   │   └── nodejs_examples.md
│   ├── annexes/
│   │   ├── references.md
│   │   ├── glossary.md
│   │   ├── faq.md
│   │   └── additional_resources.md
│   └── contributions/
│       ├── guide.md
│       ├── contributors.md
│       └── code_of_conduct.md
├── mix.exs
└── README.md
