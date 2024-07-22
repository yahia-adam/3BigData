# Application Web de Classification des Déchets

Cette application web développée en Flask permet d'uploader une image d'un déchet et de retourner le résultat de la classification indiquant si le déchet doit être du métal, papier ou plastique.

## Fonctionnalités

- Upload d'image de déchet.
- Classification automatique du déchet.
- Affichage du résultat de la classification.

## Prérequis

- Python 3.7+
- pip (gestionnaire de paquets Python)
- Virtualenv (recommandé)

## Installation

1. Clonez le dépôt

```bash
git clone https://github.com/yahia-adam/3BigData.git
cd 3BigData/web-app
```

2. Créez un environnement virtuel et activez-le
```bash
python -m venv venv
source venv/bin/activate  # sur windows, utilisez `venv\Scripts\activate`
```

3. Installez les dépendances
```bash
pip install -r requirements.txt
```

4. Configurez les variables d'environnement

Créez un fichier ```.env``` à la racine du projet et ajoutez les variables nécessaires :
```bash
FLASK_APP=app.py
FLASK_ENV=development
SECRET_KEY=your_secret_key
```
5. Exécution de l'Application
Pour démarrer le serveur de développement, utilisez la commande suivante :
```bash 
flask run
```
Par défaut, l'application sera accessible à l'adresse http://127.0.0.1:5000.


6. Utilisation
   - Accédez à l'application via votre navigateur web.
   - Uploadez une image du déchet en utilisant le formulaire prévu à cet effet.
   - Recevez le résultat de la classification indiquant la catégorie du déchet (métal, papier ou plastique).


7. Développement
Pour ajouter de nouvelles fonctionnalités ou modifier l'existant, vous pouvez éditer les fichiers suivants :
   - ```app.py``` : Point d'entrée principal de l'application
   - ```templates/``` : Dossier contenant les templates HTML
   - ```static/``` : Dossier pour les fichiers statiques (CSS, JS, images)


8. N'oubliez pas de mettre à jour le fichier requirements.txt si vous ajoutez de nouvelles dépendances :
```bash
pip freeze > requirements.txt
```