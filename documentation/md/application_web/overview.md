# Application Web de Classification des Déchets

Cette application web développée en Laravel permet d'uploader une image d'un déchet et de retourner le résultat de la classification indiquant si le déchet doit être mis dans la poubelle rouge, verte ou jaune.

## Fonctionnalités

- Upload d'image de déchet.
- Classification automatique du déchet.
- Affichage du résultat de la classification.

## Prérequis

- PHP >= 7.4
- Composer
- Serveur Web (Apache, Nginx, etc.)
- Base de données (MySQL, PostgreSQL, etc.)

## Installation

1. Clonez le dépôt

```bash
git clone https://github.com/yahia-adam/3BigData.git
cd 3BigData/web-app
```
2. Installez les dépendances

   [composer install](https://getcomposer.org/)


3. Mettez à jour votre fichier .env avec les informations de votre base de données

```
DB_CONNECTION=mysql
DB_HOST=127.0.0.1
DB_PORT=3306
DB_DATABASE=nom_de_votre_base_de_donnees
DB_USERNAME=votre_nom_d_utilisateur
DB_PASSWORD=votre_mot_de_passe
```

4. Migrate la base de données

```
php artisan migrate:fresh --seed
```

5. Exécution de l'Application

Pour démarrer le serveur de développement, utilisez la commande suivante :

```
php artisan serve
```

Par défaut, l'application sera accessible à l'adresse http://127.0.0.1:8000.

6. Utilisation

    a. Accédez à l'application via votre navigateur web.
    
    b. Uploadez une image du déchet en utilisant le formulaire prévu à cet effet.
    
    c. Recevez le résultat de la classification indiquant la couleur de la poubelle appropriée (rouge, verte ou jaune).