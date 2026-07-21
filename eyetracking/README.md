# Expérience d’oculométrie

Ce dossier contient le code nécessaire à l’exécution de l’expérience d’oculométrie en vision libre. L’expérience présente des stimuli visuels ou infrarouges avec PsychoPy, enregistre les données du regard avec un système EyeLink et produit un fichier EDF par session.

Le protocole sélectionne automatiquement les stimuli associés au participant, conserve le ratio d’aspect des images, randomise leur ordre de manière reproductible et ajoute dans le fichier EDF les informations nécessaires pour un traitement ultérieur.

## 1. Fonctionnalités principales

- sélection des stimuli correspondant à l'id du participant ;
- présentation des stimuli après redimensionnement en plein écran avec PsychoPy ;
- ordre des stimuli déterministe pour chaque participant ;
- acquisition EyeLink des fixations du regard ;
- prise en charge des modalités visible et infrarouge ;
- croix de fixation centrale d’une durée aléatoire comprise entre 0,8 et 1,2 seconde ;
- présentation de chaque image pendant 3 secondes ;
- redimensionnement des stimuli sans déformation dans une zone maximale configurable ;
- calibration initiale, correction de dérive périodique et recalibration complète périodique ;
- création automatique d’un dossier de résultats par session ;
- messages EyeLink structurés pour faciliter l’extraction et la reprojection des données ;


## 2. Organisation du dossier

```text
eyetracking/
├── main.py
├── README.md
├── src/
│   ├── EyeLinkCoreGraphicsPsychoPy.py
│   └── fixTarget.bmp
├── stimuli/
│   ├── visible/
│   └── infrared/
├── tables/
│   ├── participants.csv
│   └── metadata.csv
└── results/
    ├── visible/
    └── infrared/
```

Après une session, les résultats sont enregistrés dans un dossier de la forme :

```text
results/<modalite>/<participant>_<date>_<heure>/
└── ID-<participant>-<date>_<heure>.EDF
```

En mode de test, les entrées sont recherchées dans `test/tables/` et `test/stimuli/`, tandis que les sorties sont enregistrées dans `results/test/<modalite>/`.

## 3. Prérequis

L’exécution nécessite :

- Python ;
- PsychoPy ;
- Pandas ;
- Pillow ;
- PyLink et les composants du EyeLink Developers Kit ;
- le module `EyeLinkCoreGraphicsPsychoPy.py` ;

## 4. Configuration

Les principaux paramètres sont définis au début de `main.py`.

- `DUMMY_MODE` : Utilise une connexion EyeLink simulée. 
- `FULL_SCREEN` : Ouvre la fenêtre PsychoPy en plein écran.
- `TEST_MODE` : Effectue l'expérience en mode Test en utilisant les dossiers consacrés.
- `STIMULUS_MAX_SIZE` : Zone maximale d’affichage d’un stimulus, en pixels.
- `IMG_TIME` : Durée d’affichage d’une image, en secondes.
- `FIXATION_MIN_TIME` : Durée minimale de la croix de fixation.
- `FIXATION_MAX_TIME` : Durée maximale de la croix de fixation.
- `CENTRAL_RECALIBRATION_EVERY`: Correction centrale après chaque bloc de 25 images.
- `FULL_RECALIBRATION_EVERY`: Recalibration complète après chaque bloc de 50 images.
- `HOST_IP` | `100: Adresse IP du poste EyeLink Host.
- `MONITOR_WIDTH_CM` |: Largeur physique du moniteur.
- `MONITOR_DISTANCE_CM` : Distance entre le participant et le moniteur.

Pour effectuer une acquisition réelle, il faut au minimum vérifier au minimum :

```python
DUMMY_MODE = False
FULL_SCREEN = True
TEST_MODE = False
```

La largeur du moniteur et la distance d’observation doivent correspondre à l’installation expérimentale. La résolution de l’écran est détectée automatiquement au démarrage.

## 5. Tables d’entrée

Le dossier contient des fichiers `csv` qui ont différents rôles.

#### `participants.csv`

Cette table associe chaque participant à un sous-ensemble de stimuli.

Colonnes requises :

| Colonne | Description |
|---|---|
| `id` | Identifiant saisi au lancement de l’expérience. |
| `subset` | Sous-ensemble attribué au participant, par exemple `A`, `B` ou `C`. |

Exemple :

```csv
id;subset
1;A
2;B
3;C
```

#### `metadata.csv`

Cette table décrit les stimuli et leur répartition entre les sous-ensembles.

Colonnes actuellement présentes :

| Colonne | Utilisation |
|---|---|
| `image` | Nom du fichier image affiché et nom transmis au pipeline de traitement. |
| `modality` | Modalité du stimulus. |
| `subset` | Sous-ensemble auquel appartient l’image. |
| `unique_label` | Annotation sémantique ; non utilisée par le script d’acquisition actuel. |
| `multi_labels` | Annotations multiples ; non utilisées par le script d’acquisition actuel. |
| `description` | Description textuelle ; non utilisée par le script d’acquisition actuel. |


Chaque sous-ensemble doit contenir une seule modalité. Les fichiers doivent être présents dans :

```text
- stimuli/visible/
- stimuli/infrared/
```


## 6. Lancement de l’expérience

Depuis le dossier `eyetracking/`, exécuter :

```bash
python main.py
```

Le déroulement est le suivant :

1. saisie de l’identifiant du participant ;
2. recherche du sous-ensemble dans `participants.csv` ;
3. chargement des stimuli correspondants depuis `metadata.csv` ;
5. randomisation déterministe de l’ordre des essais ;
6. ouverture de la fenêtre PsychoPy ;
7. calibration initiale en acquisition ;
8. exécution des essais ;
9. transfert du fichier EDF dans le dossier de résultats.


## 7. Déroulement d’un essai

Pour chaque stimulus :

1. l’image est chargée et redimensionnée en conservant son ratio d’aspect ;
2. sa géométrie est exprimée dans les coordonnées complètes de l’écran ;
3. l’image est envoyée comme fond au poste EyeLink Host ;
4. l’enregistrement EyeLink démarre ;
5. une croix de fixation centrale est affichée pendant une durée aléatoire ;
6. le stimulus est présenté pendant 3 secondes ;
7. des messages décrivant l’essai et la géométrie du stimulus sont écrits dans l’EDF ;
8. l’écran est effacé et l’enregistrement de l’essai est arrêté.

La taille maximale d’affichage est limitée par `STIMULUS_MAX_SIZE`, sans dépasser la résolution réelle de l’écran.

## 8. Messages enregistrés dans l’EDF

Le script écrit notamment les messages suivants :

```text
TRIALID
NUM_IMAGE
Affichage_Image
image_onset
time_out
blank_screen
TRIAL_RESULT
DISPLAY_COORDS
```

Il enregistre également des variables Data Viewer relatives :

- au nom de l’image ;
- à la modalité ;
- à la largeur affichée ;
- à la hauteur affichée ;
- au rectangle occupé par l’image sur l’écran.

Ces informations permettent de replacer les coordonnées du regard dans l’espace propre au stimulus pendant le traitement.

