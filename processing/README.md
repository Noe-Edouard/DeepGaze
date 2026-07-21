# Traitement des données d’oculométrie

Ce dossier contient le pipeline de transformation des fichiers EyeLink exportés au format ASC en tableaux structurés et nettoyés. Il assure quatre étapes principales :

1. lecture et interprétation des événements EyeLink ;
2. extraction des essais, fixations et clignements ;
3. évaluation de la qualité de chaque essai ;
4. nettoyage des fixations et reprojection dans l’espace du stimulus.

Le traitement est conçu pour être compatible avec les messages écrits dans les fichiers EDF par le module `eyetracking`, notamment `DISPLAY_COORDS`, `Affichage_Image`, `time_out` et les variables `TRIAL_VAR` décrivant le nom et les dimensions du stimulus.

## 1. Organisation du dossier

```text
processing/
├── __init__.py
├── parser.py
├── extractor.py
├── profiler.py
├── cleaner.py
└── README.md
```

Le code dépend également de modules communs situés dans un package `helpers` :

```text
helpers/
├── indexer.py
├── models.py
└── utils.py
```

Ces fichiers doivent être disponibles dans l’environnement Python pour que le pipeline fonctionne.

## 2. Vue d’ensemble du pipeline

Fichiers EDF $\rightarrow$ conversion externe EDF → Fichiers ASC $\rightarrow$ Parser (RawTrial) $\rightarrow$ Extractor (fixations brutes, clignements) $\rightarrow$ Profiler (qualité des essais, essais invalides, résumé par participant) $\rightarrow$ Cleaner (fixations nettoyées, résumé par images)

La conversion des fichiers EDF en fichiers ASC n’est pas effectuée par ce dossier. Elle doit être réalisée au préalable avec l'outil Eyelink EDF2ASCII.



## 3. Format des données d’entrée

### Fichiers ASC

`Extractor.run()` traite tous les fichiers portant l’extension `.asc` présents directement dans un dossier donné :

```text
data/
└── asc/
    ├── ID-001-2026_04_20_10_00.asc
    ├── ID-002-2026_04_20_11_00.asc
    └── ID-003-2026_04_20_12_00.asc
```

Le parseur attend notamment les lignes suivantes :

```text
MSG <timestamp> DISPLAY_COORDS 0 0 <x_max> <y_max>
MSG <timestamp> Affichage_Image
MSG <timestamp> !V TRIAL_VAR name <image>
MSG <timestamp> !V TRIAL_VAR stimulus_width <largeur>
MSG <timestamp> !V TRIAL_VAR stimulus_height <hauteur>
EFIX <oeil> <début> <fin> <durée> <x> <y> <pupille>
EBLINK <oeil> <début> <fin> <durée>
MSG <timestamp> time_out
```

Les modalités acceptées sont exactement : `visible` et `ìnfrared`

### Métadonnées des stimuli

Le `Parser` reçoit un objet `Metadata`. Celui-ci est utilisé par `Indexer` pour :

- associer le nom d’un stimulus à un identifiant numérique ;
- retrouver le nom canonique d’une image ;
- récupérer les dimensions originales de l’image ;
- distinguer les modalités visible et infrarouge.

La construction exacte de l’objet `Metadata` dépend de l’implémentation présente dans `helpers.models`.

## 4. Utilisation générale

Exemple schématique :

```python
from pathlib import Path

from helpers.models import Metadata
from processing.parser import Parser
from processing.extractor import Extractor
from processing.profiler import Profiler
from processing.cleaner import Cleaner


# PATHS
asc_dir = root_dir / Path(f"asc/{modality}/")
images_dir = root_dir / Path(f"images/{modality}/")
dataframes_dir = root_dir / Path(f"dataframes/{modality}")
maps_dir = root_dir / Path(f"maps/{modality}")
metadata_path = root_dir / Path("metadata/metadata.json")

# EXTRACTION
loader = Loader(metadata_path)
parser = Parser(loader.metadata)
extractor = Extractor(parser)

raw_fixations_df, raw_blinks_df, raw_summary_df = extractor.run(
    asc_dir=asc_dir,
    modality=modality,
    verbose=True,
    save=True,
    output_dir=dataframes_dir / 'raw'
)

# ANALYSIS
profiler = Profiler(
    min_fix_duration=80,
    max_blinks_valid=4,
    center_tolerance_deg=2.0,
)

trial_summary_df, participant_summary_df, invalid_trials_df = profiler.run(
    summary_df=raw_summary_df,
    fixations_df=raw_fixations_df,
    blinks_df=raw_blinks_df,
    verbose=True,
    save=True,
    output_dir=dataframes_dir / 'analysis'
)


# CLEANING
cleaner = Cleaner(
    min_fix_duration=80,
    keep_only_inside_screen=True,
)

clean_fixations_df, clean_summary_df = cleaner.run(
    raw_fixations_df=raw_fixations_df,
    trial_summary_df=trial_summary_df,
    verbose=True,
    save=True,
    output_dir=dataframes_dir / 'clean'
)

```


## 5. `Parser`

`Parser` lit un fichier ASC et produit une liste d’objets `RawTrial`.

### Initialisation

```python
parser = Parser(metadata)
```


### Lecture d’un fichier

```python
trials = parser.parse_file(
    filepath="data/asc/ID-001-session.asc",
    participant_id="001",
    modality="visible",
)
```

Pour chaque essai, le parseur extrait :

- l’identifiant du participant ;
- la modalité ;
- le nom de l’image ;
- le préfixe du nom de l’image ;
- l’identifiant de l’image ;
- le début et la fin de la présentation ;
- la résolution de l’écran ;
- la taille affichée du stimulus ;
- les dimensions originales de l’image ;
- les fixations ;
- les clignements.

### Détection des essais

Le début d’un essai correspond à une ligne `MSG` contenant `Affichage_Image`. La fin correspond à une ligne `MSG` contenant `time_out`.

Le timestamp enregistré comme `image_onset` est celui du message `Affichage_Image`. 

### Fixations

Le format attendu est :

```text
EFIX <eye> <start> <end> <duration> <x> <y> <pupil>
```

La valeur de pupille n’est pas conservée.

Une ligne `EFIX` incomplète ou non convertible est ignorée sans interrompre le traitement.

### Clignements

Le format attendu est :

```text
EBLINK <eye> <start> <end> <duration>
```

Les champs conservés sont l’œil, le début, la fin et la durée.

Une ligne incorrecte est ignorée.


## 6. `Extractor`

`Extractor` applique un `Parser` à l’ensemble des fichiers ASC d’un dossier et transforme les objets `RawTrial` en trois DataFrames.

### Initialisation

```python
extractor = Extractor(parser)
```

### Exécution

```python
fixations_df, blinks_df, summary_df = extractor.run(
    asc_dir="data/asc/visible",
    modality="visible",
    verbose=True,
    save=False,
)
```

### Clé d’essai

Chaque essai reçoit une clé de la forme :

```text
<image_stem>_<participant_id>_<modality>_<trial_index>
```

Exemple :

```text
123_007_visible_0042
```

`trial_index` est attribué après concaténation des essais de tous les fichiers. Il est donc global au lot traité et ne recommence pas à zéro pour chaque participant.

### Tableau `summary_df`

Une ligne est créée par essai.

Principales colonnes :

| Colonne | Description |
|---|---|
| `trial_key` | Clé unique de l’essai dans le lot. |
| `trial_index` | Index global de l’essai. |
| `participant_id` | Identifiant extrait du nom du fichier. |
| `image_modality` | Modalité fournie à l’extracteur. |
| `image_name` | Nom canonique obtenu via l’index. |
| `image_stem` | Préfixe extrait du nom du stimulus. |
| `image_id` | Identifiant numérique du stimulus. |
| `image_onset` | Timestamp de début. |
| `image_offset` | Timestamp de fin. |
| `image_duration` | Différence entre fin et début. |
| `screen_width`, `screen_height` | Dimensions de l’écran. |
| `stimuli_width`, `stimuli_height` | Dimensions affichées. |
| `image_width`, `image_height` | Dimensions originales. |
| `n_fixations_raw` | Nombre de fixations extraites. |
| `n_blinks_raw` | Nombre de clignements extraits. |

### Tableau `fixations_df`

Une ligne est créée par fixation.

Colonnes supplémentaires principales :

| Colonne | Description |
|---|---|
| `fixation_id` | Position de la fixation dans l’essai, à partir de `0`. |
| `eye` | Œil enregistré. |
| `start`, `end` | Timestamps absolus. |
| `duration` | Durée en millisecondes. |
| `x`, `y` | Coordonnées dans l’espace écran. |
| `relative_start` | Début relatif au début de l’image. |
| `relative_end` | Fin relative au début de l’image. |

Si aucun essai ne contient de fixation, le DataFrame peut être vide.

### Tableau `blinks_df`

Une ligne est créée par clignement.

Colonnes principales :

- `blink_id` ;
- `eye` ;
- `start` ;
- `end` ;
- `duration` ;
- `relative_start` ;
- `relative_end`.

### Sorties enregistrées

Lorsque `save=True`, les noms logiques transmis à `save_dataframe` sont :

```text
summary
fixations
blinks
```

L’extension, l’éventuel horodatage et la convention exacte de nommage dépendent de l’implémentation de `helpers.utils.save_dataframe`.

## 7. `Profiler`

`Profiler` évalue la qualité des essais à partir des tableaux bruts.

### Paramètres

```python
profiler = Profiler(
    min_fix_duration=80,
    max_blinks_valid=4,
    center_tolerance_deg=2.0,
    n_min_fixation_per_trial=3,
    viewing_distance_cm=60.0,
    screen_width_cm=52.0,
    screen_height_cm=32.5,
)
```

### Exécution

```python
trial_quality_df, participant_summary_df, invalid_trials_df = profiler.run(
    summary_df=summary_df,
    fixations_df=fixations_df,
    blinks_df=blinks_df,
    verbose=True,
    save=True,
    output_dir="outputs/analysis/visible",
)
```

### Fixations valides pour le profilage

Une fixation est considérée comme valide lorsqu’elle respecte simultanément :

```text
duration >= min_fix_duration
coordonnées à l’intérieur de l’écran
```

Les fixations situées en dehors de l’écran sont comptabilisées séparément.

### Critères d’invalidité d’un essai

Un essai est déclaré invalide lorsqu’au moins une des conditions suivantes est vraie :

| Code | Condition |
|---|---|
| `no_fixation` | Aucune fixation brute n’est présente. |
| `too_many_blinks` | Le nombre de clignements est strictement supérieur à `max_blinks_valid`. |
| `first_fixation_not_centered` | La première fixation brute dépasse la tolérance centrale. |
| `no_valid_fixation` | Aucune fixation ne respecte les critères de durée et de position écran. |
| `insufficient_number_of_fixations` | Le nombre de fixations valides est inférieur au minimum configuré. |

Plusieurs causes sont concaténées avec `;` dans `invalid_reasons`.


## 8. `Cleaner`

`Cleaner` élimine les essais invalides et applique les critères de nettoyage aux fixations.

### Paramètres

```python
cleaner = Cleaner(
    min_fix_duration=80,
    keep_only_inside_screen=True,
    remove_initial_fixation=True,
    initial_fixation_id=0,
)
```

### Exécution

```python
clean_fixations_df, clean_summary_df = cleaner.run(
    raw_fixations_df=fixations_df,
    trial_summary_df=trial_quality_df,
    verbose=True,
    save=True,
    output_dir="outputs/clean/visible",
)
```

### Étapes de nettoyage

Les opérations sont appliquées dans l’ordre suivant :

1. conservation des essais pour lesquels `is_invalid_trial` vaut `False` ;
2. suppression optionnelle de la fixation initiale ;
3. suppression des fixations dont la durée est inférieure au seuil ;
4. calcul de l’indicateur `is_inside_screen` ;
5. suppression optionnelle des fixations hors écran ;
6. calcul des coordonnées relatives au stimulus ;
7. tri des lignes.

Les limites de l’écran sont testées avec des comparaisons inclusives :

```text
0 <= x <= screen_width
0 <= y <= screen_height
```

Dans un système où les pixels valides vont de `0` à `screen_width - 1` et de `0` à `screen_height - 1`, les coordonnées exactement égales à la largeur ou à la hauteur sont donc considérées comme internes par le code actuel.

### Reprojection dans l’espace du stimulus

Le stimulus est supposé centré sur l’écran.

Les coordonnées de son coin supérieur gauche sont calculées ainsi :

```text
stimulus_left = (screen_width - stimuli_width) / 2
stimulus_top  = (screen_height - stimuli_height) / 2
```

Puis :

```text
x_stimulus = x - stimulus_left
y_stimulus = y - stimulus_top
```

Le DataFrame reçoit également `is_inside_stimulus`, calculé avec :

```text
0 <= x_stimulus <= stimuli_width
0 <= y_stimulus <= stimuli_height
```

Par défaut, les fixations extérieures au stimulus ne sont pas supprimées. Elles restent dans le tableau avec `is_inside_stimulus=False`. Le paramètre `keep_only_inside_screen` concerne uniquement les limites de l’écran.


