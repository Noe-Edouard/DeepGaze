# Construction des cartes de saillance

Ce dossier contient les outils utilisés pour construire les cartes de fixations, les cartes de densité de fixations humaines et les cartes de saillance pour le projet.

Il couvre trois fonctions principales :

1. construction de cartes de fixation humaines, individuelles ou agrégées ;
2. construction du biais central ;
3. production de cartes de saillance avec DeepGaze IIE.

Toutes les cartes finales sont représentées par des tableaux NumPy bidimensionnels. Les cartes probabilistes sont normalisées de manière à avoir une somme égale à `1`, sauf lorsqu’aucune fixation n’est disponible.

## 1. Organisation du dossier

```text
saliency/
├── builder.py
├── centerbias.py
├── deepgaze.py
└── README.md
```

Les modules remplissent les rôles suivants :

| Module | Rôle |
|---|---|
| `builder.py` | Construction des cartes de fixation et des densités humaines. |
| `centerbias.py` | Estimation d’un biais central à partir de fixations humaines. |
| `deepgaze.py` | Inférence avec le modèle préentraîné DeepGaze IIE. |

`builder.py` dépend également des structures `Dataset` et `FixationData` définies dans `helpers.models`.


## 2. Prérequis

Les modules utilisent :

- Python 3.10 ou une version ultérieure ;
- NumPy ;
- pandas ;
- SciPy ;
- OpenCV ;
- PyTorch ;
- le package fournissant `deepgaze_pytorch` ;
- les structures du package local `helpers`.


Le package `deepgaze_pytorch` et ses poids préentraînés doivent être installés ou rendus disponibles dans l’environnement utilisé pour l’inférence.


## 3. Cartes humaines avec `Builder`

### Données attendues

`Builder.build_fixation_map()` reçoit un objet `Dataset` indexable par `stem` :

```python
stem_data = data[stem]
```

Chaque entrée doit fournir au minimum :

```text
stem_data.dimensions_data.image_width
stem_data.dimensions_data.image_height
stem_data.dimensions_data.screen_width
stem_data.dimensions_data.screen_height
stem_data.dimensions_data.stimuli_width
stem_data.dimensions_data.stimuli_height
stem_data.participants_data
```

Les coordonnées d’entrée sont exprimées dans l’espace complet de l’écran.

### Construction d’une carte individuelle

```python
from saliency.builder import Builder

builder = Builder()

participant_map = builder.build_fixation_map(
    data=dataset,
    stem="123",
    participant_id="007",
)
```

### Projection écran vers image

Le stimulus est supposé centré sur l’écran.

Le coin supérieur gauche du stimulus est calculé par :

```text
offset_x = (screen_width - stimuli_width) / 2
offset_y = (screen_height - stimuli_height) / 2
```

Les coordonnées dans l’espace du stimulus sont ensuite :

```text
x_stimulus = x_screen - offset_x
y_stimulus = y_screen - offset_y
```

Les fixations sont conservées uniquement lorsque :

```text
0 <= x_stimulus < stimuli_width
0 <= y_stimulus < stimuli_height
```

La conversion vers les dimensions originales de l’image est :

```text
x_image = floor(x_stimulus × image_width / stimuli_width)
y_image = floor(y_stimulus × image_height / stimuli_height)
```

Cette transformation suppose que l’image a été affichée sans déformation et centrée, ce qui correspond au comportement du module `eyetracking`.

### Agrégation de cartes individuelles

```python
participant_maps = [
    map_participant_1,
    map_participant_2,
    map_participant_3,
]

aggregate_map = builder.aggregate_fixation_maps(participant_maps)
```

### Construction d’une densité humaine

```python
human_density = builder.build_saliency_map(
    fixation_map=aggregate_map,
    sigma=20.0,
)
```

Le traitement applique un filtre gaussien avec `scipy.ndimage.gaussian_filter`, puis normalise la carte :

```text
density = gaussian_filter(fixation_map, sigma)
density = density / density.sum()
```

La somme est égale à `1` lorsqu’au moins une fixation est présente.

Le paramètre `sigma` est exprimé en pixels de l’image originale. Sa valeur doit donc être adaptée à la résolution des images ou calculée à partir d’un angle visuel défini dans la méthodologie.


### Exemple de génération complète


```python
from pathlib import Path
from saliency.builder import Builder

builder = Builder()
loader = Loader("data/metadata/metadata.json")

stem = "000123"
modality = "visible"
dataframe = loader.load_dataframe("data/fixation.csv")
dataset = loader.load_dataset(dataframe)

participant_maps = []

for participant_id in dataset[stem].participants_data:
    participant_map = builder.build_fixation_map(
        data=dataset,
        stem=stem,
        participant_id=participant_id,
    )

    participant_maps.append(participant_map)

aggregate_map = builder.aggregate_fixation_maps(participant_maps)

density_map = builder.build_saliency_map(
    aggregate_map,
    sigma=20.0,
)

```

## 4. Construction du biais central

`build_centerbias()` construit une distribution spatiale globale à partir d’une ou plusieurs sources de fixations.

### Sources acceptées

Exemple avec un DataFrame :

```python
from saliency.centerbias import build_centerbias

centerbias = build_centerbias(
    fixation_sources=clean_fixations_df,
    output_path="outputs/centerbias/global.npy",
)
```

Exemple avec plusieurs fichiers :

```python
centerbias = build_centerbias(
    fixation_sources=[
        "outputs/clean/visible/fixations.csv",
        "outputs/clean/infrared/fixations.csv",
    ],
    output_path="outputs/centerbias/all_modalities.npy",
)
```

### Pondération optionnelle

Une colonne de poids peut être fournie :

```python
centerbias = build_centerbias(
    fixation_sources=fixations_df,
    output_path="outputs/centerbias/weighted.npy",
    weight_col="weight",
)
```

Les poids négatifs sont ramenés à zéro. Les valeurs non finies deviennent nulles.

La pondération est utile lorsque chaque image ou chaque participant doit contribuer de manière équivalente. Sans pondération, chaque fixation contribue avec le même poids ; une image ou un participant ayant davantage de fixations influence donc davantage le biais estimé.

### Leave-one-out

Pour évaluer une image sans utiliser ses propres fixations dans le biais central :

```python
centerbias = build_centerbias(
    fixation_sources=fixations_df,
    output_path="outputs/centerbias/leave_one_out/123.npy",
    leave_out_stem="123",
)
```

Cette procédure évite d’introduire directement les fixations de l’image évaluée dans le prior utilisé par le modèle.

Le leave-one-out doit être appliqué de manière cohérente lors de la comparaison entre DeepGaze et les données humaines.


## 5. Prédictions DeepGaze

La classe `DeepGaze` encapsule le modèle préentraîné `DeepGazeIIE`.

### Initialisation

```python
from saliency.deepgaze import DeepGaze

model = DeepGaze()
```

Par défaut :

- CUDA est utilisé lorsqu’il est disponible ;
- le CPU est utilisé sinon ;
- un biais central uniforme de `1024 × 1024` est créé ;
- le modèle DeepGaze IIE préentraîné est chargé ;
- le modèle est placé en mode évaluation.

L’instanciation doit être effectuée une seule fois avant de parcourir les images, car le chargement du modèle est coûteux.

### Types de biais central

#### Biais uniforme

```python
model = DeepGaze(
    centerbias_type="uniform",
    centerbias_size=(1024, 1024),
)
```

Un tableau nul est utilisé comme log-biais avant normalisation, ce qui correspond à une distribution uniforme.

#### Biais gaussien

```python
model = DeepGaze(
    centerbias_type="gaussian",
    centerbias_size=(1024, 1024),
)
```

Dans la version actuelle, toute valeur de `centerbias_type` différente de `"uniform"` déclenche la construction d’un biais gaussien. Il n’existe pas de validation explicite des valeurs.

#### Biais chargé depuis un fichier

```python
model = DeepGaze(
    centerbias_path="outputs/centerbias/visible.npy",
)
```

Le biais est redimensionné automatiquement à la taille de chaque image. Les versions redimensionnées sont mises en cache selon leur couple `(hauteur, largeur)`.

### Inférence

```python
density = model.predict(image)
```

La sortie :

- est un tableau NumPy bidimensionnel ;
- a la même hauteur et la même largeur que l’image d’entrée ;
- utilise une densité positive ;
- est normalisée par une opération `logsumexp` ;
- possède une somme numériquement proche de `1`.
