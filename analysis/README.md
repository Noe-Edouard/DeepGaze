# Analyse et comparaison des cartes de saillance

Ce dossier contient les outils utilisés pour comparer les cartes de saillance humaines entre modalités, évaluer les prédictions de DeepGaze et appliquer les tests statistiques associés.

Le pipeline prend en entrée des fixations nettoyées, construit les cartes nécessaires par l’intermédiaire du dossier `saliency`, calcule plusieurs métriques de similarité ou de prédictivité, puis produit des tableaux de résultats et des analyses statistiques.

## 1. Organisation du dossier

```text
analysis/
├── analyzer.py
├── comparator.py
├── metrics.py
├── saver.py
├── stats.py
└── README.md
```

Les modules ont les responsabilités suivantes :

| Module | Rôle |
|---|---|
| `metrics.py` | Définition des métriques de similarité, de prédictivité et de région d’intérêt. |
| `comparator.py` | Comparaisons humain–humain et DeepGaze–humain. |
| `stats.py` | Tests statistiques élémentaires et corrections pour comparaisons multiples. |
| `analyzer.py` | Organisation des tests statistiques par expérience et par métrique. |
| `saver.py` | Conversion des résultats en CSV et JSON. |


## 2. Prérequis

Le dossier utilise notamment :

- Python 3.10 ou une version ultérieure ;
- NumPy ;
- pandas ;
- SciPy ;
- scikit-learn ;
- Pingouin ;
- tqdm ;
- PyTorch et DeepGaze par l’intermédiaire du dossier `saliency` ;
- POT, importé sous le nom `ot`, pour la distance de Wasserstein approchée ;
- les modules locaux `helpers` et `saliency`.


## 3. Métriques

La classe `Metrics` regroupe les fonctions dans un registre :

```python
metrics.registry
```

Structure actuelle :

```text
similarity
├── cc
├── sim
└── kl

predictivity
├── nss
├── auc_judd
└── ig

roi
├── mass
├── density
├── ratio
└── normalized_ratio
```

Les métriques `emd` et `ll` sont implémentées, mais désactivées dans le registre.
Les métriques roi ne sont pas utilisées dans le pipeline.

L'initisation se fait de la manière suivante.

```python
from analysis.metrics import Metrics

metrics = Metrics(epsilon=1e-12)
```

`epsilon` évite les divisions par zéro et les logarithmes nuls.


## 4. `Comparator`

La classe `Comparator` est responsable des comparaison humain-humain et deepgaze-humain.
### Initialisation

```python
from analysis.comparator import Comparator

comparator = Comparator(
    metadata_json="metadata/metadata.json",
    annotations_json="metadata/annotations.json",
    sigma=20.0,
    epsilon=1e-12,
)
```


### Cohérence intra-modale

`compute_intra_scores()` évalue la cohérence entre participants pour une image.

```python
scores = comparator.compute_intra_scores(
    data=dataset,
    stem="123",
    n_splits=50,
    seed=42,
)
```

Pour chaque répétition :

1. les participants sont mélangés ;
2. ils sont divisés en deux groupes ;
3. une carte agrégée est construite par groupe ;
4. chaque carte est lissée ;
5. les groupes sont comparés dans les deux directions ;
6. la moyenne des deux directions est enregistrée.

Le résultat final est la moyenne sur les répétitions.


### Expérience humain visible–humain infrarouge

```python
results = comparator.run_expe1(
    infrared_df=infrared_fixations_df,
    visible_df=visible_fixations_df,
    n_splits=50,
)
```

Pour chaque image, le comparateur calcule :

| Variable | Description |
|---|---|
| `vv` | Cohérence split-half visible–visible ($M_{H-H}^{RGB}$ dans la présentation)|
| `ii` | Cohérence split-half infrarouge–infrarouge ($M_{H-H}^{LWIR}$ dans la présentation) |
| `vi` | Visible humain évalué avec la carte infrarouge. |
| `iv` | Infrarouge humain évalué avec la carte visible. |
| `intra` | Moyenne de `vv` et `ii` ($M_{H-H}^{intra}$ dans la présentation) |
| `inter` | Moyenne de `vi` et `iv` ($M_{H-H}^{inter}$ dans la présentation) |
| `delta` | Avantage intra-modal orienté positivement ($\Delta_{H-H}$ dans la présentation) |
| `daynight` | Condition de visibilité obtenue depuis les annotations. |

Pour les métriques où une valeur élevée est meilleure, `delta=intra-inter` et pour les autres, `delta=inter-intra` si bien que `delta>0`  représente normalement une meilleure correspondance intra-modale qu’inter-modale.

Les annotations visible et infrarouge sont récupérées dans la méthode, mais elles ne sont pas utilisées dans les résultats actuels.

### Expérience DeepGaze–humain


```python
from pathlib import Path

results = comparator.run_expe2(
    infrared_df=infrared_fixations_df,
    visible_df=visible_fixations_df,
    infrared_image_dir=Path("stimuli/infrared"),
    visible_image_dir=Path("stimuli/visible"),
    infrared_centerbias_path=Path("centerbias/infrared.npy"),
    visible_centerbias_path=Path("centerbias/visible.npy"),
    n_splits=50,
)
```

Deux modèles `DeepGaze` sont instanciés :

- un avec le biais central visible ;
- un avec le biais central infrarouge.

Pour chaque image, les valeurs suivantes sont produites :

| Variable | Description |
|---|---|
| `vv` | Plafond humain split-half visible ($M_{H-H}^{RGB}$ dans la présentation) |
| `ii` | Plafond humain split-half infrarouge ($M_{H-H}^{RGB}$ dans la présentation) |
| `vd` | Comparaison humain visible–DeepGaze visible ($M_{DG-H}^{RGB}$ dans la présentation) |
| `id` | Comparaison humain infrarouge–DeepGaze infrarouge ($M_{DG-H}^{LWIR}$ dans la présentation)|
| `delta_v` | Écart entre plafond humain et DeepGaze en visible ($\Delta_{M}^{RGB}$ dans la présentation) |
| `delta_i` | Écart entre plafond humain et DeepGaze en infrarouge ($\Delta_{M}^{LWIR}$ dans la présentation) |
| `delta` | Différence `delta_i - delta_v`. ($\Delta_{M}^{DG-H}$ dans la présentation)|
| `daynight` | Condition de visibilité. |

Pour les métriques où une valeur élevée est meilleure, `delta_v = vv - vd`, sinon, `delta_i = ii - id`. Dans les deux cas, une valeur positive indique que DeepGaze est moins performant que le plafond humain.

La différence finale est `delta = delta_i - delta_v`. Un `delta > 0` indique un déficit relatif de DeepGaze plus important en infrarouge
tandis qu'un `delta < 0` indique un déficit relatif plus important en visible


## 5. Analyse statistique

La classe `Stats` fournit les tests élémentaires. La classe `Analyzer` organise ces tests pour les deux expériences.

### Initialisation

```python
from analysis.analyzer import Analyzer

analyzer = Analyzer(
    alpha=0.05,
    correction_method="holm",
)
```

### Test t

`Stats.run_ttest()` prend en charge :

- un test à un échantillon contre une constante ;
- un test apparié ;
- un test indépendant.

Exemple apparié :

```python
result = analyzer.stats.run_ttest(
    x=df["ii"],
    y=df["vv"],
    test_name="ii_vs_vv",
    paired=True,
    alternative="two-sided",
)
```

Exemple à un échantillon :

```python
result = analyzer.stats.run_ttest(
    x=df["delta_i"],
    y=0,
    test_name="delta_i_vs_0",
    alternative="greater",
)
```

Le résultat peut contenir :

```text
test
test_type
alternative
n
mean_x
mean_y
mean_diff
std_x
std_y
std_diff
T
dof
p_value
cohen_d
CI95
power
BF10
significant
```

Lorsque l’effectif est insuffisant, le dictionnaire descriptif est retourné sans statistique de test.

### Correction multiple

```python
analyzer.stats.apply_correction(
    test_results=list(results.values()),
    correction_method="holm",
)
```

Les champs ajoutés sont :

```text
p_value_corrected
significant_corrected
correction_method
```

Les corrections sont appliquées séparément à chaque famille définie dans `Analyzer`.

### ANOVA

```python
result = analyzer.stats.run_anova(
    df=df,
    value_col="delta",
    factor_col="daynight",
)
```

La méthode réalise une ANOVA intergroupes avec Pingouin et retourne notamment :

```text
F
ddof1
ddof2
p_value
np2
```

Elle n’est pas utilisée par les méthodes principales de `Analyzer` dans la version actuelle.



## 6. Analyse statistique de l’expérience 1

### Entrée attendue

`Analyzer.analyze_expe1()` attend un DataFrame contenant :

```text
metric_type
metric_name
vv
ii
intra
inter
delta
daynight
```

Les valeurs `vv`, `ii`, `intra`, `inter` et `delta` sont converties en valeurs numériques.

### Exécution

```python
analysis = analyzer.analyze_expe1(expe1_results_df)
```

Trois familles sont produites :

```text
consistency
divergence
daynight
```

#### Cohérence visible–infrarouge

```text
ii contre vv
```

Test t apparié bilatéral.

Il évalue une différence de cohérence humaine entre les modalités.

#### Intra contre inter

```text
intra contre inter
```

Test t apparié unilatéral avec :

```text
alternative = "greater"
```

Cette alternative est cohérente pour les métriques où une valeur élevée est meilleure, telles que CC, SIM, NSS, AUC ou IG.

Elle n’est pas cohérente pour `kl`, où une valeur faible représente une meilleure correspondance. Pour `kl`, l’hypothèse directionnelle devrait normalement être inversée, ou le test devrait porter sur le `delta` déjà orienté.

Une approche uniforme consiste à tester directement :

```text
delta > 0
```

pour toutes les métriques.

#### Effet du moment de la journée

Les valeurs de `delta` sont comparées entre les deux conditions de luminosité avec un test indépendant bilatéral.

Le code suppose qu’il existe exactement deux valeurs non nulles :

```python
vis_a, vis_b = metric_df["daynight"].dropna().unique()
```

Zéro, une ou plus de deux conditions provoquent une erreur de déballage.

La correction de Welch est demandée à Pingouin avec `correction=True`, puis une correction pour comparaisons multiples est appliquée entre métriques.

## 7. Analyse statistique de l’expérience 2

### Entrée attendue

`Analyzer.analyze_expe2()` attend :

```text
metric_type
metric_name
delta_v
delta_i
delta
daynight
```

### Tests prévus

Pour chaque métrique :

#### Visible

```text
delta_v > 0
```

Test à un échantillon unilatéral.

#### Infrarouge

```text
delta_i > 0
```

Test à un échantillon unilatéral.

#### Différence de déficit

```text
delta != 0
```

Test à un échantillon bilatéral.

#### Effet du moment de la journée

Pour chacune des trois variables, comparaison indépendante entre les deux conditions de luminosité.


## 8. Exemple de pipeline complet

```python
from pathlib import Path

import pandas as pd

from analysis.analyzer import Analyzer
from analysis.comparator import Comparator
from analysis.saver import Saver


visible_df = pd.read_csv("data/visible_fixations.csv")
infrared_df = pd.read_csv("data/infrared_fixations.csv")

comparator = Comparator(
    metadata_json="metadata/metadata.json",
    annotations_json="metadata/annotations.json",
    sigma=20.0,
    epsilon=1e-12,
)

saver = Saver("outputs")
analyzer = Analyzer(
    alpha=0.05,
    correction_method="holm",
)

# Expérience humain–humain
expe1_results = comparator.run_expe1(
    infrared_df=infrared_df,
    visible_df=visible_df,
    n_splits=50,
)

expe1_table = saver.save_expe1_results(
    expe1_results,
)

expe1_analysis = analyzer.analyze_expe1(
    expe1_table,
)

saver.save_expe1_analysis(
    expe1_analysis,
)

# Expérience DeepGaze–humain
expe2_results = comparator.run_expe2(
    infrared_df=infrared_df,
    visible_df=visible_df,
    infrared_image_dir=Path("stimuli/infrared"),
    visible_image_dir=Path("stimuli/visible"),
    infrared_centerbias_path=Path("centerbias/infrared.npy"),
    visible_centerbias_path=Path("centerbias/visible.npy"),
    n_splits=50,
)

# Après correction de save_expe2_results()
expe2_table = saver.save_expe2_results(
    expe2_results,
)

# Après correction des clés daynight/visibility
expe2_analysis = analyzer.analyze_expe2(
    expe2_table,
)

saver.save_expe2_analysis(
    expe2_analysis,
)
```