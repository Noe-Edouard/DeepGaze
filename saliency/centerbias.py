from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import pandas as pd

FixationSource = str | Path | pd.DataFrame


# À placer au niveau du module (statique), pas dans la fonction
_LOADERS = {
    ".csv": pd.read_csv,
    ".tsv": lambda p: pd.read_csv(p, sep="\t"),
    ".parquet": pd.read_parquet,
    ".pkl": pd.read_pickle,
    ".pickle": pd.read_pickle,
    ".xlsx": pd.read_excel,
    ".xls": pd.read_excel,
}

def _load_df(source: FixationSource) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    
    path = Path(source)
    ext = path.suffix.lower()
    
    if ext not in _LOADERS:
        raise ValueError(f"Format non supporté : {path}")
    return _LOADERS[ext](path)


def build_centerbias(
    fixation_sources: FixationSource | Sequence[FixationSource],
    output_path: str | Path,
    template_shape: tuple[int, int] = (512, 512),
    x_col: str = "x",
    y_col: str = "y",
    screen_width_col: str = "screen_width",
    screen_height_col: str = "screen_height",
    stimuli_width_col: str = "stimuli_width",
    stimuli_height_col: str = "stimuli_height",
    image_width_col: str = "image_width",
    image_height_col: str = "image_height",
    image_stem_col: str = "image_stem",
    weight_col: Optional[str] = None,
    sigma: float = 24.0,
    epsilon: float = 1e-12,
    leave_out_stem: Optional[str] = None,
    clip_outside: bool = False,
) -> np.ndarray:
    """Construit un centerbias normalisé à partir des sources de fixations données."""
    
    # 1. Chargement et fusion des données
    if isinstance(fixation_sources, (str, Path, pd.DataFrame)):
        fixation_sources = [fixation_sources]
    df = pd.concat([_load_df(src) for src in fixation_sources], ignore_index=True)

    # 2. Validation des colonnes requises (mise à jour avec le référentiel écran/stimulus)
    required = {
        x_col, y_col, 
        screen_width_col, screen_height_col, 
        stimuli_width_col, stimuli_height_col, 
        image_width_col, image_height_col
    }
    if leave_out_stem is not None:
        required.add(image_stem_col)
    if weight_col is not None:
        required.add(weight_col)
        
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {sorted(missing)}")

    # 3. Filtrage par image_stem (Leave-one-out)
    if leave_out_stem is not None:
        df = df[df[image_stem_col].astype(str) != str(leave_out_stem)]

    # 4. Nettoyage des coordonnées et dimensions manquantes/invalides
    df = df.dropna(subset=list(required))
    df = df[
        (df[screen_width_col] > 0) & (df[screen_height_col] > 0) &
        (df[stimuli_width_col] > 0) & (df[stimuli_height_col] > 0) &
        (df[image_width_col] > 0) & (df[image_height_col] > 0)
    ]

    if df.empty:
        raise ValueError("Aucune donnée valide après filtrage.")

    # 5. CORRECTION : Projection des coordonnées Écran -> Image (Copie de la logique du Visualizer)
    left = (df[screen_width_col] - df[stimuli_width_col]) / 2
    top = (df[screen_height_col] - df[stimuli_height_col]) / 2

    x_image = (df[x_col] - left) * df[image_width_col] / df[stimuli_width_col]
    y_image = (df[y_col] - top) * df[image_height_col] / df[stimuli_height_col]

    # Maintenant on normalise de manière safe entre 0.0 et 1.0 en utilisant les coordonnées "Image"
    x_norm = x_image.to_numpy() / df[image_width_col].to_numpy()
    y_norm = y_image.to_numpy() / df[image_height_col].to_numpy()

    if clip_outside:
        x_norm = np.clip(x_norm, 0.0, 1.0 - 1e-12)
        y_norm = np.clip(y_norm, 0.0, 1.0 - 1e-12)
    else:
        valid = (x_norm >= 0.0) & (x_norm < 1.0) & (y_norm >= 0.0) & (y_norm < 1.0)
        x_norm, y_norm = x_norm[valid], y_norm[valid]
        df = df[valid]

    if len(x_norm) == 0:
        raise ValueError("Aucune fixation restante dans les limites de l'image.")

    # 6. Gestion des poids optionnels
    weights = None
    if weight_col is not None:
        weights = np.nan_to_num(df[weight_col].to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
        weights = np.maximum(weights, 0.0)
        if weights.sum() <= 0:
            raise ValueError(f"Tous les poids de la colonne '{weight_col}' sont invalides ou nuls.")

    # 7. Création de la matrice d'accumulation via un histogramme 2D numérique
    h, w = template_shape
    fixation_count, _, _ = np.histogram2d(
        y_norm, x_norm, bins=[h, w], range=[[0, 1], [0, 1]], weights=weights
    )

    # 8. Floutage Gaussien et Normalisation de la densité
    centerbias = cv2.GaussianBlur(
        fixation_count, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma), borderType=cv2.BORDER_REPLICATE
    )
    centerbias = (centerbias + float(epsilon))
    centerbias /= centerbias.sum()

    # 9. Sauvegarde au format .npy et retour
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, centerbias.astype(np.float32))

    return centerbias.astype(np.float32)