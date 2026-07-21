from __future__ import annotations
from pathlib import Path
from typing import Tuple

import pandas as pd

from helpers.utils import save_dataframe



class Cleaner:

    def __init__(
        self,
        min_fix_duration: int = 80,
        keep_only_inside_screen: bool = True,
        remove_initial_fixation: bool = True,
        initial_fixation_id: int = 0,
    ):
        self.min_fix_duration = min_fix_duration
        self.keep_only_inside_screen = keep_only_inside_screen
        self.remove_initial_fixation = remove_initial_fixation
        self.initial_fixation_id = initial_fixation_id
        
    def run(
        self,
        raw_fixations_df: pd.DataFrame,
        trial_summary_df: pd.DataFrame,
        verbose: bool = False,
        save: bool = False,
        output_dir: str | Path = Path("outputs/clean"),
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        self._check_inputs(raw_fixations_df, trial_summary_df)
        
        clean_fixations_df = self.clean_fixations(
            raw_fixations_df=raw_fixations_df,
            trial_summary_df=trial_summary_df,
        )
        clean_summary_df = self.build_summary(
            raw_fixations_df=raw_fixations_df,
            clean_fixations_df=clean_fixations_df,
            trial_summary_df=trial_summary_df,
        )
        
        
        if save:
            save_dataframe(clean_fixations_df, output_dir, "fixations")
            save_dataframe(clean_summary_df, output_dir, "summary")
        
        if verbose:
            n_raw_fixations = len(raw_fixations_df)
            n_raw_trials = raw_fixations_df["trial_key"].nunique()
            self._print_cleaning_summary(
                clean_fixations_df=clean_fixations_df,
                trial_summary_df=trial_summary_df,
                clean_summary_df=clean_summary_df,
                n_raw_fixations=n_raw_fixations,
                n_raw_trials=n_raw_trials,
            )
            
        return clean_fixations_df, clean_summary_df
    
    def clean_fixations(
        self,
        raw_fixations_df: pd.DataFrame,
        trial_summary_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Nettoie les fixations brutes et construit éventuellement deux fichiers :
        - clean_fixations.csv : fixations conservées après nettoyage ;
        - clean_summary.csv : résumé par image et modalité.

        Le dataframe retourné reste clean_fixations_df pour garder la compatibilité
        avec le pipeline existant.
        """

        valid_trial_keys = trial_summary_df.loc[
            ~trial_summary_df["is_invalid_trial"],
            "trial_key",
        ]

        clean_df = raw_fixations_df[
            raw_fixations_df["trial_key"].isin(valid_trial_keys)
        ].copy()
        
        if self.remove_initial_fixation:
            clean_df = clean_df[
                clean_df["fixation_id"].astype(int) != self.initial_fixation_id
            ].copy()


        clean_df = clean_df[
            clean_df["duration"] >= self.min_fix_duration
        ].copy()

        

        clean_df["is_inside_screen"] = (
            (clean_df["x"] >= 0)
            & (clean_df["x"] <= clean_df["screen_width"])
            & (clean_df["y"] >= 0)
            & (clean_df["y"] <= clean_df["screen_height"])
        )

        if self.keep_only_inside_screen:
            clean_df = clean_df[clean_df["is_inside_screen"]].copy()

        

        clean_df = self._add_stimulus_coordinates(clean_df)

        clean_df = clean_df.sort_values(
            by=[
                "participant_id",
                "image_modality",
                "image_id",
                "trial_index",
                "fixation_id",
            ]
        ).reset_index(drop=True)


        return clean_df

    def build_summary(
        self,
        raw_fixations_df: pd.DataFrame,
        clean_fixations_df: pd.DataFrame,
        trial_summary_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Construit un résumé par image et par modalité.

        Colonnes principales :
        - n_raw_fixations ;
        - n_raw_trials ;
        - n_raw_participants ;
        - n_valid_trials ;
        - n_invalid_trials ;
        - n_valid_participants ;
        - n_clean_fixations ;
        - n_clean_participants ;
        - mean_clean_fixations_per_participant ;
        - invalid_trial_rate ;
        - clean_fixation_rate.
        """

        image_keys = ["image_modality", "image_id", "image_name"]

        raw_summary_per_image = (
            raw_fixations_df
            .groupby(image_keys)
            .agg(
                n_raw_fixations=("fixation_id", "count"),
                n_raw_trials=("trial_key", "nunique"),
                n_raw_participants=("participant_id", "nunique"),
            )
            .reset_index()
        )

        trial_summary_per_image = (
            trial_summary_df
            .groupby(image_keys)
            .agg(
                n_trials=("trial_key", "count"),
                n_valid_trials=("is_invalid_trial", lambda s: int((~s).sum())),
                n_invalid_trials=("is_invalid_trial", lambda s: int(s.sum())),
                n_participants_total=("participant_id", "nunique"),
            )
            .reset_index()
        )

        valid_trials = trial_summary_df[
            ~trial_summary_df["is_invalid_trial"]
        ].copy()

        if valid_trials.empty:
            valid_participants_per_image = pd.DataFrame(
                columns=[*image_keys, "n_valid_participants"]
            )
        else:
            valid_participants_per_image = (
                valid_trials
                .groupby(image_keys)
                .agg(
                    n_valid_participants=("participant_id", "nunique"),
                )
                .reset_index()
            )

        if clean_fixations_df.empty:
            clean_summary_per_image = pd.DataFrame(
                columns=[
                    *image_keys,
                    "n_clean_fixations",
                    "n_clean_participants",
                    "mean_clean_fixations_per_participant",
                ]
            )
        else:
            clean_summary_per_image = (
                clean_fixations_df
                .groupby(image_keys)
                .agg(
                    n_clean_fixations=("fixation_id", "count"),
                    n_clean_participants=("participant_id", "nunique"),
                )
                .reset_index()
            )

            fixations_per_participant = (
                clean_fixations_df
                .groupby([*image_keys, "participant_id"])
                .size()
                .reset_index(name="n_fixations")
            )

            mean_fixations_per_participant = (
                fixations_per_participant
                .groupby(image_keys)
                .agg(
                    mean_clean_fixations_per_participant=("n_fixations", "mean"),
                )
                .reset_index()
            )

            clean_summary_per_image = clean_summary_per_image.merge(
                mean_fixations_per_participant,
                on=image_keys,
                how="left",
            )

        summary = (
            raw_summary_per_image
            .merge(trial_summary_per_image, on=image_keys, how="outer")
            .merge(valid_participants_per_image, on=image_keys, how="outer")
            .merge(clean_summary_per_image, on=image_keys, how="outer")
        )

        numeric_columns = [
            "n_raw_fixations",
            "n_raw_trials",
            "n_raw_participants",
            "n_trials",
            "n_valid_trials",
            "n_invalid_trials",
            "n_participants_total",
            "n_valid_participants",
            "n_clean_fixations",
            "n_clean_participants",
        ]

        for column in numeric_columns:
            if column in summary.columns:
                summary[column] = summary[column].fillna(0).astype(int)

        if "mean_clean_fixations_per_participant" in summary.columns:
            summary["mean_clean_fixations_per_participant"] = (
                summary["mean_clean_fixations_per_participant"]
                .fillna(0.0)
                .astype(float)
            )

        summary["invalid_trial_rate"] = summary.apply(
            lambda row: (
                row["n_invalid_trials"] / row["n_trials"]
                if row["n_trials"] > 0
                else 0.0
            ),
            axis=1,
        )

        summary["clean_fixation_rate"] = summary.apply(
            lambda row: (
                row["n_clean_fixations"] / row["n_raw_fixations"]
                if row["n_raw_fixations"] > 0
                else 0.0
            ),
            axis=1,
        )

        return summary.sort_values(
            by=["image_modality", "image_id"]
        ).reset_index(drop=True)

    def _print_cleaning_summary(
        self,
        clean_fixations_df: pd.DataFrame,
        trial_summary_df: pd.DataFrame,
        clean_summary_df: pd.DataFrame,
        n_raw_fixations: int,
        n_raw_trials: int,
    ) -> None:
        
        n_after_cleaning = len(clean_fixations_df)
        n_invalid_trials = int(trial_summary_df["is_invalid_trial"].sum())
        n_valid_trials = int((~trial_summary_df["is_invalid_trial"]).sum())

        print("\n----- TRIALS -----")
        print(f"Trials (raw):     {n_raw_trials}")
        print(f"Trials (valid):   {n_valid_trials}")
        print(f"Trials (invalid): {n_invalid_trials}")

        print("\n----- FIXATIONS -----")
        print(f"Raw fixations:   {n_raw_fixations}")
        print(f"Clean fixations: {n_after_cleaning}")

        if n_raw_fixations > 0:
            kept_rate = len(clean_fixations_df) / n_raw_fixations * 100
            print(f"Keeping rate: {kept_rate:.2f}%")


        print("\n----- PER IMAGE -----")

        n_images = clean_summary_df[["image_modality", "image_id"]].drop_duplicates().shape[0]
        print(f"\nAnalysed images: {n_images}")
        print(
            f"\nValid participant per image (mean): "
            f"{clean_summary_df['n_valid_participants'].mean():.2f}"
        )
        print(
            f"Valid participant per image (std): "
            f"{clean_summary_df['n_valid_participants'].std():.2f}"
        )
        print(
            f"Valid participant per image (min): "
            f"{clean_summary_df['n_valid_participants'].min()}"
        )
        print(
            f"Valid participant per image (max): "
            f"{clean_summary_df['n_valid_participants'].max()}"
        )

        print(
            f"\nClean fixations per image (mean): "
            f"{clean_summary_df['n_clean_fixations'].mean():.2f}"
        )
        print(
            f"Clean fixations per image (std): "
            f"{clean_summary_df['n_clean_fixations'].std():.2f}"
        )
        print(
            f"Clean fixations per image (min): "
            f"{clean_summary_df['n_clean_fixations'].min()}"
        )
        print(
            f"Clean fixations per image (max): "
            f"{clean_summary_df['n_clean_fixations'].max()}"
        )

        print(
            f"\nRatio of invalid trials per image (mean): "
            f"{clean_summary_df['invalid_trial_rate'].mean() * 100:.2f}%"
        )

        print(
            f"Ratio of kept fixations per image (mean): "
            f"{clean_summary_df['clean_fixation_rate'].mean() * 100:.2f}%"
        )

       

    def _add_stimulus_coordinates(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        df = df.copy()

        df["stimulus_left"] = (
            df["screen_width"] - df["stimuli_width"]
        ) / 2

        df["stimulus_top"] = (
            df["screen_height"] - df["stimuli_height"]
        ) / 2

        df["x_stimulus"] = df["x"] - df["stimulus_left"]
        df["y_stimulus"] = df["y"] - df["stimulus_top"]

        df["is_inside_stimulus"] = (
            (df["x_stimulus"] >= 0)
            & (df["x_stimulus"] <= df["stimuli_width"])
            & (df["y_stimulus"] >= 0)
            & (df["y_stimulus"] <= df["stimuli_height"])
        )

        return df

    @staticmethod
    def _check_inputs(
        raw_fixations_df: pd.DataFrame,
        trial_summary_df: pd.DataFrame,
    ) -> None:

        required_fixation_columns = {
            "trial_key",
            "participant_id",
            "image_modality",
            "image_id",
            "image_name",
            "trial_index",
            "fixation_id",
            "duration",
            "x",
            "y",
            "screen_width",
            "screen_height",
            "stimuli_width",
            "stimuli_height",
        }

        required_quality_columns = {
            "trial_key",
            "participant_id",
            "image_modality",
            "image_id",
            "image_name",
            "is_invalid_trial",
        }

        missing_fixation_columns = (
            required_fixation_columns - set(raw_fixations_df.columns)
        )

        missing_quality_columns = (
            required_quality_columns - set(trial_summary_df.columns)
        )

        if missing_fixation_columns:
            raise ValueError(
                f"Missing columns in raw_fixations_df: "
                f"{sorted(missing_fixation_columns)}"
            )

        if missing_quality_columns:
            raise ValueError(
                f"Missing columns in trial_summary_df: "
                f"{sorted(missing_quality_columns)}"
            )
