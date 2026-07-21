from __future__ import annotations
from tqdm import tqdm
from pathlib import Path

import math
import pandas as pd

from helpers.utils import save_dataframe 



class Profiler:

    def __init__(
        self,
        min_fix_duration: int = 80,
        max_blinks_valid: int = 4,
        center_tolerance_deg: float = 2.0,
        n_min_fixation_per_trial: int = 3, 
        viewing_distance_cm: float = 60.0,
        screen_width_cm: float = 52.0,
        screen_height_cm: float = 32.5,
    ):
        self.min_fix_duration = min_fix_duration
        self.max_blinks_valid = max_blinks_valid
        self.center_tolerance_deg = center_tolerance_deg
        self.n_min_fixation_per_trial = n_min_fixation_per_trial
        self.viewing_distance_cm = viewing_distance_cm
        self.screen_width_cm = screen_width_cm
        self.screen_height_cm = screen_height_cm

    def run(
        self,
        summary_df: pd.DataFrame,
        fixations_df: pd.DataFrame,
        blinks_df: pd.DataFrame,
        verbose: bool = False,
        save: bool = False,
        output_dir: str | Path = Path("outputs/analysis/")
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        trial_summary_df = self.build_trial_summary_df(
            summary_df=summary_df,
            fixations_df=fixations_df,
            blinks_df=blinks_df,
        )

        participant_summary_df = self.build_participant_summary_df(
            trial_summary_df=trial_summary_df,
        )

        invalid_trials_df = trial_summary_df[
            trial_summary_df["is_invalid_trial"]
        ].copy()

        # SAVES 
        if save:
            save_dataframe(trial_summary_df, output_dir , "trial_quality")
            save_dataframe(invalid_trials_df, output_dir , "invalid_trials")
            save_dataframe(participant_summary_df, output_dir , "participant_summary")
            

        # STATS
        if verbose:
            n_trials = trial_summary_df.shape[0]
            n_invalid = invalid_trials_df.shape[0]
            invalid_rate = (n_invalid / n_trials) * 100 if n_trials > 0 else 0

            # GLOBAL
            print(f"\n----- GLOBAL -----")
            print(f"Trials: {n_trials}")
            print(f"Invalid trials: {n_invalid} ({invalid_rate:.2f}%)")

            print(f"Blink per trial (mean): {trial_summary_df['n_blinks'].mean():.2f}")
            print(f"Blink per trial (std):  {trial_summary_df['n_blinks'].std():.2f}")

            print(f"Total fixations (mean): {trial_summary_df['n_fixations_total'].mean():.2f}")
            print(f"Total fixations (std):  {trial_summary_df['n_fixations_total'].std():.2f}")
            print(f"Total fixations (mean): {trial_summary_df['n_fixations_total'].mean():.2f}")
            print(f"Valid fixations (std):  {trial_summary_df['n_fixations_valid'].std():.2f}")
            n_bad_center = (~trial_summary_df["first_fixation_centered"]).sum()
            print(f"Not centered: {n_bad_center} ({(n_bad_center / n_trials * 100):.2f}%)")

            # DISTANCES 
            print(f"\n----- DISTANCES -----")
            print(f"Centre distance (mean): {trial_summary_df['mean_distance_to_center'].mean():.2f}")
            print(f"Centre distance (std): {trial_summary_df['std_distance_to_center'].mean():.2f}")
            print(f"Successive distance (mean): {trial_summary_df['mean_successive_distance'].mean():.2f}")
            print(f"Successive distance (std): {trial_summary_df['std_successive_distance'].mean():.2f}")

            # INVALID REASONS
            print(f"\n----- INVALIDITY -----")
            if not invalid_trials_df.empty:
                print(invalid_trials_df["invalid_reasons"].value_counts())
            else:
                print("No valid trials")

          
        return trial_summary_df, participant_summary_df, invalid_trials_df


    def build_trial_summary_df(
        self,
        summary_df: pd.DataFrame,
        fixations_df: pd.DataFrame,
        blinks_df: pd.DataFrame,
    ) -> pd.DataFrame:

        rows = []

        iterator = tqdm(
            summary_df.iterrows(),
            total=len(summary_df),
            desc="Profiling trials",
        )

        for _, trial in iterator:
            trial_key = trial["trial_key"]

            trial_fixations = fixations_df[
                fixations_df["trial_key"] == trial_key
            ].copy()

            trial_blinks = blinks_df[
                blinks_df["trial_key"] == trial_key
            ].copy() if not blinks_df.empty else pd.DataFrame()

            rows.append(
                self._profile_trial(
                    trial=trial,
                    fixations=trial_fixations,
                    blinks=trial_blinks,
                )
            )

        return pd.DataFrame(rows)

    def _profile_trial(
        self,
        trial: pd.Series,
        fixations: pd.DataFrame,
        blinks: pd.DataFrame,
    ) -> dict:

        screen_width = trial["screen_width"]
        screen_height = trial["screen_height"]

        center_x = screen_width / 2
        center_y = screen_height / 2

        n_blinks = len(blinks)
        n_fixations_total = len(fixations)

        if fixations.empty:
            return self._empty_trial_row(
                trial=trial,
                n_blinks=n_blinks,
                invalid_reason="no_fixation",
            )

        fixations = fixations.sort_values("fixation_id").copy()

        fixations["is_valid_duration"] = (
            fixations["duration"] >= self.min_fix_duration
        )

        fixations["is_outside_screen"] = (
            (fixations["x"] < 0)
            | (fixations["x"] > screen_width)
            | (fixations["y"] < 0)
            | (fixations["y"] > screen_height)
        )

        valid_fixations = fixations[
            fixations["is_valid_duration"]
            & ~fixations["is_outside_screen"]
        ].copy()

        n_fixations_valid = len(valid_fixations)
        n_fixations_outside_screen = int(fixations["is_outside_screen"].sum())

        first_fixation = fixations.iloc[0]

        first_dx_deg = self._px_to_deg_x(
            first_fixation["x"] - center_x,
            screen_width,
        )

        first_dy_deg = self._px_to_deg_y(
            first_fixation["y"] - center_y,
            screen_height,
        )

        first_fixation_centered = (
            abs(first_dx_deg) <= self.center_tolerance_deg
            and abs(first_dy_deg) <= self.center_tolerance_deg
        )

        cumulative_distance_to_center = None
        mean_distance_to_center = None
        std_distance_to_center = None
        cumulative_successive_distance = None
        mean_successive_distance = None
        std_successive_distance = None

        if not valid_fixations.empty:
            distances_to_center = valid_fixations.apply(
                lambda row: self._euclidean_distance(
                    row["x"],
                    row["y"],
                    center_x,
                    center_y,
                ),
                axis=1,
            )

            cumulative_distance_to_center = float(distances_to_center.sum())
            mean_distance_to_center = float(distances_to_center.mean())
            std_distance_to_center = float(distances_to_center.std())

            successive_distances = []

            ordered = valid_fixations.sort_values("fixation_id")

            for (_, previous), (_, current) in zip(
                ordered.iloc[:-1].iterrows(),
                ordered.iloc[1:].iterrows(),
            ):
                successive_distances.append(
                    self._euclidean_distance(
                        previous["x"],
                        previous["y"],
                        current["x"],
                        current["y"],
                    )
                )

            if successive_distances:
                successive_distances_series = pd.Series(successive_distances)

                cumulative_successive_distance = float(successive_distances_series.sum())
                mean_successive_distance = float(successive_distances_series.mean())
                std_successive_distance = float(successive_distances_series.std())

        invalid_reasons = []

        if n_blinks > self.max_blinks_valid:
            invalid_reasons.append("too_many_blinks")

        if not first_fixation_centered:
            invalid_reasons.append("first_fixation_not_centered")

        if n_fixations_valid == 0:
            invalid_reasons.append("no_valid_fixation")
            
        if (
            self.n_min_fixation_per_trial is not None
            and n_fixations_valid < self.n_min_fixation_per_trial
        ):
            invalid_reasons.append("insufficient_number_of_fixations")

        is_invalid_trial = len(invalid_reasons) > 0

        return {
            "trial_key": trial["trial_key"],
            "trial_index": trial["trial_index"],
            "participant_id": trial["participant_id"],
            "image_modality": trial["image_modality"],
            "image_id": trial["image_id"],
            "image_name": trial["image_name"],
            "image_stem": trial["image_stem"],

            "n_blinks": n_blinks,
            "n_fixations_total": n_fixations_total,
            "n_fixations_valid": n_fixations_valid,
            "n_fixations_outside_screen": n_fixations_outside_screen,

            "first_fixation_centered": first_fixation_centered,
            "first_fixation_dx_deg": first_dx_deg,
            "first_fixation_dy_deg": first_dy_deg,

            "cumulative_distance_to_center": cumulative_distance_to_center,
            "mean_distance_to_center": mean_distance_to_center,
            "std_distance_to_center": std_distance_to_center,

            "cumulative_successive_distance": cumulative_successive_distance,
            "mean_successive_distance": mean_successive_distance,
            "std_successive_distance": std_successive_distance,

            "is_invalid_trial": is_invalid_trial,
            "invalid_reasons": ";".join(invalid_reasons),
        }

    def build_participant_summary_df(
        self,
        trial_summary_df: pd.DataFrame,
    ) -> pd.DataFrame:

        summary = (
            trial_summary_df
            .groupby("participant_id")
            .agg(
                n_trials=("trial_key", "count"),
                n_invalid_trials=("is_invalid_trial", "sum"),
                total_blinks=("n_blinks", "sum"),
                mean_blinks_per_trial=("n_blinks", "mean"),
                mean_fixations_total=("n_fixations_total", "mean"),
                mean_fixations_valid=("n_fixations_valid", "mean"),
                mean_fixations_outside_screen=("n_fixations_outside_screen", "mean"),
                n_trials_bad_centering=(
                    "first_fixation_centered",
                    lambda s: int((~s).sum()),
                ),
                mean_distance_to_center=("mean_distance_to_center", "mean"),
                mean_successive_distance=("mean_successive_distance", "mean"),
                std_distance_to_center=("std_distance_to_center", "std"),
                std_successive_distance=("std_successive_distance", "std"),
            )
            .reset_index()
        )

        summary["invalid_trial_rate"] = (
            summary["n_invalid_trials"] / summary["n_trials"]
        )

        return summary

    def _empty_trial_row(
        self,
        trial: pd.Series,
        n_blinks: int,
        invalid_reason: str,
    ) -> dict:

        return {
            "trial_key": trial["trial_key"],
            "trial_index": trial["trial_index"],
            "participant_id": trial["participant_id"],
            "image_modality": trial["image_modality"],
            "image_id": trial["image_id"],
            "image_name": trial["image_name"],
            "image_stem": trial["image_stem"],

            "n_blinks": n_blinks,
            "n_fixations_total": 0,
            "n_fixations_valid": 0,
            "n_fixations_outside_screen": 0,

            "first_fixation_centered": False,
            "first_fixation_dx_deg": None,
            "first_fixation_dy_deg": None,

            "cumulative_distance_to_center": None,
            "mean_distance_to_center": None,
            "std_distance_to_center": None,

            "cumulative_successive_distance": None,
            "mean_successive_distance": None,
            "std_successive_distance": None,
            
            "is_invalid_trial": True,
            "invalid_reasons": invalid_reason,
        }

    def _px_to_deg_x(self, px: float, screen_width_px: float) -> float:
        cm = px * self.screen_width_cm / screen_width_px
        return math.degrees(
            2 * math.atan(cm / (2 * self.viewing_distance_cm))
        )

    def _px_to_deg_y(self, px: float, screen_height_px: float) -> float:
        cm = px * self.screen_height_cm / screen_height_px
        return math.degrees(
            2 * math.atan(cm / (2 * self.viewing_distance_cm))
        )

    @staticmethod
    def _euclidean_distance(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> float:
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)