from __future__ import annotations
from pathlib import Path

import re
import pandas as pd

from processing.parser import Parser
from helpers.models import RawTrial
from helpers.utils import save_dataframe




class Extractor:

    def __init__(self, parser: Parser):
        self.parser = parser

    def run(
        self,
        asc_dir: str | Path,
        modality: str,
        verbose: bool = False,
        save: bool = False,
        output_dir: str | Path = Path("outputs/raw/"),
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        from tqdm import tqdm

        asc_dir = Path(asc_dir)

        if not asc_dir.exists():
            raise FileNotFoundError(f"ASC directory not found: {asc_dir}")

        filepaths = sorted(asc_dir.glob("*.asc"))

        if not filepaths:
            raise ValueError(f"No ASC files found in: {asc_dir}")

        all_trials: list[RawTrial] = []

        for filepath in tqdm(filepaths, desc="Processing participants"):
            participant_id = self._get_participant_id(filepath.name)

            trials = self.parser.parse_file(
                filepath=filepath,
                participant_id=participant_id,
                modality=modality,
            )

            all_trials.extend(trials)

        if not all_trials:
            raise ValueError("No trial data extracted from ASC files.")

        summary_df = self.extract_summary(all_trials)
        fixations_df = self.extract_fixations(all_trials)
        blinks_df = self.extract_blinks(all_trials)
        

        # SAVES
        if save:
            save_dataframe(summary_df, output_dir , "summary")
            save_dataframe(fixations_df, output_dir , "fixations")
            save_dataframe(blinks_df, output_dir , "blinks")
            

        # STATS
        if verbose:
            n_trials = summary_df.shape[0]
            n_fixations = fixations_df.shape[0]
            n_blinks = blinks_df.shape[0]

            n_participants = summary_df["participant_id"].nunique()
            n_images = summary_df["image_id"].nunique()

            print(f"\n----- GENERAL -----")
            print(f"Participants: {n_participants}")
            print(f"Trials: {n_trials}")
            print(f"Images uniques: {n_images}")
            print(f"Fixations: {n_fixations}")
            print(f"Blinks: {n_blinks}")

            # Fixations per trial
            fixations_per_trial = fixations_df.groupby("trial_key").size()

            print(f"\n----- FIXATIONS PER TRIAL -----")
            print(f"Moyenne: {fixations_per_trial.mean():.2f}")
            print(f"Std: {fixations_per_trial.std():.2f}")
            print(f"Min: {fixations_per_trial.min()}")
            print(f"Max: {fixations_per_trial.max()}")

            # Blinks per trial 
            if not blinks_df.empty:
                blinks_per_trial = blinks_df.groupby("trial_key").size()

                print(f"\n----- BLINKS PER TRIAL -----")
                print(f"Moyenne: {blinks_per_trial.mean():.2f}")
                print(f"Std: {blinks_per_trial.std():.2f}")
                print(f"Min: {blinks_per_trial.min()}")
                print(f"Max: {blinks_per_trial.max()}")

            # Fixations duration
            print(f"\n----- FIXATIONS DURATION (ms) -----")
            print(f"Moyenne: {fixations_df['duration'].mean():.2f}")
            print(f"Std: {fixations_df['duration'].std():.2f}")
            print(f"Min: {fixations_df['duration'].min()}")
            print(f"Max: {fixations_df['duration'].max()}")

    
        return fixations_df, blinks_df, summary_df

    def extract_summary(
        self,
        trials: list[RawTrial],
    ) -> pd.DataFrame:

        rows = []

        for trial_index, trial in enumerate(trials):
            trial_key = self._make_trial_key(trial, trial_index)
            dimensions = trial.dimensions

            rows.append(
                {
                    "trial_key": trial_key,
                    "trial_index": trial_index,
                    "participant_id": trial.participant_id,
                    "image_modality": trial.image_modality,
                    "image_name": trial.image_name,
                    "image_stem": trial.image_stem,
                    "image_id": trial.image_id,
                    "image_onset": trial.image_onset,
                    "image_offset": trial.image_offset,
                    "image_duration": trial.image_offset - trial.image_onset,
                    "screen_width": dimensions.screen_width,
                    "screen_height": dimensions.screen_height,
                    "stimuli_width": dimensions.stimuli_width,
                    "stimuli_height": dimensions.stimuli_height,
                    "image_width": dimensions.image_width,
                    "image_height": dimensions.image_height,
                    "n_fixations_raw": len(trial.fixations),
                    "n_blinks_raw": len(trial.blinks),
                }
            )

        df = pd.DataFrame(rows)

        return df.sort_values(
            by=["participant_id", "image_modality", "trial_index"]
        ).reset_index(drop=True)

    def extract_fixations(
        self,
        trials: list[RawTrial],
    ) -> pd.DataFrame:

        rows = []

        for trial_index, trial in enumerate(trials):
            trial_key = self._make_trial_key(trial, trial_index)
            dimensions = trial.dimensions

            for fixation in trial.fixations:
                rows.append(
                    {
                        "trial_key": trial_key,
                        "trial_index": trial_index,
                        "participant_id": trial.participant_id,
                        "image_modality": trial.image_modality,
                        "image_name": trial.image_name,
                        "image_stem": trial.image_stem,
                        "image_id": trial.image_id,
                        "fixation_id": fixation.id,
                        "eye": fixation.eye,
                        "start": fixation.start,
                        "end": fixation.end,
                        "duration": fixation.duration,
                        "x": fixation.x,
                        "y": fixation.y,
                        "relative_start": fixation.start - trial.image_onset,
                        "relative_end": fixation.end - trial.image_onset,
                        "screen_width": dimensions.screen_width,
                        "screen_height": dimensions.screen_height,
                        "stimuli_width": dimensions.stimuli_width,
                        "stimuli_height": dimensions.stimuli_height,
                        "image_width": dimensions.image_width,
                        "image_height": dimensions.image_height,
                    }
                )

        df = pd.DataFrame(rows)

        if df.empty:
            return df

        return df.sort_values(
            by=[
                "participant_id",
                "image_modality",
                "trial_index",
                "fixation_id",
            ]
        ).reset_index(drop=True)

    def extract_blinks(
        self,
        trials: list[RawTrial],
    ) -> pd.DataFrame:

        rows = []

        for trial_index, trial in enumerate(trials):
            trial_key = self._make_trial_key(trial, trial_index)

            for blink_id, blink in enumerate(trial.blinks):
                rows.append(
                    {
                        "trial_key": trial_key,
                        "trial_index": trial_index,
                        "participant_id": trial.participant_id,
                        "image_modality": trial.image_modality,
                        "image_name": trial.image_name,
                        "image_stem": trial.image_stem,
                        "image_id": trial.image_id,
                        "blink_id": blink_id,
                        "eye": blink.eye,
                        "start": blink.start,
                        "end": blink.end,
                        "duration": blink.duration,
                        "relative_start": blink.start - trial.image_onset,
                        "relative_end": blink.end - trial.image_onset,
                    }
                )

        df = pd.DataFrame(rows)

        if df.empty:
            return df

        return df.sort_values(
            by=[
                "participant_id",
                "image_modality",
                "trial_index",
                "blink_id",
            ]
        ).reset_index(drop=True)

    
        

    @staticmethod
    def _make_trial_key(
        trial: RawTrial,
        trial_index: int,
    ) -> str:
        return (
            f"{trial.image_stem}_"
            f"{trial.participant_id}_"
            f"{trial.image_modality}_"
            f"{trial_index:04d}"
        )

    @staticmethod
    def _get_participant_id(filename: str) -> str:
        stem = Path(filename).stem

        match = re.search(r"ID-(\d+)", stem)
        if match:
            return f"{int(match.group(1)):03d}"

        digits = re.findall(r"\d+", stem)
        if digits:
            return f"{int(digits[0]):03d}"

        return stem