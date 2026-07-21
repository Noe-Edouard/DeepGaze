from __future__ import annotations
from abc import ABC
from pathlib import Path
from typing import Optional, Tuple

from helpers.models import Metadata, FixationData, DimensionData, BlinkData, RawTrial, TrialState
from helpers.indexer import Indexer



class Parser(ABC):

    def __init__(self, metadata: Metadata):
        super().__init__()

        self.metadata = metadata
        self.index = Indexer(metadata)

        self.valid_modalities = ['visible', 'infrared']
        self.start_msg = " Affichage_Image"
        self.end_msg = " time_out"
        self.display_msg = " DISPLAY_COORDS"
        self.screen_height = 1200
        self.screen_width = 1600

    def parse_file(
        self,
        filepath: str | Path,
        participant_id: str,
        modality: str,
    ) -> list[RawTrial]:

        if modality not in self.valid_modalities:
            raise ValueError(
                f"Modality not valid, must be in: {self.valid_modalities}"
            )

        raw_trials: list[RawTrial] = []

        with open(filepath, "r", encoding="utf-8", errors="ignore") as file:
            lines = file.readlines()

        self.screen_width, self.screen_height = self._get_screen_size(lines)

        state = TrialState()

        for line_index, raw_line in enumerate(lines):
            line = raw_line.strip()

            if self._is_start_trial(line):
                state.reset()
                
                state.image_onset = self._get_timestamp(line)
                state.image_name = self._get_image_name(lines, line_index)
                state.image_stem = self._get_image_stem(state.image_name)
                state.image_id = self._get_image_id(state.image_stem, modality)
                state.dimensions = self._get_dimensions(state.image_id, lines, line_index)

                continue


            if self._is_fix_line(line): 
                if not state.is_started():
                    continue
                
                fixation = self._parse_efix_line(line, len(state.fixations))

                if fixation is not None:
                    state.fixations.append(fixation)

                continue

            if self._is_blink_line(line):
                if not state.is_started():
                    continue
                
                blink = self._parse_eblink_line(line)

                if blink is not None:
                    state.blinks.append(blink)

                continue

            if self._is_end_trial(line):
                state.image_offset = self._get_timestamp(line)

                if state.is_valid():
                    raw_trials.append(
                        RawTrial(
                            participant_id=str(participant_id),
                            image_modality=str(modality),
                            image_name=str(self.index.get_name(state.image_id)),
                            image_stem=str(state.image_stem),
                            image_id=int(state.image_id),
                            image_onset=int(state.image_onset),
                            image_offset=int(state.image_offset),
                            dimensions=state.dimensions,
                            fixations=list(state.fixations),
                            blinks=list(state.blinks),
                        )
                    )
                    
                else:
                    raise ValueError(
                        f"Incomplete trial detected for participant '{participant_id}' "
                        f"(image='{state.image_name}', stem='{state.image_stem}'). "
                        f"Missing fields: "
                        f"{'image_offset ' if state.image_offset is None else ''}"
                        f"{'image_name ' if state.image_name is None else ''}"
                        f"{'image_id ' if state.image_id is None else ''}"
                        f"{'dimensions' if state.dimensions is None else ''}. "
                        f"Trial start timestamp={state.image_onset}."
                    )
                    
                state.reset()

                continue

        return raw_trials

    def _is_start_trial(self, line: str) -> bool:
        return line.startswith("MSG") and self.start_msg in line

    def _is_end_trial(self, line: str) -> bool:
        return line.startswith("MSG") and self.end_msg in line

    def _is_display_line(self, line: str) -> bool:
        return line.startswith("MSG") and self.display_msg in line

    def _is_fix_line(self, line: str) -> bool:
        return line.startswith("EFIX")
    
    def _is_blink_line(self, line: str) -> bool:
        return line.startswith("EBLINK")
    
    @staticmethod
    def _parse_msg_line(line: str) -> tuple[Optional[int], Optional[str]]:
        parts = line.split(maxsplit=2)

        if len(parts) < 2 or parts[0] != "MSG":
            return None, None

        try:
            timestamp = int(parts[1])
        except ValueError:
            return None, None

        message = parts[2] if len(parts) > 2 else None

        return timestamp, message

    @staticmethod
    def _parse_efix_line(
        line: str,
        fixation_id: int,
    ) -> Optional[FixationData]:
        """
        EyeLink format:
        EFIX <eye> <start> <end> <duration> <x> <y> <pupil>
        """

        parts = line.split()

        if len(parts) < 7:
            return None

        try:
            return FixationData(
                id=fixation_id,
                eye=str(parts[1]),
                start=int(parts[2]),
                end=int(parts[3]),
                duration=int(parts[4]),
                x=float(parts[5]),
                y=float(parts[6]),
            )
        except (IndexError, ValueError):
            return None

    @staticmethod
    def _parse_eblink_line(line: str) -> Optional[BlinkData]:
        """
        EyeLink format:
        EBLINK <eye> <start> <end> <duration>
        """

        parts = line.split()

        if len(parts) < 5:
            return None

        try:
            return BlinkData(
                eye=str(parts[1]),
                start=int(parts[2]),
                end=int(parts[3]),
                duration=int(parts[4]),
            )
        except (IndexError, ValueError):
            return None

    def _get_timestamp(self, line: str) -> int:
        timestamp, _ = self._parse_msg_line(line)

        if timestamp is None:
            raise ValueError(f"Invalid MSG line, cannot extract timestamp: '{line}'")

        return timestamp

    def _get_screen_size(
        self,
        lines: list[str],
    ) -> Tuple[int, int]:

        for line in lines:
            if self._is_display_line(line):
                parts = line.split()

                try:
                    screen_width = int(parts[-2])
                    screen_height = int(parts[-1])
                except (IndexError, ValueError):
                    raise ValueError(f"Invalid {self.display_msg} line format: '{line.strip()}'")

                return screen_width+1, screen_height+1

        raise RuntimeError(f'"{self.display_msg}" message not found in file.')

    def _get_image_stem(self, image_name: str) -> str:
        stem = Path(image_name).stem
        parts = stem.split("_")

        if not parts:
            raise ValueError(f"Invalid image name: {image_name}")

        return parts[0]
    
    def _get_dimensions(self, image_id: int, lines: list[str], start_index: int) -> DimensionData:
        image_width, image_height = self.index.get_size(image_id)
        stimuli_width, stimuli_height = self._get_stimuli_size(lines, start_index)
        
        return DimensionData(
                screen_width=self.screen_width,
                screen_height=self.screen_height,
                stimuli_height=stimuli_height,
                stimuli_width=stimuli_width,
                image_height=image_height,
                image_width=image_width,
            )

    def _get_image_id(
        self,
        image_name: str,
        image_modality: str,
    ) -> int:
        image_stem = Path(image_name).stem
        return self.index.get_id(image_stem, image_modality)
    

    def _get_stimuli_size(self, lines: list[str], start_index: int) -> tuple[int, int]:
        stimulus_width = None
        stimulus_height = None

        for j in range(start_index + 1, len(lines)):
            current_line = lines[j].strip()

            if self._is_end_trial(current_line):
                break

            if "!V TRIAL_VAR stimulus_width " in current_line:
                try:
                    stimulus_width = int(current_line.split()[-1])
                except (ValueError, IndexError):
                    raise ValueError(f"Invalid stimulus_width line: {current_line}")

            elif "!V TRIAL_VAR stimulus_height " in current_line:
                try:
                    stimulus_height = int(current_line.split()[-1])
                except (ValueError, IndexError):
                    raise ValueError(f"Invalid stimulus_height line: {current_line}")

            if stimulus_width is not None and stimulus_height is not None:
                return stimulus_width, stimulus_height

        raise ValueError(f"Stimulus size not found after trial start at line {start_index}")
    

    def _get_image_name(self, lines: list[str], start_index: int) -> str:
        

        for j in range(start_index + 1, len(lines)):
            line = lines[j].strip()

            if self._is_end_trial(line):
                break

            if "!V TRIAL_VAR name " in line:
                parts = line.split()
                if not parts:
                    continue

                name = parts[-1].lower()

                if name.endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    return name

        raise ValueError(f"Image name not found after trial start at line {start_index}")
            
            

