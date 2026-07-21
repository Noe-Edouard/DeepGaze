from __future__ import annotations

from dataclasses import dataclass
from typing import List, TypeAlias, Optional, Dict
from dataclasses import dataclass, field


@dataclass
class DimensionData:
    screen_width: int
    screen_height: int
    stimuli_width: int
    stimuli_height: int
    image_width: int
    image_height: int


@dataclass
class ParticipantInfo:
    id: str
    modality: str
    n_images: int 
    n_fixation_per_image: int
    n_valid_fixation_per_image: int
    mean_valid: str

@dataclass
class ImageData:
    id: int
    stem: str
    name: str
    extension: str
    modality: str
    height: int
    width: int
    daynight: Optional[str] = None


@dataclass
class BlinkData:
    eye: str
    start: int
    end: int
    duration: int

@dataclass
class FixationData:
    id: int
    eye: str
    start: int
    end: int
    duration: int
    x: float
    y: float


@dataclass
class TrialState:
    image_onset: Optional[int] = None
    image_offset: Optional[int] = None
    image_name: Optional[str] = None
    image_stem: Optional[str] = None
    image_id: Optional[str] = None
    dimensions: Optional[DimensionData] = None
    fixations: list[FixationData] = field(default_factory=list)
    blinks: list[BlinkData] = field(default_factory=list)
    

    def reset(self):
        self.image_onset = None
        self.image_offset = None
        self.image_name = None
        self.image_stem = None
        self.image_id = None
        self.dimensions = None
        self.fixations = []
        self.blinks = []
        
    def is_started(self) -> bool:
        return self.image_onset is not None
        
    def is_valid(self) -> bool:
        return (
            self.image_onset is not None
            and self.image_offset is not None
            and self.image_name is not None
            and self.image_stem is not None
            and self.image_id is not None
            and self.dimensions is not None
        )


@dataclass
class RawTrial:
    dimensions: DimensionData
    participant_id: str
    
    image_modality: str
    image_name: str
    image_stem: str
    image_id: str
    image_onset: int
    image_offset: int
    
    fixations: list[FixationData]
    blinks: list[BlinkData]


@dataclass
class CleanTrial:
    dimensions: DimensionData
    participant_id: int
    
    image_modality: str
    image_name: str
    image_stem: str
    image_id: int
    
    fixation_data: FixationData
    
    
    
@dataclass(frozen=True)
class RoiBox:
    class_id: int
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min
    
    @property
    def area(self) -> int:
        return max(0, self.x_max - self.x_min) * max(0, self.y_max - self.y_min)

@dataclass(frozen=True)
class RoiData:
    image_id: int
    image_stem: str
    image_modality: str
    rois: list[RoiBox]


@dataclass
class TrialData:
    image_data: ImageData
    dimensions_data: DimensionData
    participants_data: Dict[str, list[FixationData]]
    
    

Dataset: TypeAlias = Dict[str, TrialData]
Metadata: TypeAlias = List[ImageData]
Annotations: TypeAlias = List[RoiData] # {modality: stem: [annotations]}
Results: TypeAlias = Dict[str, Dict[str, Dict[str, Dict]]] # {metric_type: {metric_name: stem: {results}}}