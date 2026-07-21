from pathlib import Path
import json

from helpers.models import Metadata, ImageData, Annotations, RoiData



class Indexer:
    def __init__(self, metadata: Metadata, annotations: Annotations = None):
        self.metadata = metadata
        self.annotations = annotations or []
        
        self._metadata_by_id: dict[int, ImageData] = {}
        self._annotations_by_stem_modality: dict[
            tuple[str, str],
            RoiData,
        ] = {}
        
        self._index_metadata()
        self._index_annotations()

    def _index_metadata(self) -> None:
        for image in self.metadata:
            image_id = self.normalize_id(image.id)

            if image_id in self._metadata_by_id:
                raise ValueError(f"Duplicate image id found: {image_id}")

            self._metadata_by_id[image_id] = image

    def _index_annotations(self) -> None:
        for annotation in self.annotations:
            stem = self.normalize_stem(annotation.image_stem)
            modality = self.normalize_modality(
                annotation.image_modality
            )
            image_id = self.normalize_id(annotation.image_id)

            key = (stem, modality)

            if key in self._annotations_by_stem_modality:
                raise ValueError(
                    f"Duplicate annotations for (stem, modality): {key}"
                )

            expected_id = self.get_id(stem, modality)

            if image_id != expected_id:
                raise ValueError(
                    f"Inconsistent annotation image_id for {key}: "
                    f"got {image_id}, expected {expected_id}"
                )

            self._annotations_by_stem_modality[key] = annotation

    @staticmethod
    def normalize_modality(modality: str) -> str:
        value = str(modality).lower().strip()
        if value not in ("infrared", "visible"):
            raise ValueError(f"Invalid modality: {modality}")
        return value

    @staticmethod
    def normalize_id(image_id: int | str) -> int:
        try:
            value = int(image_id)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid image id: {image_id}") from e

        if value <= 0:
            raise ValueError(f"Image id must be positive, got: {value}")

        return value

    @staticmethod
    def normalize_stem(stem: str | int) -> str:
        raw = str(stem).strip()
        if not raw:
            raise ValueError("Image stem cannot be empty")

        # Accepte "1", "001", "000001" et normalise vers "001"
        if raw.isdigit():
            return f"{int(raw):06d}"

        # Cas plus défensif si un nom de fichier complet arrive par erreur
        stem_only = Path(raw).stem.strip()
        if stem_only.isdigit():
            return f"{int(stem_only):03d}"

        raise ValueError(f"Invalid image stem: {stem}")

    def get_id(self, image_stem: str | int, image_modality: str) -> int:

        image = self.get_metadata(
            image_modality=self.normalize_modality(image_modality), 
            image_stem=self.normalize_stem(image_stem),
        )

        if image is None:
            raise ValueError(
                f"No image found with stem={image_stem} and modality={image_modality}"
            )

        return self.normalize_id(image.id)
    
    def get_size(self, image_id: int | str) -> tuple[int, int]:
        image = self.get_metadata(image_id)

        if image is None:
            raise ValueError(f"No image found with id={image_id}")

        width = int(image.width)
        height = int(image.height)

        if width <= 0 or height <= 0:
            raise ValueError(
                f"Invalid dimensions for image id={image_id}: "
                f"width={width}, height={height}"
            )

        return width, height

    def get_modality(self, image_id: int | str) -> str:
        image = self.get_metadata(image_id)

        if image is None:
            raise ValueError(f"No image found with id={image_id}")

        return self.normalize_modality(image.modality)

    def get_name(self, image_id: int | str) -> str:
        image = self._metadata_by_id.get(image_id)

        if image is None:
            raise ValueError(f"No image found with id={image_id}")

        name = str(image.name).strip()
        if not name:
            raise ValueError(f"Image with id={image_id} has an empty name")

        return name
    
    def get_visibility(self, image_id: int | str) -> str:
        image = self.get_metadata(image_id)

        if image is None:
            raise ValueError(f"No image found with id={image_id}")

        return str(image.daynight)

    def get_metadata(
        self,
        image_id: int | str | None = None,
        image_modality: str | None = None,
        image_stem: str | int | None = None,
    ):
        if image_id is not None:
            normalized_id = self.normalize_id(image_id)

            for image in self.metadata:
                if image.id == normalized_id:
                    return image

            return None

        elif image_modality is not None and image_stem is not None:
            stem = self.normalize_stem(image_stem)
            modality = self.normalize_modality(image_modality)

            for image in self.metadata:
                if (
                    self.normalize_stem(image.stem) == stem
                    and self.normalize_modality(image.modality) == modality
                ):
                    return image

            return None

        else:
            raise ValueError(
                "get_metadata requires either image_id or both image_modality and image_stem"
            )
    

    def get_annotation(
        self,
        modality: str,
        stem: str | int,
    ) -> RoiData:
        stem = self.normalize_stem(stem)
        modality = self.normalize_modality(modality)

        return self._annotations_by_stem_modality.get(
            (stem, modality)
        )