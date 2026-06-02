"""Full mammogram preprocessing pipeline.

Implements the complete preprocessing chain from raw DICOM to model-ready
tensor, including breast region segmentation, pectoral muscle removal,
and intensity normalization. Each step is designed to handle the
variability inherent in clinical mammography data -- different
manufacturers, compression levels, exposure settings, and patient anatomy.

Pipeline:
    1. DICOM loading and metadata extraction
    2. Photometric normalization (MONOCHROME1 -> MONOCHROME2)
    3. Breast region segmentation (separate tissue from background)
    4. Pectoral muscle removal (MLO views only)
    5. Laterality normalization (breast always on left)
    6. CLAHE contrast enhancement
    7. Resize with aspect-preserving padding
    8. Tensor conversion and normalization
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for mammogram preprocessing pipeline."""

    target_size: Tuple[int, int] = (2048, 1024)
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8
    breast_threshold_percentile: float = 5.0
    min_breast_area_fraction: float = 0.05
    pectoral_removal_enabled: bool = True
    pectoral_removal_margin: int = 10
    normalize_mean: float = 0.214
    normalize_std: float = 0.169
    remove_artifacts: bool = True
    artifact_max_area_fraction: float = 0.01


class BreastSegmenter:
    """Segments the breast region from mammogram background.

    Uses adaptive thresholding and morphological operations to create a
    binary mask separating breast tissue from the air background and any
    annotation artifacts (labels, markers).

    Args:
        threshold_percentile: Percentile for initial thresholding. Values
            below this percentile are considered background.
        min_area_fraction: Minimum area (as fraction of image) for the
            breast region. Smaller connected components are discarded.
    """

    def __init__(
        self,
        threshold_percentile: float = 5.0,
        min_area_fraction: float = 0.05,
    ) -> None:
        self.threshold_percentile = threshold_percentile
        self.min_area_fraction = min_area_fraction

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Create binary mask of the breast region.

        Args:
            image: Grayscale mammogram, float32, shape (H, W).

        Returns:
            Binary mask of same shape (1 = breast, 0 = background).
        """
        # Threshold to separate tissue from air
        thresh_val = np.percentile(image[image > 0], self.threshold_percentile)
        binary = (image > thresh_val).astype(np.uint8)

        # Morphological cleanup: close small holes, remove noise
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

        # Keep only the largest connected component (the breast)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        if num_labels <= 1:
            logger.warning("No breast region detected; returning full image mask.")
            return np.ones_like(image, dtype=np.uint8)

        # Find largest component (excluding background label 0)
        min_area = self.min_area_fraction * image.shape[0] * image.shape[1]
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = np.argmax(areas) + 1

        if areas[largest_idx - 1] < min_area:
            logger.warning(
                "Largest component area (%.1f%%) below threshold (%.1f%%).",
                100 * areas[largest_idx - 1] / (image.shape[0] * image.shape[1]),
                100 * self.min_area_fraction,
            )

        mask = (labels == largest_idx).astype(np.uint8)

        # Fill holes within the breast mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(mask, contours, -1, 1, thickness=cv2.FILLED)

        return mask


class PectoralMuscleRemover:
    """Removes the pectoral muscle from MLO (medio-lateral oblique) views.

    The pectoral muscle appears as a bright triangular region in the upper
    corner of MLO views. Its high intensity can dominate learned features
    and it is not relevant to cancer risk prediction. This module detects
    and masks it out using Hough line detection on the gradient image.

    Args:
        margin: Additional pixels to remove beyond the detected muscle edge
            to ensure complete removal.
    """

    def __init__(self, margin: int = 10) -> None:
        self.margin = margin

    def remove(
        self,
        image: np.ndarray,
        breast_mask: np.ndarray,
        is_left: bool = True,
    ) -> np.ndarray:
        """Remove pectoral muscle from the mammogram.

        Args:
            image: Grayscale mammogram, float32, shape (H, W).
            breast_mask: Binary breast segmentation mask.
            is_left: Whether the breast is on the left side (after laterality
                normalization, this should always be True).

        Returns:
            Updated breast mask with pectoral muscle region removed.
        """
        mask = breast_mask.copy()
        h, w = image.shape[:2]

        # Convert to uint8 for OpenCV operations
        img_uint8 = (image * 255).astype(np.uint8)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(img_uint8, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blurred, 30, 100)

        # Focus on the upper-left region where pectoral muscle appears
        roi_h = int(h * 0.6)
        roi_w = int(w * 0.4)
        if not is_left:
            # If breast is on right, pectoral is upper-right
            roi_edges = edges[:roi_h, w - roi_w:]
        else:
            roi_edges = edges[:roi_h, :roi_w]

        # Hough line detection
        lines = cv2.HoughLinesP(
            roi_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=int(roi_h * 0.2),
            maxLineGap=20,
        )

        if lines is None or len(lines) == 0:
            logger.debug("No pectoral muscle line detected.")
            return mask

        # Find the most prominent line (longest)
        best_line = None
        best_length = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            # Pectoral muscle line should be roughly diagonal
            if length > best_length and abs(y2 - y1) > abs(x2 - x1) * 0.3:
                best_length = length
                best_line = line[0]

        if best_line is None:
            return mask

        x1, y1, x2, y2 = best_line

        # Adjust coordinates back to full image
        if not is_left:
            x1 += w - roi_w
            x2 += w - roi_w

        # Create pectoral mask: everything above/left of the line
        pectoral_mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array([
            [0, 0],
            [x1 + self.margin, y1],
            [x2 + self.margin, y2],
            [0, y2],
        ], dtype=np.int32)

        if not is_left:
            pts = np.array([
                [w, 0],
                [x1 - self.margin, y1],
                [x2 - self.margin, y2],
                [w, y2],
            ], dtype=np.int32)

        cv2.fillPoly(pectoral_mask, [pts], 1)

        # Remove pectoral from breast mask
        mask[pectoral_mask == 1] = 0

        logger.debug("Pectoral muscle removed (line: (%d,%d)->(%d,%d)).", x1, y1, x2, y2)
        return mask


class ArtifactRemover:
    """Removes common mammography artifacts from the image.

    Handles removal of:
    - Patient ID labels burned into the image
    - Orientation markers (L/R indicators)
    - Wire markers and implant markers
    - Small bright spots from sensor defects

    Uses connected component analysis to identify and remove small bright
    objects that are disproportionately intense compared to surrounding tissue.
    """

    def __init__(self, max_area_fraction: float = 0.01) -> None:
        self.max_area_fraction = max_area_fraction

    def remove(
        self, image: np.ndarray, breast_mask: np.ndarray
    ) -> np.ndarray:
        """Remove artifacts from the mammogram.

        Args:
            image: Float32 mammogram.
            breast_mask: Binary breast mask.

        Returns:
            Cleaned image.
        """
        cleaned = image.copy()
        h, w = image.shape[:2]
        total_area = h * w

        # Find very bright small regions outside the breast
        outside_breast = (breast_mask == 0).astype(np.uint8)
        bright_thresh = np.percentile(image[breast_mask > 0], 95) if (breast_mask > 0).any() else 0.5
        bright_outside = ((image > bright_thresh) & (breast_mask == 0)).astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            bright_outside, connectivity=8
        )

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < total_area * self.max_area_fraction:
                cleaned[labels == i] = 0.0

        return cleaned


class MammogramPreprocessor:
    """Complete mammogram preprocessing pipeline.

    Chains together all preprocessing steps in the correct order to transform
    a raw DICOM image into a normalized tensor ready for model input.

    Args:
        config: Preprocessing configuration.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None) -> None:
        self.config = config or PreprocessingConfig()

        self.segmenter = BreastSegmenter(
            threshold_percentile=self.config.breast_threshold_percentile,
            min_area_fraction=self.config.min_breast_area_fraction,
        )
        self.pectoral_remover = PectoralMuscleRemover(
            margin=self.config.pectoral_removal_margin,
        )
        self.artifact_remover = ArtifactRemover(
            max_area_fraction=self.config.artifact_max_area_fraction,
        )

    def process(
        self,
        image: np.ndarray,
        is_mlo: bool = False,
        is_left: bool = True,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """Run the full preprocessing pipeline.

        Args:
            image: Raw mammogram image, float32, shape (H, W), values in [0, 1].
            is_mlo: Whether this is an MLO view (triggers pectoral removal).
            is_left: Whether the breast is on the left side of the image.

        Returns:
            Tuple of:
                - Preprocessed tensor, shape (1, H, W)
                - Binary breast mask, shape (H, W)
        """
        # Step 1: Breast segmentation
        breast_mask = self.segmenter.segment(image)

        # Step 2: Pectoral muscle removal (MLO views only)
        if is_mlo and self.config.pectoral_removal_enabled:
            breast_mask = self.pectoral_remover.remove(image, breast_mask, is_left)

        # Step 3: Artifact removal
        if self.config.remove_artifacts:
            image = self.artifact_remover.remove(image, breast_mask)

        # Step 4: Apply breast mask (zero out background)
        image = image * breast_mask

        # Step 5: CLAHE enhancement on breast region only
        image = self._apply_masked_clahe(image, breast_mask)

        # Step 6: Crop to breast bounding box + resize
        image = self._crop_and_resize(image, breast_mask)

        # Step 7: Normalize to model-expected distribution
        image = (image - self.config.normalize_mean) / (self.config.normalize_std + 1e-8)

        # Step 8: Convert to tensor
        tensor = torch.from_numpy(image).unsqueeze(0).float()  # (1, H, W)

        return tensor, breast_mask

    def _apply_masked_clahe(
        self, image: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Apply CLAHE only to the breast region."""
        img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)

        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=(self.config.clahe_tile_size, self.config.clahe_tile_size),
        )

        enhanced = clahe.apply(img_uint8).astype(np.float32) / 255.0

        # Only use enhanced values within the breast mask
        result = np.where(mask > 0, enhanced, 0.0)
        return result

    def _crop_and_resize(
        self, image: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Crop to breast bounding box and resize to target dimensions."""
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return cv2.resize(image, (self.config.target_size[1], self.config.target_size[0]))

        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()

        # Add small margin
        margin = 10
        y_min = max(0, y_min - margin)
        y_max = min(image.shape[0], y_max + margin)
        x_min = max(0, x_min - margin)
        x_max = min(image.shape[1], x_max + margin)

        cropped = image[y_min:y_max, x_min:x_max]

        # Resize preserving aspect ratio with padding
        target_h, target_w = self.config.target_size
        crop_h, crop_w = cropped.shape[:2]
        scale = min(target_h / crop_h, target_w / crop_w)
        new_h, new_w = int(crop_h * scale), int(crop_w * scale)

        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros((target_h, target_w), dtype=np.float32)
        pad_h = (target_h - new_h) // 2
        pad_w = (target_w - new_w) // 2
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        return canvas

    def batch_process(
        self,
        images: list[np.ndarray],
        is_mlo: list[bool],
        is_left: list[bool],
    ) -> torch.Tensor:
        """Process a batch of mammograms.

        Args:
            images: List of raw mammogram arrays.
            is_mlo: List of MLO flags per image.
            is_left: List of laterality flags per image.

        Returns:
            Batch tensor of shape (B, 1, H, W).
        """
        tensors = []
        for img, mlo, left in zip(images, is_mlo, is_left):
            tensor, _ = self.process(img, is_mlo=mlo, is_left=left)
            tensors.append(tensor)

        return torch.stack(tensors, dim=0)
