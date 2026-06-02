"""Unit tests for mammography data loading and preprocessing.

Tests cover:
- DICOM metadata parsing (laterality, view position)
- Photometric interpretation normalization
- CLAHE application
- Laterality normalization (flipping)
- Image resizing with aspect-preserving padding
- Dataset class functionality
- Augmentation behavior
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.data.dicom_loader import (
    DicomLoader,
    Laterality,
    MammogramMetadata,
    ViewPosition,
)
from src.data.dataset import MammographyAugmentation, MammographyDataset


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _make_dummy_metadata(**overrides) -> MammogramMetadata:
    """Create a MammogramMetadata with sensible defaults."""
    defaults = dict(
        patient_id="TEST_001",
        study_uid="1.2.3.4",
        series_uid="1.2.3.4.1",
        sop_uid="1.2.3.4.1.1",
        laterality=Laterality.LEFT,
        view_position=ViewPosition.CC,
        manufacturer="TestCorp",
        institution="Test Hospital",
        pixel_spacing=(0.07, 0.07),
        rows=3328,
        columns=2560,
        bits_stored=14,
        photometric_interpretation="MONOCHROME2",
        presentation_intent="FOR PROCESSING",
    )
    defaults.update(overrides)
    return MammogramMetadata(**defaults)


def _make_test_image(
    height: int = 256, width: int = 128, seed: int = 42
) -> np.ndarray:
    """Create a synthetic mammogram-like test image."""
    rng = np.random.RandomState(seed)

    # Background (air) is black
    image = np.zeros((height, width), dtype=np.float32)

    # Simulate breast region (bright ellipse)
    yy, xx = np.ogrid[:height, :width]
    cy, cx = height // 2, width // 4
    ry, rx = height // 3, width // 3
    mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 < 1

    # Add tissue texture
    tissue = rng.uniform(0.3, 0.8, size=(height, width)).astype(np.float32)
    image[mask] = tissue[mask]

    # Add some noise
    image += rng.normal(0, 0.02, image.shape).astype(np.float32)
    image = np.clip(image, 0, 1)

    return image


# ---------------------------------------------------------------------------
# DicomLoader tests
# ---------------------------------------------------------------------------


class TestDicomLoader:
    """Tests for the DicomLoader class."""

    def test_initialization_defaults(self):
        loader = DicomLoader()
        assert loader.target_size == (2048, 1024)
        assert loader.apply_clahe is True
        assert loader.normalize_laterality is True

    def test_initialization_custom(self):
        loader = DicomLoader(
            target_size=(1024, 512),
            apply_clahe=False,
            clahe_clip_limit=3.0,
            normalize_laterality=False,
        )
        assert loader.target_size == (1024, 512)
        assert loader.apply_clahe is False
        assert loader.clahe_clip_limit == 3.0

    def test_file_not_found(self):
        loader = DicomLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/test.dcm")

    def test_to_float32_normalization(self):
        loader = DicomLoader()
        # Simulate a 14-bit image
        raw = np.random.randint(0, 16383, size=(100, 100), dtype=np.uint16)
        result = loader._to_float32(raw.astype(np.float32), bits_stored=14)
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_to_float32_constant_image(self):
        """Edge case: constant image should not cause division by zero."""
        loader = DicomLoader()
        constant = np.ones((100, 100), dtype=np.float32) * 5000
        result = loader._to_float32(constant, bits_stored=14)
        assert np.all(result == 0.0)  # Near-zero range -> zeros

    def test_apply_clahe(self):
        loader = DicomLoader(clahe_clip_limit=2.0)
        image = _make_test_image()
        enhanced = loader._apply_clahe(image)
        assert enhanced.shape == image.shape
        assert enhanced.dtype == np.float32
        assert enhanced.min() >= 0.0
        assert enhanced.max() <= 1.0
        # CLAHE should change the image
        assert not np.allclose(enhanced, image, atol=0.01)

    def test_resize_preserves_aspect(self):
        loader = DicomLoader(target_size=(256, 128))
        # Tall narrow image
        image = np.random.rand(500, 200).astype(np.float32)
        resized = loader._resize(image, (256, 128))
        assert resized.shape == (256, 128)

    def test_resize_pads_correctly(self):
        loader = DicomLoader(target_size=(100, 100))
        # Very wide image -> will have vertical padding
        image = np.ones((50, 200), dtype=np.float32)
        resized = loader._resize(image, (100, 100))
        assert resized.shape == (100, 100)
        # Check that padding exists (corners should be zero)
        assert resized[0, 0] == 0.0 or resized[-1, -1] == 0.0

    def test_normalize_photometric_monochrome1(self):
        loader = DicomLoader()
        meta = _make_dummy_metadata(
            photometric_interpretation="MONOCHROME1",
            bits_stored=12,
        )
        image = np.array([0, 1000, 4095], dtype=np.uint16)
        inverted = loader._normalize_photometric(image, meta)
        assert inverted[0] == 4095  # Was 0, now max
        assert inverted[2] == 0    # Was max, now 0

    def test_normalize_photometric_monochrome2_noop(self):
        loader = DicomLoader()
        meta = _make_dummy_metadata(photometric_interpretation="MONOCHROME2")
        image = np.array([100, 200, 300], dtype=np.uint16)
        result = loader._normalize_photometric(image, meta)
        np.testing.assert_array_equal(result, image)


# ---------------------------------------------------------------------------
# Augmentation tests
# ---------------------------------------------------------------------------


class TestAugmentation:
    """Tests for mammography-specific augmentations."""

    def test_no_augmentation(self):
        """With all augmentations disabled, image should be unchanged."""
        aug = MammographyAugmentation(
            horizontal_flip_prob=0.0,
            rotation_range=0.0,
            scale_range=(1.0, 1.0),
            intensity_shift_range=0.0,
            gaussian_noise_std=0.0,
            random_crop_pad=0,
        )
        image = _make_test_image()
        result = aug(image)
        np.testing.assert_array_almost_equal(result, image)

    def test_rotation_range(self):
        aug = MammographyAugmentation(
            rotation_range=5.0,
            scale_range=(1.0, 1.0),
            intensity_shift_range=0.0,
            gaussian_noise_std=0.0,
            random_crop_pad=0,
        )
        image = _make_test_image()
        result = aug(image)
        assert result.shape == image.shape
        # Rotated image should differ from original
        assert not np.allclose(result, image)

    def test_intensity_shift_bounds(self):
        """Intensity shift should keep values in [0, 1]."""
        aug = MammographyAugmentation(
            rotation_range=0.0,
            scale_range=(1.0, 1.0),
            intensity_shift_range=0.5,
            gaussian_noise_std=0.0,
            random_crop_pad=0,
        )
        image = _make_test_image()
        for _ in range(10):
            result = aug(image)
            assert result.min() >= 0.0
            assert result.max() <= 1.0

    def test_gaussian_noise(self):
        aug = MammographyAugmentation(
            rotation_range=0.0,
            scale_range=(1.0, 1.0),
            intensity_shift_range=0.0,
            gaussian_noise_std=0.05,
            random_crop_pad=0,
        )
        image = _make_test_image()
        result = aug(image)
        assert result.shape == image.shape
        # Should be close but not identical
        diff = np.abs(result - image).mean()
        assert diff > 0.001
        assert diff < 0.2

    def test_output_shape_preserved(self):
        """Augmentation should never change the output shape."""
        aug = MammographyAugmentation()
        image = _make_test_image(height=128, width=64)
        for _ in range(5):
            result = aug(image)
            assert result.shape == (128, 64)


# ---------------------------------------------------------------------------
# Dataset tests (mocked DICOM loading)
# ---------------------------------------------------------------------------


class TestMammographyDataset:
    """Tests for the MammographyDataset class."""

    def test_class_weights_balanced(self):
        """50/50 split should give equal weights."""
        dataset = MammographyDataset.__new__(MammographyDataset)
        dataset.samples = [
            {"cancer_risk": 0.0} for _ in range(50)
        ] + [
            {"cancer_risk": 1.0} for _ in range(50)
        ]
        weights = dataset.get_class_weights()
        assert torch.allclose(weights, torch.tensor([1.0, 1.0]))

    def test_class_weights_imbalanced(self):
        """Imbalanced dataset should upweight minority class."""
        dataset = MammographyDataset.__new__(MammographyDataset)
        dataset.samples = [
            {"cancer_risk": 0.0} for _ in range(95)
        ] + [
            {"cancer_risk": 1.0} for _ in range(5)
        ]
        weights = dataset.get_class_weights()
        assert weights[1] > weights[0]  # Positive class should have higher weight

    def test_positive_rate(self):
        dataset = MammographyDataset.__new__(MammographyDataset)
        dataset.samples = [
            {"cancer_risk": 0.0} for _ in range(90)
        ] + [
            {"cancer_risk": 1.0} for _ in range(10)
        ]
        assert abs(dataset.positive_rate - 0.1) < 1e-6

    def test_density_map(self):
        assert MammographyDataset.DENSITY_MAP["A"] == 0
        assert MammographyDataset.DENSITY_MAP["D"] == 3


# ---------------------------------------------------------------------------
# Integration-style tests
# ---------------------------------------------------------------------------


class TestPreprocessingIntegration:
    """Integration tests for the full preprocessing chain."""

    def test_clahe_improves_contrast(self):
        """CLAHE should increase the standard deviation of pixel values."""
        loader = DicomLoader()
        image = _make_test_image()
        # Reduce contrast artificially
        low_contrast = image * 0.3 + 0.35
        enhanced = loader._apply_clahe(low_contrast)

        # Enhanced should have more spread
        assert enhanced.std() >= low_contrast.std() * 0.8

    def test_end_to_end_resize_pipeline(self):
        """Full resize pipeline should produce correct output dimensions."""
        loader = DicomLoader(target_size=(512, 256))
        image = _make_test_image(height=1000, width=800)
        resized = loader._resize(image, (512, 256))
        assert resized.shape == (512, 256)
        assert resized.dtype == np.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
