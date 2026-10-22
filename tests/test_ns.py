"""Unit tests for Neutrosophic Set (T, I, F) transformation and entropy."""

import numpy as np
import pytest
from src.ns import neutrosophic_transform, compute_entropy


def test_ns_transform_range_and_complement():
    """Checks that T, I, F lie in [0, 1] and F == 1 - T everywhere."""
    np.random.seed(42)
    img = np.random.uniform(0.0, 1.0, size=(64, 64))

    T, I, F = neutrosophic_transform(img, window_size=5)

    assert T.shape == (64, 64)
    assert I.shape == (64, 64)
    assert F.shape == (64, 64)

    assert np.all(T >= 0.0) and np.all(T <= 1.0)
    assert np.all(I >= 0.0) and np.all(I <= 1.0)
    assert np.all(F >= 0.0) and np.all(F <= 1.0)

    # Check F == 1 - T
    np.testing.assert_allclose(F, 1.0 - T, atol=1e-7)


def test_ns_transform_uniform_image():
    """A constant uniform image has zero variation, so I should be zero."""
    img = np.ones((50, 50), dtype=float) * 0.5
    T, I, F = neutrosophic_transform(img, window_size=5)

    # For constant image, delta = 0 everywhere
    assert np.all(I == 0.0)


def test_ns_transform_edge_indeterminacy():
    """Indeterminacy I should be highest at sharp edges."""
    img = np.zeros((60, 60), dtype=float)
    img[:, 30:] = 1.0  # vertical step edge

    T, I, F = neutrosophic_transform(img, window_size=5)

    # Pixels far from edge have delta == 0
    far_left_I = I[30, 5]
    far_right_I = I[30, 55]
    edge_I = I[30, 30]

    assert edge_I > far_left_I
    assert edge_I > far_right_I
    assert edge_I == pytest.approx(1.0, abs=1e-2)


def test_entropy_properties():
    """Checks entropy bounds: zero for delta distribution, ln(K) for uniform."""
    # 1. Delta distribution (all pixels have identical value)
    uniform_arr = np.full((100, 100), 0.75)
    en_delta = compute_entropy(uniform_arr, bins=256)
    assert en_delta == pytest.approx(0.0, abs=1e-6)

    # 2. Perfectly uniform distribution across all 256 bins
    # Each bin has equal probability 1/256 -> entropy is ln(256)
    vals = np.linspace(0.001, 0.999, 256 * 100)
    en_uniform = compute_entropy(vals, bins=256)
    expected_max = np.log(256)
    assert en_uniform == pytest.approx(expected_max, abs=1e-2)


def test_entropy_with_mask():
    """Checks that entropy only accounts for pixels within mask."""
    arr = np.zeros((100, 100), dtype=float)
    # Put uniform values in the right half
    arr[:, 50:] = np.tile(np.linspace(0.001, 0.999, 256), (100, 1))[:, :50]
    
    # Mask covers only left half (all zeros)
    mask_left = np.zeros((100, 100), dtype=bool)
    mask_left[:, :50] = True

    en_masked = compute_entropy(arr, mask=mask_left, bins=256)
    assert en_masked == pytest.approx(0.0, abs=1e-6)


def test_enhanced_ns_transform_properties():
    """Checks bounds and edge sensitivity of enhanced_neutrosophic_transform."""
    from src.ns import enhanced_neutrosophic_transform
    img = np.zeros((80, 80), dtype=float)
    img[:, 40:] = 1.0  # step edge
    mask = np.ones((80, 80), dtype=bool)

    T, I, F = enhanced_neutrosophic_transform(img, brain_mask=mask)

    assert T.shape == (80, 80)
    assert I.shape == (80, 80)
    assert F.shape == (80, 80)

    assert np.all(T >= 0.0) and np.all(T <= 1.0)
    assert np.all(I >= 0.0) and np.all(I <= 1.0)
    assert np.all(F >= 0.0) and np.all(F <= 1.0)

    # Edge should have significantly higher indeterminacy than homogeneous areas
    assert I[40, 40] > I[40, 10]
    assert I[40, 40] > I[40, 70]

