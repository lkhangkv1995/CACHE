import numpy as np
from cache import analyze


def test_analysis_shapes_and_finite_features():
    rng = np.random.default_rng(42)
    im = np.full((96, 128, 3), [240, 183, 204], dtype=np.uint8)
    im = np.clip(im.astype(int) + rng.integers(-15, 16, size=im.shape), 0, 255).astype(np.uint8)
    result = analyze(im, mpp=None, scales_um=(2, 4), window_um=16)
    for k in ('candidate', 'structure_valid', 'bundles', 'theta', 'coherence', 'alignment'):
        assert result[k].shape == im.shape[:2]
    for k in ('candidate_fraction', 'orientation_coverage', 'ridge_coverage', 'global_alignment'):
        assert 0 <= result['features'][k] <= 1


def test_orientation_axial_range():
    im = np.full((96, 96, 3), [245, 195, 212], dtype=np.uint8)
    im[::8, :, :] = [195, 83, 130]
    result = analyze(im, scales_um=(2, 4))
    assert np.all((result['theta'] >= 0) & (result['theta'] < np.pi))
