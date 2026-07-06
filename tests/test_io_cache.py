# =============================================================================
# test_io_cache.py — smoke test for the cache load/seed-average helpers.
# Run:  python -m pytest tests/
# =============================================================================

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ksz2_21cm.io.cache import load_seed_caches, seed_average


def test_load_and_average_roundtrip(tmp_path):
    seeds = [1, 2, 3]
    for seed in seeds:
        seed_dir = tmp_path / f"seed_{seed}"
        seed_dir.mkdir()
        data = {9.0: {"D_cross": np.full(5, float(seed))}}
        np.save(seed_dir / f"cross_corr_sq_seed{seed}.npy", data)

    loaded = load_seed_caches("cross_corr_sq_seed{seed}.npy", seeds, str(tmp_path))
    assert len(loaded) == 3

    mean, std = seed_average(loaded, 9.0, "D_cross")
    assert np.allclose(mean, 2.0)  # mean of [1, 2, 3]
    assert np.allclose(std, np.std([1.0, 2.0, 3.0]))


def test_missing_seed_is_skipped_not_crashed(tmp_path):
    loaded = load_seed_caches("cross_corr_sq_seed{seed}.npy", [42], str(tmp_path))
    assert loaded == {}
