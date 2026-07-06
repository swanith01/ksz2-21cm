# =============================================================================
# cache.py — generic .npy cache loading + seed-averaging helpers.
#
# The same "_load_cache" / "_load" / "seed_avg" pattern was copy-pasted with
# small variations into: the embedded SNR cell, cell_snr_with_noise.py,
# SNR_rigorous_v2.py, and overlay_zhou25.py. This is the one copy.
# =============================================================================

import os
import numpy as np


def load_seed_caches(filename_pattern, seeds, cache_dir):
    """
    Load one .npy dict-of-dicts cache per seed, e.g.
        cross_corr_sq_seed{seed}.npy  ->  {z0: {...}, ...}

    Parameters
    ----------
    filename_pattern : str
        e.g. "cross_corr_sq_seed{seed}.npy" — must contain "{seed}".
    seeds : iterable of int
    cache_dir : str
        The run's main cache_dir; each seed's file is expected at
        cache_dir/seed_<seed>/<filename_pattern.format(seed=seed)>.

    Returns
    -------
    dict[int, dict] — {seed: loaded_result_dict}, only for seeds whose
    cache file exists and loads successfully.
    """
    out = {}
    for seed in seeds:
        path = os.path.join(
            cache_dir, f"seed_{seed}", filename_pattern.format(seed=seed)
        )
        if os.path.exists(path):
            try:
                out[seed] = np.load(path, allow_pickle=True).item()
            except Exception as e:
                print(f"  \u2717 seed {seed}: failed to load {path}: {e}")
    return out


def seed_average(results_by_seed, z0, field):
    """
    Average one field across seeds at a given z0/chunk key.

    Parameters
    ----------
    results_by_seed : dict[int, dict[float, dict]]
        {seed: {z0: {field: array, ...}}}
    z0 : float
    field : str

    Returns
    -------
    (mean, std) as np.ndarray, or (None, None) if no seed has usable data.
    """
    vals = []
    for seed, per_z0 in results_by_seed.items():
        if z0 not in per_z0:
            continue
        v = per_z0[z0][field]
        if np.all(~np.isfinite(v)):
            continue
        vals.append(v)
    if not vals:
        return None, None
    arr = np.array(vals)
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)
