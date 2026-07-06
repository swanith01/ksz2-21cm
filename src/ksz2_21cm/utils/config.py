# =============================================================================
# config.py — load configs/*.yaml into a plain dict, with env-var path overrides
# and light validation. Scripts should do:
#
#   from ksz2_21cm.utils.config import load_config
#   cfg = load_config("configs/fiducial.yaml")
#   cache_dir = cfg["paths"]["cache_dir"]
#
# This replaces `from run_config import *`. Nothing here is 21cmFAST-specific,
# so it lives in utils/, not simulate/.
# =============================================================================

import os
import yaml

_ENV_OVERRIDES = {
    ("paths", "cache_dir"): "KSZ2_21CM_CACHE_DIR",
    ("paths", "plot_dir"):  "KSZ2_21CM_PLOT_DIR",
}

_REQUIRED_KEYS = [
    ("paths", "cache_dir"),
    ("paths", "plot_dir"),
    ("simulation", "hii_dim"),
    ("simulation", "box_len"),
    ("simulation", "z_min"),
    ("simulation", "z_max"),
    ("simulation", "random_seeds"),
    ("ksz", "z_obs"),
    ("cross_correlation", "k_par_min"),
    ("cross_correlation", "delta_z"),
    ("cross_correlation", "z_chunk_centres"),
]


def load_config(path):
    """
    Load a YAML config file and return a plain nested dict.

    - Expands `~` in any path-like string field.
    - Applies KSZ2_21CM_CACHE_DIR / KSZ2_21CM_PLOT_DIR env var overrides,
      so the same config file works on a laptop and on pride/swarm.
    - Fails loudly (KeyError) if a required key is missing, rather than
      silently proceeding like `from run_config import *` used to.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    for (section, key), env_name in _ENV_OVERRIDES.items():
        env_val = os.environ.get(env_name)
        if env_val:
            cfg.setdefault(section, {})[key] = env_val

    if "paths" in cfg:
        for key, val in cfg["paths"].items():
            if isinstance(val, str):
                cfg["paths"][key] = os.path.expanduser(val)

    missing = []
    for section, key in _REQUIRED_KEYS:
        if section not in cfg or key not in cfg[section]:
            missing.append(f"{section}.{key}")
    if missing:
        raise KeyError(
            f"Config '{path}' is missing required keys: {missing}. "
            f"Did you copy the real values over from run_config.py? "
            f"See the '# CHECK' comments in configs/fiducial.yaml."
        )

    return cfg


def ensure_dirs(cfg):
    """Create cache_dir and plot_dir (and standard subdirs) if they don't exist."""
    os.makedirs(cfg["paths"]["cache_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["plot_dir"], exist_ok=True)
    os.makedirs(os.path.join(cfg["paths"]["cache_dir"], "kSZ_maps"), exist_ok=True)
