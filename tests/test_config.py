# =============================================================================
# test_config.py
# A minimal smoke test — this is the one thing the checklist actually needs
# ("no tests" is fine to ship v0.1, but this one is cheap and catches the
# most common failure mode: a config with a typo'd or missing key).
#
# Run:  python -m pytest tests/
# =============================================================================

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ksz2_21cm.utils.config import load_config

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_fiducial_config_loads():
    cfg = load_config(os.path.join(REPO_ROOT, "configs", "fiducial.yaml"))
    assert "paths" in cfg
    assert "simulation" in cfg
    assert isinstance(cfg["simulation"]["random_seeds"], list)
    assert len(cfg["simulation"]["random_seeds"]) > 0


def test_missing_required_key_raises(tmp_path):
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("paths:\n  cache_dir: /tmp/x\n  plot_dir: /tmp/y\n")
    try:
        load_config(str(bad_config))
        assert False, "expected KeyError for missing required keys"
    except KeyError:
        pass
