# =============================================================================
# lightcone_worker.py
# Worker function for concurrent lightcone simulations.
# Runs in forked subprocesses — fully self-contained with all imports.
#
# [repo note] This is the module containing the heavy py21cmfast call
# (run_lightcone). Moved here unchanged from the top-level script of the
# same name — this is exactly the "heavy 21cmFAST calls as a separate
# module" split that was requested; no logic changed.
# Imported by scripts/01_run_lightcones.py.
# =============================================================================

import os
import glob
import time as _time
import py21cmfast as _p21c


def run_or_load_seed(seed, seed_cache_dir, z_min, z_max,
                     hii_dim, box_len, n_threads):
    """
    Run or load one lightcone simulation for a given random seed.

    Parameters
    ----------
    seed           : int   — random seed
    seed_cache_dir : str   — directory to store/retrieve the LightCone HDF5
    z_min          : float — lightcone lower redshift bound
    z_max          : float — lightcone upper redshift bound
    hii_dim        : int   — HII_DIM (grid resolution)
    box_len        : float — BOX_LEN in Mpc
    n_threads      : int   — threads for this worker

    Returns
    -------
    (seed, cache_file_path, sim_time_seconds, status_string)
    status_string is one of "cached", "computed", or "failed: <msg>"
    """

    os.makedirs(seed_cache_dir, exist_ok=True)

    # ── [FIX — found during desktop testing] ─────────────────────────────────
    # py21cmfast derives its FFTW-wisdom / cache directory from a GLOBAL,
    # machine-wide config value: config["direc"] (normally ~/21cmFAST-cache,
    # stored in ~/.21cmfast/config.yml). On at least one desktop this value
    # was corrupted to the literal string "<class 'pathlib.Path'>" — a
    # py21cmfast first-run config bug, not something in this code — and
    # py21cmfast's own wisdoms_path.mkdir() call has no exist_ok=True, so two
    # seeds racing to create that same broken path for the first time throws
    # FileExistsError for whichever loses the race. It may also explain a
    # second symptom seen on the same run: a seed reporting "computed"
    # successfully but no LightCone_*.h5 turning up in seed_cache_dir — if
    # the global config is what actually controls where results land, it may
    # not be seed_cache_dir at all.
    #
    # Fix: don't trust the machine-wide default. Force it to our own
    # seed-specific directory before doing anything else. This also
    # eliminates the race as a side effect, since no two seeds now share the
    # same directory to fight over.
    _p21c.config['direc'] = seed_cache_dir

    # ── Build py21cmfast params inside the subprocess ────────────────────────
    # CFFI objects cannot survive a fork reliably; always reconstruct them here.
    up = _p21c.UserParams(
        HII_DIM=hii_dim,
        BOX_LEN=box_len,
        USE_INTERPOLATION_TABLES=True,
        N_THREADS=n_threads,
    )
    ap = _p21c.AstroParams()

    # ── Cache check ──────────────────────────────────────────────────────────
    # FIX 6: Validate the cache by simply reading the HDF5 file, NOT by
    # re-running run_lightcone(write=False) which costs almost as much as
    # a full recompute.
    cached = sorted(glob.glob(os.path.join(seed_cache_dir, "LightCone_*.h5")))
    valid_cached = [f for f in cached if os.path.getsize(f) > 1e6]  # > 1 MB

    if valid_cached:
        cache_file = valid_cached[0]
        try:
            _ = _p21c.LightCone.read(cache_file)
            return (seed, cache_file, 0.0, "cached")
        except Exception:
            # File exists but is corrupt — fall through to recompute
            pass

    # ── Run new simulation ───────────────────────────────────────────────────
    sim_start = _time.time()
    try:
        lc = _p21c.run_lightcone(
            redshift=z_min,
            max_redshift=z_max,
            lightcone_quantities=('brightness_temp', 'density',
                                  'xH_box', 'velocity'),
            user_params=up,
            astro_params=ap,
            random_seed=seed,
            direc=seed_cache_dir,   # intermediate box cache goes in the per-seed dir
            write=True,             # caches INTERMEDIATE boxes (init, perturb_field,
                                    # etc.) for faster re-runs — NOT the final LightCone
        )

        # ── [FIX — the actual bug] ────────────────────────────────────────────
        # Every py21cmfast tutorial example calls lightcone.save() EXPLICITLY,
        # even when write=True was passed to run_lightcone — because write=True
        # only controls intermediate-box caching, not the final LightCone file.
        # The old code's "FIX 7" comment ("do NOT call lc.save() again —
        # write=True already saved it") was wrong; that's exactly why every
        # worker was reporting "computed" while writing no retrievable file
        # anywhere. save() also conveniently returns the exact path it wrote
        # to, so there's no need to glob-guess afterward.
        cache_file = str(lc.save(direc=seed_cache_dir))

        sim_time = _time.time() - sim_start
        return (seed, cache_file, sim_time, "computed")

    except Exception as e:
        sim_time = _time.time() - sim_start
        return (seed, None, sim_time, f"failed: {str(e)}")
