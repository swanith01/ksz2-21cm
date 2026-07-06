# data/ — manifest, not the data itself

Per the "no huge data files in git" rule, this directory holds only a
manifest of where real data lives, plus (once you start populating it)
small final products.

## Where the real data lives

CHECK and fill in — placeholders below based on the paths used across your
old scripts:

| What | Where (cluster: pride/swarm, user: swanith) | Committed? |
|---|---|---|
| 21cmFAST lightcone HDF5 (`LightCone_*.h5`), per seed | `<cache_dir>/seed_<N>/` — CHECK actual path, was e.g. `~/1Jun2026_kSZ_sqr_21cm_sqr/cache/seed_<N>/` or `/user1/swanith/1Jun2026_kSZ_Sqr_21cm_sqr_code/.../cache/seed_<N>/` in different scripts (these don't agree — pick one and put it in `configs/fiducial.yaml`) | No — multi-GB each, `.gitignore`d |
| kSZ maps (`kSZ_map_z<zobs>_seed<N>.npy`) | `<cache_dir>/kSZ_maps/` | Could be — small (a few MB each) |
| Cross-corr caches (`cross_corr_seed<N>.npy`, `cross_corr_sq_seed<N>.npy`) | `<cache_dir>/seed_<N>/` | Yes, recommended — these are the numbers the paper figures are made from; committing them under `data/products/` means someone can reproduce your figures without re-running 21cmFAST at all |
| Noisy cross-corr cache (`cross_corr_sq_CMBS4_SKA_seed<N>.npy`) | `<cache_dir>/seed_<N>/` | Yes, recommended, same reasoning |

## `data/products/`

Once you're ready, copy the small final products here (or symlink from
the real cache_dir) so a fresh clone of this repo can run
`paper/figure_scripts/*.py` and get the same figures without touching
21cmFAST or `pride`/`swarm` at all. This mirrors the `ksz-pipeline`
convention: "notebooks only load pre-computed products from
`data/products/`, no heavy computation inline."

Nothing is here yet — this is a placeholder directory.
