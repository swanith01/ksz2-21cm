# ksz2-21cm status and checklist (updated 2026-10-12, step 10)

Goal: reproduce Zhou+25 Fig. 5 (D_l^{kSZ2 x 21cm2} vs z0, x_HII on top axis) and Ma+18 (kSZ x 21cm) with our own
validated stitched-lightcone method, then study how the signal changes with reionization history.

## DONE
- [x] Vendored the stitched construction (density, xH, velocity_z, 21cmFAST brightness_temp share one rotation); kSZ code untouched.
- [x] Stitched pipeline 01 -> 02 -> 04 for many seeds: per-seed .npz cache, power-cut safe, h5 auto-cleanup (~830 MB/seed).
- [x] 15 seeds at 256^3 / 800 Mpc (job 1738182): tau = 0.0607 +/- 0.00004, z(x_e=0.5) = 7.76.
- [x] Main plot (06): D_l vs z0 at l = 400/800/1600, x_HII top axis, Zhou overlay, s.e.m., significance, optional 1/chi(z0)^2 (commit dba5153).
- [x] Plots + README saved on usha: data/plots/2026-10-10 and 2026-10-12.
- [x] Zhou units checked: Fig. 5 is muK^4 (mK^4 is Fig. 7, the 21cm^2 auto-spectrum).
- [x] Read Zhou's methods: zreion, 2 Gpc/h, 1024^3, 30 realizations, T21 = T0(1+delta)(1-x), HERA/SO filters.
- [x] AMBER checked: it has a 21cm module (Tb_*.dat, mK; Ts >> Tgamma, no RSD, no lightcone).

## FINDINGS SO FAR
- Peak at z0 = 9.5 (x_HII ~ 0.2) is real (3-4.5 sigma) and matches Zhou.
- Negative lobe is NOT detected (<= 1.7 sigma); a Zhou-sized lobe is excluded (~4 / 9 / 25 sigma at l = 400 / 800 / 1600).
- Zero-crossing z ~ 7.1-7.4 (x_HII ~ 0.65) vs Zhou z ~ 8.4 (x_HII ~ 0.35); x_HII(z) histories are close, so history is not the cause.
- Amplitude: ~7e4 x above Zhou after 1/chi^2 and correct units. Filters or a normalization bug (unresolved).
- Our reionization: 21cmFAST defaults, R_BUBBLE_MAX = 15 Mpc (small bubbles), HII_EFF_FACTOR = 30, no Ts fluctuations.

## TODO (suggested order)
- [ ] 1. Cluster cleanup: delete unneeded 21cmFAST coeval caches (keep lightcones, stitched .npz, kSZ maps, results).
- [ ] 2. Synthetic-map test of the cross-corr worker normalization (known input -> known C_l). Decides filters vs bug for the 7e4 x.
- [ ] 3. Decide default for --chi-norm (recommend z0) and flip it in 06.
- [ ] 4. Bubble-size test: add astro_params to the config; 3 seeds with larger R_BUBBLE_MAX (e.g. 50 Mpc); compare lobe depth and zero-crossing.
- [ ] 5. Zhou-style HERA noise + SO filters (worker/plot option; kSZ code untouched); re-compare shape and amplitude.
- [ ] 6. Post-reionization test: add --z-min-ksz to 02; cut the kSZ line of sight at z_min = 4, 5, 6, 7; compare mean and scatter of D.
- [x] 7a. Ma+18 Figs. 2/3 digitized (both kSZ-21cm and kSZ^2-21cm) -> plotting/ma18_data.py; workers 04b + plot 08 (kSZ^2 x 21cm with ell<100 removed, kSZ x 21cm linear, cross-seed null). RUN 04b + 08 on swarm.
- [ ] 7b. Read the 08 null-test table: is kSZ x 21cm (linear) zero? does kSZ^2 x 21cm (21cm not squared) go negative near x_e=0.5 like Ma?
- [ ] 7c. Later: Ts-on 21cmFAST run (Ma's z~13 / high-z sign flip needs Ts fluctuations); check APPLY_RSDS in our T21.
- [ ] 7d. Wedge/foreground test (step 11): run 04c + 09; does kSZ^2 x 21cm (21cm not squared) survive k_par / horizon-wedge filtering vs the squared Zhou statistic? (pedagogical plot wedge_survival.*)
- [ ] 8. AMBER: check the cluster build writes Tb_*.dat (21cm block set to write, z grid matching our snapshots); write a reader; sweep z_mid / dz / A_z.

## HOUSEKEEPING
- [ ] Replace the 05 native-vs-stitched ratio statistic (mean of per-bin ratios is meaningless); consider a native 256^3 baseline.
- [ ] kSZ at l < ~700 in the stitched construction is unvalidated; l = 400 points are in that regime.
- [ ] Script 04: make worker count respect NCPUS (PBS Pro exports NCPUS, not PBS_NCPUS).
- [ ] Commit configs/variants/stitched_tiny_cluster.yaml (untracked on swarm).
- [ ] Investigate the stray ~/Desktop/Research/kSZ-pipeline/src/ksz2_21cm directory.
- [ ] Archive the old native modules once the comparison is trusted.
- [ ] Set up an SSH key usha <-> swarm (no password prompts).
