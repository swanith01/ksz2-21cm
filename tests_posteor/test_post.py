import sys, importlib.util, numpy as np
S = "/tmp/claude-0/-home-claude/cd4b2d82-d0d2-583c-9208-b8b93b82bcbd/scratchpad/t18"
sys.path.insert(0, S + "/tree/src")
from ksz2_21cm.simulate.stitch_from_coeval import build_los_z_grid, comoving_distance_mpc
from ksz2_21cm.simulate.stitch_below import build_low_z_grid
from ksz2_21cm.utils.constants import MPC_CM, ne0_cgs
from ksz2_21cm.ksz.optical_depth import analytic_tau_below, compute_tau, compute_patchy_mask, compute_visibility
from ksz2_21cm.ksz.lightcone_integral import compute_ksz_map
spec = importlib.util.spec_from_file_location("p02b", "/home/claude/ksz2-21cm-deploy18/scripts/02b_compute_ksz_posteor.py")
m02b = importlib.util.module_from_spec(spec); spec.loader.exec_module(m02b)

rng = np.random.default_rng(5); N, L = 16, 160.0; cell = L / N; ne0 = ne0_cgs()
z_arr = build_los_z_grid(4.0, 20.0, cell); pos = np.array([comoving_distance_mpc(z) for z in z_arr]); n = len(z_arr)
xH_1d = np.clip((z_arr - 5.5) / 6.0, 0, 1)                 # fully ionized below z=5.5, neutral above 11.5
xH = np.clip(xH_1d[None, None, :] + 0.03 * rng.standard_normal((N, N, n)) * (xH_1d > 0)[None, None, :], 0, 1).astype(np.float32)
xH[:, :, xH_1d == 0] = 0.0
main = dict(z_arr=z_arr, pos_axis=pos, xH_box=xH,
            density=(0.2 * rng.standard_normal((N, N, n))).astype(np.float32),
            velocity_z=(3e7 * rng.standard_normal((N, N, n))).astype(np.float32))
z_low = build_low_z_grid(4.0, 0.001, cell); pos_low = np.array([comoving_distance_mpc(z) for z in z_low])
low = dict(z_arr=z_low, pos_axis=pos_low,
           density=(0.4 * rng.standard_normal((N, N, len(z_low)))).astype(np.float32),
           velocity_z=(3e7 * rng.standard_normal((N, N, len(z_low)))).astype(np.float32))

# --- script 02's own EoR-only map
x_e = 1 - xH.mean(axis=(0, 1)); tau0 = analytic_tau_below(float(z_arr.min()))
z_mid, ds, dtau, tau = compute_tau(x_e, z_arr, pos, ne0=ne0, tau0=tau0)
_, vis_main, vis3 = compute_visibility(tau, z_arr, z_mid); mask, mask3 = compute_patchy_mask(x_e)
eor = compute_ksz_map(1 + main["density"], 1 - xH, main["velocity_z"] / MPC_CM, red_axis=z_arr, ds=ds,
                      visibility_3D=vis3, ne0=ne0, patchy_mask_3D=mask3)
# --- post map from the new code, two block sizes
post_a, info = m02b.posteor_map(main, low, ne0, block=7)
post_b, _ = m02b.posteor_map(main, low, ne0, block=100000)
print("block-size independence: max|a-b| / rms =", np.max(np.abs(post_a - post_b)) / post_b.std())
print("patchy window ends at z=%.2f (index %d); existing post slices kept: %d; low slices: %d" % (info["z_patchy_end"], info["i_p"], info["n_keep"] - 1, info["n_low"]))
print("visibility at the join: low-z analytic %.5f vs existing %.5f" % info["vis_join"])
# --- additivity: total over everything (patchy + post slices) == eor + post
n_keep = info["n_keep"]
mask_full_main = np.where(np.arange(n) < n_keep, 1.0, mask)                       # 1 on post slices, patchy mask elsewhere
zg = np.linspace(0, 4.0, 300); vis_low = np.exp(-np.interp(z_low, zg, [analytic_tau_below(z) for z in zg]))
z_all = np.concatenate([z_low, z_arr]); pos_all = np.concatenate([pos_low, pos]); vis_main = vis_main.copy(); vis_main[0] = np.exp(-tau0); vis_all = np.concatenate([vis_low, vis_main])
full = m02b.ksz_chunked(1 + np.concatenate([low["density"], main["density"]], 2),
                        np.concatenate([np.ones_like(low["density"]), 1 - xH], 2),
                        np.concatenate([low["velocity_z"], main["velocity_z"]], 2) / MPC_CM,
                        z_all, np.diff(pos_all), vis_all, ne0,
                        np.concatenate([np.ones(len(z_low)), mask_full_main]), block=50)
resid = full - (eor + post_b)
print("additivity: max|full - (EoR+post)| / rms(full) = %.2e" % (np.max(np.abs(resid)) / full.std()))
print("rms: EoR %.3e  post %.3e  total %.3e" % (eor.std(), post_b.std(), (eor + post_b).std()))
# --- tau consistency: numeric x_e=1 integral vs analytic (He-III in the analytic one)
xe1 = np.ones(len(z_low) + 1); zz = np.concatenate([z_low, [4.0]]); pp = np.concatenate([pos_low, [pos[0]]])
_, _, _, tnum = compute_tau(xe1, zz, pp, ne0=ne0, tau0=analytic_tau_below(float(z_low.min())))
print("tau(0->4): numeric (H+HeII, x_e=1) %.5f vs analytic (incl. HeIII transition) %.5f" % (tnum[-1], analytic_tau_below(4.0)))
assert np.max(np.abs(post_a - post_b)) / post_b.std() < 1e-6 and np.max(np.abs(resid)) / full.std() < 1e-6
print("ALL OK")
