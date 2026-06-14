"""
postprocess.py  —  Plots & animation for Crazyflie 2.1 NED-LQR simulation.

Loads sim_out.npz produced by sim.py.
Run independently: python postprocess.py
All figures saved under figures/.  Animation shown in a live window.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from typing import Any, Optional, cast

import params as P
import lqr_control as lqr
import control as ctrl

os.makedirs("figures", exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading sim_out.npz …")
data         = np.load("sim_out.npz")
t            = data["t"]
xs           = data["xs"]                  # (N, 23)
ctrl_hist    = data["ctrl_hist"]           # (N, 5): Fc, Lc, Mc, Nc, sat
wind_hist    = data["wind_hist"]           # (N, 3): F_ned
ref_hist     = data["ref_hist"]            # (N, 4): pd, pn, pe, psi refs
WIND_INTENSITY = float(data["wind_intensity"][0])
T_END        = float(data["T_END"][0])

N = len(t)

# ── Unpack states ─────────────────────────────────────────────────────────────
pn_s, pe_s, pd_s   = xs[:,0], xs[:,1], xs[:,2]
u_s, v_s, w_s      = xs[:,3], xs[:,4], xs[:,5]
phi_s, theta_s, psi_s = xs[:,6], xs[:,7], xs[:,8]
p_s, q_s, r_s      = xs[:,9], xs[:,10], xs[:,11]
pwm_act             = xs[:,12:16]
alt_s               = -pd_s

# ── Unpack control history ────────────────────────────────────────────────────
Fc_h, Lc_h, Mc_h, Nc_h = ctrl_hist[:,0], ctrl_hist[:,1], ctrl_hist[:,2], ctrl_hist[:,3]

# ── Reference signals ─────────────────────────────────────────────────────────
pd_ref_a, pn_ref_a, pe_ref_a, psi_ref_a = [ref_hist[:,i] for i in range(4)]
alt_ref_a = -pd_ref_a

# ── Derived ───────────────────────────────────────────────────────────────────
def _wrap(err):
    return (err + np.pi) % (2 * np.pi) - np.pi

e_alt = alt_s - alt_ref_a
e_pn  = pn_s  - pn_ref_a
e_pe  = pe_s  - pe_ref_a
e_psi = np.array([_wrap(psi_s[i] - psi_ref_a[i]) for i in range(N)])

ei_pd_a, ei_pn_a, ei_pe_a = xs[:,16], xs[:,17], xs[:,18]
dF_a = Fc_h - P.MASS * P.GRAVITY

pwm_pct    = pwm_act * 100.
pwm_mean_a = pwm_pct.mean(axis=1)
pwm_sprd_a = pwm_pct.max(axis=1) - pwm_pct.min(axis=1)

# Wind in % of weight
wind_pct = wind_hist / (P.MASS * P.GRAVITY) * 100.
wind_mag = np.linalg.norm(wind_hist, axis=1) / (P.MASS * P.GRAVITY) * 100.  # scalar % of weight


###############################################################################
# Fig 1 — Position tracking
fig1, axs = plt.subplots(3, 2, figsize=(14, 10))
fig1.suptitle("Position Tracking — NED LQR (Crazyflie 2.1)", fontsize=14, fontweight="bold")
pairs = [(alt_s, alt_ref_a, e_alt, "Altitude (m↑)", "Altitude"),
         (pn_s,  pn_ref_a,  e_pn,  "North pn (m)",  "North"),
         (pe_s,  pe_ref_a,  e_pe,  "East pe (m)",   "East")]
for row, (act, ref, err, yl, tl) in enumerate(pairs):
    ax = axs[row, 0]
    ax.plot(t, act, "C0", lw=2, label="Actual")
    ax.plot(t, ref, "C3--", lw=1.5, label="Ref")
    ax.set(ylabel=yl, title=tl); ax.legend(fontsize=8); ax.grid(True)
    ax = axs[row, 1]
    ax.plot(t, err, "C3", lw=2)
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.fill_between(t, err, 0, alpha=0.12, color="C3")
    ax.set(ylabel="Error (m)", title=tl + " Error"); ax.grid(True)
axs[2, 0].set_xlabel("Time (s)"); axs[2, 1].set_xlabel("Time (s)")
fig1.tight_layout(); fig1.savefig("figures/fig_position.png", dpi=150)
print("Saved figures/fig_position.png")


###############################################################################
# Fig 2 — Attitude & rates
fig2, axs2 = plt.subplots(2, 3, figsize=(14, 8))
fig2.suptitle("Attitude & Body Rates", fontsize=13, fontweight="bold")
att_sigs  = [phi_s, theta_s, psi_s]
att_refs  = [None, None, psi_ref_a]
att_lbls  = ["Roll φ (°)", "Pitch θ (°)", "Yaw ψ (°)"]
rate_sigs = [p_s, q_s, r_s]
rate_lbls = ["Roll rate p (°/s)", "Pitch rate q (°/s)", "Yaw rate r (°/s)"]
cols      = ["C2", "C4", "C5"]
for i in range(3):
    ax = axs2[0, i]
    ax.plot(t, np.degrees(att_sigs[i]), cols[i], lw=2, label="Actual")
    if att_refs[i] is not None:
        ax.plot(t, np.degrees(att_refs[i]), "k--", lw=1.5, label="Ref")
        ax.legend(fontsize=8)
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set(title=att_lbls[i]); ax.grid(True)
    axs2[1, i].plot(t, np.degrees(rate_sigs[i]), cols[i], lw=2)
    axs2[1, i].axhline(0, color="k", ls="--", lw=0.8)
    axs2[1, i].set(xlabel="Time (s)", title=rate_lbls[i]); axs2[1, i].grid(True)
fig2.tight_layout(); fig2.savefig("figures/fig_attitude.png", dpi=150)
print("Saved figures/fig_attitude.png")


###############################################################################
# Fig 3 — Motors
fig3, axs3 = plt.subplots(2, 1, figsize=(12, 7))
fig3.suptitle("Motor Actuator States (PWM)", fontsize=13, fontweight="bold")
mcols = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
for i in range(4):
    axs3[0].plot(t, pwm_pct[:, i], color=mcols[i], lw=1.5, label=f"M{i+1}")
axs3[0].axhline(P.PWM_HOVER * 100, color="k", ls="--", lw=1,
                label=f"Hover ({P.PWM_HOVER*100:.1f}%)")
axs3[0].set(ylabel="PWM (%)", title="Per-motor"); axs3[0].legend(); axs3[0].grid(True)
axs3[1].plot(t, pwm_mean_a, "k", lw=2, label="Mean")
axs3[1].fill_between(t, pwm_mean_a - pwm_sprd_a/2, pwm_mean_a + pwm_sprd_a/2,
                     alpha=0.25, color="steelblue", label="Spread (attitude effort)")
axs3[1].axhline(P.PWM_HOVER * 100, color="k", ls="--", lw=1)
axs3[1].set(xlabel="Time (s)", ylabel="PWM (%)", title="Mean ± Spread")
axs3[1].legend(); axs3[1].grid(True)
fig3.tight_layout(); fig3.savefig("figures/fig_motors.png", dpi=150)
print("Saved figures/fig_motors.png")


###############################################################################
# Fig 4 — Wind disturbance
fig4, axs4 = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
fig4.suptitle(f"Wind Disturbance  (intensity = {WIND_INTENSITY})", fontsize=13,
              fontweight="bold")
ned_lbls = ["N", "E", "D"]
ned_cols = ["C0", "C1", "C2"]
for j in range(3):
    axs4[0].plot(t, wind_hist[:, j] * 1e3, color=ned_cols[j],
                 lw=1.2, alpha=0.85, label=f"F_{ned_lbls[j]}")
axs4[0].axhline(0, color="k", ls="--", lw=0.7)
axs4[0].set(ylabel="Force (mN)", title="Wind force in NED frame")
axs4[0].legend(); axs4[0].grid(True)

# Magnitude as % of weight for context
axs4[1].plot(t, wind_mag, color="C3", lw=1.5, label="|F_wind|")
axs4[1].fill_between(t, 0, wind_mag, alpha=0.2, color="C3")
axs4[1].set(xlabel="Time (s)", ylabel="% of weight",
            title="Gust magnitude relative to drone weight")
axs4[1].legend(); axs4[1].grid(True)
fig4.tight_layout(); fig4.savefig("figures/fig_wind.png", dpi=150)
print("Saved figures/fig_wind.png")


###############################################################################
# Fig 5 — Full diagnostic dashboard
fig5 = plt.figure(figsize=(20, 15))
fig5.suptitle("Full Controller Diagnostic Dashboard — NED LQR (Crazyflie 2.1)",
              fontsize=14, fontweight="bold")
gs = GridSpec(4, 4, figure=fig5, hspace=0.44, wspace=0.35)
_ax_cache: dict = {}
def _ax(r, c):
    if (r, c) not in _ax_cache:
        _ax_cache[(r, c)] = fig5.add_subplot(gs[r, c])
    return _ax_cache[(r, c)]
def _plot(ax, y, col, ref=None, title="", ylabel=""):
    ax.plot(t, y, col, lw=2)
    if ref is not None: ax.plot(t, ref, "k--", lw=1.5)
    ax.axhline(0, color="k", ls="--", lw=0.7); ax.grid(True)
    ax.set(title=title, ylabel=ylabel)
def _fill(ax, y, col, title="", ylabel=""):
    ax.plot(t, y, col, lw=2); ax.axhline(0, color="k", ls="--", lw=0.7)
    ax.fill_between(t, y, 0, alpha=0.15, color=col); ax.grid(True)
    ax.set(title=title, ylabel=ylabel)

_plot(_ax(0,0), alt_s,               "C0", alt_ref_a,               title="Altitude (m↑)")
_fill(_ax(0,1), e_alt,               "C3",                           title="Alt Error (m)")
_plot(_ax(0,2), ei_pd_a,             "C1",                           title="pd Integrator")
_fill(_ax(0,3), dF_a,                "C6",                           title="ΔThrust (N)")
_plot(_ax(1,0), pn_s,                "C0", pn_ref_a,                title="North pn (m)")
_fill(_ax(1,1), e_pn,                "C3",                           title="North Error (m)")
_plot(_ax(1,2), np.degrees(theta_s), "C4",                           title="Pitch θ (°)")
_fill(_ax(1,3), Mc_h,                "C4",                           title="Pitch Moment (N·m)")
_plot(_ax(2,0), pe_s,                "C0", pe_ref_a,                title="East pe (m)")
_fill(_ax(2,1), e_pe,                "C3",                           title="East Error (m)")
_plot(_ax(2,2), np.degrees(phi_s),   "C2",                           title="Roll φ (°)")
_fill(_ax(2,3), Lc_h,                "C2",                           title="Roll Moment (N·m)")
_plot(_ax(3,0), np.degrees(psi_s),   "C5", np.degrees(psi_ref_a),   title="Yaw ψ (°)")
_fill(_ax(3,1), np.degrees(e_psi),   "C5",                           title="Yaw Error (°)")

# Wind magnitude on dashboard (replaces empty slot)
ax_wd = _ax(3, 2)
ax_wd.plot(t, wind_mag, "C3", lw=1.5)
ax_wd.fill_between(t, 0, wind_mag, alpha=0.2, color="C3")
ax_wd.axhline(0, color="k", ls="--", lw=0.7); ax_wd.grid(True)
ax_wd.set(title=f"Wind |F| (% weight)  int={WIND_INTENSITY}")

ax_pw = _ax(3, 3)
ax_pw.plot(t, pwm_mean_a, "k", lw=2, label="Mean")
ax_pw.fill_between(t, pwm_mean_a - pwm_sprd_a/2, pwm_mean_a + pwm_sprd_a/2,
                   alpha=0.25, color="steelblue", label="Spread")
ax_pw.axhline(P.PWM_HOVER * 100, color="gray", ls="--", lw=1.2)
ax_pw.set(title="Motor PWM mean±spread"); ax_pw.legend(fontsize=7); ax_pw.grid(True)
for r, lbl in enumerate(["ALTITUDE", "FORWARD (pn)", "LATERAL (pe)", "YAW"]):
    _ax(r, 0).set_ylabel(lbl, fontsize=10, fontweight="bold")
fig5.savefig("figures/fig_dashboard.png", dpi=150)
print("Saved figures/fig_dashboard.png")


###############################################################################
# Fig 6 — 3D path history
fig6 = plt.figure(figsize=(9, 7))
ax6  = fig6.add_subplot(111, projection="3d")
sc   = ax6.scatter(pe_s, pn_s, alt_s, c=t, cmap="plasma", s=3)
plt.colorbar(sc, ax=ax6, label="Time (s)", shrink=0.6)
ax6.set(xlabel="East (m)", ylabel="North (m)", zlabel="Alt (m)",
        title="3-D Flight Path  [colour = time]")
fig6.tight_layout(); fig6.savefig("figures/fig_3d_path.png", dpi=150)
print("Saved figures/fig_3d_path.png")


###############################################################################
# Fig 7 — Stability margins (Bode)
fig7, axs7 = plt.subplots(2, 4, figsize=(18, 7))
fig7.suptitle("Open-Loop Bode — Stability Margins", fontsize=13, fontweight="bold")
def _bode(ax_m, ax_p, A, B, K, title):
    try:
        sys_ol = ctrl.ss(A, B, K, np.zeros((1, 1)))
        om     = np.logspace(-2, 4, 1000)
        mag, phase, om_out = ctrl.bode(sys_ol, om, plot=False)
        mdb = 20 * np.log10(np.squeeze(mag))
        ph  = np.degrees(np.squeeze(phase))
        ax_m.semilogx(om_out, mdb, "b", lw=2)
        ax_m.axhline(0, color="k", ls="--", lw=0.8)
        ax_m.grid(which="both"); ax_m.set(ylabel="Mag (dB)")
        ax_p.semilogx(om_out, ph, "r", lw=2)
        ax_p.axhline(-180, color="k", ls="--", lw=0.8)
        ax_p.set(xlabel="rad/s", ylabel="Phase (°)"); ax_p.grid(which="both")
        gm, pm, _, _ = ctrl.margin(sys_ol)
        gm_db = 20 * np.log10(gm) if np.isfinite(gm) and gm > 0 else float("inf")
        ax_m.set_title(f"{title}\nGM={gm_db:.1f}dB  PM={pm:.1f}°", fontsize=9)
    except Exception as exc:
        ax_m.text(0.1, 0.5, str(exc), transform=ax_m.transAxes, fontsize=7)
_bode(axs7[0,0], axs7[1,0], lqr.A_z,   lqr.B_z,   lqr.K_z,   "Altitude (pd)")
_bode(axs7[0,1], axs7[1,1], lqr.A_pn,  lqr.B_pn,  lqr.K_pn,  "North (pn)")
_bode(axs7[0,2], axs7[1,2], lqr.A_pe,  lqr.B_pe,  lqr.K_pe,  "East (pe)")
_bode(axs7[0,3], axs7[1,3], lqr.A_yaw, lqr.B_yaw, lqr.K_yaw, "Yaw rate")
fig7.tight_layout(); fig7.savefig("figures/fig_stability_margins.png", dpi=150)
print("Saved figures/fig_stability_margins.png")

plt.close("all")


###############################################################################
# Animation

def _R(phi, theta, psi):
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cy, sy = np.cos(psi), np.sin(psi)
    return np.array([[ct*cy, sy*sp*ct - cp*sy, cp*sy*ct + sp*sy],
                     [ct*sy, sp*sy*st + cp*cy, cp*sy*st - sp*cy],
                     [  -st,           sp*ct,           cp*ct  ]])

def _wrap_s(err): return (err + np.pi) % (2 * np.pi) - np.pi

ARM_VIZ = P.ARM_LENGTH * 8
dg = ARM_VIZ * np.sqrt(2) / 2
mb = np.array([[ dg, -dg, -dg,  dg],
               [ dg,  dg, -dg, -dg],
               [  0,   0,   0,   0]])

_ref_t   = np.linspace(0, T_END, 400)
_ref_pd  = np.array([ref_hist[np.searchsorted(t, ti, side="right").clip(0, N-1), 0] for ti in _ref_t])
_ref_pn  = np.array([ref_hist[np.searchsorted(t, ti, side="right").clip(0, N-1), 1] for ti in _ref_t])
_ref_pe  = np.array([ref_hist[np.searchsorted(t, ti, side="right").clip(0, N-1), 2] for ti in _ref_t])
_ref_alt = -_ref_pd

_PHASES = [
    (0.0,  0.5,  "Takeoff"),
    (0.5,  8.0,  "Hover"),
    (8.0,  8.5,  "Move North"),
    (8.5, 10.0,  "Hold North"),
    (10.0, 16.0, "Yaw slew →90°"),
    (16.0, 16.5, "Move East"),
    (16.5, T_END, "Hold position"),
]
def _phase_label(tc):
    for t0, t1, label in _PHASES:
        if t0 <= tc < t1: return label
    return ""

matplotlib.use("QtAgg")
print("Generating 3D animation …")

fig_a = plt.figure(figsize=(13, 7))
fig_a.patch.set_facecolor("#0d0d0d")
ax_a = cast(Axes3D, fig_a.add_axes((0.01, 0.04, 0.62, 0.88), projection="3d"))
ax_a.set_facecolor("#0d0d0d")
for pane in (getattr(ax_a.xaxis, 'pane', None), getattr(ax_a.yaxis, 'pane', None), getattr(ax_a.zaxis, 'pane', None)):
    if pane is not None:
        pane.fill = False
        pane.set_edgecolor("#333333")
ax_a.tick_params(colors="#888888", labelsize=7)
ax_a.xaxis.label.set_color("#888888"); ax_a.xaxis.label.set_fontsize(8)
ax_a.yaxis.label.set_color("#888888"); ax_a.yaxis.label.set_fontsize(8)
ax_a.zaxis.label.set_color("#888888"); ax_a.zaxis.label.set_fontsize(8)

ghost_path, = ax_a.plot(_ref_pe, _ref_pn, _ref_alt,
                         color="#444444", lw=1, ls="--", alpha=0.6, zorder=1)
path_line,  = ax_a.plot([], [], [], color="#3a7bd5", lw=1.2, alpha=0.55, zorder=2)
arm1_line,  = ax_a.plot([], [], [], color="#eeeeee", lw=3, zorder=5)
arm2_line,  = ax_a.plot([], [], [], color="#eeeeee", lw=3, zorder=5)
motors_plt, = ax_a.plot([], [], [], 'o', markersize=7, color="#e41a1c", zorder=6)
target_pt,  = ax_a.plot([], [], [], marker="+", markersize=14, lw=0,
                         color="#00e676", zorder=4)
wind_quiver: list[Optional[Any]] = [None]
vel_quiver:  list[Optional[Any]] = [None]

# HUD panel
ax_hud = fig_a.add_axes((0.645, 0.38, 0.34, 0.56))
ax_hud.set_facecolor("#111111"); ax_hud.set_xticks([]); ax_hud.set_yticks([])
for sp in ax_hud.spines.values(): sp.set_edgecolor("#333333")
hud_text = ax_hud.text(0.05, 0.97, "", transform=ax_hud.transAxes,
                        fontsize=8.5, va="top", ha="left",
                        fontfamily="monospace", color="#e0e0e0", linespacing=1.6)

# PWM bars
ax_pwm = fig_a.add_axes((0.645, 0.10, 0.34, 0.22))
ax_pwm.set_facecolor("#111111")
ax_pwm.set_xlim(0, 100); ax_pwm.set_ylim(-0.5, 3.5)
ax_pwm.set_xlabel("PWM (%)", color="#888888", fontsize=8)
ax_pwm.set_yticks([0, 1, 2, 3])
ax_pwm.set_yticklabels(["M1", "M2", "M3", "M4"], color="#cccccc", fontsize=8)
ax_pwm.tick_params(axis="x", colors="#888888", labelsize=7)
for sp in ax_pwm.spines.values(): sp.set_edgecolor("#333333")
ax_pwm.axvline(P.PWM_HOVER * 100, color="#ffb300", lw=1, ls="--", alpha=0.7,
               label=f"Hover ({P.PWM_HOVER*100:.0f}%)")
ax_pwm.legend(fontsize=7, loc="lower right", labelcolor="#ffb300", framealpha=0)
mcols_pwm = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
pwm_bars  = ax_pwm.barh([0,1,2,3], [0,0,0,0], color=mcols_pwm, height=0.55, alpha=0.85)

title_txt = fig_a.text(0.33, 0.97, "Crazyflie 2.1  —  NED LQR",
                        ha="center", va="top", fontsize=12,
                        color="#ffffff", fontweight="bold")
time_txt  = fig_a.text(0.33, 0.025, "", ha="center", va="bottom", fontsize=10,
                        color="#aaaaaa", fontfamily="monospace")
phase_txt = fig_a.text(0.33, 0.935, "", ha="center", va="top", fontsize=9.5,
                        color="#00e676", fontstyle="italic")
wind_txt  = fig_a.text(0.645, 0.365, "", ha="left", va="top", fontsize=8,
                        color="#ff7043", fontfamily="monospace")

def _set3d(line, xs, ys, zs):
    if hasattr(line, "set_data_3d"):
        line.set_data_3d(xs, ys, zs)
    else:
        line.set_data(xs, ys); line.set_3d_properties(zs)

anim_fps    = 30
sim_dt      = np.mean(np.diff(t))
stride      = max(1, int((1.0 / anim_fps) / sim_dt))
frame_idx_a = list(range(0, N, stride))
pwm_frames  = pwm_pct[frame_idx_a]

def init_anim():
    ax_a.set_xlim(-1, 5); ax_a.set_ylim(-1, 5); ax_a.set_zlim(0, 5)
    ax_a.set_xlabel("East (m)"); ax_a.set_ylabel("North (m)"); ax_a.set_zlabel("Alt (m)")
    return (path_line, arm1_line, arm2_line, motors_plt, target_pt, hud_text, time_txt, phase_txt)

def update_anim(fi):
    idx = frame_idx_a[fi]
    tc  = t[idx]

    # Path
    _set3d(path_line, pe_s[:idx+1], pn_s[:idx+1], alt_s[:idx+1])

    # Drone geometry
    Rb     = _R(phi_s[idx], theta_s[idx], psi_s[idx])
    mp_ned = Rb @ mb
    cg_enu = np.array([pe_s[idx], pn_s[idx], alt_s[idx]])
    mp_enu = np.array([mp_ned[1], mp_ned[0], -mp_ned[2]]) + cg_enu[:, None]
    _set3d(arm1_line, [mp_enu[0,0], mp_enu[0,2]],
                      [mp_enu[1,0], mp_enu[1,2]],
                      [mp_enu[2,0], mp_enu[2,2]])
    _set3d(arm2_line, [mp_enu[0,1], mp_enu[0,3]],
                      [mp_enu[1,1], mp_enu[1,3]],
                      [mp_enu[2,1], mp_enu[2,3]])
    _set3d(motors_plt, mp_enu[0], mp_enu[1], mp_enu[2])

    # Target
    pd_r, pn_r, pe_r, psi_r = ref_hist[idx]
    _set3d(target_pt, [pe_r], [pn_r], [-pd_r])

    # Velocity arrow
    if vel_quiver[0] is not None: vel_quiver[0].remove()
    speed = np.linalg.norm([u_s[idx], v_s[idx], w_s[idx]])
    if speed > 0.05:
        vel_enu = np.array([v_s[idx], u_s[idx], -w_s[idx]])
        sc_v    = 0.5 / max(speed, 0.5)
        vel_quiver[0] = ax_a.quiver(*cg_enu, *(vel_enu * sc_v),
                                     color="#ffb300", lw=1.5,
                                     arrow_length_ratio=0.35, alpha=0.9)
    else:
        vel_quiver[0] = None

    # Wind force arrow (cyan, NED→ENU)
    if wind_quiver[0] is not None: wind_quiver[0].remove()
    F_wind = wind_hist[idx]
    w_mag  = np.linalg.norm(F_wind)
    if w_mag > 0.002 * P.MASS * P.GRAVITY:    # show if > 0.2 % of weight
        w_enu  = np.array([F_wind[1], F_wind[0], -F_wind[2]])
        sc_w   = 0.8 / max(w_mag, 0.01)
        wind_quiver[0] = ax_a.quiver(*cg_enu, *(w_enu * sc_w),
                                      color="#00bcd4", lw=1.5,
                                      arrow_length_ratio=0.4, alpha=0.85)
    else:
        wind_quiver[0] = None

    # PWM bars
    pwm_now = pwm_frames[fi]
    for bar, val in zip(pwm_bars, pwm_now):
        bar.set_width(val)
        bar.set_alpha(0.6 + min(0.35, abs(val - P.PWM_HOVER * 100) / 30))

    # HUD
    pos_err_3d = np.sqrt((pn_s[idx]-pn_r)**2 + (pe_s[idx]-pe_r)**2 +
                         (alt_s[idx] - (-pd_r))**2)
    w_pct = w_mag / (P.MASS * P.GRAVITY) * 100.
    hud_lines = [
        "── POSITION ──────────────",
        f"  N  {pn_s[idx]:+6.3f} m   ref {pn_r:+5.2f}",
        f"  E  {pe_s[idx]:+6.3f} m   ref {pe_r:+5.2f}",
        f"  Alt {alt_s[idx]:5.3f} m   ref {-pd_r:5.2f}",
        f"  3D err  {pos_err_3d:.4f} m",
        "",
        "── ATTITUDE ──────────────",
        f"  φ  {np.degrees(phi_s[idx]):+6.2f}°",
        f"  θ  {np.degrees(theta_s[idx]):+6.2f}°",
        f"  ψ  {np.degrees(psi_s[idx]):+6.2f}°  ref {np.degrees(psi_r):+5.1f}°",
        "",
        "── DYNAMICS ──────────────",
        f"  Speed   {speed:.3f} m/s",
        f"  p {np.degrees(p_s[idx]):+5.1f}  q {np.degrees(q_s[idx]):+5.1f}"
        f"  r {np.degrees(r_s[idx]):+5.1f} °/s",
        f"  F_cmd   {Fc_h[idx]:.4f} N",
        "",
        "── WIND ──────────────────",
        f"  |F|  {w_mag*1e3:5.1f} mN  ({w_pct:.1f}% wt)",
        f"  N {F_wind[0]*1e3:+5.1f}  E {F_wind[1]*1e3:+5.1f}"
        f"  D {F_wind[2]*1e3:+5.1f} mN",
    ]
    hud_text.set_text("\n".join(hud_lines))
    time_txt.set_text(f"t = {tc:5.2f} s  /  {T_END:.0f} s")
    phase_txt.set_text(_phase_label(tc))

    return (path_line, arm1_line, arm2_line, motors_plt,
            target_pt, hud_text, time_txt, phase_txt)

anim = FuncAnimation(fig_a, update_anim,
                     frames=len(frame_idx_a),
                     init_func=init_anim,
                     blit=False,
                     interval=1000 / anim_fps)
plt.show()

# Uncomment to save:
# from matplotlib.animation import PillowWriter
# anim.save("figures/flight_animation.gif", writer=PillowWriter(fps=anim_fps))
# print("Saved figures/flight_animation.gif")