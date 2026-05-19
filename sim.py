"""
sim.py  —  Crazyflie 2.1 closed-loop simulation, NED frame.

State (20,):
  [0:3]   pn, pe, pd           NED position [m]
  [3:6]   u, v, w              body velocity [m/s]
  [6:9]   phi, theta, psi      ZYX Euler angles [rad]
  [9:12]  p, q, r              body angular rates [rad/s]
  [12:16] pwm1..4              motor actuator states [0,1]
  [16]    e_int_pd = integral(pd - pd_ref) dt
  [17]    e_int_pn = integral(pn - pn_ref) dt
  [18]    e_int_pe = integral(pe - pe_ref) dt
  [19]    pad

Reference trajectory (smooth ramp — avoids step-command overshoot):
  Altitude:  0 -> 1 m  (pd: 0 -> -1 m) over 3 s, starting t=0
  North:     0 -> 2 m  over 3 s, starting t=8 s
  East:      0 -> 1 m  over 3 s, starting t=18 s

Anti-windup: integrator clamped when actuators saturate.
"""

import os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib.animation import FuncAnimation, PillowWriter

import params as P
import lqr_control as lqr
from dynamics import f_continuous
from mixer import wrench_to_pwm

os.makedirs("figures", exist_ok=True)


###################################################
# Reference trajectory generation - with smoothed cubic ramps
def _ramp(t, t0, dur, v0, v1):
    """Smooth cubic (smoothstep) ramp, zero velocity at both ends."""
    if t <= t0:       return v0
    if t >= t0 + dur: return v1
    s = (t - t0) / dur
    s = s * s * (3. - 2. * s)
    return v0 + s * (v1 - v0)

def reference(t):
    """Returns (pd_ref, pn_ref, pe_ref, psi_ref) — NED [m, m, m, rad]."""
    pd_ref  = _ramp(t,  0., 0.5,  0.,  -1.)
    pn_ref  = _ramp(t,  8., 0.5,  0.,   2.)
    pe_ref  = _ramp(t, 18., 0.5,  0.,   1.)
    psi_ref = _ramp(t, 24., 2.,  0., np.pi/2.)
    return pd_ref, pn_ref, pe_ref, psi_ref


# def reference(t):
#     pd_ref = -1.0 if t > 0. else 0.0
#     pn_ref = 2.0 if t > 8. else 0.0
#     pe_ref = 1.0 if t > 18. else 0.0
#     return pd_ref, pn_ref, pe_ref

# yaw helper - wraps error to +/- pi
def _wrap(err):
    return (err + np.pi) % (2 * np.pi) - np.pi




##############################################################
# Controller

# Anti-windup limits for integrators (m*s)
INT_PD_LIM  = 2.0
INT_POS_LIM = 5.0

def compute_control(t, s):
    """
    Returns (pwm[4], F_total, L_roll, M_pitch, N_yaw, saturated).

    u = -K @ error_state
    Gravity feedforward applied to altitude (hover thrust + control)
    """
    pn, pe, pd     = s[0],  s[1],  s[2]
    u_b, v_b, w_b  = s[3],  s[4],  s[5]
    phi, theta, psi= s[6],  s[7], s[8]
    p, q, r        = s[9],  s[10], s[11]
    e_int_pd       = np.clip(s[16], -INT_PD_LIM,  INT_PD_LIM)
    e_int_pn       = np.clip(s[17], -INT_POS_LIM, INT_POS_LIM)
    e_int_pe       = np.clip(s[18], -INT_POS_LIM, INT_POS_LIM)

    pd_ref, pn_ref, pe_ref, psi_ref = reference(t)

    # Altitude error = actual - ref
    e_z  = np.array([pd - pd_ref, w_b, e_int_pd])
    dF   = float(-(lqr.K_z  @ e_z).ravel()[0])
    F_cmd = P.MASS * P.GRAVITY + dF

    # North/X
    e_pn = np.array([pn - pn_ref, u_b, theta, q, e_int_pn])
    M_cmd = float(-(lqr.K_pn @ e_pn).ravel()[0])

    # East/Y
    e_pe = np.array([pe - pe_ref, v_b, phi, p, e_int_pe])
    L_cmd = float(-(lqr.K_pe @ e_pe).ravel()[0])

    # Yaw full tracking now
    e_yaw = np.array([_wrap(psi - psi_ref), r])
    N_cmd = float(-(lqr.K_yaw @ e_yaw).ravel()[0])



    pwm = wrench_to_pwm(F_cmd, L_cmd, M_cmd, N_cmd)

    # Detect saturation (for anti windup in integrator update)
    saturated = np.any(pwm >= 0.999) or np.any(pwm <= 0.001)

    return pwm, F_cmd, L_cmd, M_cmd, N_cmd, saturated


###############################################################
# ODE
def ode(t, s):
    pwm_cmd, _, _, _, _, sat = compute_control(t, s)
    xdot = f_continuous(t, s[:16], pwm_cmd)

    pd_ref, pn_ref, pe_ref, _ = reference(t)

    # Integrator update: e_int_dot = (actual - ref)
    # NOTE freeze integrator if saturated AND error pushes further into sat
    ei_pd_dot = s[2] - pd_ref
    ei_pn_dot = s[0] - pn_ref
    ei_pe_dot = s[1] - pe_ref

    if sat:
        # Clamp don't let integrator grow in winding-up direction
        if abs(s[16]) >= INT_PD_LIM  and np.sign(s[16]) == np.sign(ei_pd_dot):
            ei_pd_dot = 0.
        if abs(s[17]) >= INT_POS_LIM and np.sign(s[17]) == np.sign(ei_pn_dot):
            ei_pn_dot = 0.
        if abs(s[18]) >= INT_POS_LIM and np.sign(s[18]) == np.sign(ei_pe_dot):
            ei_pe_dot = 0.

    return np.concatenate([xdot, [ei_pd_dot, ei_pn_dot, ei_pe_dot, 0.]])



#########################################################################
# run simulation / solve ode
T_END  = 30.
x_init = np.zeros(20)
x_init[12:16] = P.PWM_HOVER

print("Running closed-loop simulation …")
t0 = time.time()
sol = solve_ivp(ode, [0., T_END], x_init, method="RK45",
                max_step=0.005, rtol=1e-7, atol=1e-9)
print(f"  Done — {len(sol.t)} steps in {time.time()-t0:.1f}s  |  {sol.message}")

t  = sol.t
xs = sol.y.T




######################################
##############################################
## Post Processing only


pn_s    = xs[:,0];  pe_s   = xs[:,1];  pd_s   = xs[:,2]
u_s     = xs[:,3];  v_s    = xs[:,4];  w_s    = xs[:,5]
phi_s   = xs[:,6];  theta_s= xs[:,7];  psi_s  = xs[:,8]
p_s     = xs[:,9];  q_s    = xs[:,10]; r_s    = xs[:,11]
pwm_act = xs[:,12:16]
alt_s   = -pd_s

N = len(t)
Fc_h = np.zeros(N); Lc_h = np.zeros(N); Mc_h = np.zeros(N); Nc_h = np.zeros(N)
for i in range(N):
    _, Fc, Lc, Mc, Nc, _ = compute_control(t[i], xs[i])
    Fc_h[i]=Fc; Lc_h[i]=Lc; Mc_h[i]=Mc; Nc_h[i]=Nc

pd_ref_a   = np.array([reference(ti)[0] for ti in t])
pn_ref_a   = np.array([reference(ti)[1] for ti in t])
pe_ref_a   = np.array([reference(ti)[2] for ti in t])
psi_ref_a  = np.array([reference(ti)[3] for ti in t])
alt_ref_a  = -pd_ref_a

e_alt = alt_s         - alt_ref_a
e_pn  = pn_s          - pn_ref_a
e_pe  = pe_s          - pe_ref_a
e_psi = np.array([_wrap(psi_s[i] - psi_ref_a[i]) for i in range(N)])

ei_pd_a = xs[:,16]; ei_pn_a = xs[:,17]; ei_pe_a = xs[:,18]
dF_a    = Fc_h - P.MASS * P.GRAVITY

pwm_pct    = pwm_act * 100.
pwm_mean_a = pwm_pct.mean(axis=1)
pwm_sprd_a = pwm_pct.max(axis=1) - pwm_pct.min(axis=1)


####################################################
# Fig 1 position tracking

fig1, axs = plt.subplots(3, 2, figsize=(14, 10))
fig1.suptitle("Position Tracking — NED LQR (Crazyflie 2.1)", fontsize=14, fontweight="bold")
pairs = [(alt_s, alt_ref_a, e_alt, "Altitude (m↑)",  "Altitude"),
         (pn_s,  pn_ref_a,  e_pn,  "North pn (m)",   "North"),
         (pe_s,  pe_ref_a,  e_pe,  "East pe (m)",    "East")]
for row,(act,ref,err,yl,tl) in enumerate(pairs):
    ax=axs[row,0]; ax.plot(t,act,"C0",lw=2,label="Actual"); ax.plot(t,ref,"C3--",lw=1.5,label="Ref")
    ax.set(ylabel=yl,title=tl); ax.legend(fontsize=8); ax.grid(True)
    ax=axs[row,1]; ax.plot(t,err,"C3",lw=2); ax.axhline(0,color="k",ls="--",lw=0.8)
    ax.fill_between(t,err,0,alpha=0.12,color="C3"); ax.set(ylabel="Error (m)",title=tl+" Error"); ax.grid(True)
axs[2,0].set_xlabel("Time (s)"); axs[2,1].set_xlabel("Time (s)")
fig1.tight_layout(); fig1.savefig("figures/fig_position.png", dpi=150)
print("Saved figures/fig_position.png")


########################################################
# Fig 2 Attitude
fig2, axs2 = plt.subplots(2, 3, figsize=(14, 8))
fig2.suptitle("Attitude & Body Rates", fontsize=13, fontweight="bold")
att_sigs  = [phi_s, theta_s, psi_s]
att_refs  = [None,  None,  psi_ref_a]
att_lbls  = ["Roll φ (°)", "Pitch θ (°)", "Yaw ψ (°)"]
rate_sigs = [p_s, q_s, r_s]
rate_lbls = ["Roll rate p (°/s)", "Pitch rate q (°/s)", "Yaw rate r (°/s)"]
cols      = ["C2", "C4", "C5"]
for i in range(3):
    ax = axs2[0,i]
    ax.plot(t, np.degrees(att_sigs[i]), cols[i], lw=2, label="Actual")
    if att_refs[i] is not None:
        ax.plot(t, np.degrees(att_refs[i]), "k--", lw=1.5, label="Ref")
        ax.legend(fontsize=8)
    ax.axhline(0, color="k", ls="--", lw=0.8); ax.set(title=att_lbls[i]); ax.grid(True)
    axs2[1,i].plot(t, np.degrees(rate_sigs[i]), cols[i], lw=2)
    axs2[1,i].axhline(0, color="k", ls="--", lw=0.8)
    axs2[1,i].set(xlabel="Time (s)", title=rate_lbls[i]); axs2[1,i].grid(True)
fig2.tight_layout(); fig2.savefig("figures/fig_attitude.png", dpi=150)
print("Saved figures/fig_attitude.png")


######################################################
# Fig 3 motors
fig3, axs3 = plt.subplots(2,1, figsize=(12,7))
fig3.suptitle("Motor Actuator States (PWM)", fontsize=13, fontweight="bold")
mcols = ["#e41a1c","#377eb8","#4daf4a","#984ea3"]
for i in range(4):
    axs3[0].plot(t, pwm_pct[:,i], color=mcols[i], lw=1.5, label=f"M{i+1}")
axs3[0].axhline(P.PWM_HOVER*100, color="k", ls="--", lw=1, label=f"Hover ({P.PWM_HOVER*100:.1f}%)")
axs3[0].set(ylabel="PWM (%)", title="Per-motor"); axs3[0].legend(); axs3[0].grid(True)
axs3[1].plot(t, pwm_mean_a, "k", lw=2, label="Mean")
axs3[1].fill_between(t, pwm_mean_a-pwm_sprd_a/2, pwm_mean_a+pwm_sprd_a/2,
                     alpha=0.25, color="steelblue", label="Spread (attitude effort)")
axs3[1].axhline(P.PWM_HOVER*100, color="k", ls="--", lw=1)
axs3[1].set(xlabel="Time (s)", ylabel="PWM (%)", title="Mean ± Spread")
axs3[1].legend(); axs3[1].grid(True)
fig3.tight_layout(); fig3.savefig("figures/fig_motors.png", dpi=150)
print("Saved figures/fig_motors.png")


###############################################
# Fig 4 dashboard with extra useful plots
fig4 = plt.figure(figsize=(20,15))
fig4.suptitle("Full Controller Diagnostic Dashboard — NED LQR (Crazyflie 2.1)",
              fontsize=14, fontweight="bold")
gs = GridSpec(4, 4, figure=fig4, hspace=0.44, wspace=0.35)
def _ax(r,c): return fig4.add_subplot(gs[r,c])
def _plot(ax, y, col, ref=None, title="", ylabel=""):
    ax.plot(t, y, col, lw=2)
    if ref is not None: ax.plot(t, ref, "k--", lw=1.5)
    ax.axhline(0, color="k", ls="--", lw=0.7); ax.grid(True)
    ax.set(title=title, ylabel=ylabel)
def _fill(ax, y, col, title="", ylabel=""):
    ax.plot(t, y, col, lw=2); ax.axhline(0, color="k", ls="--", lw=0.7)
    ax.fill_between(t, y, 0, alpha=0.15, color=col); ax.grid(True)
    ax.set(title=title, ylabel=ylabel)

_plot(_ax(0,0), alt_s,                  "C0", alt_ref_a,          title="Altitude (m↑)")
_fill(_ax(0,1), e_alt,                  "C3",                     title="Alt Error (m)")
_plot(_ax(0,2), ei_pd_a,                "C1",                     title="pd Integrator")
_fill(_ax(0,3), dF_a,                   "C6",                     title="ΔThrust (N)")
_plot(_ax(1,0), pn_s,                   "C0", pn_ref_a,           title="North pn (m)")
_fill(_ax(1,1), e_pn,                   "C3",                     title="North Error (m)")
_plot(_ax(1,2), np.degrees(theta_s),    "C4",                     title="Pitch θ (°)")
_fill(_ax(1,3), Mc_h,                   "C4",                     title="Pitch Moment (N·m)")
_plot(_ax(2,0), pe_s,                   "C0", pe_ref_a,           title="East pe (m)")
_fill(_ax(2,1), e_pe,                   "C3",                     title="East Error (m)")
_plot(_ax(2,2), np.degrees(phi_s),      "C2",                     title="Roll φ (°)")
_fill(_ax(2,3), Lc_h,                   "C2",                     title="Roll Moment (N·m)")
_plot(_ax(3,0), np.degrees(psi_s),      "C5", np.degrees(psi_ref_a), title="Yaw ψ (°)")
_fill(_ax(3,1), np.degrees(e_psi),      "C5",                     title="Yaw Error (°)")
_fill(_ax(3,2), Nc_h,                   "C5",                     title="Yaw Moment (N·m)")
ax_pw = _ax(3,3)
ax_pw.plot(t, pwm_mean_a, "k", lw=2, label="Mean")
ax_pw.fill_between(t, pwm_mean_a-pwm_sprd_a/2, pwm_mean_a+pwm_sprd_a/2,
                   alpha=0.25, color="steelblue", label="Spread")
ax_pw.axhline(P.PWM_HOVER*100, color="gray", ls="--", lw=1.2)
ax_pw.set(title="Motor PWM mean±spread"); ax_pw.legend(fontsize=7); ax_pw.grid(True)
for r,lbl in enumerate(["ALTITUDE","FORWARD (pn)","LATERAL (pe)","YAW"]):
    fig4.add_subplot(gs[r,0]).set_ylabel(lbl, fontsize=10, fontweight="bold")
fig4.savefig("figures/fig_dashboard.png", dpi=150)
print("Saved figures/fig_dashboard.png")



##################################################
# fig 5 3D path history

fig5 = plt.figure(figsize=(9,7))
ax5 = fig5.add_subplot(111, projection="3d")
sc = ax5.scatter(pe_s, pn_s, alt_s, c=t, cmap="plasma", s=3)
plt.colorbar(sc, ax=ax5, label="Time (s)", shrink=0.6)
ax5.set(xlabel="East (m)", ylabel="North (m)", zlabel="Alt (m)",
        title="3-D Flight Path  [colour = time]")
fig5.tight_layout(); fig5.savefig("figures/fig_3d_path.png", dpi=150)
print("Saved figures/fig_3d_path.png")

##########################################################################
# Fig 6 Stability Margins

import control as ctrl
fig6, axs6 = plt.subplots(2, 4, figsize=(18,7))
fig6.suptitle("Open-Loop Bode — Stability Margins", fontsize=13, fontweight="bold")
def _bode(ax_m, ax_p, A, B, K, title):
    try:
        sys_ol = ctrl.ss(A, B, K, np.zeros((1, K.shape[1])))
        om = np.logspace(-2, 4, 1000)
        mag, phase, om_out = ctrl.bode(sys_ol, om, plot=False)
        mdb = 20*np.log10(np.squeeze(mag)); ph = np.degrees(np.squeeze(phase))
        ax_m.semilogx(om_out, mdb, "b", lw=2); ax_m.axhline(0,color="k",ls="--",lw=0.8)
        ax_m.grid(which="both"); ax_m.set(ylabel="Mag (dB)")
        ax_p.semilogx(om_out, ph, "r", lw=2); ax_p.axhline(-180,color="k",ls="--",lw=0.8)
        ax_p.set(xlabel="rad/s", ylabel="Phase (°)"); ax_p.grid(which="both")
        gm,pm,_,_ = ctrl.margin(sys_ol)
        gm_db = 20*np.log10(gm) if np.isfinite(gm) and gm>0 else float("inf")
        ax_m.set_title(f"{title}\nGM={gm_db:.1f}dB  PM={pm:.1f}°", fontsize=9)
    except Exception as exc:
        ax_m.text(0.1,0.5,str(exc),transform=ax_m.transAxes,fontsize=7)
_bode(axs6[0,0],axs6[1,0], lqr.A_z,   lqr.B_z,   lqr.K_z,   "Altitude (pd)")
_bode(axs6[0,1],axs6[1,1], lqr.A_pn,  lqr.B_pn,  lqr.K_pn,  "North (pn)")
_bode(axs6[0,2],axs6[1,2], lqr.A_pe,  lqr.B_pe,  lqr.K_pe,  "East (pe)")
_bode(axs6[0,3],axs6[1,3], lqr.A_yaw, lqr.B_yaw, lqr.K_yaw, "Yaw rate")
fig6.tight_layout(); fig6.savefig("figures/fig_stability_margins.png", dpi=150)
print("Saved figures/fig_stability_margins.png")


############################################################################
def _R(phi,theta,psi):
    cp,sp=np.cos(phi),np.sin(phi); ct,st=np.cos(theta),np.sin(theta); cy,sy=np.cos(psi),np.sin(psi)
    return np.array([[ct*cy,sy*sp*ct-cp*sy,cp*sy*ct+sp*sy],
                     [ct*sy,sp*sy*st+cp*cy,cp*sy*st-sp*cy],
                     [  -st,       sp*ct,         cp*ct  ]])

ARM_VIZ = P.ARM_LENGTH * 8
dg = ARM_VIZ * np.sqrt(2) / 2
mb = np.array([[ dg,-dg,-dg, dg],[ dg, dg,-dg,-dg],[0,0,0,0]])  # motor body offsets



############################################################################
# Fig 8 Animation (live window)
from matplotlib.animation import FuncAnimation, PillowWriter
matplotlib.use("QtAgg")


print("Generating 3D Animation...")

fig_anim = plt.figure(figsize=(9, 7))
ax_anim = fig_anim.add_subplot(111, projection="3d")

# Plot elements that will be updated in the animation loop
path_line, = ax_anim.plot([], [], [], 'b--', lw=1, alpha=0.4, label="Flight Path")
arm1_line, = ax_anim.plot([], [], [], 'k-', lw=3)
arm2_line, = ax_anim.plot([], [], [], 'k-', lw=3)
motors,    = ax_anim.plot([], [], [], 'o', markersize=6, color='#e41a1c')

# Target 30 FPS playback
sim_dt = np.mean(np.diff(t))
anim_fps = 30
stride = max(1, int((1.0 / anim_fps) / sim_dt))
frame_indices = list(range(0, len(t), stride))

def _set_line_3d(line, xs, ys, zs):
    if hasattr(line, "set_data_3d"):
        line.set_data_3d(xs, ys, zs)
    else:
        line.set_data(xs, ys)
        if hasattr(line, "set_3d_properties"):
            line.set_3d_properties(zs)
        else:
            raise AttributeError("Line object does not support 3D updates")


def init_anim():
    """Initialize the animation background/axes"""
    # Set static limits based on the overall flight bounds
    # pad = 0.5
    # ax_anim.set_xlim(np.min(pe_s) - pad, np.max(pe_s) + pad)
    # ax_anim.set_ylim(np.min(pn_s) - pad, np.max(pn_s) + pad)
    # ax_anim.set_zlim(0, np.max(alt_s) + pad)
    ax_anim.set_xlim(-5, 5)
    ax_anim.set_ylim(-5, 5)
    ax_anim.set_zlim(0, 5)

    ax_anim.set_xlabel("East (m)")
    ax_anim.set_ylabel("North (m)")
    ax_anim.set_zlabel("Alt (m)")
    ax_anim.set_title("Crazyflie 2.1 — 3D Flight Animation")
    ax_anim.legend(loc="upper left")

    return path_line, arm1_line, arm2_line, motors

def update_anim(frame_idx):
    """Update loop for FuncAnimation"""
    idx = frame_indices[frame_idx]

    # 1. Update trailing path
    _set_line_3d(path_line, pe_s[:idx], pn_s[:idx], alt_s[:idx])

    # 2. Compute current 3D drone geometry
    Rb = _R(phi_s[idx], theta_s[idx], psi_s[idx])
    mp_ned = Rb @ mb  # 'mb' is the motor body offsets from your snapshot code
    cg_enu = np.array([pe_s[idx], pn_s[idx], alt_s[idx]])

    # Convert NED geometry to ENU (East-North-Up) for plotting
    mp_enu = np.array([mp_ned[1], mp_ned[0], -mp_ned[2]]) + cg_enu[:, None]

    # 3. Update Cross Arms (Motors 0 to 2, Motors 1 to 3)
    _set_line_3d(arm1_line,
                 [mp_enu[0, 0], mp_enu[0, 2]],
                 [mp_enu[1, 0], mp_enu[1, 2]],
                 [mp_enu[2, 0], mp_enu[2, 2]])
    _set_line_3d(arm2_line,
                 [mp_enu[0, 1], mp_enu[0, 3]],
                 [mp_enu[1, 1], mp_enu[1, 3]],
                 [mp_enu[2, 1], mp_enu[2, 3]])

    # 4. Update Motors
    _set_line_3d(motors, mp_enu[0], mp_enu[1], mp_enu[2])

    return path_line, arm1_line, arm2_line, motors

# Create the animation
anim = FuncAnimation(
    fig_anim,
    update_anim,
    frames=len(frame_indices),
    init_func=init_anim,
    blit=False,
    interval=1000/anim_fps
)

# Option A: View interactively (ensure your QtAgg backend is running)
plt.show()

# Option B: Save as a GIF (uncomment below if you want to export it)
# gif_path = "figures/flight_animation.gif"
# anim.save(gif_path, writer=PillowWriter(fps=anim_fps))
# print(f"Saved {gif_path}")




