"""
sim.py  —  Crazyflie 2.1 closed-loop simulation, NED frame.

State (23,):
  [0:3]   pn, pe, pd           NED position [m]
  [3:6]   u, v, w              body velocity [m/s]
  [6:9]   phi, theta, psi      ZYX Euler angles [rad]
  [9:12]  p, q, r              body angular rates [rad/s]
  [12:16] pwm1..4              motor actuator states [0,1]
  [16]    e_int_pd             ∫(pd - pd_ref) dt
  [17]    e_int_fwd            ∫(body-forward pos error) dt
  [18]    e_int_rgt            ∫(body-right   pos error) dt
  [19]    pad
  [20:23] wind filter states   shaped turbulence force in NED [N]

Outputs saved to sim_out.npz (loaded by postprocess.py).
Wind can be disabled with WIND_INTENSITY = 0.0.
"""

import os, time
import numpy as np
from scipy.integrate import solve_ivp
import subprocess

import params as P
import lqr_control as lqr
from dynamics import f_continuous, R_body2ned
from mixer import wrench_to_pwm
from wind import WindModel, apply_wind_to_xdot

os.makedirs("figures", exist_ok=True)

# ── Wind configuration ────────────────────────────────────────────────────────
WIND_INTENSITY = 50.0    # 0 = off, 0.5 = gentle, 1.0 = moderate, 2.0 = strong
wind = WindModel(intensity=1.0)


###############################################################################
# Reference trajectory — smoothed cubic ramps
# (step inputs cause transients that saturate actuators)

def _ramp(t, t0, dur, v0, v1):
    """Smooth cubic (smoothstep) ramp, zero velocity at both ends."""
    if t <= t0:        return v0
    if t >= t0 + dur:  return v1
    s = (t - t0) / dur
    s = s * s * (3. - 2. * s)
    return v0 + s * (v1 - v0)

def reference(t):
    """Returns (pd_ref, pn_ref, pe_ref, psi_ref) — NED [m, m, m, rad]."""
    pd_ref  = _ramp(t,  0.,  0.5,  0., -3.)
    pn_ref  = _ramp(t,  8.,  0.5,  0.,  3.)
    pe_ref  = _ramp(t, 16.,  0.5,  0.,  3.)
    psi_ref = _ramp(t, 16.,  2.,   0.,  np.pi / 2.)
    return pd_ref, pn_ref, pe_ref, psi_ref


###############################################################################
# Helpers

def _wrap(err):
    """Wrap angle error to (−π, π]."""
    return (err + np.pi) % (2 * np.pi) - np.pi

def _yaw_rot2d(psi):
    """NED horizontal → body-forward/right (yaw rotation only)."""
    cy, sy = np.cos(psi), np.sin(psi)
    return np.array([[ cy, sy],
                     [-sy, cy]])


###############################################################################
# Controller

INT_PD_LIM  = 2.0   # anti-windup clamp for altitude integrator  [m·s]
INT_POS_LIM = 5.0   # anti-windup clamp for horizontal integrators [m·s]

def compute_control(t, s):
    """
    Returns (pwm[4], F_cmd, L_cmd, M_cmd, N_cmd, saturated).

    Convention: u = −K @ e   (regulator around error state)
    Gravity feedforward: F_total = m*g + δF
    Horizontal loops rotated into yaw-aligned body frame so controllers
    remain valid at arbitrary heading (see yaw fix in previous revision).
    """
    pn, pe, pd      = s[0],  s[1],  s[2]
    u_b, v_b, w_b   = s[3],  s[4],  s[5]
    phi, theta, psi = s[6],  s[7],  s[8]
    p, q, r         = s[9],  s[10], s[11]
    e_int_pd  = np.clip(s[16], -INT_PD_LIM,  INT_PD_LIM)
    e_int_fwd = np.clip(s[17], -INT_POS_LIM, INT_POS_LIM)
    e_int_rgt = np.clip(s[18], -INT_POS_LIM, INT_POS_LIM)

    pd_ref, pn_ref, pe_ref, psi_ref = reference(t)

    # Altitude
    e_z   = np.array([pd - pd_ref, w_b, e_int_pd])
    dF    = float(-(lqr.K_z @ e_z).ravel()[0])
    F_cmd = P.MASS * P.GRAVITY + dF

    # Rotate NED position error into yaw-aligned body frame
    R2 = _yaw_rot2d(psi)
    e_fwd, e_rgt = R2 @ np.array([pn - pn_ref, pe - pe_ref])

    # Pitch (body-forward / North-aligned)
    e_pn  = np.array([e_fwd, u_b, theta, q, e_int_fwd])
    M_cmd = float(-(lqr.K_pn @ e_pn).ravel()[0])

    # Roll (body-right / East-aligned)
    e_pe  = np.array([e_rgt, v_b, phi, p, e_int_rgt])
    L_cmd = float(-(lqr.K_pe @ e_pe).ravel()[0])

    # Yaw
    e_yaw = np.array([_wrap(psi - psi_ref), r])
    N_cmd = float(-(lqr.K_yaw @ e_yaw).ravel()[0])

    pwm       = wrench_to_pwm(F_cmd, L_cmd, M_cmd, N_cmd)
    saturated = np.any(pwm >= 0.999) or np.any(pwm <= 0.001)

    return pwm, F_cmd, L_cmd, M_cmd, N_cmd, saturated


###############################################################################
# ODE  (state length = 23)

def ode(t, s):
    pwm_cmd, _, _, _, _, sat = compute_control(t, s)

    # Wind disturbance
    w_filt           = s[20:23]
    F_ned, M_body, w_dot = wind.step(t, w_filt)

    # Base dynamics
    xdot = f_continuous(t, s[:16], pwm_cmd)

    # Inject wind
    omega  = s[9:12]
    R_b2n  = R_body2ned(s[6], s[7], s[8])
    xdot   = apply_wind_to_xdot(xdot, F_ned, M_body, R_b2n, P.MASS, P.J, omega)

    # Integrator derivatives (yaw-rotated body-frame position errors)
    pd_ref, pn_ref, pe_ref, _ = reference(t)
    psi = s[8]
    pos_err_ned           = np.array([s[0] - pn_ref, s[1] - pe_ref])
    ei_fwd_dot, ei_rgt_dot = _yaw_rot2d(psi) @ pos_err_ned
    ei_pd_dot             = s[2] - pd_ref

    # Anti-windup: freeze integrator if saturated and error grows in same direction
    if sat:
        if abs(s[16]) >= INT_PD_LIM  and np.sign(s[16]) == np.sign(ei_pd_dot):
            ei_pd_dot  = 0.
        if abs(s[17]) >= INT_POS_LIM and np.sign(s[17]) == np.sign(ei_fwd_dot):
            ei_fwd_dot = 0.
        if abs(s[18]) >= INT_POS_LIM and np.sign(s[18]) == np.sign(ei_rgt_dot):
            ei_rgt_dot = 0.

    return np.concatenate([xdot,
                           [ei_pd_dot, ei_fwd_dot, ei_rgt_dot, 0.],
                           w_dot])


###############################################################################
# Run

T_END  = 30.
x_init = np.zeros(23)
x_init[12:16] = P.PWM_HOVER          # motors at hover trim
x_init[20:23] = wind.initial_state() # wind filter at zero

print(f"Running simulation ...")
t0_wall = time.time()
sol = solve_ivp(ode, [0., T_END], x_init,
                method="RK45", max_step=0.005, rtol=1e-7, atol=1e-9)
print(f"  Done — {len(sol.t)} steps in {time.time()-t0_wall:.1f}s  |  {sol.message}")

t  = sol.t
xs = sol.y.T       # (N, 23)


###############################################################################
# Save outputs for postprocess.py

# Re-evaluate control history (needed for plots / HUD)
N = len(t)
ctrl_hist = np.zeros((N, 5))   # [F, L, M, N, sat]
for i in range(N):
    _, Fc, Lc, Mc, Nc, sat = compute_control(t[i], xs[i])
    ctrl_hist[i] = [Fc, Lc, Mc, Nc, float(sat)]

# Wind force history
wind_hist = np.zeros((N, 3))
_w = wind.initial_state()
for i in range(N):
    F_ned, _, _ = wind.step(t[i], xs[i, 20:23])
    wind_hist[i] = F_ned

# Reference history
ref_hist = np.array([reference(ti) for ti in t])   # (N, 4): pd, pn, pe, psi

np.savez("sim_out.npz",
         t=t, xs=xs,
         ctrl_hist=ctrl_hist,
         wind_hist=wind_hist,
         ref_hist=ref_hist,
         wind_intensity=np.array([WIND_INTENSITY]),
         T_END=np.array([T_END]))

print("Saved sim_out.npz")


################################
################################
# run postprocess.py as a subprocess
result = subprocess.run(["uv", "run", "postprocess.py"], capture_output=True, text=True)
print("Output:\n", result.stdout)
if result.returncode != 0:
    print("Error:\n", result.stderr)
