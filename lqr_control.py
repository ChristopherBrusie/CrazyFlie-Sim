"""
lqr_control.py  —  NED LQR gains for Crazyflie 2.1.

REGULATOR CONVENTION
  All systems designed as  xdot = A*x + B*u,  u = -K*x  (x = ERROR state)
  Integrator row: e_int_dot = +e   ->   A[int, e] = +1

CONTROLLER CALL  (in sim.py, each axis):
  e_z   = [pd-pd_ref,  w,        e_int_pd]   dF      = -(K_z   @ e_z)
  e_pn  = [pn-pn_ref,  u, theta, q, e_int_pn]  M_pitch = -(K_pn  @ e_pn)
  e_pe  = [pe-pe_ref,  v, phi,   p, e_int_pe]  L_roll  = -(K_pe  @ e_pe)
  e_yaw = [r]                                   N_yaw   = -(K_yaw @ e_yaw)
  F_total = m*g + dF   (gravity feedforward)

GAINS TUNED for:
  - No reference shaping needed: the simulation uses a ramp reference
  - Overdamped response (zeta > 1) to avoid overshoot on step commands
  - Bandwidth well below actuator bandwidth (30 ms motor lag)
"""

import numpy as np
import control

from params import MASS, GRAVITY, J, K_AERO, PWM_HOVER, RPM_C0, RPM_C1, RPM_C2, RPM_SCALE

# ── Aero drag at hover ───────────────────────────────────────────────────────
_n_h     = (RPM_C0 + PWM_HOVER*RPM_C1 - RPM_C2*PWM_HOVER**2) / RPM_SCALE
_sigma_h = 4 * 2*np.pi * abs(_n_h)

K_DZ = K_AERO[2, 2] * _sigma_h / MASS    # ≈ -0.221 s^-1
K_DU = K_AERO[0, 0] * _sigma_h / MASS    # ≈ -0.300 s^-1

# ── Altitude ─────────────────────────────────────────────────────────────────
# error state: [e_pd, w, e_int_pd]
# e_pd_dot  = w
# w_dot     = K_DZ*w  + (-1/m)*dF
# e_int_dot = e_pd
A_z = np.array([[0.,    1.,   0.],
                 [0.,  K_DZ,   0.],
                 [1.,    0.,   0.]], float)
B_z = np.array([[0.], [-1./MASS], [0.]], float)

# Conservative: overdamped, ~0.5 Hz bandwidth
Q_z = np.diag([80., 15., 3.])
R_z = np.array([[5e4]])
K_z, _, E_z = control.lqr(A_z, B_z, Q_z, R_z)

# ── North / Pitch ─────────────────────────────────────────────────────────────
# error state: [e_pn, u, theta, q, e_int_pn]
A_pn = np.array([[0.,    1.,        0., 0., 0.],
                  [0.,  K_DU, -GRAVITY,  0., 0.],
                  [0.,    0.,        0., 1., 0.],
                  [0.,    0.,        0., 0., 0.],
                  [1.,    0.,        0., 0., 0.]], float)
B_pn = np.array([[0.], [0.], [0.], [1./J[1,1]], [0.]], float)

Q_pn = np.diag([50., 20., 800., 20., 1.])
R_pn = np.array([[3e5]])
K_pn, _, E_pn = control.lqr(A_pn, B_pn, Q_pn, R_pn)

# ── East / Roll ───────────────────────────────────────────────────────────────
# error state: [e_pe, v, phi, p, e_int_pe]
A_pe = np.array([[0.,    1.,       0., 0., 0.],
                  [0.,  K_DU, GRAVITY,  0., 0.],
                  [0.,    0.,       0., 1., 0.],
                  [0.,    0.,       0., 0., 0.],
                  [1.,    0.,       0., 0., 0.]], float)
B_pe = np.array([[0.], [0.], [0.], [1./J[0,0]], [0.]], float)

Q_pe = np.diag([50., 20., 500., 20., 1.])
R_pe = np.array([[3e5]])
K_pe, _, E_pe = control.lqr(A_pe, B_pe, Q_pe, R_pe)

# ── Yaw rate ─────────────────────────────────────────────────────────────────
A_yaw = np.array([[0.]])
B_yaw = np.array([[1./J[2,2]]])
Q_yaw = np.array([[300.]])
R_yaw = np.array([[1.]])
K_yaw, _, E_yaw = control.lqr(A_yaw, B_yaw, Q_yaw, R_yaw)

A_yaw = np.array([[0., 1.],
                  [0., 0.]], float)
B_yaw = np.array([[0.],
                  [1./J[2,2]]], float)
Q_yaw = np.diag([200., 30.])
R_yaw = np.array([[1.]])
K_yaw, _, E_yaw = control.lqr(A_yaw, B_yaw, Q_yaw, R_yaw)

if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    print(f"K_DZ={K_DZ:.4f},  K_DU={K_DU:.4f}")
    for name, K, E in [("K_z", K_z, E_z), ("K_pn", K_pn, E_pn),
                        ("K_pe", K_pe, E_pe), ("K_yaw", K_yaw, E_yaw)]:
        print(f"\n{name} = {K.ravel()}")
        print(f"  eigs: {E}")