"""
dynamics.py — Nonlinear 6-DOF + actuator dynamics for Crazyflie 2.1 (NED frame).

State vector  x ∈ ℝ¹⁶:
  [0:3]   pn, pe, pd          NED position [m]
  [3:6]   u, v, w             Body velocity [m/s]
  [6:9]   phi, theta, psi     ZYX Euler angles [rad]
  [9:12]  p, q, r             Body angular rates [rad/s]
  [12:16] pwm1..pwm4          Motor actuator states (ratio ∈ [0,1])

Input u ∈ ℝ⁴:
  pwm_cmd1..4  commanded motor PWM ratios ∈ [0,1]

NED conventions:
  • Gravity vector in NED inertial: [0, 0, g]
  • Body z-axis points down at hover → thrust decreases pd
  • Positive pd = drone below origin
"""

import numpy as np
from params import (MASS, GRAVITY, J, K_AERO, ARM_DIAG, C_TAU,
                    A_THRUST, B_THRUST, A_ACT, B_ACT,
                    RPM_C0, RPM_C1, RPM_C2, RPM_SCALE, PWM_HOVER)


# ── Rotation utilities ──────────────────────────────────────────────────────

def R_body2ned(phi: float, theta: float, psi: float) -> np.ndarray:
    """ZYX Euler: body frame → NED inertial frame (3x3)."""
    cp, sp = np.cos(phi),   np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cy, sy = np.cos(psi),   np.sin(psi)
    # R = Rz(psi) @ Ry(theta) @ Rx(phi)
    return np.array([
        [ct*cy, sy*sp*ct - cp*sy, cp*sy*ct + sp*sy],
        [ct*sy, sp*sy*st + cp*cy, cp*sy*st - sp*cy],
        [  -st,           sp*ct,           cp*ct  ],
    ])


def W_euler(phi: float, theta: float) -> np.ndarray:
    """Euler kinematic matrix: ω_body → Euler angle rates (3x3)."""
    cp, sp = np.cos(phi), np.sin(phi)
    ct, tt = np.cos(theta), np.tan(theta)
    return np.array([
        [1., sp*tt,  cp*tt],
        [0., cp,    -sp   ],
        [0., sp/ct,  cp/ct],
    ])


# ── Motor helpers ───────────────────────────────────────────────────────────

def motor_thrust(pwm: np.ndarray) -> np.ndarray:
    """PWM ratio → thrust [N] for each motor."""
    return A_THRUST * pwm**2 + B_THRUST * pwm


def motor_rpm(pwm: float) -> float:
    """PWM ratio → rotational speed [rev/s]."""
    return (RPM_C0 + pwm*RPM_C1 - RPM_C2*pwm**2) / RPM_SCALE


# ── Nonlinear ODE (continuous-time) ─────────────────────────────────────────

def f_continuous(t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Continuous-time state derivative.

    Parameters
    ----------
    t : time [s]  (unused, kept for solve_ivp compatibility)
    x : state vector (16,)
    u : commanded PWM ratios (4,)

    Returns
    -------
    xdot : (16,)
    """
    pn, pe, pd        = x[0],  x[1],  x[2]
    u_b, v_b, w_b     = x[3],  x[4],  x[5]
    phi, theta, psi   = x[6],  x[7],  x[8]
    p,   q,   r       = x[9],  x[10], x[11]
    pwm               = x[12:16]

    # ── Actuator dynamics ──────────────────────────────────────────────────
    pwm_dot = A_ACT * pwm + B_ACT * np.clip(u, 0., 1.)

    # ── Thrust & moments ──────────────────────────────────────────────────
    F = motor_thrust(pwm)                   # per-motor force [N]
    F_total = F.sum()

    d = ARM_DIAG
    L_roll  = d * (F[2] + F[3] - F[0] - F[1])      # M3+M4 − M1−M2
    M_pitch = d * (F[0] + F[3] - F[1] - F[2])      # M1+M4 − M2−M3
    N_yaw   = C_TAU * (F[0] - F[1] + F[2] - F[3])  # CCW - CW

    # ── Aerodynamic drag (body frame) ─────────────────────────────────────
    n_motors = np.array([motor_rpm(pw) for pw in pwm])
    sigma    = 2*np.pi * np.sum(np.abs(n_motors))   # Σ|ω_i|  [rad/s]
    v_body   = np.array([u_b, v_b, w_b])
    F_aero   = K_AERO @ (sigma * v_body)             # body-frame drag [N]

    # ── Gravity in body frame ─────────────────────────────────────────────
    R_b2n    = R_body2ned(phi, theta, psi)
    g_ned    = np.array([0., 0., GRAVITY])
    g_body   = R_b2n.T @ g_ned                       # [gx, gy, gz] body

    # ── Translational dynamics ────────────────────────────────────────────
    #  F_thrust in body frame: acts along -z_body (lifts drone = opposes +pd)
    F_thrust_body = np.array([0., 0., -F_total])
    omega    = np.array([p, q, r])
    vel_body = np.array([u_b, v_b, w_b])
    vdot     = (F_thrust_body + F_aero) / MASS + g_body - np.cross(omega, vel_body) # thrust, drag, gravity, Coriolis

    # ── Translational kinematics (position in NED) ─────────────────────
    pos_dot  = R_b2n @ vel_body

    # ── Rotational kinematics ─────────────────────────────────────────────
    euler_dot = W_euler(phi, theta) @ omega

    # ── Rotational dynamics ───────────────────────────────────────────────
    moments   = np.array([L_roll, M_pitch, N_yaw])
    Jomega    = J @ omega
    omega_dot = np.linalg.solve(J, moments - np.cross(omega, Jomega))

    return np.concatenate([pos_dot, vdot, euler_dot, omega_dot, pwm_dot])


# ── Linearization about hover ────────────────────────────────────────────────

def linearize_hover(verbose: bool = False):
    """
    Numerically linearize f_continuous about hover equilibrium.

    Returns
    -------
    A : (16,16) state matrix
    B : (16,4)  input matrix
    x0 : hover state
    u0 : hover input
    """
    x0 = np.zeros(16)
    x0[12:16] = PWM_HOVER
    u0 = np.full(4, PWM_HOVER)

    eps = 1e-6
    A = np.zeros((16, 16))
    B = np.zeros((16, 4))
    f0 = f_continuous(0., x0, u0)

    for i in range(16):
        xe = x0.copy(); xe[i] += eps
        A[:, i] = (f_continuous(0., xe, u0) - f0) / eps

    for i in range(4):
        ue = u0.copy(); ue[i] += eps
        B[:, i] = (f_continuous(0., x0, ue) - f0) / eps

    if verbose:
        np.set_printoptions(precision=4, suppress=True, linewidth=130)
        print("A (hover linearization):\n", A)
        print("\nB (hover linearization):\n", B)

    return A, B, x0, u0


if __name__ == "__main__":
    A, B, x0, u0 = linearize_hover(verbose=True)
    ev = np.linalg.eigvals(A)
    print(f"\nOpen-loop eigenvalues:\n{np.sort_complex(ev)}")