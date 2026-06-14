"""
wind.py  —  Stochastic wind disturbance model for Crazyflie 2.1.

Architecture
────────────
Turbulence forces (NED, 3 states):
    Dryden-style first-order shaping filter driven by white noise.
    The filter pole sets the gust bandwidth:  τ_gust ~ 0.3 s  →  ~0.5 Hz corner.
    Force magnitude scaled so peak gusts are O(10–30 % of drone weight).

Turbulence moments (body, no extra states):
    Unfiltered white noise scaled by σ_moment.  Rotational disturbances on a
    33 g frame are high-frequency (prop gyro damps low-freq roll/pitch quickly),
    so pure white noise is a reasonable approximation here.

ODE integration:
    The 3 filter states are appended to the sim state vector at indices [20:23].
    f_wind() returns the state derivative contribution (3,).
    wind_wrench() returns (F_ned[3], M_body[3]) for injection into f_continuous.

Usage in sim.py:
    # state layout: [x_dyn(20), w_filt(3)]   total = 23
    from wind import WindModel
    wind = WindModel(intensity=1.0, seed=42)

    def ode(t, s):
        w_filt  = s[20:23]
        F_ned, M_body, w_dot = wind.step(t, w_filt)
        xdot = f_continuous_with_wind(t, s[:16], pwm_cmd, F_ned, M_body)
        ...
        return np.concatenate([xdot, integrator_dots, [0.], w_dot])

Intensity parameter
───────────────────
  intensity = 0.0   →  no wind
  intensity = 0.5   →  gentle breeze  (peak gust ~15 % of weight)
  intensity = 1.0   →  moderate gusts (peak gust ~30 % of weight)
  intensity = 2.0   →  strong gusts   (peak gust ~60 % of weight)
"""

import numpy as np

# ── Physical reference ────────────────────────────────────────────────────────
_WEIGHT     = 0.033 * 9.81        # N  ≈ 0.324 N
_ARM        = 0.046                # m

# ── Turbulence shaping filter ─────────────────────────────────────────────────
# First-order: F_dot = -F/τ + σ/τ * w(t)
# Steady-state std = σ, bandwidth = 1/(2π τ) Hz
_TAU_GUST   = 1.0               # s  → ~0.5 Hz corner frequency
_A_FILT     = -1.0 / _TAU_GUST    # filter pole

# ── Moment arm for turbulence moment ─────────────────────────────────────────
# Scales moment noise relative to force noise so attitudes stay small
_MOMENT_ARM = _ARM * 0.5           # effective lever arm for random moments


class WindModel:
    """
    Stochastic wind disturbance model.

    Parameters
    ----------
    intensity : float
        Overall wind strength scalar (see module docstring).
    seed : int | None
        RNG seed for reproducibility.  None = random.
    dt_noise : float
        Internal noise sample interval [s].  Must be << ODE max_step.
        Default 0.002 s (500 Hz) is well above the 200 Hz ODE limit.
    """

    def __init__(self, intensity: float = 0.1,
                 seed: int | None = None,
                 dt_noise: float = 0.002):

        self.intensity  = float(intensity)
        self.rng        = np.random.default_rng(seed)
        self.dt_noise   = dt_noise

        # Noise std levels at intensity = 1.0
        self._sigma_F   = 0.10 * _WEIGHT   # per-axis force std  [N]
        self._sigma_M   = self._sigma_F * _MOMENT_ARM  # moment std  [N·m]

        # Cache for piecewise-constant noise (updated every dt_noise)
        self._t_noise    : float       = -1.0
        self._w_cache    : np.ndarray  = np.zeros(3)  # turbulence drive
        self._m_cache    : np.ndarray  = np.zeros(3)  # moment noise

    # ── Public API ────────────────────────────────────────────────────────────

    def n_states(self) -> int:
        """Number of ODE filter states contributed by wind (append to sim state)."""
        return 3

    def initial_state(self) -> np.ndarray:
        """Zero-initialised filter states."""
        return np.zeros(3)

    def step(self, t: float, w_filt: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate wind disturbance at time t.

        Parameters
        ----------
        t      : current simulation time [s]
        w_filt : current filter state (3,) — turbulence force in NED [N]

        Returns
        -------
        F_ned   : (3,) total wind force in NED frame [N]
        M_body  : (3,) total wind moment in body frame [N·m]
        w_dot   : (3,) filter state derivative (for ODE)
        """
        if self.intensity == 0.0:
            return np.zeros(3), np.zeros(3), np.zeros(3)

        sc = self.intensity

        # Refresh white-noise samples on a fixed sub-grid to give the ODE
        # solver a piecewise-constant (but dense) noise signal.  This avoids
        # noise that varies inside a single RK45 step (which would make the
        # adaptive solver shrink dt enormously trying to track it).
        if t >= self._t_noise + self.dt_noise:
            self._t_noise  = t
            self._w_cache  = self.rng.standard_normal(3) * self._sigma_F * sc
            self._m_cache  = self.rng.standard_normal(3) * self._sigma_M * sc

        # ── Turbulence shaping filter ─────────────────────────────────────────
        # w_dot = A_filt * w_filt + (sigma/tau) * white_noise
        # Steady-state gives w_filt ~ N(0, sigma^2)
        drive  = self._w_cache / _TAU_GUST
        w_dot  = _A_FILT * w_filt + drive     # (3,) filter derivative

        F_ned  = w_filt.copy()            # filtered turbulence force [N]
        M_body = self._m_cache.copy()     # unfiltered moment [N·m]

        return F_ned, np.zeros(3), w_dot


# ── Injection helper for dynamics ─────────────────────────────────────────────

def apply_wind_to_xdot(xdot: np.ndarray,
                        F_ned: np.ndarray,
                        M_body: np.ndarray,
                        R_b2n: np.ndarray,
                        mass: float,
                        J: np.ndarray,
                        omega: np.ndarray) -> np.ndarray:
    """
    Add wind force and moment contributions to an existing xdot vector.

    Parameters
    ----------
    xdot   : (16,) state derivative from f_continuous (modified in-place copy)
    F_ned  : (3,)  wind force in NED frame [N]
    M_body : (3,)  wind moment in body frame [N·m]
    R_b2n  : (3,3) body-to-NED rotation matrix (from R_body2ned)
    mass   : drone mass [kg]
    J      : (3,3) inertia matrix [kg·m²]
    omega  : (3,)  body angular rate [rad/s]

    Returns
    -------
    xdot_wind : (16,) modified state derivative
    """
    xdot = xdot.copy()

    # Rotate NED wind force to body frame, add to body velocity derivatives
    F_body_wind = R_b2n.T @ F_ned       # [N] in body frame
    xdot[3:6]  += F_body_wind / mass    # velocity derivatives [m/s²]

    # Add wind moment to angular acceleration.
    # Cross-coupling term (ω × Jω) is already accounted for in f_continuous,
    # so we only add the direct moment contribution here.
    xdot[9:12] += np.linalg.solve(J, M_body)

    return xdot


if __name__ == "__main__":
    # Quick sanity plot
    import matplotlib.pyplot as plt

    dt   = 0.002
    T    = 15.0
    ts   = np.arange(0, T, dt)
    wm   = WindModel(intensity=1.0, seed=0)
    W    = np.zeros(3)
    Fs   = np.zeros((len(ts), 3))
    Ms   = np.zeros((len(ts), 3))

    for i, ti in enumerate(ts):
        F, M, Wdot = wm.step(ti, W)
        Fs[i] = F; Ms[i] = M
        W = W + Wdot * dt

    weight = 0.033 * 9.81
    fig, axs = plt.subplots(2, 1, figsize=(11, 6))
    fig.suptitle("Wind model preview  (intensity=1.0)", fontsize=12)
    for j, (lbl, col) in enumerate([("N","C0"),("E","C1"),("D","C2")]):
        axs[0].plot(ts, Fs[:,j]/weight*100, col, lw=1, label=f"F_{lbl}")
    axs[0].set(ylabel="% of weight", title="Turbulence force (NED)")
    axs[0].axhline(0, color="k", lw=0.7, ls="--"); axs[0].legend(); axs[0].grid(True)
    for j, (lbl, col) in enumerate([("x","C0"),("y","C1"),("z","C2")]):
        axs[1].plot(ts, np.degrees(Ms[:,j]/29e-6), col, lw=1, label=f"M_{lbl}")
    axs[1].set(xlabel="Time (s)", ylabel="≈ angular accel (°/s²)",
               title="Random moment (body)"); axs[1].legend(); axs[1].grid(True)
    fig.tight_layout(); plt.show()