"""
motor mixer. maps desired wrench -> motor PWM [0,1]

wrench: [Ftot, L, M, N]

Saturation strategy (in order):
  1. Shift all forces to keep max ≤ F_MOTOR_MAX (preserves moment ratios).
  2. Shift all forces to keep min ≥ 0 (may push max over limit temporarily).
  3. If spread > physical range, scale moments down around midpoint.
  4. Hard clip to [0, F_MOTOR_MAX].
"""

import numpy as np
from params import (A_THRUST, B_THRUST, F_MOTOR_MIN, F_MOTOR_MAX,
                    M_MIX_INV, C_TAU)


def wrench_to_pwm(F_total: float, L_roll: float, M_pitch: float, N_yaw: float
                  ) -> np.ndarray:
    """
    Parameters
    ----------
    F_total : total thrust  [N]
    L_roll  : roll moment   [N·m]  (+right wing up)
    M_pitch : pitch moment  [N·m]  (+nose up)
    N_yaw   : yaw moment    [N·m]  (+CCW from above)

    Returns
    -------
    pwm : shape (4,), motor PWM ratios in [0, 1].  Order: M1..M4.
    """
    U = np.array([F_total, L_roll, M_pitch, N_yaw])
    F = M_MIX_INV @ U          # desired force per motor [N]

    # 1. Push down if any motor exceeds max
    excess = np.max(F) - F_MOTOR_MAX
    if excess > 0:
        F -= excess

    # 2. Push up if any motor goes below zero
    deficit = F_MOTOR_MIN - np.min(F)
    if deficit > 0:
        F += deficit

    # 3. If spread is still too wide, scale moments (keep mean thrust)
    span = np.max(F) - np.min(F)
    phys_span = F_MOTOR_MAX - F_MOTOR_MIN
    if span > phys_span:
        mid = np.mean(F)
        scale = phys_span / span
        F = mid + scale * (F - mid)

    # 4. Safety clip
    F = np.clip(F, F_MOTOR_MIN, F_MOTOR_MAX)

    # Invert thrust map: quad form
    disc = np.maximum(B_THRUST**2 + 4.0*A_THRUST*F, 0.0)
    pwm  = (-B_THRUST + np.sqrt(disc)) / (2.0*A_THRUST)
    return np.clip(pwm, 0.0, 1.0)


if __name__ == "__main__":
    from params import MASS, GRAVITY
    F_hov = MASS * GRAVITY
    pwm = wrench_to_pwm(F_hov, 0.0, 0.0, 0.0)
    print(f"Hover PWM: {pwm*100}")