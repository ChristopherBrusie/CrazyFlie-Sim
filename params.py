# see ReadMe for motor ordering and directions

import numpy as np

MASS        = 0.033          # kg
GRAVITY     = 9.81           # m/s^2
ARM_LENGTH  = 0.046          # m
ARM_DIAG    = ARM_LENGTH * np.sqrt(2) / 2   # projected arm length for moments

# Inertia tensor [kg*m^2]
J = np.array([[16.571710, 0.830806, 0.718277],
              [0.830806, 16.655602, 1.800197],
              [0.718277, 1.800197, 29.261652]]) * 1e-6

# constants for PWM -> thrust map
A_THRUST = 0.091492681
B_THRUST = 0.067673604

# constant for thrust -> torque map
C_TAU = 0.005964552

# Motor force limits (per motor)
F_MOTOR_MIN = 0.0
F_MOTOR_MAX = A_THRUST + B_THRUST   # Ax^2 + Bx


# aerodynamic drag matrix (body)
K_AERO = np.array([[-10.2506, -0.3177, -0.4332],
                   [-0.3177, -10.2506, -0.4332],
                   [-7.7050,  -7.7050, -7.5530]]) * 10e-7

# Mixer matrix: maps motor thrusts -> wrench
d = ARM_DIAG
M_MIX = np.array([
    [1,      1,      1,      1     ],
    [-d,    -d,      d,      d     ],
    [ d,    -d,     -d,      d     ],
    [ C_TAU, -C_TAU, C_TAU, -C_TAU],
])

# wrench -> motor thrusts
M_MIX_INV = np.linalg.inv(M_MIX)

# Hover equilibrium: quad form
F_HOVER   = MASS * GRAVITY / 4
PWM_HOVER = (-B_THRUST + np.sqrt(B_THRUST**2 + 4*A_THRUST*F_HOVER)) / (2*A_THRUST)

# Motor RPM map: n [rev/s] = (C0 + C1*pwm - C2*pwm^2) / 60
RPM_C0 = 2073.0
RPM_C1 = 358.1
RPM_C2 = 1.434
RPM_SCALE = 60.0

# Motor actuator dynamics (continuous-time first-order lag)
# Original discrete pole: z = 0.9695404 at Ts = 0.01s => tau ~ 323ms (too slow for NL sim)
# Use a physically reasonable 30ms time constant for the brushed motors+ESC
# Real CF motor bandwidth from sys-id is ~30-50 rad/s in thrust
_TAU_MOTOR = 0.030           # 30 ms time constant
A_ACT = -1.0 / _TAU_MOTOR   # = -33.3 s^-1
B_ACT = -A_ACT               # = +33.3  (unity steady-state gain)

if __name__ == "__main__":
    print(f"MASS       = {MASS} kg")
    print(f"J diag     = {np.diag(J)*1e6} x1e-6 kg*m^2")
    print(f"F_MOTOR_MAX= {F_MOTOR_MAX:.4f} N  (total {4*F_MOTOR_MAX:.4f} N, weight {MASS*GRAVITY:.4f} N)")
    print(f"PWM_HOVER  = {PWM_HOVER:.4f} ({PWM_HOVER*100:.2f}%)")
    print(f"A_ACT      = {A_ACT:.2f}  (tau = {_TAU_MOTOR*1000:.0f} ms)")
    print(f"M_MIX:\n{M_MIX}")