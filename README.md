
- sim.py: runs the simulation via RK45 solver, post-processing for plots/animation.


## params.py
- physical parameters of the drone itself, with other helpful constants

## mixer.py
- nonlinear motor mixing algorithm
- maps from commanded wrench to motor PWM values
- Saturation strategy (in order):
  1. Shift all forces to keep max ≤ F_MOTOR_MAX (preserves moment ratios).
  2. Shift all forces to keep min ≥ 0 (may push max over limit temporarily).
  3. If spread > physical range, scale moments down around midpoint.
  4. Hard clip to [0, F_MOTOR_MAX].

## dynamics.py
- contains the full continuous nonlinear dynamics of the 16-state model as ```f_continuous()```
- computes a hover trim linearization for LQR derivation.

## lqr_control.py
- an error-state position controller with integral terms
- uses the hover-linearized model for derivation
- offers North, East, Down, and Yaw control

## wind.py
- a Dryden-style stochastic wind disturbance model
- currently only applies NED forces, no torques.
- how it works:
    - passes white noise through a low-pass filter: $\dot{w} = -\frac{w}{\tau} + \frac{\sigma}{\tau} \cdot \eta(t)$
    - where $\eta(t)$ is white noise, $\tau = 0.3s$ is the time constant, and $\sigma$ is the target standard deviation.
    - $\sigma$ controls how hard the wind blows
    - $\tau$ controls how slowly it changes

## postprocess.py
- loads ```sim_out.npz``` produced by ```sim.py```
- creates usefull figures and visualizations from the simulation output.

## sim.py
- closed-loop simulation via RK45 solver
- computes control, feeds to mixer, computes dynamics, injects wind.

#### waypoint_following.py
- an unrelated example of different guidance laws for dubin bank-to-turn aircraft (NLGL, Vector Field, PLOS, carrot-chasing)





Drone motor ordering and directions:
![drone motor order + directions](figures/image.png)


NED coordinate system example:
![NED example](figures/image-1.png)

