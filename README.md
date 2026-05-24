- params.py: contains physical parameters of the drone and other helpful constants
- mixer.py: motor mixing algorithm, nonlinear map from commanded wrench -> motor PWMs
- dynamics.py: contains full continuous nonlinear dynamics of the drone, and hover linearization
- lqr_control.py: derivation of feedback gains using LQR with integral action
- sim.py: runs the simulation via RK45 solver, post-processing for plots/animation.
- waypoint_following.py: example of different guidance laws for dubin bank-to-turn aircraft (NLGL, Vector Field, PLOS, carrot-chasing)



crazyflie_sim/
│
├── params.py
├── dynamics.py
├── mixer.py
├── lqr_control.py
│
├── trajectory/
│   ├── __init__.py
│   ├── base.py         ← NEW: Trajectory ABC with reference() interface
│   ├── waypoint.py     ← NEW: WaypointTrajectory + hover_then_waypoints()
│   └── collocation.py  ← NEW: solve_collocation() + CollocationTrajectory
│
├── controllers/
│   ├── __init__.py
│   └── lqr_tracker.py  ← NEW: LQRTracker class (stateful, integrators inside)
│
└── sim.py              ← MODIFIED: now ~30 lines of wiring, all logic extracted


Drone motor ordering and directions:
![drone motor order + directions](figures/image.png)


NED coordinate system example:
![NED example](figures/image-1.png)

