import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate

###############
#### TODO: re-add comments from matlab, Fillet for smooth waypoint switching (chapter 11)
###############

# ── PARAMETERS ────────────────────────────────────────────────────────────
V_airspeed = 15
gravity = 9.81
phi_max = np.radians(45)
u_max = gravity * np.tan(phi_max)
R_min = V_airspeed**2 / u_max
tau_phi = 0.5     # roll lag (s)
k_psi = 1.0       # heading p gain
dt = 0.05         # 20hz
T_max = 600
wp_cap = 20       # waypoint capture radius (m)

# NLGL/L1
L1_distance = 18

# Vector Field
chi_inf = np.radians(70)
k_vf = 0.015

# PLOS
k2_plos = 0.02    # cross track gain

# Carrot Following
delta_c = 60      # lookahead (m)

# ── WAYPOINTS ─────────────────────────────────────────────────────────────
waypoints = np.array([
    [100,  100],
    [562,  1426],
    [1098, 898],
    [87,   1299],
    [902,  1062],
    [1249, 319],
    [273,  275],
    [456,  787],
    [648,  437]
])
n_wp = waypoints.shape[0]
print(f'Loaded {n_wp} waypoints.')

# ── INITIAL CONDITIONS ────────────────────────────────────────────────────
p0 = waypoints[0]
psi0 = np.arctan2(waypoints[1,1]-p0[1], waypoints[1,0]-p0[0])
phi0 = 0

# ── SIMULATION LOOP ───────────────────────────────────────────────────────
law_names = ['NLGL (L1)', 'Vector Field', 'PLOS', 'Carrot-Chasing']
n_laws = len(law_names)

class SimResult:
    pass

R = [] # List to store simulation results for each law

for i in range(n_laws):
    # state
    x = p0[0]
    y = p0[1]
    psi = psi0
    phi = phi0

    t = 0
    wi = 1 # current target waypoint index (0-indexed array, so 1 is the 2nd waypoint)

    Lx = [x]
    Ly = [y]
    Lxte = [0]
    Lu = [0]
    Lt = [0]

    while t < T_max and wi < n_wp:
        p = np.array([x, y])

        # constructing path line segment
        Wi = waypoints[wi-1]
        Wi1 = waypoints[wi]

        path_segment = Wi1 - Wi # path segment
        seg_mag = np.linalg.norm(path_segment) # segment length
        path_segment_unit = path_segment / seg_mag # univ vector along segment
        theta = np.arctan2(path_segment[1], path_segment[0]) #

        ep = p - Wi
        s_along = np.dot(ep, path_segment_unit)

        # Signed XTE: positive = UAV is to the right of path
        xte = path_segment_unit[0]*ep[1] - path_segment_unit[1]*ep[0]

        # Waypoint Switching Logic
        if s_along >= seg_mag or np.linalg.norm(p - Wi1) < wp_cap:
            wi += 1
            continue

        # ── Guidance Laws ────────────────────────────────────────────────
        # 1. NLGL
        if i == 0:
            if np.abs(xte) > L1_distance:
                s_vtp = s_along + L1_distance
            else:
                s_vtp = s_along + np.sqrt(L1_distance**2 - xte**2)

            s_vtp = np.clip(s_vtp, 0, seg_mag)
            vtp = Wi + s_vtp * path_segment_unit
            bear = np.arctan2(vtp[1]-y, vtp[0]-x)
            eta = np.arctan2(np.sin(bear - psi), np.cos(bear - psi))
            a_cmd = 2 * V_airspeed**2 * np.sin(eta) / L1_distance
            phi_cmd = np.clip(np.arctan(a_cmd / gravity), -phi_max, phi_max)

        # 2. Vector Field
        elif i == 1:
            psi_d = theta - chi_inf * (2/np.pi) * np.arctan(k_vf * xte)
            psi_err = np.arctan2(np.sin(psi_d - psi), np.cos(psi_d - psi))
            phi_cmd = np.clip(np.arctan(k_psi * psi_err), -phi_max, phi_max)

        # 3. PLOS
        elif i == 2:
            psi_d = theta - k2_plos * xte
            psi_err = np.arctan2(np.sin(psi_d - psi), np.cos(psi_d - psi))
            phi_cmd = np.clip(np.arctan(k_psi * psi_err), -phi_max, phi_max)

        # 4. Carrot Following
        elif i == 3:
            s_c = np.clip(s_along + delta_c, 0, seg_mag)
            vtp_c = Wi + s_c * path_segment_unit
            psi_d = np.arctan2(vtp_c[1]-y, vtp_c[0]-x)
            psi_err = np.arctan2(np.sin(psi_d - psi), np.cos(psi_d - psi))
            phi_cmd = np.clip(np.arctan(k_psi * psi_err), -phi_max, phi_max)

        # Roll dynamics
        phi_dot = (phi_cmd - phi) / tau_phi
        phi = np.clip(phi + phi_dot * dt, -phi_max, phi_max)
        u_act = gravity * np.tan(phi)

        # Kinematics
        x += V_airspeed * np.cos(psi) * dt
        y += V_airspeed * np.sin(psi) * dt
        psi += (gravity/V_airspeed) * np.tan(phi) * dt
        t += dt

        # Logging
        Lx.append(x)
        Ly.append(y)
        Lxte.append(np.abs(xte)) # Log absolute xte for metrics plotting
        Lu.append(np.abs(u_act)) # Log absolute control effort
        Lt.append(t)

    # ── Calculate Metrics and Store in R ─────────────────────────────────
    res = SimResult()
    res.name = law_names[i]
    res.x = np.array(Lx)
    res.y = np.array(Ly)
    res.xte = np.array(Lxte)
    res.u = np.array(Lu)
    res.t = np.array(Lt)
    res.wp_done = wi - 1

    # Metrics
    res.rms_xte = np.sqrt(np.mean(res.xte**2))
    res.max_xte = np.max(res.xte)
    res.ctrl_eff = integrate.trapezoid(res.u, res.t)
    R.append(res)

# ── TERMINAL SUMMARY TABLE ────────────────────────────────────────────────
print('\n' + '═'*62)
print(f'  {"Law":<18} {"RMS_XTE":>8} {"Max_XTE":>8} {"CtrlEffort":>12} {"WP_Done":>9}')
print('  ' + '-'*60)
for i in range(n_laws):
    print(f'  {R[i].name:<18} {R[i].rms_xte:8.1f} {R[i].max_xte:8.1f} {R[i].ctrl_eff:12.0f} {R[i].wp_done:6d}/{n_wp-1}')
print('═'*62)
print(f'Va={V_airspeed:.0f} m/s | R_min={R_min:.1f} m | phi_max={np.degrees(phi_max):.0f} deg | tau_phi={tau_phi:.1f} s')
print(f'NLGL L={L1_distance:.0f}m | VF chi_inf={np.degrees(chi_inf):.0f}deg k={k_vf:.3f} | PLOS k2={k2_plos:.3f} | Carrot delta={delta_c:.0f}m')


# ── PLOTS ─────────────────────────────────────────────────────────────────
clrs = ['#0072BD', '#D95319', '#77AC30', '#7E2F8E']
mkrs = ['-', '-', '--', '-.']
lw   = 1.8

# ── Fig 1: Trajectory Map ─────────────────────────────────────────────────
fig1 = plt.figure('Trajectories', figsize=(9.5, 8), facecolor='w')
ax1 = fig1.add_subplot(111)
ax1.grid(True)
ax1.set_aspect('equal', adjustable='box')

for s in range(n_wp - 1):
    ax1.plot([waypoints[s, 0], waypoints[s+1, 0]],
             [waypoints[s, 1], waypoints[s+1, 1]],
             'k--', linewidth=1)

ax1.plot(waypoints[:, 0], waypoints[:, 1], 'ks', markerfacecolor='k',
         markersize=9, label='Waypoints')

for i in range(n_wp):
    ax1.text(waypoints[i, 0] + 18, waypoints[i, 1] + 18, f'WP{i+1}',
             fontsize=8, fontweight='bold')

th = np.linspace(0, 2 * np.pi, 80)
for i in range(1, n_wp):
    ax1.plot(waypoints[i, 0] + wp_cap * np.cos(th),
             waypoints[i, 1] + wp_cap * np.sin(th),
             color=[0.75, 0.75, 0.75], linewidth=0.6)

for i in range(n_laws):
    ax1.plot(R[i].x, R[i].y, linestyle=mkrs[i], color=clrs[i], linewidth=lw,
             label=f'{R[i].name} (WP {R[i].wp_done}/{n_wp-1})')

ax1.plot(p0[0], p0[1], 'g^', markersize=13, markerfacecolor='g', label='Start')

ax1.legend(loc='best', fontsize=9)
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_title(f'UAV Path Following — Va={V_airspeed:.0f} m/s, $R_{{min}}$={R_min:.0f} m, $\\phi_{{max}}$=45°', fontsize=12)
ax1.tick_params(labelsize=11)

# ── Fig 2: Cross-Track Error vs Time ──────────────────────────────────────
fig2 = plt.figure('XTE vs Time', figsize=(11, 5.2), facecolor='w')
max_xte_all = max([np.max(r.xte) for r in R])
ymax = min(max_xte_all * 1.05, 400)

for i in range(n_laws):
    ax = fig2.add_subplot(2, 2, i + 1)
    ax.plot(R[i].t, R[i].xte, color=clrs[i], linewidth=lw)
    ax.grid(True)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('|XTE| (m)')
    ax.set_title(R[i].name, fontsize=11)
    ax.set_ylim([0, ymax])
    ax.tick_params(labelsize=10)

    text_str = f"RMS = {R[i].rms_xte:.1f} m\nMax = {R[i].max_xte:.1f} m\nWP: {R[i].wp_done}/{n_wp-1}"
    ax.text(0.97, 0.94, text_str, transform=ax.transAxes,
            ha='right', va='top', fontsize=9,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

fig2.suptitle('Absolute Cross-Track Error vs Time', fontsize=13, fontweight='bold')
fig2.tight_layout()

# ── Fig 3: Bar Charts (Metrics) ───────────────────────────────────────────
fig3 = plt.figure('Metrics', figsize=(10, 3.8), facecolor='w')
mdata = np.array([[r.rms_xte, r.max_xte, r.ctrl_eff] for r in R])
mlabels = ['RMS XTE (m)', 'Max XTE (m)', r'Control Effort ($\int |u| dt$)']
short_names = ['NLGL', 'VF', 'PLOS', 'Carrot']

for m in range(3):
    ax = fig3.add_subplot(1, 3, m+1)
    bars = ax.bar(range(n_laws), mdata[:, m], width=0.65, color=clrs)

    ax.set_xticks(range(n_laws))
    ax.set_xticklabels(short_names, fontsize=10)
    ax.set_ylabel(mlabels[m])
    ax.set_title(mlabels[m], fontsize=11)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Add text labels on top of the bars
    for j, bar in enumerate(bars):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.015 * np.max(mdata[:,m]),
                f'{yval:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

fig3.suptitle('Performance Comparison', fontsize=13, fontweight='bold')
fig3.tight_layout()

# ── Fig 4: Bank Angle Demand ───────────────────────────────────────────────
fig4 = plt.figure('Bank Angle', figsize=(10.5, 3.8), facecolor='w')
ax4 = fig4.add_subplot(111)
ax4.grid(True)

for i in range(n_laws):
    phi_log = np.rad2deg(np.arctan(R[i].u / gravity)) # Reconstruct bank angle
    # Apply sign back if needed (since we logged absolute u, we approximate visual envelope here)
    # Note: To perfectly match MATLAB, log signed u_act instead of np.abs(u_act) in the loop!
    ax4.plot(R[i].t, phi_log, linestyle=mkrs[i], color=clrs[i],
             linewidth=lw, label=R[i].name)

phi_max_deg = np.rad2deg(phi_max)
ax4.axhline(phi_max_deg, color='r', linestyle='--', linewidth=1.2, label=r'$\phi_{max}$')
ax4.axhline(-phi_max_deg, color='r', linestyle='--', linewidth=1.2)

ax4.legend(loc='best', fontsize=10)
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Bank Angle (deg)')
ax4.set_title('Bank Angle Demand vs Time', fontsize=12)
ax4.set_ylim([-50, 50])
ax4.tick_params(labelsize=11)

plt.show()


