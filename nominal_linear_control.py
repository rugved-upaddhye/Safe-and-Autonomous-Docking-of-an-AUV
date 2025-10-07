import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LightSource

num_episodes = 5

# Controller gains
p_gains = {
    'px': 0.15, 'py': 0.15, 'pz': 0.15, 'pyaw': 0.15
}

dt = 0.1
total_time = 60.0
num_steps = int(total_time / dt)

# ranges for random starting points
start_ranges = {
    'x': [-6.0, -4.0], 'y': [-5.0, 5.0], 'z': [-4.0, 4.0], 'yaw': [0, 2 * np.pi]
}

# Fixed Target
target_state = np.array([0.0, 0.0, 0.0, np.pi])

# --- 1. Pre-compute All Episodes ---

all_trajectories = []
print(f"Pre-computing {num_episodes} simulation episodes...")

for episode in range(num_episodes):
    rx_0 = np.random.uniform(*start_ranges['x'])
    ry_0 = np.random.uniform(*start_ranges['y'])
    rz_0 = np.random.uniform(*start_ranges['z'])
    yaw_0 = np.random.uniform(*start_ranges['yaw'])
    initial_state = np.array([rx_0, ry_0, rz_0, yaw_0])

    states = np.zeros((num_steps + 1, 4))
    states[0] = initial_state

    for i in range(num_steps):
        current_state = states[i]
        error = current_state - target_state
        vx = -p_gains['px'] * error[0]
        vy = -p_gains['py'] * error[1]
        vz = -p_gains['pz'] * error[2]
        omega_z = -p_gains['pyaw'] * error[3]
        states[i+1] = current_state + np.array([vx, vy, vz, omega_z]) * dt

    all_trajectories.append(states)

print("All episodes computed. Starting visualization...")

# --- 2. Setup 3D Visualization ---
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("AUV Docking Simulation: Episode by Episode", fontsize=16)
ax.set_xlabel("X (m)"), ax.set_ylabel("Y (m)"), ax.set_zlabel("Z (m)")
ax.set_xlim([-7, 2]), ax.set_ylim([-6, 6]), ax.set_zlim([-5, 5])
ax.view_init(elev=25., azim=-145)

# Target docking port
ax.plot([0], [0], [0], 'ro', markersize=10, label="Docking Port")

# AUV Model Definition
def create_auv_model(sections=12, body_len=1.2, body_rad=0.15, nose_len=0.4, fin_h=0.4, fin_w=0.4):
    theta = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    circ_pts = np.array([np.cos(theta), np.sin(theta)]).T * body_rad
    front_ring = np.insert(circ_pts, 2, body_len / 2, axis=1)
    back_ring = np.insert(circ_pts, 2, -body_len / 2, axis=1)
    nose_tip = np.array([[0, 0, body_len / 2 + nose_len]])
    fin_base_z = -body_len / 2
    fins = np.array([
        [0, body_rad, fin_base_z], [0, body_rad + fin_h, fin_base_z], [-fin_w, body_rad, fin_base_z],
        [0, -body_rad, fin_base_z], [0, -(body_rad + fin_h), fin_base_z], [-fin_w, -body_rad, fin_base_z],
        [body_rad, 0, fin_base_z], [body_rad + fin_h, 0, fin_base_z], [-fin_w, body_rad, fin_base_z],
        [-body_rad, 0, fin_base_z], [-(body_rad + fin_h), 0, fin_base_z], [-fin_w, -body_rad, fin_base_z]
    ])
    fins[6:9, [0, 1]] = fins[6:9, [0, 1]][:, [1, 0]]
    fins[9:12, [0, 1]] = fins[9:12, [0, 1]][:, [1, 0]]
    vertices = np.vstack([front_ring, back_ring, nose_tip, fins])
    faces = []
    for i in range(sections):
        j = (i + 1) % sections
        faces.extend([[i, j, i + sections], [j, j + sections, i + sections]])
    nose_idx = 2 * sections
    for i in range(sections):
        faces.append([i, (i + 1) % sections, nose_idx])
    fin_start_idx = nose_idx + 1
    for i in range(4):
        base = fin_start_idx + i * 3
        faces.append([base, base + 1, base + 2])
    R_orient = np.array([[0,0,1],[0,1,0],[-1,0,0]])
    vertices = (R_orient @ vertices.T).T
    return vertices, np.array(faces)

auv_verts, auv_faces = create_auv_model()
auv_body_plot = None
light = LightSource(azdeg=225, altdeg=25)
trajectory_line, = ax.plot([], [], [], 'c--', lw=1.5, label="Current Trajectory")
episode_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=12)

# --- 3. Animation Function ---
def animate(frame):
    global auv_body_plot

    # Determine current episode and the time step within it
    episode_num = frame // (num_steps + 1)
    time_step = frame % (num_steps + 1)
    
    # Get the state from the pre-computed data
    rx, ry, rz, yaw = all_trajectories[episode_num][time_step]

    # Rotate and translate AUV vertices
    R_yaw = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
    transformed_v = (R_yaw @ auv_verts.T).T + np.array([rx, ry, rz])
    
    if auv_body_plot:
        auv_body_plot.remove()
        
    auv_body_plot = ax.plot_trisurf(
        transformed_v[:,0], transformed_v[:,1], transformed_v[:,2], 
        triangles=auv_faces, color='deepskyblue', lightsource=light,
        edgecolor='black', linewidth=0.2
    )
    
    # Update trajectory for the current episode
    current_trajectory_data = all_trajectories[episode_num][:time_step+1]
    trajectory_line.set_data(current_trajectory_data[:, 0], current_trajectory_data[:, 1])
    trajectory_line.set_3d_properties(current_trajectory_data[:, 2])

    # Update text
    episode_text.set_text(f'Episode: {episode_num + 1}/{num_episodes}\nTime: {time_step*dt:.1f}s')
    
    return auv_body_plot, trajectory_line, episode_text

# --- 4. Run Animation ---
total_frames = num_episodes * (num_steps + 1)
ani = FuncAnimation(fig, animate, frames=total_frames, interval=30, blit=False)
ax.legend()
plt.show()
# print("Saving animation as MP4... This may take a moment.")
# ani.save('auv_docking_simulation.mp4', writer='ffmpeg', fps=30)