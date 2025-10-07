import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

# --- Parameters for Multi-Episode Simulation ---

num_episodes = 15
p_gains = {
    'px': 0.15, 'py': 0.15, 'pz': 0.15, 'pyaw': 0.15
}
dt = 0.1
total_time = 70.0
num_steps = int(total_time / dt)
start_ranges = {
    'x': [-6.0, -4.0], 'y': [-5.0, 5.0], 'z': [-4.0, 4.0], 'yaw': [0, 2 * np.pi]
}
target_state = np.array([0.0, 0.0, 0.0, np.pi])

# --- Run All Episodes and Collect Data for Analysis ---

all_trajectories = []
final_errors = []
times_to_target = []
target_threshold = 0.5

print(f"Running {num_episodes} simulation episodes for analysis...")

for episode in range(num_episodes):
    rx_0, ry_0, rz_0, yaw_0 = (
        np.random.uniform(*start_ranges['x']),
        np.random.uniform(*start_ranges['y']),
        np.random.uniform(*start_ranges['z']),
        np.random.uniform(*start_ranges['yaw'])
    )
    initial_state = np.array([rx_0, ry_0, rz_0, yaw_0])
    
    states = np.zeros((num_steps + 1, 4))
    states[0] = initial_state
    
    time_reached = None
    
    for i in range(num_steps):
        current_state = states[i]
        error = current_state - target_state
        
        # Check for time-to-target
        pos_error = np.linalg.norm(error[:3])
        if pos_error < target_threshold and time_reached is None:
            time_reached = i * dt
            
        vx = -p_gains['px'] * error[0]
        vy = -p_gains['py'] * error[1]
        vz = -p_gains['pz'] * error[2]
        omega_z = -p_gains['pyaw'] * error[3]
        
        states[i+1] = current_state + np.array([vx, vy, vz, omega_z]) * dt
        
    all_trajectories.append(states)
    
    # Calculate final position error for this episode
    final_pos_error = np.linalg.norm(states[-1, :3] - target_state[:3])
    final_errors.append(final_pos_error)
    
    # Store time-to-target, use total_time if never reached
    times_to_target.append(time_reached if time_reached is not None else total_time)
    
    print(f"  - Episode {episode + 1} complete. Final Error: {final_pos_error:.3f}m")

print("All episodes simulated.")


# --- 1. 3D Trajectories Plot with Color-Coded Endpoints ---

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title(f"All Trajectories with Color-Coded Final Error", fontsize=16)
ax.set_xlabel("X (m)"), ax.set_ylabel("Y (m)"), ax.set_zlabel("Z (m)")
ax.view_init(elev=30., azim=-125)

# Create a color map based on final errors
colors = cm.plasma(np.array(final_errors) / max(final_errors))

ax.scatter(target_state[0], target_state[1], target_state[2], c='red', marker='*', s=400, label="Docking Port", depthshade=False)

for i, trajectory in enumerate(all_trajectories):
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color='gray', alpha=0.5, lw=1)
    ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], c='g', marker='o', s=40)
    # Color the endpoint based on its final error
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], c=[colors[i]], marker='x', s=100)

# Create a color bar legend
sm = plt.cm.ScalarMappable(cmap=cm.plasma, norm=plt.Normalize(vmin=min(final_errors), vmax=max(final_errors)))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.5)
cbar.set_label('Final Docking Error (m)')

ax.legend()
plt.show()


# --- 2. Final Docking Error Plot ---

fig2, ax2 = plt.subplots(figsize=(12, 7))
episode_indices = np.arange(1, num_episodes + 1)
ax2.bar(episode_indices, final_errors, color=colors)
ax2.set_title("Final Docking Error per Episode", fontsize=16)
ax2.set_xlabel("Episode Number", fontsize=12)
ax2.set_ylabel("Euclidean Distance Error (m)", fontsize=12)
ax2.set_xticks(episode_indices)
ax2.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# --- 3. Time-to-Target Analysis ---

fig3, ax3 = plt.subplots(figsize=(12, 7))
ax3.hist(times_to_target, bins=10, edgecolor='black', color='skyblue')
ax3.set_title(f"Distribution of Time to Reach {target_threshold}m from Target", fontsize=16)
ax3.set_xlabel("Time (seconds)", fontsize=12)
ax3.set_ylabel("Number of Episodes", fontsize=12)
ax3.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()