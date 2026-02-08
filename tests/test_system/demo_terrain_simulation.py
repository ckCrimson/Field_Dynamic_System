import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# --- IMPORT YOUR ARCHITECTURE ---
# (Adjust these paths to match your project structure)
from src.field_dynamic_system.core.state.discrete import LazyDiscreteStateSpace
from src.field_dynamic_system.core.field.mappings import DiscreteFieldMapper
from src.field_dynamic_system.core.field.algebra import RealFieldAlgebra, VectorFieldAlgebra
from src.field_dynamic_system.neighbor.discrete import DiscreteTopology
from src.field_dynamic_system.systems.static import StaticFieldTopologySystem
#from src.field_dynamic_system.systems.dynamic import DynamicSystem  # Assuming you have the engine class

# --- 1. SETUP: THE 1 MILLION STATE WORLD ---
print("Initializing 1000x1000 World (1 Million States)...")
t0 = time.time()

# A. Create Raw Coordinates (Lazy Space)
N = 1000
x = np.linspace(0, 100, N)
y = np.linspace(0, 100, N)
xv, yv = np.meshgrid(x, y)
# Shape: (1,000,000, 2)
raw_coords = np.column_stack((xv.ravel(), yv.ravel()))

# We wrap it in LazySpace (Zero object overhead)
space = LazyDiscreteStateSpace(raw_data=raw_coords, wrapper_class=tuple)


# B. Generate Terrain (The "Fields")
# Using JAX for fast vectorized generation
def generate_terrain(coords):
    x, y = coords[:, 0], coords[:, 1]
    # Simple procedrual terrain: 3 sin waves combined
    h = 10 * jnp.sin(x * 0.1) * jnp.cos(y * 0.1) \
        + 5 * jnp.sin(x * 0.3 + y * 0.2) \
        + 20 * jnp.exp(-((x - 50) ** 2 + (y - 50) ** 2) / 100.0)  # Big mountain in center
    return h.reshape(-1, 1)


def generate_wind(coords, height):
    # Wind blows East (1, 0) but is deflected by terrain height
    # Slope calculation (approximate)
    flat_wind = jnp.column_stack((jnp.ones(len(coords)), jnp.zeros(len(coords))))
    # Wind speeds up on hills (height factor)
    return flat_wind * (1.0 + height / 10.0)


# C. Create Field Mappers
print("  - Generating Field Data...")
# Convert numpy coords to JAX for generation
j_coords = jnp.array(raw_coords)

# Height Field
height_data = generate_terrain(j_coords)
height_mapper = DiscreteFieldMapper(space, RealFieldAlgebra())
height_mapper.apply_vector(height_data)

# Wind Field
wind_data = generate_wind(j_coords, height_data)
wind_mapper = DiscreteFieldMapper(space, VectorFieldAlgebra(dim=2))
wind_mapper.apply_vector(wind_data)


# D. Create Topology (Grid)
# We mock a 4-connected grid for movement logic
# D. Create Topology (Grid)
class GridTopology(DiscreteTopology):
    def get_adjacency_matrix(self):
        # Optimized Path: Returns the Identity Matrix (or real adjacency)
        # The System uses THIS for the vector physics.
        return jnp.eye(N*N)

    def compute_neighbors(self, state_val):
        # Abstract Method Implementation (Required by Python ABC)
        # Since we are using the vectorized matrix path above,
        # this method is technically unused in this specific demo,
        # but we must define it to instantiate the class.
        return [state_val] # Just return self (dummy)

topology = GridTopology(space)


topology = GridTopology(space)

# E. COMPILE THE SYSTEM
print("  - Compiling StaticFieldTopologySystem...")
world_system = StaticFieldTopologySystem(
    state=jnp.array([500500]),  # Start in the middle (Index ~500,000)
    topology=topology,
    space=space,
    field_mappers={
        "height": height_mapper,
        "wind": wind_mapper
    }
)

print(f"World Created in {time.time() - t0:.4f} seconds.")


# --- 2. THE PHYSICS ENGINE ---

# Define the Operator (The Move Logic)
# Input: Current Index (int)
# Output: New Index (int)
@jax.jit
def move_agent(current_idx_array):
    idx = current_idx_array[0]  # scalar index

    # 1. Get current position vector from Space (We assume we pass coords in)
    # Optimization: In a real engine, we'd look up coords from index.
    # For this demo, let's pretend state is actually (x, y) continuous for smoothness
    pass
    # WAIT! Our system state is an INDEX (Discrete).
    # Let's switch logic: The agent "Surfs" the grid.

    # Simple Logic:
    # New_Index = Old_Index + Wind_Direction_Offset
    # Since wind is (1, 0) mostly, agent moves +1 in X (Index +1)

    # To make it visible, let's just move +N (Up) +1 (Right) based on wind
    return current_idx_array + 1 + N  # Move Diagonal


# Let's use a simpler Dynamic System loop for visualization
# We will manually step it to update the plot

# --- 3. VISUALIZATION ---
print("Visualizing...")

# Prepare Data for Plotting
H = np.array(height_data).reshape(N, N)
WX = np.array(wind_data[:, 0]).reshape(N, N)
WY = np.array(wind_data[:, 1]).reshape(N, N)

fig, ax = plt.subplots(figsize=(10, 8))

# 1. Plot Height Map
im = ax.imshow(H, cmap='terrain', origin='lower', extent=[0, 100, 0, 100])
plt.colorbar(im, label="Terrain Height")

# 2. Plot Wind Vectors (Subsampled for readability)
# We verify the wind field exists by plotting arrows
skip = 40  # Only plot every 40th vector
ax.quiver(xv[::skip, ::skip], yv[::skip, ::skip],
          WX[::skip, ::skip], WY[::skip, ::skip],
          color='white', alpha=0.5, label="Wind Velocity")

# 3. Plot The Agent (Red Dot)
agent_idx = 500500  # Centerish
agent_pos = raw_coords[agent_idx]
dot, = ax.plot(agent_pos[0], agent_pos[1], 'ro', markersize=10, markeredgecolor='black', label="System")

ax.set_title(f"1,000,000 State Simulation\nSystem with Height & Wind Fields")
ax.legend(loc='upper right')

# SIMULATION LOOP (Animation)
try:
    for step in range(100):
        # 1. Physics Step (Approximate movement)
        # Get wind at current index
        current_wind = wind_data[agent_idx]  # (vx, vy)

        # Update logical position (Move along wind)
        # Scaling speed for visibility
        agent_pos[0] += current_wind[0] * 0.5
        agent_pos[1] += current_wind[1] * 0.5

        # Wrap around world (Torus topology)
        agent_pos[0] %= 100
        agent_pos[1] %= 100

        # 2. Find nearest discrete state index (Inverse lookup)
        # (In raw physics we'd keep continuous state, but here we snap to grid)
        col = int(agent_pos[0] / (100 / N))
        row = int(agent_pos[1] / (100 / N))
        agent_idx = row * N + col

        # 3. Update Plot
        dot.set_data([agent_pos[0]], [agent_pos[1]])
        ax.set_title(f"Step {step}: Agent at {agent_pos} | Height: {H[row, col]:.2f}")

        plt.pause(0.01)  # Short pause to animate

except KeyboardInterrupt:
    pass

plt.show()