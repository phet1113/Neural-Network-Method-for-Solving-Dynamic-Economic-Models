from os import name
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# 1. CONFIG
class EconomicConfig:
    def __init__(self):
        self.beta = 0.96
        self.delta = 0.15
        self.theta = 0.7
        self.psi_0 = 0.01
        self.rho = 0.7
        self.sigma_eps = 0.15

        # Anchors for Normalization
        r_implied = (1.0 / self.beta) - 1.0
        self.k_ss = (self.theta / (r_implied + self.delta)) ** (1.0 / (1.0 - self.theta))
        print(f"Target k_ss: {self.k_ss:.2f}")

    # for Consistency Check
    def profit(self, k, z):
        return z * tf.pow(k, self.theta)
    def adjustment_cost(self, i, k):
        return (self.psi_0 / 2.0) * tf.square(i) / (k + 1e-8)

    def marginal_cost(self, i, k):
        # MC = 1 + psi * (I/k)
        return 1.0 + self.psi_0 * (i / (k + 1e-8))

    def marginal_benefit_future(self, k_next, k_next_next, z_next):
        # The "Benefit" of having k_next tomorrow is:
        # 1. Extra Profit: theta * z * k^(theta-1)
        # 2. Saved Adjustment Cost: psi_0/2 * (I_next / k_next)^2
        # 3. Resale Value: (1 - delta) * (1 + psi_I_next)

        inv_next = k_next_next - (1 - self.delta) * k_next

        mpk = self.theta * z_next * tf.pow(k_next + 1e-8, self.theta - 1.0)

        # Term from reducing future adjustment costs
        adj_savings = (self.psi_0 / 2.0) * tf.square(inv_next / (k_next + 1e-8))

        # Term from determining future investment cost
        mc_next = self.marginal_cost(inv_next, k_next)

        return mpk + adj_savings + (1 - self.delta) * mc_next

# 2. DEEP AGENT (Policy Only)
class DeepAgent(tf.keras.Model):
    def __init__(self, cfg):
        super(DeepAgent, self).__init__()
        self.cfg = cfg

        # Policy Network
        # We use a wide Sigmoid cage [5, 450] to prevent explosion
        # Init bias 0.55 -> Softplus(0.55) ~= 1.0 -> Starts at k_ss
        self.policy_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(2,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='softplus',
                                  bias_initializer=tf.keras.initializers.Constant(0.55))
        ])

    def call(self, inputs):
        k = inputs[:, 0:1]
        z = inputs[:, 1:2]

        # Normalize
        k_norm = k / self.cfg.k_ss
        z_norm = tf.math.log(z)
        inp = tf.concat([k_norm, z_norm], axis=1)

        # Predict Ratio
        ratio = self.policy_net(inp)

        # De-normalize & Clip
        # Start exactly at k_ss (ratio=1.0)
        k_prime = ratio * self.cfg.k_ss

        return tf.clip_by_value(k_prime, 5.0, 450.0)

# 3. Euler training (Method 2)
@tf.function
def train_step(k_batch, z_batch):
    N = tf.shape(k_batch)[0]

    # 1. Generate Future Shocks (AiO: 2 paths)
    eps_1 = tf.random.normal((N, 1))
    eps_2 = tf.random.normal((N, 1))

    log_z = tf.math.log(z_batch)
    z_next_1 = tf.exp(config.rho * log_z + config.sigma_eps * eps_1)
    z_next_2 = tf.exp(config.rho * log_z + config.sigma_eps * eps_2)

    with tf.GradientTape() as tape:
        # --- Time t ---
        # Agent chooses k_{t+1}
        k_next = agent(tf.concat([k_batch, z_batch], axis=1))

        inv_curr = k_next - (1 - config.delta) * k_batch

        # Marginal Cost today
        MC_t = config.marginal_cost(inv_curr, k_batch)

        # --- Time t+1 (Two Branches) ---
        # Agent chooses k_{t+2} based on future state (k_{t+1}, z_{t+1})
        k_next_next_1 = agent(tf.concat([k_next, z_next_1], axis=1))
        k_next_next_2 = agent(tf.concat([k_next, z_next_2], axis=1))

        # Marginal Benefit tomorrow (Calculated purely from Physics + Next Choice)
        MB_1 = config.marginal_benefit_future(k_next, k_next_next_1, z_next_1)
        MB_2 = config.marginal_benefit_future(k_next, k_next_next_2, z_next_2)

        # --- Euler Residual (AiO) ---
        # Equation: MC_t = Beta * E[MB_{t+1}]
        # Residual: MC_t - Beta * MB_{t+1}

        err_1 = MC_t - config.beta * MB_1
        err_2 = MC_t - config.beta * MB_2

        # AiO Squared Error: E[err_1 * err_2]
        loss = tf.reduce_mean(err_1 * err_2)

    grads = tape.gradient(loss, agent.trainable_variables)
    optimizer.apply_gradients(zip(grads, agent.trainable_variables))
    return loss

def train_model():
    print("Starting Training (Method 2: Euler Only)...")
    loss_hist = []

    for epoch in range(30000):
        # Wide Sampling around 77
        k = tf.random.uniform((64, 1), 20.0, 150.0)
        z = tf.exp(tf.random.normal((64, 1), 0.0, 0.2))

        loss = train_step(k, z)

        if epoch % 2000 == 0:
            # We use abs() because AiO loss can be negative noise
            print(f"Epoch {epoch}: Euler Loss={np.abs(loss):.6f}")
            loss_hist.append(loss)

    return loss_hist

# 4. VFI (Ground Truth)
class Wide_VFI:
    def __init__(self, cfg):
        self.cfg = cfg
        self.nk = 200
        self.nz = 15
        # Grid covers the full dynamic range [5, 450]
        self.k_grid = np.linspace(5.0, 450.0, self.nk)

        log_z_std = cfg.sigma_eps / np.sqrt(1 - cfg.rho**2)
        self.z_grid = np.exp(np.linspace(-4*log_z_std, 4*log_z_std, self.nz))

        gh_nodes, gh_weights = np.polynomial.hermite.hermgauss(5)
        self.eps_nodes = gh_nodes * np.sqrt(2) * cfg.sigma_eps
        self.eps_weights = gh_weights / np.sqrt(np.pi)

    def solve(self):
        print("Solving VFI...")
        V = np.zeros((self.nz, self.nk))
        K_curr = self.k_grid[None, :, None]
        K_next = self.k_grid[None, None, :]
        Z_curr = self.z_grid[:, None, None]

        Invest = K_next - (1 - self.cfg.delta) * K_curr
        AdjCost = (self.cfg.psi_0 / 2.0) * (Invest**2) / (K_curr + 1e-8)
        Profit = Z_curr * (K_curr ** self.cfg.theta)
        Rewards = Profit - Invest - AdjCost

        for i in range(2000):
            # Interpolate V
            interp = RegularGridInterpolator((self.z_grid, self.k_grid), V, bounds_error=False, fill_value=None)

            # Vectorized Expectations
            log_z = np.log(self.z_grid)
            z_futures = np.exp(self.cfg.rho * log_z[:, None] + self.eps_nodes[None, :]) # (nz, 5)

            EV = np.zeros((self.nz, self.nk))
            for iz in range(self.nz):
                zf = z_futures[iz, :] # (5,)
                # Grid for interpolation: (5, nk)
                mesh_z, mesh_k = np.meshgrid(zf, self.k_grid, indexing='ij')
                points = np.stack([mesh_z.ravel(), mesh_k.ravel()], axis=1)
                v_vals = interp(points).reshape(5, self.nk)
                EV[iz, :] = np.dot(self.eps_weights, v_vals)

            RHS = Rewards + self.cfg.beta * EV[:, None, :]
            V_new = np.max(RHS, axis=2)

            if np.max(np.abs(V_new - V)) < 1e-4:
                pol_idx = np.argmax(RHS, axis=2)
                # Return interpolator for cleaner plotting
                return RegularGridInterpolator((self.z_grid, self.k_grid), self.k_grid[pol_idx], bounds_error=False, fill_value=None)
            V = V_new
        return None

# 5. Euler Residual Validation
def calculate_euler_residual(policy_func, k_val, z_val, config):
    """
    Calculates the Euler residual for a given policy function at a single state (k, z).
    """
    # Use the same quadrature nodes as VFI for a fair comparison
    gh_nodes, gh_weights = np.polynomial.hermite.hermgauss(5)
    eps_nodes = gh_nodes * np.sqrt(2) * config.sigma_eps
    eps_weights = gh_weights / np.sqrt(np.pi)

    # --- Time t ---
    # Agent chooses k_{t+1} based on the policy
    # We need to handle both TF agent and Scipy interpolator
    if isinstance(policy_func, DeepAgent):
        k_tensor = tf.constant([[k_val]], dtype=tf.float32)
        z_tensor = tf.constant([[z_val]], dtype=tf.float32)
        k_next = policy_func(tf.concat([k_tensor, z_tensor], axis=1)).numpy().item()
    else: # It's the VFI interpolator
        k_next = policy_func((z_val, k_val)).item()

    k_batch = tf.constant([[k_val]], dtype=tf.float32)
    k_next_batch = tf.constant([[k_next]], dtype=tf.float32)

    inv_curr = k_next_batch - (1 - config.delta) * k_batch
    MC_t = config.marginal_cost(inv_curr, k_batch)

    # --- Time t+1 (Expectation over future shocks) ---
    log_z = np.log(z_val)
    z_futures = np.exp(config.rho * log_z + eps_nodes) # Shape: (5,)

    mb_futures = []
    for z_next_val in z_futures:
        # Agent chooses k_{t+2} based on future state
        if isinstance(policy_func, DeepAgent):
            k_next_tensor = tf.constant([[k_next]], dtype=tf.float32)
            z_next_tensor = tf.constant([[z_next_val]], dtype=tf.float32)
            k_next_next = policy_func(tf.concat([k_next_tensor, z_next_tensor], axis=1)).numpy().item()
        else: # VFI interpolator
            k_next_next = policy_func((z_next_val, k_next)).item()

        # Calculate Marginal Benefit for this specific future state
        k_next_b = tf.constant([[k_next]], dtype=tf.float32)
        k_next_next_b = tf.constant([[k_next_next]], dtype=tf.float32)
        z_next_b = tf.constant([[z_next_val]], dtype=tf.float32)

        mb = config.marginal_benefit_future(k_next_b, k_next_next_b, z_next_b)
        mb_futures.append(mb.numpy().item())
        
    # Calculate the expected marginal benefit
    expected_MB = np.dot(eps_weights, mb_futures)

    # --- Euler Residual ---
    # Residual: MC_t - Beta * E[MB_{t+1}]
    residual = MC_t.numpy().item() - config.beta * expected_MB
    return residual

def validate_residuals(agent, vfi_policy, config):
    """
    Generates a grid of states, calculates Euler residuals for both DL and VFI,
    and visualizes them as heatmaps.
    """
    print("\n--- Starting Euler Residual Validation ---")
    
    # Create a validation grid
    k_grid_val = np.linspace(38, 115, 50)
    z_grid_val = np.exp(np.linspace(-2 * (config.sigma_eps / np.sqrt(1 - config.rho**2)), 
                                    2 * (config.sigma_eps / np.sqrt(1 - config.rho**2)), 40))

    residuals_dl = np.zeros((len(z_grid_val), len(k_grid_val)))
    residuals_vfi = np.zeros((len(z_grid_val), len(k_grid_val)))

    # Calculate residuals for each point on the grid
    for i, z in enumerate(z_grid_val):
        for j, k in enumerate(k_grid_val):
            residuals_dl[i, j] = calculate_euler_residual(agent, k, z, config)
            residuals_vfi[i, j] = calculate_euler_residual(vfi_policy, k, z, config)

    # Print summary statistics
    mae_dl = np.mean(np.abs(residuals_dl))
    mae_vfi = np.mean(np.abs(residuals_vfi))
    print(f"Mean Absolute Euler Residual (DL):  {mae_dl:.6f}")
    print(f"Mean Absolute Euler Residual (VFI): {mae_vfi:.6f}")

    # Plotting the heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Determine a common color scale for fair comparison
    max_abs_res = max(np.max(np.abs(residuals_dl)), np.max(np.abs(residuals_vfi)))
    vmin, vmax = -max_abs_res, max_abs_res

    # Plot for DL Agent
    im1 = axes[0].imshow(residuals_dl, origin='lower', aspect='auto', 
                         extent=[k_grid_val[0], k_grid_val[-1], z_grid_val[0], z_grid_val[-1]],
                         cmap='coolwarm', vmin=vmin, vmax=vmax)
    axes[0].set_title('DL Agent Euler Residuals')
    axes[0].set_xlabel('Current Capital k')
    axes[0].set_ylabel('Current Shock z')

    # Plot for VFI
    im2 = axes[1].imshow(residuals_vfi, origin='lower', aspect='auto',
                         extent=[k_grid_val[0], k_grid_val[-1], z_grid_val[0], z_grid_val[-1]],
                         cmap='coolwarm', vmin=vmin, vmax=vmax)
    axes[1].set_title('VFI Euler Residuals (Ground Truth)')
    axes[1].set_xlabel('Current Capital k')
    axes[1].set_ylabel('Current Shock z')

    fig.colorbar(im2, ax=axes.ravel().tolist(), label='Residual Value (MC - βE[MB])')
    plt.tight_layout()
    plt.savefig('./results/euler_residuals.pdf', dpi=300)
    # Note: We will call plt.show() at the very end.


# 6. Plotting & Mape
def plot_final(agent, config,vfi_policy):
    # vfi_solver = Wide_VFI(config)
    # vfi_policy = vfi_solver.solve()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    z_targets = [0.606, 1.013, 1.694]

    # Plot range matches the "Cage" region
    k_plot = np.linspace(38, 115, 100)

    mape_scores = []

    for i, z_val in enumerate(z_targets):
        ax = axes[i]

        # 1. VFI (Ground Truth)
        points = np.stack([np.full_like(k_plot, z_val), k_plot], axis=1)
        k_vfi = vfi_policy(points)
        ax.plot(k_plot, k_vfi, color='orange', linestyle='--', linewidth=3, label='VFI')

        # 2. DL (Prediction)
        k_tens = tf.cast(k_plot[:, None], tf.float32)
        z_tens = tf.ones_like(k_tens) * z_val

        # Safe unpacking (handles if agent returns tuple or single tensor)
        raw_out = agent(tf.concat([k_tens, z_tens], axis=1))
        if isinstance(raw_out, (list, tuple)):
            _, k_dl_raw = raw_out
        else:
            k_dl_raw = raw_out

        k_dl = k_dl_raw.numpy().flatten()

        # 3. Calculate MAPE
        # Mean Absolute Percentage Error = |(True - Pred) / True| * 100
        abs_error = np.abs(k_vfi - k_dl)
        pct_error = (abs_error / (np.abs(k_vfi) + 1e-8)) * 100
        mape = np.mean(pct_error)
        mape_scores.append(mape)

        ax.plot(k_plot, k_dl, color='blue', linewidth=2, label=f'DL (MAPE={mape:.2f}%)')

        # 4. 45-degree line
        ax.plot(k_plot, k_plot, color='gray', linestyle=':', label="k'=k")

        ax.set_title(f"Policy at z = {z_val:.3f}")
        ax.set_xlabel("Current Capital k")
        ax.set_ylabel("Next Capital k'")
        ax.legend()
        ax.grid(True, alpha=0.3)

    overall_mape = np.mean(mape_scores)
    print(f"Overall MAPE: {overall_mape:.4f}%")
    plt.tight_layout()
    plt.savefig('./results/policy_comparison.pdf', dpi=300) 
    # plt.show()

if __name__ == '__main__':
    tf.random.set_seed(42)
    np.random.seed(42)
    config = EconomicConfig()
    tf.keras.backend.clear_session()
    agent = DeepAgent(config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)

    # EXECUTE
    hist = train_model()

    vfi_solver = Wide_VFI(config)
    vfi_policy_interpolator = vfi_solver.solve()

    # --- Now run analysis using the trained agent and VFI solution ---
    # 1. Plot the policy functions and calculate MAPE
    plot_final(agent, config, vfi_policy_interpolator)

    # 2. Validate and plot the Euler residuals
    validate_residuals(agent, vfi_policy_interpolator, config)

    plt.show()