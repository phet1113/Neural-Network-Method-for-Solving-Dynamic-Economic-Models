import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import time
# 1. CONFIG (Risky Debt Setup)
class EconomicConfig:
    def __init__(self):
        self.beta = 0.96
        self.delta = 0.15
        self.theta = 0.7
        self.psi_0 = 0.01
        self.rho = 0.7

        # [TUNING] Higher Volatility & Penalty to make Debt risky
        self.sigma_eps = 0.30
        self.alpha = 0.50     # 50% Bankruptcy Cost

        self.tau = 0.2

        # Anchors
        r_implied = (1.0 / self.beta) - 1.0
        self.k_ss = (self.theta / (r_implied + self.delta)) ** (1.0 / (1.0 - self.theta))
        self.val_ss = (self.k_ss ** self.theta) / (1 - self.beta)

        print(f"Risky Model: k_ss={self.k_ss:.2f}")

    def profit(self, k, z):
        return (1 - self.tau) * z * tf.pow(k, self.theta)

    def adjustment_cost(self, i, k):
        return (self.psi_0 / 2.0) * tf.square(i) / (k + 1e-8)

    def recovery_value(self, k, z):
        # Liquidation value
        val = (1 - self.tau) * self.profit(k, z) + (1 - self.delta) * k
        return (1 - self.alpha) * val

# 2. RISKY AGENT (Split Heads + Smart Init)
class RiskyAgent(tf.keras.Model):
    def __init__(self, cfg):
        super(RiskyAgent, self).__init__()
        self.cfg = cfg

        self.price_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(3,)),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        self.value_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(3,)),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(1, activation='linear')
        ])

        # We split the output to control initialization separately
        input_layer = tf.keras.layers.Input(shape=(3,))
        h1 = tf.keras.layers.Dense(64, activation='relu')(input_layer)
        h2 = tf.keras.layers.Dense(64, activation='relu')(h1)

        # Head 1: Capital (k') -> Softplus -> Init to Steady State (0.55)
        k_out = tf.keras.layers.Dense(1, activation='softplus',
                                      bias_initializer=tf.keras.initializers.Constant(0.55),
                                      name='k_head')(h2)

        # Head 2: Debt (b') -> Sigmoid -> Init to ZERO (-3.0)
        # This forces the agent to start with 0 debt and learn to borrow slowly
        b_out = tf.keras.layers.Dense(1, activation='sigmoid',
                                      bias_initializer=tf.keras.initializers.Constant(-3.0),
                                      name='b_head')(h2)

        self.policy_model = tf.keras.Model(inputs=input_layer, outputs=[k_out, b_out])

    # Standard Keras call (returns everything)
    def call(self, inputs):
        k = inputs[:, 0:1]
        b = inputs[:, 1:2]
        z = inputs[:, 2:3]

        val = self.get_value(k, b, z)
        kp, bp = self.get_policy(k, b, z)
        q = self.get_price(kp, bp, z)

        return val, kp, bp, q

    def get_price(self, k_prime, b_prime, z):
        kp_norm = k_prime / self.cfg.k_ss
        bp_norm = b_prime / self.cfg.k_ss
        z_norm = tf.math.log(z)
        inp = tf.concat([kp_norm, bp_norm, z_norm], axis=1)

        q = self.price_net(inp)
        return q * self.cfg.beta # Max price is 1/(1+r)

    def get_value(self, k, b, z):
        k_norm = k / self.cfg.k_ss
        b_norm = b / self.cfg.k_ss
        z_norm = tf.math.log(z)
        inp = tf.concat([k_norm, b_norm, z_norm], axis=1)

        v_raw = self.value_net(inp)
        return v_raw * self.cfg.val_ss

    def get_policy(self, k, b, z):
        k_norm = k / self.cfg.k_ss
        b_norm = b / self.cfg.k_ss
        z_norm = tf.math.log(z)
        inp = tf.concat([k_norm, b_norm, z_norm], axis=1)

        # Get both outputs from the split model
        k_ratio, b_ratio = self.policy_model(inp)

        # De-normalize
        k_prime = k_ratio * self.cfg.k_ss
        k_prime = tf.clip_by_value(k_prime, 5.0, 450.0)

        # Debt Range [0, 300]
        # b_ratio starts near 0.05 because of init -3.0
        b_prime = b_ratio * 300.0

        return k_prime, b_prime

# 3. TRAIN STEP (With Agency Penalty)

@tf.function
def train_step(k, b, z):
    N = tf.shape(k)[0]
    eps = tf.random.normal((N, 1))
    log_z = tf.math.log(z)
    z_next = tf.exp(config.rho * log_z + config.sigma_eps * eps)

    # --- Generate Random Debt for Pricing Training ---
    # The Bank needs to learn the price for ALL debt levels, not just what the agent picks.
    # We simulate a "Hypothetical" loan book.
    b_hypothetical = tf.random.uniform((N, 1), 10.0, 250.0) # Random debt 10 to 250

    with tf.GradientTape(persistent=True) as tape:
        # 1. Policy Decisions
        k_prime, b_prime = agent.get_policy(k, b, z)

        # 2. Pricing (Hypothetical)
        # Train the Price Net on the HYPOTHETICAL debt, not the agent's choice.
        # This ensures the Red Line (Yield Curve) forms correctly even if agent borrows 0.
        # Note: We use k_prime (agent's size choice) but b_hypothetical
        q_hyp = agent.get_price(k_prime, b_hypothetical, z)

        # Check Default for Hypothetical Debt
        v_next_hyp = agent.get_value(k_prime, b_hypothetical, z_next)
        survive_hyp = tf.sigmoid(v_next_hyp * 10.0)

        # Recovery for Hypothetical Debt
        rec_val = config.recovery_value(k_prime, z_next)
        rec_rate = rec_val / (b_hypothetical + 1e-8)
        rec_rate = tf.clip_by_value(rec_rate, 0.0, 1.0)

        payoff_hyp = survive_hyp * 1.0 + (1 - survive_hyp) * rec_rate

        # LOSS 1: Price Network (Fit the Curve)
        loss_price = tf.reduce_mean(tf.square(q_hyp - config.beta * payoff_hyp))

        # 3. Value & Policy (Actual Agent)
        # Now we use the Real Choice (b_prime) for the Firm's objectives
        q_actual = agent.get_price(k_prime, b_prime, z)

        inv = k_prime - (1 - config.delta) * k
        prof = config.profit(k, z)
        adj = config.adjustment_cost(inv, k)

        # Net Borrowing
        net_borrowing = q_actual * b_prime - b
        dividend = prof - adj - inv + net_borrowing

        v_next_real = agent.get_value(k_prime, b_prime, z_next)
        v_next_lim = tf.nn.relu(v_next_real)

        # LOSS 2: Value (Bellman)
        target_v = tf.stop_gradient(dividend + config.beta * v_next_lim)
        v_curr = agent.get_value(k, b, z)
        loss_val = tf.reduce_mean(tf.square((v_curr - target_v)/config.val_ss))

        # LOSS 3: Policy (Maximize Wealth)
        # with Penalty
        agency_cost = 0.002 * tf.reduce_mean(b_prime / config.k_ss)

        total_wealth = dividend + config.beta * v_next_lim
        loss_pol = -tf.reduce_mean(total_wealth / config.val_ss) + agency_cost

    # Gradients
    grad_price = tape.gradient(loss_price, agent.price_net.trainable_variables)
    opt_price.apply_gradients(zip(grad_price, agent.price_net.trainable_variables))

    grad_val = tape.gradient(loss_val, agent.value_net.trainable_variables)
    opt_val.apply_gradients(zip(grad_val, agent.value_net.trainable_variables))

    grad_pol = tape.gradient(loss_pol, agent.policy_model.trainable_variables)
    grad_pol, _ = tf.clip_by_global_norm(grad_pol, 0.5)
    opt_pol.apply_gradients(zip(grad_pol, agent.policy_model.trainable_variables))

    del tape
    return loss_price, loss_val, loss_pol, tf.reduce_mean(b_prime)


# 4. TRAINING LOOP
def train_risky_model():
    print("Starting Risky Debt Model Training...")
    hist_price = []
    hist_debt = []

    for epoch in range(30000):
        # Sample State (k, b, z)
        k = tf.random.uniform((64, 1), 20.0, 150.0)
        b = tf.random.uniform((64, 1), 0.0, 100.0) # Random current debt
        z = tf.exp(tf.random.normal((64, 1), 0.0, 0.2))

        l_p, l_v, l_pol, avg_debt = train_step(k, b, z)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: PriceLoss={l_p:.5f} | ValLoss={l_v:.5f} | Avg New Debt={avg_debt:.1f}")
            hist_price.append(l_p)
            hist_debt.append(avg_debt)

    return hist_debt

# 5. VALIDATION PLOTS (Economic Logic Check)
def plot_risky_logic(agent, config):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot A: Credit Spread Curve
    # Does Price q go down as Debt increases?
    b_grid = np.linspace(0, 200, 100)
    k_fixed = np.ones_like(b_grid) * 77.0 # At steady state capital
    z_fixed = np.ones_like(b_grid) * 1.0

    k_tens = tf.cast(k_fixed[:, None], tf.float32)
    b_tens = tf.cast(b_grid[:, None], tf.float32)
    z_tens = tf.cast(z_fixed[:, None], tf.float32)

    # Get Price q(k, b, z) directly
    q_preds = agent.get_price(k_tens, b_tens, z_tens)

    # Implied Interest Rate = (1/q) - 1
    implied_r = (1.0 / (q_preds + 1e-8)) - 1.0

    axes[0].plot(b_grid, implied_r, color='red', linewidth=2)
    axes[0].set_title("Supply of Funds: Interest Rate vs Debt Amount")
    axes[0].set_xlabel("Debt Issued (b')")
    axes[0].set_ylabel("Implied Interest Rate")
    axes[0].axhline(1/config.beta - 1, color='gray', linestyle=':', label="Risk Free")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot B: Policy (Leverage Choice)
    # Does firm choose positive debt?
    z_vals = [0.606, 1.013, 1.694]
    k_range = np.linspace(40, 120, 50)

    for z in z_vals:
        k_in = tf.cast(k_range[:, None], tf.float32)
        b_in = tf.zeros_like(k_in) # Assume starting with 0 debt
        z_in = tf.ones_like(k_in) * z

        _, b_prime = agent.get_policy(k_in, b_in, z_in)
        axes[1].plot(k_range, b_prime, label=f"z={z}")

    axes[1].set_title("Demand for Funds: Debt Choice")
    axes[1].set_xlabel("Current Capital")
    axes[1].set_ylabel("New Debt Chosen (b')")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./results/debt.pdf', dpi=300) 
    plt.show()


class Risky_VFI:
    def __init__(self, cfg):
        self.cfg = cfg
        self.nk = 40  
        self.nb = 25  
        self.nz = 11  
        
        self.k_grid = np.linspace(20.0, 150.0, self.nk)
        self.b_grid = np.linspace(0.0, 120.0, self.nb)
        
        log_z_std = cfg.sigma_eps / np.sqrt(1 - cfg.rho**2)
        self.z_grid = np.exp(np.linspace(-3 * log_z_std, 3 * log_z_std, self.nz))
 
        gh_nodes, gh_weights = np.polynomial.hermite.hermgauss(5)
        self.eps_nodes = gh_nodes * np.sqrt(2) * cfg.sigma_eps
        self.eps_weights = gh_weights / np.sqrt(np.pi)
 
    def solve(self, max_iter=1000, tol=1e-4):
        print("\n--- Solving with Value Function Iteration (VFI) ---")
        
        V = np.zeros((self.nz, self.nk, self.nb))
        q = np.full((self.nz, self.nk, self.nb), self.cfg.beta) # (k',b',z) -> q
        
        K_prime_grid = self.k_grid
        B_prime_grid = self.b_grid
        
        # 维度: (State_z, State_k, State_b, Choice_k', Choice_b')
        Z_s = self.z_grid[:, None, None, None, None]
        K_s = self.k_grid[None, :, None, None, None]
        B_s = self.b_grid[None, None, :, None, None]
        Kp_c = K_prime_grid[None, None, None, :, None]
        Bp_c = B_prime_grid[None, None, None, None, :]

        Inv = Kp_c - (1 - self.cfg.delta) * K_s
        Profit = (1-self.cfg.tau) * Z_s * (K_s ** self.cfg.theta)
        AdjCost = (self.cfg.psi_0 / 2.0) * (Inv**2) / (K_s + 1e-8)
        Dividend_part = Profit - Inv - AdjCost - B_s

        for i in range(max_iter):
            start_time = time.time()
            V_old = V.copy()
            q_old = q.copy()
            
            interp_V = RegularGridInterpolator((self.z_grid, self.k_grid, self.b_grid), V, bounds_error=False, fill_value=None)
            log_z = np.log(self.z_grid)
            z_futures = np.exp(self.cfg.rho * log_z[:, None] + self.eps_nodes[None, :]) # (nz, 5)
            
            # (nz, nk', nb')
            EV = np.zeros((self.nz, self.nk, self.nb))
            for iz in range(self.nz):
                zf = z_futures[iz, :] # (5,)
                # 为插值创建点 (k', b', z')
                mesh_k, mesh_b, mesh_z = np.meshgrid(K_prime_grid, B_prime_grid, zf, indexing='ij')
                points = np.stack([mesh_z.ravel(), mesh_k.ravel(), mesh_b.ravel()], axis=1)
                v_vals = interp_V(points).reshape(self.nk, self.nb, 5)
                # 计算期望 (nk', nb')
                EV[iz, :, :] = np.dot(v_vals, self.eps_weights)

            # q(k', b', z) = beta * E[Payoff(k',b',z')]
            interp_recov = lambda k, z: (1 - self.cfg.alpha) * ((1-self.cfg.tau)*z*k**self.cfg.theta + (1-self.cfg.delta)*k)
            
            v_next_expected = EV # E[V(k',b',z')]
            prob_survive = np.where(v_next_expected > 0, 1.0, 0.0) # V>0则存活


            recov_vals = interp_recov(K_prime_grid[None, :, None], z_futures[:, None, :]) # (nz, nk, 5)
            

            recov_vals_expanded = recov_vals[:, :, None, :]      # Shape (nz, nk, 1, 5)
            b_prime_expanded = B_prime_grid[None, None, :, None] # Shape (1, 1, nb, 1)
            recov_rates = np.clip(recov_vals_expanded / (b_prime_expanded + 1e-8), 0.0, 1.0) # Shape (nz, nk, nb, 5)
            # #############################################################

            E_rec_rate = np.dot(recov_rates, self.eps_weights) # (nz, nk, nb)

            E_payoff = prob_survive * 1.0 + (1 - prob_survive) * E_rec_rate
            q = self.cfg.beta * E_payoff

            q_expanded = q[:, None, None, :, :]
            NetBorrowing = q_expanded * Bp_c
            
            RHS = Dividend_part + NetBorrowing + self.cfg.beta * EV[:, None, None, :, :] # (nz, nk, nb, nk', nb')
            
            V = np.max(RHS, axis=(3, 4))
            pol_indices = np.argmax(RHS.reshape(self.nz, self.nk, self.nb, -1), axis=3)
            pol_idx_k, pol_idx_b = np.unravel_index(pol_indices, (self.nk, self.nb))

            v_diff = np.max(np.abs(V - V_old))
            q_diff = np.max(np.abs(q - q_old))
            
            elapsed = time.time() - start_time
            print(f"Iter {i}: V_diff={v_diff:.6f}, q_diff={q_diff:.6f} ({elapsed:.2f}s)")

            if v_diff < tol and q_diff < tol:
                print(f"VFI converged in {i+1} iterations.")
                break
        k_policy = RegularGridInterpolator((self.z_grid, self.k_grid, self.b_grid), self.k_grid[pol_idx_k], bounds_error=False, fill_value=None)
        b_policy = RegularGridInterpolator((self.z_grid, self.k_grid, self.b_grid), self.b_grid[pol_idx_b], bounds_error=False, fill_value=None)
        
        return k_policy, b_policy
def plot_policy_comparison(agent, vfi_k_policy, vfi_b_policy, config):
    print("\n--- Comparing DL Agent vs. VFI Ground Truth ---")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5)) 

    z_targets_idx = [0, len(vfi_solver.z_grid) // 2, len(vfi_solver.z_grid) - 1]
    z_targets = [vfi_solver.z_grid[i] for i in z_targets_idx]

    k_plot = np.linspace(30, 140, 50)
    
    b_initial = 0.0

    mape_scores_k = []

    for i, z_val in enumerate(z_targets):
        points = np.stack([np.full_like(k_plot, z_val), k_plot, np.full_like(k_plot, b_initial)], axis=1)
        k_vfi = vfi_k_policy(points)
        
        k_tens = tf.cast(k_plot[:, None], tf.float32)
        z_tens = tf.ones_like(k_tens) * z_val
        b_tens = tf.zeros_like(k_tens)
        
        k_dl_raw, _ = agent.get_policy(k_tens, b_tens, z_tens) 
        k_dl = k_dl_raw.numpy().flatten()
        
        mape_k = np.mean(np.abs((k_vfi - k_dl) / (k_vfi + 1e-8))) * 100
        mape_scores_k.append(mape_k)
        ax_k = axes[i]
        ax_k.plot(k_plot, k_vfi, color='orange', linestyle='--', linewidth=3, label='VFI (Ground Truth)')
        ax_k.plot(k_plot, k_dl, color='blue', linewidth=2, label=f'DL (MAPE={mape_k:.2f}%)')
        ax_k.plot(k_plot, k_plot, color='gray', linestyle=':', label="k'=k")
        ax_k.set_title(f"Next Capital k' (z={z_val:.3f})")
        ax_k.set_xlabel("Current Capital k")
        ax_k.set_ylabel("Next Capital k'")
        ax_k.legend()
        ax_k.grid(True, alpha=0.3)
        
    overall_mape_k = np.mean(mape_scores_k)
    print(f"Overall k' MAPE: {overall_mape_k:.4f}%")
    
    plt.tight_layout()
    plt.savefig('./results/policy_comparison_risky_k.pdf', dpi=300) 
    plt.show()


if __name__ == '__main__':
    tf.random.set_seed(0)
    np.random.seed(0)
    config = EconomicConfig()
    tf.keras.backend.clear_session()
    agent = RiskyAgent(config)

    # Distinct optimizers for stability
    opt_price = tf.keras.optimizers.Adam(learning_rate=0.0003)
    opt_val = tf.keras.optimizers.Adam(learning_rate=0.0003)
    opt_pol = tf.keras.optimizers.Adam(learning_rate=0.0001) # Slow policy learning

    # Run
    hist = train_risky_model()

    plot_risky_logic(agent, config)

     # --- VFI部分 ---
    vfi_solver = Risky_VFI(config)
    vfi_k_policy, vfi_b_policy = vfi_solver.solve()

    # --- 对比分析 ---
    plot_policy_comparison(agent, vfi_k_policy, vfi_b_policy, config)