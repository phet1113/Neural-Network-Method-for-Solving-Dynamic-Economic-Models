'''import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize
import matplotlib
from scipy.interpolate import RegularGridInterpolator
matplotlib.use('TkAgg')

# 深度学习求解部分 (来自abc)
class EconomicConfig:
    def __init__(self, beta=0.96, delta=0.15, theta=0.7, psi_0=0.01, rho=0.7, sigma_eps=0.15, tau=0.2):
        self.beta = beta
        self.delta = delta
        self.theta = theta
        self.psi_0 = psi_0
        self.rho = rho
        self.sigma_eps = sigma_eps
        self.tau = tau
        r_implied = (1.0 / self.beta) - 1.0
        self.k_ss = (self.theta / (r_implied + self.delta)) ** (1.0 / (1.0 - self.theta))
        self.val_ss = (self.k_ss ** self.theta) / (1 - self.beta)
        print(f"Steady-state capital: {self.k_ss:.2f}")

    def profit(self, k, z):
        return (1 - self.tau) * z * tf.pow(k, self.theta)

    def adjustment_cost(self, i, k):
        return (self.psi_0 / 2.0) * tf.square(i) / (k + 1e-8)

    def marginal_cost(self, i, k):
        return 1.0 + self.psi_0 * (i / (k + 1e-8))

    def marginal_benefit(self, k_next, k_next_next, z_next):
        inv_next = k_next_next - (1 - self.delta) * k_next
        mpk = self.theta * z_next * tf.pow(k_next + 1e-8, self.theta - 1.0)
        adj_savings = (self.psi_0 / 2.0) * tf.square(inv_next / (k_next + 1e-8))
        mc_next = self.marginal_cost(inv_next, k_next)
        return mpk + adj_savings + (1 - self.delta) * mc_next

class DeepAgent(tf.keras.Model):
    #深度学习代理模型 (来自abc)

    def __init__(self, cfg):
        super(DeepAgent, self).__init__()
        self.cfg = cfg
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
        k_norm = k / self.cfg.k_ss
        z_norm = tf.math.log(z)
        inp = tf.concat([k_norm, z_norm], axis=1)
        ratio = self.policy_net(inp)
        k_prime = ratio * self.cfg.k_ss
        return tf.clip_by_value(k_prime, 5.0, 450.0)


class DeepSolver:
#深度学习求解器 (封装abc部分)
    def __init__(self, config):
        self.config = config
        self.agent = DeepAgent(config)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)

    @tf.function
    def train_step(self, k_batch, z_batch):
        """训练步骤 (使用AiO方法)"""
        N = tf.shape(k_batch)[0]

        eps_1 = tf.random.normal((N, 1))
        eps_2 = tf.random.normal((N, 1))

        log_z = tf.math.log(z_batch)
        z_next_1 = tf.exp(self.config.rho * log_z + self.config.sigma_eps * eps_1)
        z_next_2 = tf.exp(self.config.rho * log_z + self.config.sigma_eps * eps_2)

        with tf.GradientTape() as tape:
            k_next = self.agent(tf.concat([k_batch, z_batch], axis=1))
            inv_curr = k_next - (1 - self.config.delta) * k_batch
            MC_t = self.config.marginal_cost(inv_curr, k_batch)

            k_next_next_1 = self.agent(tf.concat([k_next, z_next_1], axis=1))
            k_next_next_2 = self.agent(tf.concat([k_next, z_next_2], axis=1))
            MB_1 = self.config.marginal_benefit(k_next, k_next_next_1, z_next_1)
            MB_2 = self.config.marginal_benefit(k_next, k_next_next_2, z_next_2)
            err_1 = MC_t - self.config.beta * MB_1
            err_2 = MC_t - self.config.beta * MB_2
            loss = tf.reduce_mean(err_1 * err_2)

        grads = tape.gradient(loss, self.agent.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.agent.trainable_variables))
        return loss

    def train(self, epochs=30000):
        """训练深度学习代理"""
        print("Training Deep Agent...")
        loss_hist = []

        for epoch in range(epochs):

            k = tf.random.uniform((64, 1), 20.0, 150.0)
            z = tf.exp(tf.random.normal((64, 1), 0.0, 0.2))

            loss = self.train_step(k, z)

            if epoch % 2000 == 0:
                print(f"Epoch {epoch}: Euler Loss={np.abs(loss.numpy()):.6f}")
                loss_hist.append(loss.numpy())

        return loss_hist, self.agent


# 数据生成部分 (整合深度学习代理)
class DeepDataGenerator:
    # 基于深度学习代理的数据生成器

    def __init__(self, deep_agent):
        self.deep_agent = deep_agent  # 传入训练好的深度学习代理

    def generate_data(self, config, num_firms, num_periods, burn_in=50):
        #使用深度学习代理生成模拟数据
        # 初始化状态
        k = tf.ones((num_firms, 1)) * config.k_ss
        z = tf.exp(tf.random.normal((num_firms, 1), 0.0, config.sigma_eps))

        # 存储数据
        data = {
            'k': np.zeros((num_firms, num_periods)),
            'z': np.zeros((num_firms, num_periods)),
            'investment': np.zeros((num_firms, num_periods)),
            'profit': np.zeros((num_firms, num_periods)) }

        # 预烧期（burn-in）使数据达到稳态
        for _ in range(burn_in):
            # 使用深度学习代理预测下一期资本
            states = tf.concat([k, z], axis=1)
            k_next = self.deep_agent(states)

            # 计算投资
            investment = k_next - (1 - config.delta) * k

            # 更新冲击
            log_z = tf.math.log(z)
            eps = tf.random.normal(z.shape, 0.0, config.sigma_eps)
            z_next = tf.exp(config.rho * log_z + eps)

            # 更新状态
            k = k_next
            z = z_next

        # 主模拟期
        for t in range(num_periods):
            # 保存当前状态
            data['k'][:, t] = k.numpy().flatten()
            data['z'][:, t] = z.numpy().flatten()
            data['profit'][:, t] = config.profit(k, z).numpy().flatten()

            # 使用深度学习代理预测下一期资本
            states = tf.concat([k, z], axis=1)
            k_next = self.deep_agent(states)

            # 计算投资
            investment = k_next - (1 - config.delta) * k
            data['investment'][:, t] = investment.numpy().flatten()

            # 更新冲击
            log_z = tf.math.log(z)
            eps = tf.random.normal(z.shape, 0.0, config.sigma_eps)
            z_next = tf.exp(config.rho * log_z + eps)

            # 更新状态
            k = k_next
            z = z_next

        # 计算投资率
        data['investment_rate'] = data['investment'] / (data['k'] + 1e-8)
        return data


# 结构估计部分 (GMM/SMM)
class GMM:
    def __init__(self, config):
        self.config = config

    def euler(self, beta, delta, theta, psi_0, k_t, z_t, inv_t, k_t1, z_t1, inv_t1):
        # 欧拉方程残差计算
        # 当前期边际成本
        mc_t = 1.0 + psi_0 * (inv_t / (k_t + 1e-8))

        # 下期资本
        k_t2 = k_t1 * (1 - delta) + inv_t1

        # 下期边际产品
        mpk_t1 = theta * z_t1 * tf.pow(k_t1 + 1e-8, theta - 1)

        # 下期边际成本
        mc_t1 = 1.0 + psi_0 * (inv_t1 / (k_t1 + 1e-8))

        # 下期调整成本节省
        adj_savings = (psi_0 / 2.0) * tf.square(inv_t1) / tf.square(k_t1 + 1e-8)

        # 下期边际收益
        mb_t1 = mpk_t1 + adj_savings + (1 - delta) * mc_t1

        # 欧拉残差
        residual = mc_t - beta * mb_t1
        return residual

    def objective(self, params, data):
        # GMM目标函数
        # 解包参数
        beta, delta, theta, psi_0 = params[0], params[1], params[2], params[3]

        # 提取数据 - 使用t和t+1期数据
        k_t = tf.constant(data['k'][:, :-1], dtype=tf.float32)  # t期
        z_t = tf.constant(data['z'][:, :-1], dtype=tf.float32)
        inv_t = tf.constant(data['investment'][:, :-1], dtype=tf.float32)

        k_t1 = tf.constant(data['k'][:, 1:], dtype=tf.float32)  # t+1期
        z_t1 = tf.constant(data['z'][:, 1:], dtype=tf.float32)
        inv_t1 = tf.constant(data['investment'][:, 1:], dtype=tf.float32)

        # 计算所有残差
        residuals = self.euler(
            beta, delta, theta, psi_0,
            k_t, z_t, inv_t,
            k_t1, z_t1, inv_t1)

        # 创建工具变量（常数、资本、冲击、投资率）
        instruments = tf.stack([
            tf.ones_like(k_t),
            k_t,
            z_t,
            inv_t / (k_t + 1e-8)  # 投资率
        ], axis=2)

        # 计算矩条件 (N x T x 4)
        moments = instruments * tf.expand_dims(residuals, axis=2)

        # 平均矩条件 (4,)
        avg_moments = tf.reduce_mean(moments, axis=[0, 1])

        # GMM目标函数（二次型） - 使用单位权重矩阵
        W = tf.eye(4)
        return tf.tensordot(tf.tensordot(avg_moments, W, axes=1), avg_moments, axes=1)

    def estimate(self, data, initial_params):
        # 转换为TensorFlow变量
        params = tf.Variable(initial_params, dtype=tf.float32)

        # 优化器
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

        # 跟踪损失历史
        loss_history = []
        param_history = []

        for step in range(500):
            with tf.GradientTape() as tape:
                # 计算损失
                loss = self.objective(params, data)

            # 计算梯度
            grads = tape.gradient(loss, params)

            if grads is not None:
                optimizer.apply_gradients([(grads, params)])

            # 记录损失和参数
            loss_history.append(loss.numpy())
            current_params = params.numpy().copy()
            param_history.append(current_params)

            if step % 10 == 0:
                print(f"Step {step}: Loss={loss.numpy():.6f}, Params={current_params}")

            # 提前停止条件
            if step > 10 and len(loss_history) > 1 and np.abs(loss_history[-1] - loss_history[-2]) < 1e-8:
                print(f"Early stopping at step {step}")
                break

        return params.numpy(), loss_history, param_history


class SMM:
    def __init__(self, config, data_generator):
        self.config = config
        self.data_generator = data_generator
        self.target_moments = None
        self.loss_history = []
        self.param_history = []

    # 计算关键矩
    def compute_moments(self, data):
        # 投资率
        inv_rate = data['investment'] / (data['k'] + 1e-8)

        # 资本增长率
        k_growth = np.zeros_like(data['k'])
        k_growth[:, 1:] = (data['k'][:, 1:] - data['k'][:, :-1]) / (data['k'][:, :-1] + 1e-8)

        # 矩计算
        moments = np.array([
            np.mean(inv_rate),  # 平均投资率
            np.std(inv_rate),  # 投资率波动
            np.mean(k_growth[:, 1:]),  # 平均资本增长率
            np.std(k_growth[:, 1:]),  # 资本增长率波动
            np.mean(data['profit']),  # 平均利润
            np.std(data['profit']),  # 利润波动
            np.mean(data['z']),  # 平均冲击
            np.std(data['z']),  # 冲击波动
        ])

        return moments

    def set_target_moments(self, data):
        # 设置目标矩（真实数据矩）
        self.target_moments = self.compute_moments(data)
        print(f"Target Moments: {self.target_moments}")

    # SMM目标函数
    def objective(self, params):
        # 更新模型参数
        beta, delta, theta, psi_0 = params
        sim_config = EconomicConfig(
            beta=beta,
            delta=delta,
            theta=theta,
            psi_0=psi_0,
            rho=self.config.rho,
            sigma_eps=self.config.sigma_eps,
            tau=self.config.tau
        )

        # 生成模拟数据
        sim_data = self.data_generator.generate_data(
            config=sim_config,
            num_firms=100,
            num_periods=50
        )

        # 计算模拟矩
        sim_moments = self.compute_moments(sim_data)

        # 计算矩差异
        moment_diff = sim_moments - self.target_moments

        # SMM目标函数（加权二次型）
        weights = 1.0 / (self.target_moments ** 2 + 1e-8)
        loss = np.sum(weights * moment_diff ** 2)

        # 记录参数和损失
        self.param_history.append(params.copy())
        self.loss_history.append(loss)

        if len(self.loss_history) % 10 == 0:
            print(f"Loss={loss:.6f}, Params={params}")

        return loss

    def estimate(self, initial_params):
        # 重置历史记录
        self.loss_history = []
        self.param_history = []

        # 设置参数边界
        bounds = [
            (0.9, 0.99),  # beta
            (0.05, 0.25),  # delta
            (0.5, 0.9),  # theta
            (0.001, 0.1)  # psi_0
               ]

        # 使用Scipy的minimize函数
        result = minimize(
            fun=self.objective,
            x0=initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 50, 'disp': True}
        )

        if not result.success:
            print(f"Optimization failed: {result.message}")

        return result.x, self.loss_history, np.array(self.param_history)


# 整合主函数
def i():
    # 设置随机种子
    tf.random.set_seed(42)
    np.random.seed(42)

    # 真实参数
    true_params = np.array([0.96, 0.15, 0.7, 0.01])
    print("True Parameters:", true_params)

    # 创建真实配置
    true_config = EconomicConfig(
        beta=true_params[0],
        delta=true_params[1],
        theta=true_params[2],
        psi_0=true_params[3]
    )

    # 1. 深度学习求解模型
    print("\nSTEP 1: Deep Learning Solution")
    deep_solver = DeepSolver(true_config)
    _, deep_agent = deep_solver.train(epochs=10000)

    # 2. 使用深度学习代理生成数据
    print("\nSTEP 2: Data Generation with Deep Agent")
    deep_data_gen = DeepDataGenerator(deep_agent)
    real_data = deep_data_gen.generate_data(
        config=true_config,
        num_firms=200,
        num_periods=50
    )

    # 3. 结构估计
    print("\nSTEP 3: Structural Estimation")

    # GMM估计
    print("\nGMM 估计")
    gmm_estimator = GMM(true_config)
    gmm_initial = np.array([0.95, 0.16, 0.69, 0.011])
    gmm_params, gmm_loss_history, gmm_param_history = gmm_estimator.estimate(
        data=real_data,
        initial_params=gmm_initial
    )

    # SMM估计
    print("\nSMM 估计")
    smm_estimator = SMM(true_config, deep_data_gen)
    smm_estimator.set_target_moments(real_data)
    smm_initial = np.array([0.95, 0.16, 0.69, 0.011])
    smm_params, smm_loss_history, smm_param_history = smm_estimator.estimate(
        initial_params=smm_initial
    )

    # 结果分析
    print("\n最终结果：")
    print(f"True Parameters: {true_params}")
    print(f"GMM Estimated: {gmm_params}")
    print(f"SMM Estimated: {smm_params}")

    # 科室化
    plot(true_params, gmm_params, smm_params,
                 gmm_loss_history, smm_loss_history,
                 gmm_param_history, smm_param_history)


def plot(true_params, gmm_params, smm_params,
                 gmm_loss_hist, smm_loss_hist,
                 gmm_param_hist, smm_param_hist):
    plt.figure(figsize=(15, 10))

    # 定义配色方案
    warm_colors = ['#FF7F0E', '#D62728', '#FFD700', '#8C564B']
    method_colors = ['#8B0000', '#FF8C00', '#FFD700']
    loss_colors = ['#1F77B4', '#FFD700']

    param_names = [r'$\beta$', r'$\delta$', r'$\theta$', r'$\psi_0$']
    x = np.arange(len(param_names))

    # 1. 参数比较图
    plt.subplot(2, 2, 1)
    plt.bar(x - 0.2, true_params, width=0.2, color=method_colors[0],
            edgecolor='k', alpha=0.9, label='True')
    plt.bar(x, gmm_params, width=0.2, color=method_colors[1],
            edgecolor='k', alpha=0.9, label='GMM')
    plt.bar(x + 0.2, smm_params, width=0.2, color=method_colors[2],
            edgecolor='k', alpha=0.9, label='SMM')

    plt.xticks(x, param_names)
    plt.ylabel('Parameter Value')
    plt.title('Parameter Estimation Comparison')
    plt.legend(framealpha=0.9)
    plt.grid(alpha=0.2, linestyle='--')


    # 2. 损失函数变化图
    plt.subplot(2, 2, 2)
    plt.plot(gmm_loss_hist, color=loss_colors[0], linewidth=2.5, label='GMM Loss')  # 蓝色
    plt.plot(smm_loss_hist, color=loss_colors[1], linewidth=2.5, label='SMM Loss')  # 黄色
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Optimization Progress')
    plt.yscale('log')
    plt.legend(framealpha=0.9)
    plt.grid(alpha=0.2, linestyle='--')

    # 3. GMM参数收敛路径
    plt.subplot(2, 2, 3)
    gmm_param_hist = np.array(gmm_param_hist)
    for i, name in enumerate(param_names):
        plt.plot(gmm_param_hist[:, i], color=warm_colors[i],
                 linewidth=2, label=name, marker='o', markersize=4)

    for true_val in true_params:
        plt.axhline(y=true_val, color='black', linestyle='--', alpha=0.7, linewidth=1.0)

    plt.xlabel('Iteration')
    plt.ylabel('Parameter Value')
    plt.title('GMM Parameter Convergence')
    plt.legend(framealpha=0.9)
    plt.grid(alpha=0.2, linestyle='--')


    # 4. SMM参数收敛路径
    plt.subplot(2, 2, 4)
    if len(smm_param_hist) > 0:
        smm_param_hist = np.array(smm_param_hist)
        for i, name in enumerate(param_names):
            plt.plot(smm_param_hist[:, i], color=warm_colors[i],
                     linewidth=2, label=name, marker='s', markersize=4)

        # 真实参数参考线
        for true_val in true_params:
            plt.axhline(y=true_val, color='black', linestyle='--', alpha=0.7, linewidth=1.0)

        plt.xlabel('Iteration')
        plt.ylabel('Parameter Value')
        plt.title('SMM Parameter Convergence')
        plt.legend(framealpha=0.9)
        plt.grid(alpha=0.2, linestyle='--')

    plt.tight_layout()
    plt.savefig('./results/111.pdf', dpi=300, facecolor='#FFFAF0')  # 浅橙色背景
    plt.show()


if __name__ == '__main__':
    i()'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize
import matplotlib
from scipy.interpolate import RegularGridInterpolator

matplotlib.use('Agg')


# 扩展模型配置  包含债务和破产成本
class EconomicConfig:
    def __init__(self, beta=0.96, delta=0.15, theta=0.7, psi_0=0.01, rho=0.7,
                 sigma_eps=0.30, tau=0.2, alpha=0.50):
        self.beta = beta
        self.delta = delta
        self.theta = theta
        self.psi_0 = psi_0
        self.rho = rho
        self.sigma_eps = sigma_eps
        self.tau = tau
        self.alpha = alpha  # 破产成本系数

        r_implied = (1.0 / self.beta) - 1.0
        self.k_ss = (self.theta / (r_implied + self.delta)) ** (1.0 / (1.0 - self.theta))
        self.val_ss = (self.k_ss ** self.theta) / (1 - self.beta)
        print(f"Steady-state capital: {self.k_ss:.2f}")

    def profit(self, k, z):
        return (1 - self.tau) * z * tf.pow(k, self.theta)

    def adjustment_cost(self, i, k):
        return (self.psi_0 / 2.0) * tf.square(i) / (k + 1e-8)

    def recovery_value(self, k, z):
        # 清算价值
        val = (1 - self.tau) * self.profit(k, z) + (1 - self.delta) * k
        return (1 - self.alpha) * val


# 扩展代理模型
class RiskyAgent(tf.keras.Model):
    def __init__(self, cfg):
        super(RiskyAgent, self).__init__()
        self.cfg = cfg

        # 价格网络 (预测债券价格)
        self.price_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(3,)),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # 价值网络 (预测企业价值)
        self.value_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(3,)),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(1, activation='linear')
        ])

        # 策略网络 (决策资本和债务)
        input_layer = tf.keras.layers.Input(shape=(3,))
        h1 = tf.keras.layers.Dense(64, activation='relu')(input_layer)
        h2 = tf.keras.layers.Dense(64, activation='relu')(h1)

        # 资本决策头
        k_out = tf.keras.layers.Dense(1, activation='softplus',
                                      bias_initializer=tf.keras.initializers.Constant(0.55),
                                      name='k_head')(h2)

        # 债务决策头
        b_out = tf.keras.layers.Dense(1, activation='sigmoid',
                                      bias_initializer=tf.keras.initializers.Constant(-3.0),
                                      name='b_head')(h2)

        self.policy_model = tf.keras.Model(inputs=input_layer, outputs=[k_out, b_out])

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
        return q * self.cfg.beta

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

        k_ratio, b_ratio = self.policy_model(inp)

        # 反标准化
        k_prime = k_ratio * self.cfg.k_ss
        k_prime = tf.clip_by_value(k_prime, 5.0, 450.0)

        # 债务范围 [0, 300]
        b_prime = b_ratio * 300.0

        return k_prime, b_prime


# 扩展求解器
class RiskySolver:
    def __init__(self, config):
        self.config = config
        self.agent = RiskyAgent(config)

        # 分离优化器
        self.opt_price = tf.keras.optimizers.Adam(learning_rate=0.0003)
        self.opt_val = tf.keras.optimizers.Adam(learning_rate=0.0003)
        self.opt_pol = tf.keras.optimizers.Adam(learning_rate=0.0001)

    @tf.function
    def train_step(self, k, b, z):
        N = tf.shape(k)[0]
        eps = tf.random.normal((N, 1))
        log_z = tf.math.log(z)
        z_next = tf.exp(self.config.rho * log_z + self.config.sigma_eps * eps)

        # 随机债务用于训练价格网络
        b_hyp = tf.random.uniform((N, 1), 10.0, 250.0)

        with tf.GradientTape(persistent=True) as tape:
            # 1. 策略决策
            k_prime, b_prime = self.agent.get_policy(k, b, z)

            # 2. 价格网络训练
            q_hyp = self.agent.get_price(k_prime, b_hyp, z)

            # 计算违约概率
            v_next_hyp = self.agent.get_value(k_prime, b_hyp, z_next)
            survive_hyp = tf.sigmoid(v_next_hyp * 10.0)

            # 计算回收率
            rec_val = self.config.recovery_value(k_prime, z_next)
            rec_rate = rec_val / (b_hyp + 1e-8)
            rec_rate = tf.clip_by_value(rec_rate, 0.0, 1.0)

            payoff_hyp = survive_hyp * 1.0 + (1 - survive_hyp) * rec_rate
            loss_price = tf.reduce_mean(tf.square(q_hyp - self.config.beta * payoff_hyp))

            # 3. 价值网络训练
            q_actual = self.agent.get_price(k_prime, b_prime, z)

            inv = k_prime - (1 - self.config.delta) * k
            prof = self.config.profit(k, z)
            adj = self.config.adjustment_cost(inv, k)

            # 净借款
            net_borrowing = q_actual * b_prime - b
            dividend = prof - adj - inv + net_borrowing

            v_next_real = self.agent.get_value(k_prime, b_prime, z_next)
            v_next_lim = tf.nn.relu(v_next_real)

            target_v = tf.stop_gradient(dividend + self.config.beta * v_next_lim)
            v_curr = self.agent.get_value(k, b, z)
            loss_val = tf.reduce_mean(tf.square((v_curr - target_v) / self.config.val_ss))

            # 4. 策略网络训练
            agency_cost = 0.002 * tf.reduce_mean(b_prime / self.config.k_ss)
            total_wealth = dividend + self.config.beta * v_next_lim
            loss_pol = -tf.reduce_mean(total_wealth / self.config.val_ss) + agency_cost

        # 更新价格网络
        grad_price = tape.gradient(loss_price, self.agent.price_net.trainable_variables)
        self.opt_price.apply_gradients(zip(grad_price, self.agent.price_net.trainable_variables))

        # 更新价值网络
        grad_val = tape.gradient(loss_val, self.agent.value_net.trainable_variables)
        self.opt_val.apply_gradients(zip(grad_val, self.agent.value_net.trainable_variables))

        # 更新策略网络
        grad_pol = tape.gradient(loss_pol, self.agent.policy_model.trainable_variables)
        grad_pol, _ = tf.clip_by_global_norm(grad_pol, 0.5)
        self.opt_pol.apply_gradients(zip(grad_pol, self.agent.policy_model.trainable_variables))

        del tape
        return loss_price, loss_val, loss_pol, tf.reduce_mean(b_prime)

    def train(self, epochs=30000):
        print("Training Risky Agent...")
        loss_hist = []
        debt_hist = []

        for epoch in range(epochs):
            k = tf.random.uniform((64, 1), 20.0, 150.0)
            b = tf.random.uniform((64, 1), 0.0, 100.0)
            z = tf.exp(tf.random.normal((64, 1), 0.0, 0.2))

            l_p, l_v, l_pol, avg_debt = self.train_step(k, b, z)

            if epoch % 2000 == 0:
                print(f"Epoch {epoch}: PriceLoss={l_p:.5f} | ValLoss={l_v:.5f} | AvgDebt={avg_debt:.1f}")
                loss_hist.append([l_p, l_v, l_pol])
                debt_hist.append(avg_debt)

        return loss_hist, debt_hist, self.agent


# 数据生成器
class RiskyDataGenerator:
    def __init__(self, risky_agent):
        self.risky_agent = risky_agent

    def generate_data(self, config, num_firms, num_periods, burn_in=50):
        k = tf.ones((num_firms, 1)) * config.k_ss
        b = tf.zeros((num_firms, 1))
        z = tf.exp(tf.random.normal((num_firms, 1), 0.0, config.sigma_eps))

        data = {
            'k': np.zeros((num_firms, num_periods)),
            'b': np.zeros((num_firms, num_periods)),
            'z': np.zeros((num_firms, num_periods)),
            'investment': np.zeros((num_firms, num_periods)),
            'profit': np.zeros((num_firms, num_periods)),
            'debt_issuance': np.zeros((num_firms, num_periods)),
            'bond_price': np.zeros((num_firms, num_periods))
        }

        # 预烧期
        for _ in range(burn_in):
            states = tf.concat([k, b, z], axis=1)
            _, k_next, b_next, q = self.risky_agent(states)

            log_z = tf.math.log(z)
            eps = tf.random.normal(z.shape, 0.0, config.sigma_eps)
            z_next = tf.exp(config.rho * log_z + eps)

            k = k_next
            b = b_next
            z = z_next

        # 主模拟期
        for t in range(num_periods):
            # 保存当前状态
            data['k'][:, t] = k.numpy().flatten()
            data['b'][:, t] = b.numpy().flatten()
            data['z'][:, t] = z.numpy().flatten()
            data['profit'][:, t] = config.profit(k, z).numpy().flatten()

            # 获取下一期决策
            states = tf.concat([k, b, z], axis=1)
            _, k_next, b_next, q = self.risky_agent(states)

            # 计算投资和债务发行
            investment = k_next - (1-config.delta) * k
            data['investment'][:, t] = investment.numpy().flatten()
            data['debt_issuance'][:, t] = b_next.numpy().flatten()
            data['bond_price'][:, t] = q.numpy().flatten()

            # 更新冲击
            log_z = tf.math.log(z)
            eps = tf.random.normal(z.shape, 0.0, config.sigma_eps)
            z_next = tf.exp(config.rho * log_z + eps)

            # 更新状态
            k = k_next
            b = b_next
            z = z_next

        # 计算额外指标
        data['investment_rate'] = data['investment'] / (data['k'] + 1e-8)
        data['leverage'] = data['b'] / (data['k'] + 1e-8)
        data['default_prob'] = np.where(data['bond_price'] < 0.9 * config.beta, 1, 0)

        return data


# SMM估计器
class SMM:
    def __init__(self, config, data_generator):
        self.config = config
        self.data_generator = data_generator
        self.target_moments = None
        self.loss_history = []
        self.param_history = []

    def compute_moments(self, data):
        # 投资率
        inv_rate = data['investment'] / (data['k'] + 1e-8)

        # 资本增长率
        k_growth = np.zeros_like(data['k'])
        k_growth[:, 1:] = (data['k'][:, 1:] - data['k'][:, :-1]) / (data['k'][:, :-1] + 1e-8)

        # 杠杆率
        leverage = data['leverage']

        # 违约概率
        default_prob = data['default_prob']

        # 扩展矩计算
        moments = np.array([
            np.mean(inv_rate),  # 平均投资率
            np.std(inv_rate),  # 投资率波动
            np.mean(k_growth[:, 1:]),  # 平均资本增长率
            np.std(k_growth[:, 1:]),  # 资本增长率波动
            np.mean(data['profit']),  # 平均利润
            np.std(data['profit']),  # 利润波动
            np.mean(data['z']),  # 平均冲击
            np.std(data['z']),  # 冲击波动
            np.mean(leverage),  # 平均杠杆率
            np.std(leverage),  # 杠杆率波动
            np.mean(default_prob)  # 平均违约概率
        ])

        return moments

    def set_target_moments(self, data):
        self.target_moments = self.compute_moments(data)
        print(f"Target Moments: {self.target_moments}")

    def objective(self, params):
        # 更新模型参数
        beta, delta, theta, psi_0, alpha = params
        sim_config = EconomicConfig(
            beta=beta,
            delta=delta,
            theta=theta,
            psi_0=psi_0,
            alpha=alpha,
            rho=self.config.rho,
            sigma_eps=self.config.sigma_eps,
            tau=self.config.tau
        )

        # 生成模拟数据
        sim_data = self.data_generator.generate_data(
            config=sim_config,
            num_firms=100,
            num_periods=50
        )

        # 计算模拟矩
        sim_moments = self.compute_moments(sim_data)

        # 计算矩差异
        moment_diff = sim_moments - self.target_moments

        # SMM目标函数
        weights = 1.0 / (self.target_moments ** 2 + 1e-8)
        loss = np.sum(weights * moment_diff ** 2)

        # 记录参数和损失
        self.param_history.append(params.copy())
        self.loss_history.append(loss)

        if len(self.loss_history) % 10 == 0:
            print(f"Loss={loss:.6f}, Params={params}")

        return loss

    def estimate(self, initial_params):
        self.loss_history = []
        self.param_history = []

        # 设置参数边界
        bounds = [
            (0.9, 0.99),  # beta
            (0.05, 0.25),  # delta
            (0.5, 0.9),  # theta
            (0.001, 0.1),  # psi_0
            (0.1, 0.9)  # alpha
        ]

        # 使用Scipy优化
        result = minimize(
            fun=self.objective,
            x0=initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 50, 'disp': True}
        )

        if not result.success:
            print(f"Optimization failed: {result.message}")

        return result.x, self.loss_history, np.array(self.param_history)


def main():
    tf.random.set_seed(42)
    np.random.seed(42)

    # 真实参数
    true_params = np.array([0.96, 0.15, 0.7, 0.01, 0.5])
    print("True Parameters:", true_params)

    # 创建真实配置
    true_config = EconomicConfig(
        beta=true_params[0],
        delta=true_params[1],
        theta=true_params[2],
        psi_0=true_params[3],
        alpha=true_params[4]
    )

    # 1. 训练扩展模型
    print("\nSTEP 1: Training Risky Model")
    risky_solver = RiskySolver(true_config)
    _, _, risky_agent = risky_solver.train(epochs=10000)

    # 2. 生成数据
    print("\nSTEP 2: Data Generation")
    risky_data_gen = RiskyDataGenerator(risky_agent)
    real_data = risky_data_gen.generate_data(
        config=true_config,
        num_firms=200,
        num_periods=50
    )

    # 3. SMM估计
    print("\nSTEP 3: SMM Estimation")
    smm_estimator = SMM(true_config, risky_data_gen)
    smm_estimator.set_target_moments(real_data)
    smm_initial = np.array([0.95, 0.16, 0.69, 0.011, 0.4])
    smm_params, smm_loss_history, smm_param_history = smm_estimator.estimate(
        initial_params=smm_initial
    )

    # 结果分析
    print("\nFinal Results:")
    print(f"True Parameters: {true_params}")
    print(f"SMM Estimated: {smm_params}")

    # 可视化结果
    plot_results(true_params, smm_params, smm_loss_history, smm_param_history)


def plot_results(true_params, smm_params, loss_history, param_history):
    plt.figure(figsize=(15, 10))

    # 参数名称
    param_names = [r'$\beta$', r'$\delta$', r'$\theta$', r'$\psi_0$', r'$\alpha$']
    x = np.arange(len(param_names))

    # 1. 参数比较
    plt.subplot(2, 2, 1)
    plt.bar(x - 0.1, true_params, width=0.2, color='#FF7F0E', label='True')
    plt.bar(x + 0.1, smm_params, width=0.2, color='#1F77B4', label='SMM')
    plt.xticks(x, param_names)
    plt.ylabel('Parameter Value')
    plt.title('Parameter Estimation')
    plt.legend()
    plt.grid(alpha=0.2)

    # 2. 损失函数
    plt.subplot(2, 2, 2)
    plt.plot(loss_history, color='#D62728')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Optimization Progress')
    plt.yscale('log')
    plt.grid(alpha=0.2)

    # 3. 参数收敛路径
    plt.subplot(2, 2, 3)
    param_history = np.array(param_history)
    colors = ['#FF7F0E', '#1F77B4', '#2CA02C', '#D62728', '#9467BD']

    for i in range(len(param_names)):
        plt.plot(param_history[:, i], color=colors[i], label=param_names[i])

    plt.xlabel('Iteration')
    plt.ylabel('Parameter Value')
    plt.title('Parameter Convergence')
    plt.legend()
    plt.grid(alpha=0.2)

    # 4. 参数误差
    plt.subplot(2, 2, 4)
    errors = np.abs(param_history - true_params) / true_params
    for i in range(len(param_names)):
        plt.plot(errors[:, i], color=colors[i], label=param_names[i])

    plt.xlabel('Iteration')
    plt.ylabel('Relative Error')
    plt.title('Parameter Estimation Error')
    plt.yscale('log')
    plt.legend()
    plt.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig('./results/222.pdf', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
