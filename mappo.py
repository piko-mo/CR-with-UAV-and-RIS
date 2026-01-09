import tensorflow as tf
import numpy as np
import os

class LinearArrayPhase:
    """
    线性阵列相位参数化
    θᵢ = θ₀ + i × Δθ
    """

    def __init__(self, N):
        self.N = N

    @property
    def param_dim(self):
        return 2

    def params_to_phases(self, base_phase, phase_diff):
        indices = np.arange(self.N)
        phases = base_phase + indices * phase_diff
        return np.mod(phases, 2 * np.pi)


class FourierPhase:
    """傅里叶基相位参数化"""

    def __init__(self, N, K=2):
        self.N = N
        self.K = K
        indices = np.arange(N)
        self.cos_basis = np.array([np.cos(2*np.pi*k*indices/N) for k in range(1, K+1)])
        self.sin_basis = np.array([np.sin(2*np.pi*k*indices/N) for k in range(1, K+1)])

    @property
    def param_dim(self):
        return 2 * self.K + 1

    def params_to_phases(self, params):
        dc = params[0]
        a_coeffs = params[1:self.K+1]
        b_coeffs = params[self.K+1:]
        phases = dc * np.ones(self.N)
        phases += np.dot(a_coeffs, self.cos_basis)
        phases += np.dot(b_coeffs, self.sin_basis)
        return np.mod(phases * np.pi + np.pi, 2 * np.pi)

def trig_to_phase(cos_val, sin_val):
    """三角函数编码转相位 [0, 2π]"""
    return (np.arctan2(sin_val, cos_val) + 2 * np.pi) % (2 * np.pi)

class LogStdLayer(tf.keras.layers.Layer):
    """可学习的对数标准差"""

    def __init__(self, action_dim, initial_value=-0.7, min_log_std=-2.0, max_log_std=-0.3, **kwargs):
        super().__init__(**kwargs)
        self.action_dim = action_dim
        self.initial_value = initial_value
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def build(self, input_shape):
        self.log_std = self.add_weight(
            name='log_std',
            shape=(self.action_dim,),
            initializer=tf.keras.initializers.Constant(self.initial_value),
            trainable=True
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        clipped = tf.clip_by_value(self.log_std, self.min_log_std, self.max_log_std)
        return inputs, tf.tile(tf.expand_dims(clipped, 0), [batch_size, 1])

class RolloutBuffer:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.clear()

    def add(self, state, raw_actions, scaled_action, reward, value, log_probs, done):
        self.states.append(state)
        self.raw_actions.append(raw_actions)
        self.scaled_actions.append(scaled_action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_probs)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.raw_actions = []
        self.scaled_actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def get_batch(self):
        return (
            np.array(self.states, dtype=np.float32),
            np.array(self.raw_actions, dtype=np.float32),
            np.array(self.scaled_actions, dtype=np.float32),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.values, dtype=np.float32),
            np.array(self.log_probs, dtype=np.float32),
            np.array(self.dones, dtype=np.float32)
        )

    def size(self):
        return len(self.states)


class MAPPO_Reduced:
    """
    动作空间设计 (所有维度统一用tanh到[-1,1]):
    - height_raw ∈ [-1, 1] -> height ∈ [1, H_max]
    - phase_cos ∈ [-1, 1], phase_sin ∈ [-1, 1] -> 通过atan2得到相位
    - power_raw ∈ [-1, 1] -> power ∈ [P_min, P_max]
    """

    def __init__(self, env, num_agents=3, max_episodes=500, max_steps=300,
                 phase_mode='linear', fourier_k=2):

        # ========== 超参数 ==========
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_ratio = 0.2
        self.actor_lr = 5e-5
        self.critic_lr = 1e-4
        self.update_epochs = 5
        self.mini_batch_size = 512
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.5
        self.max_grad_norm = 0.5

        self.max_episodes = max_episodes
        self.max_steps = max_steps

        self.env = env
        self.num_agents = num_agents
        self.state_dim = env.n_features

        # ========== 相位参数化 ==========
        self.phase_mode = phase_mode
        if phase_mode == 'linear':
            self.phase_param = LinearArrayPhase(env.N)
        elif phase_mode == 'fourier':
            self.phase_param = FourierPhase(env.N, K=fourier_k)
        else:
            raise ValueError(f"Unknown phase_mode: {phase_mode}")

        self.phase_raw_dim = self.phase_param.param_dim * 2
        self.action_dim = 1 + self.phase_raw_dim + 1

        print(f"[配置] 原始动作维度: {env.n_actions}")
        print(f"[配置] 降维动作维度: {self.action_dim}")
        print(f"[配置] 相位模式: {phase_mode}")

        # 动作边界
        self.height_min = 1.0
        self.height_max = float(env.H_max)
        self.power_min = 0.1
        self.power_max = float(env.P_S_max)

        # 历史
        self.height_history = []
        self.power_history = []
        self.phase_history = []

        # 网络
        self.actors = [self._build_actor() for _ in range(num_agents)]
        self.critic = self._build_critic()

        # 优化器
        self.actor_optimizers = [
            tf.keras.optimizers.Adam(learning_rate=self.actor_lr, epsilon=1e-5)
            for _ in range(num_agents)
        ]
        self.critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.critic_lr, epsilon=1e-5
        )

        self.rollout_buffer = RolloutBuffer(num_agents)

        # 奖励归一化
        self.reward_mean = 0
        self.reward_var = 1
        self.reward_count = 0
        self.warmup_steps = 300

    def _build_actor(self):
        """
        Actor网络 - 所有输出统一用tanh
        """
        inputs = tf.keras.Input(shape=(self.state_dim,))

        x = tf.keras.layers.Dense(256, kernel_initializer='orthogonal')(inputs)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Dense(256, kernel_initializer='orthogonal')(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Dense(128, kernel_initializer='orthogonal')(x)
        x = tf.keras.layers.ReLU()(x)

        mu = tf.keras.layers.Dense(
            self.action_dim,
            activation='tanh',
            kernel_initializer=tf.keras.initializers.Orthogonal(gain=0.01)
        )(x)

        mu, log_std = LogStdLayer(
            self.action_dim,
            initial_value=-0.7,
            min_log_std=-2.0,
            max_log_std=-0.3
        )(mu)

        return tf.keras.Model(inputs=inputs, outputs=[mu, log_std])

    def _build_critic(self):
        """Critic网络"""
        inputs = tf.keras.Input(shape=(self.state_dim * self.num_agents,))

        x = tf.keras.layers.Dense(512, kernel_initializer='orthogonal')(inputs)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Dense(256, kernel_initializer='orthogonal')(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Dense(128, kernel_initializer='orthogonal')(x)
        x = tf.keras.layers.ReLU()(x)

        value = tf.keras.layers.Dense(1, kernel_initializer='orthogonal')(x)

        return tf.keras.Model(inputs=inputs, outputs=value)

    def _decode_action(self, raw_action):
        """
        解码动作
        """
        height_raw = np.clip(raw_action[0], -1, 1)
        height = self.height_min + (height_raw + 1) / 2 * (self.height_max - self.height_min)

        phase_raw = raw_action[1:1+self.phase_raw_dim]
        phase_params = []

        for i in range(self.phase_param.param_dim):

            cos_val = np.clip(phase_raw[2*i], -1, 1)
            sin_val = np.clip(phase_raw[2*i + 1], -1, 1)

            norm = np.sqrt(cos_val**2 + sin_val**2)
            if norm < 1e-8:
                cos_val, sin_val = 1.0, 0.0
            else:
                cos_val /= norm
                sin_val /= norm

            phase_val = trig_to_phase(cos_val, sin_val)
            phase_params.append(phase_val)

        if self.phase_mode == 'linear':
            base_phase = phase_params[0]  # [0, 2π]
            phase_diff = phase_params[1] - np.pi  # 转换到[-π, π]
            phases = self.phase_param.params_to_phases(base_phase, phase_diff)
        else:  # fourier
            phases = self.phase_param.params_to_phases(np.array(phase_params))

        power_raw = np.clip(raw_action[-1], -1, 1)
        power = self.power_min + (power_raw + 1) / 2 * (self.power_max - self.power_min)

        return height, phases, power

    def get_action_and_value(self, states, deterministic=False):
        """获取动作和价值"""
        all_scaled_actions = []
        all_log_probs = []
        all_raw_actions = []

        for i in range(self.num_agents):
            state = np.expand_dims(states[i], axis=0).astype(np.float32)
            mu, log_std = self.actors[i](state)
            mu = mu.numpy()[0]
            log_std = log_std.numpy()[0]

            if not deterministic:
                std = np.exp(log_std)
                noise = np.random.normal(0, std)
                raw_action_unclipped = mu + noise

                raw_action = np.clip(raw_action_unclipped, -1.0, 1.0)

                log_prob = self._compute_log_prob(raw_action_unclipped, mu, std)
            else:
                raw_action = mu
                log_prob = 0.0

            all_raw_actions.append(raw_action)

            # 解码
            height, phases, power = self._decode_action(raw_action)
            scaled_action = np.concatenate([[height], phases, [power]])
            all_scaled_actions.append(scaled_action)
            all_log_probs.append(log_prob)

        # 值函数
        global_state = np.concatenate(states).reshape(1, -1).astype(np.float32)
        value = self.critic(global_state).numpy()[0, 0]

        return all_scaled_actions, all_log_probs, value, all_raw_actions

    def _compute_log_prob(self, action, mu, std):
        """高斯对数概率"""
        var = std ** 2 + 1e-8
        log_prob = -0.5 * np.sum(
            ((action - mu) ** 2) / var +
            2 * np.log(std + 1e-8) + np.log(2 * np.pi)
        )
        return log_prob

    def update_reward_stats(self, reward):
        """更新奖励统计"""
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = reward - self.reward_mean
        self.reward_var = ((self.reward_count - 1) * self.reward_var + delta * delta2) / max(self.reward_count, 1)
        self.reward_var = max(self.reward_var, 1e-6)

    def normalize_reward(self, reward):
        """归一化奖励"""
        if self.reward_count < self.warmup_steps:
            return reward
        return (reward - self.reward_mean) / (np.sqrt(self.reward_var) + 1e-8)

    def compute_gae(self, rewards, values, dones, last_value):
        """GAE计算"""
        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, last_value):
        """PPO更新"""
        (states, raw_actions, scaled_actions, rewards,
         values, old_log_probs, dones) = self.rollout_buffer.get_batch()

        advantages, returns = self.compute_gae(rewards, values, dones, last_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 分解状态
        all_agents_states = []
        for i in range(self.num_agents):
            start_idx = i * self.state_dim
            end_idx = (i + 1) * self.state_dim
            all_agents_states.append(states[:, start_idx:end_idx])

        n_samples = len(states)
        indices = np.arange(n_samples)

        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        total_ratio = 0
        update_count = 0

        for epoch in range(self.update_epochs):
            np.random.shuffle(indices)

            for start in range(0, n_samples, self.mini_batch_size):
                end = min(start + self.mini_batch_size, n_samples)
                mb_indices = indices[start:end]

                mb_states = states[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                # Critic更新
                with tf.GradientTape() as tape:
                    value_preds = tf.reshape(self.critic(mb_states), [-1])
                    critic_loss = tf.reduce_mean(tf.keras.losses.huber(mb_returns, value_preds))

                critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
                critic_grads = [tf.clip_by_norm(g, self.max_grad_norm) for g in critic_grads]
                self.critic_optimizer.apply_gradients(
                    zip(critic_grads, self.critic.trainable_variables)
                )
                total_critic_loss += critic_loss.numpy()

                # Actor更新
                for i in range(self.num_agents):
                    mb_agent_states = all_agents_states[i][mb_indices]
                    mb_agent_actions = raw_actions[mb_indices, i, :]
                    mb_agent_old_log_probs = old_log_probs[mb_indices, i]

                    with tf.GradientTape() as tape:
                        mu, log_std = self.actors[i](mb_agent_states)
                        std = tf.exp(log_std)
                        var = std ** 2

                        new_log_probs = -0.5 * tf.reduce_sum(
                            ((mb_agent_actions - mu) ** 2) / (var + 1e-8) +
                            2 * tf.math.log(std + 1e-8) + np.log(2 * np.pi),
                            axis=-1
                        )

                        log_prob_diff = new_log_probs - mb_agent_old_log_probs
                        log_prob_diff = tf.clip_by_value(log_prob_diff, -0.5, 0.5)
                        ratio = tf.exp(log_prob_diff)

                        surr1 = ratio * mb_advantages
                        surr2 = tf.clip_by_value(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * mb_advantages

                        policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

                        entropy = 0.5 * tf.reduce_mean(
                            tf.reduce_sum(tf.math.log(2 * np.pi * np.e * var + 1e-8), axis=-1)
                        )

                        actor_loss = policy_loss - self.entropy_coef * entropy

                    actor_grads = tape.gradient(actor_loss, self.actors[i].trainable_variables)
                    actor_grads = [tf.clip_by_norm(g, self.max_grad_norm) if g is not None else g
                                   for g in actor_grads]

                    grads_and_vars = [(g, v) for g, v in zip(actor_grads, self.actors[i].trainable_variables) if g is not None]
                    if grads_and_vars:
                        self.actor_optimizers[i].apply_gradients(grads_and_vars)

                    total_actor_loss += actor_loss.numpy()
                    total_entropy += entropy.numpy()
                    total_ratio += tf.reduce_mean(ratio).numpy()

                update_count += 1

        self.rollout_buffer.clear()

        n = update_count * self.num_agents
        return total_actor_loss/n, total_critic_loss/update_count, total_entropy/n, total_ratio/n

    def train(self, print_freq=50):
        """训练"""
        reward_history = []
        rate_history = []

        best_reward = -float('inf')
        best_rate = 0

        for episode in range(self.max_episodes):
            states = self.env.reset()
            ep_rewards = []
            ep_rates = []

            step_heights = []
            step_powers = []

            print(f"\n{'='*60}")
            print(f"回合 {episode}")
            print(f"{'='*60}")

            for step in range(self.max_steps):
                all_scaled, all_log_probs, value, all_raw = self.get_action_and_value(states)

                # 组合动作
                heights = [a[0] for a in all_scaled]
                phases = [a[1:-1] for a in all_scaled]
                powers = [a[-1] for a in all_scaled]
                combined = np.concatenate((heights, np.concatenate(phases), powers))

                # 环境交互
                next_states, rewards, total_rate, agent_rates, C_P_list, penalty = self.env.step(combined)

                reward = rewards[0]
                done = 1.0 if step == self.max_steps - 1 else 0.0

                self.update_reward_stats(reward)
                norm_reward = self.normalize_reward(reward)

                global_state = np.concatenate(states)
                self.rollout_buffer.add(global_state, all_raw, combined, norm_reward, value, all_log_probs, done)

                ep_rewards.append(reward)
                ep_rates.append(total_rate)
                step_heights.append(heights)
                step_powers.append(powers)

                if step % print_freq == 0:
                    print(f"\n步骤 {step}:")
                    print(f"  PU速率: {C_P_list[0]:.4f} ")
                    print(f"  SU速率: [{', '.join(f'{r:.3f}' for r in agent_rates)}]")
                    print(f"  总速率: {total_rate:.4f}, 奖励: {reward:.3f}")
                    print(f"  高度: [{', '.join(f'{h:.1f}' for h in heights)}]")
                    print(f"  功率: [{', '.join(f'{p:.2f}' for p in powers)}]")

                states = next_states

            # 更新
            global_state = np.concatenate(states).reshape(1, -1).astype(np.float32)
            last_value = self.critic(global_state).numpy()[0, 0]
            actor_loss, critic_loss, entropy, avg_ratio = self.update(last_value)

            avg_reward = np.mean(ep_rewards)
            avg_rate = np.mean(ep_rates)

            reward_history.append(avg_reward)
            rate_history.append(avg_rate)
            self.height_history.append(np.mean(step_heights, axis=0))
            self.power_history.append(np.mean(step_powers, axis=0))

            print(f"\n>>> 更新统计:")
            print(f"    Actor Loss: {actor_loss:.4f}")
            print(f"    Critic Loss: {critic_loss:.4f}")
            print(f"    Entropy: {entropy:.4f}")
            print(f"    Avg Ratio: {avg_ratio:.4f}")

            # 获取当前std
            _, log_std = self.actors[0](np.expand_dims(states[0], 0).astype(np.float32))
            std_mean = np.mean(np.exp(log_std.numpy()))
            print(f"    Std均值: {std_mean:.4f}")

            print(f"\n回合 {episode}: 平均奖励={avg_reward:.4f}, 平均速率={avg_rate:.4f}")

        return reward_history, rate_history, self.height_history, self.power_history, self.phase_history


def plot_results(reward_history, rate_history, save_path='figures'):
    """绘图"""
    import matplotlib.pyplot as plt
    os.makedirs(save_path, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    window = min(20, len(reward_history)//5) if len(reward_history) > 10 else 1

    # 奖励
    axes[0].plot(reward_history, 'b-', alpha=0.3)
    if window > 1:
        ma = np.convolve(reward_history, np.ones(window)/window, mode='valid')
        axes[0].plot(range(window-1, len(reward_history)), ma, 'b-', lw=2)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Training Rewards')
    axes[0].grid(True, alpha=0.3)

    # 速率
    axes[1].plot(rate_history, 'g-', alpha=0.3)
    if window > 1:
        ma = np.convolve(rate_history, np.ones(window)/window, mode='valid')
        axes[1].plot(range(window-1, len(rate_history)), ma, 'g-', lw=2)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Sum Rate')
    axes[1].set_title('SU Sum Rate')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_path}/training_v2.png', dpi=150)
    plt.close()
    print(f"保存到 {save_path}/")


if __name__ == "__main__":
    from environment import Cognitive_Radio

    env = Cognitive_Radio(N=10)

    mappo = MAPPO_Reduced(
        env,
        max_episodes=1000,
        max_steps=300,
        phase_mode='linear'
    )

    print(f"\n环境配置:")
    print(f"  状态维度: {env.n_features}")
    print(f"  原始动作维度: {env.n_actions}")
    print(f"  降维动作维度: {mappo.action_dim}")
    print(f"  高度范围: [{mappo.height_min}, {mappo.height_max}]")
    print(f"  功率范围: [{mappo.power_min}, {mappo.power_max}]")

    reward_history, rate_history, height_history, power_history, phase_history = mappo.train(print_freq=50)

    plot_results(reward_history, rate_history)