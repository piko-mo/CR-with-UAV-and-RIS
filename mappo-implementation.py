import tensorflow as tf
import numpy as np
import os


class LogStdLayer(tf.keras.layers.Layer):
    """可学习的对数标准差层 - 添加硬约束"""

    def __init__(self, action_dim, initial_value=-0.5, min_log_std=-2.0, max_log_std=0.0, **kwargs):
        super(LogStdLayer, self).__init__(**kwargs)
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
        super(LogStdLayer, self).build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        # 在forward时硬约束log_std范围，防止无限增长
        clipped_log_std = tf.clip_by_value(self.log_std, self.min_log_std, self.max_log_std)
        log_std_batch = tf.tile(tf.expand_dims(clipped_log_std, 0), [batch_size, 1])
        return inputs, log_std_batch


class RolloutBuffer:
    """
    轨迹缓冲区 - 修复版

    【修复】分别存储：
    - raw_actions: 每个agent的未缩放动作 (用于计算log_prob)
    - scaled_actions: 缩放后的联合动作 (用于环境交互)
    - log_probs: 每个agent的log_prob
    """

    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.states = []
        self.raw_actions = []  # 【新增】存储未缩放的原始动作
        self.scaled_actions = []  # 存储缩放后的联合动作
        self.rewards = []
        self.values = []
        self.log_probs = []  # [[agent0_lp, agent1_lp, agent2_lp], ...]
        self.dones = []

    def add(self, state, raw_actions_list, scaled_action, reward, value, log_probs_list, done):
        """
        添加一步数据

        参数:
            state: 全局状态
            raw_actions_list: 每个agent的原始动作列表 (未缩放)
            scaled_action: 缩放后的联合动作
            reward: 奖励
            value: 值函数估计
            log_probs_list: 每个agent的log_prob列表
            done: 是否终止
        """
        self.states.append(state)
        self.raw_actions.append(raw_actions_list)  # 存储原始动作
        self.scaled_actions.append(scaled_action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_probs_list)
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
            np.array(self.raw_actions, dtype=np.float32),  # shape: [T, num_agents, action_dim]
            np.array(self.scaled_actions, dtype=np.float32),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.values, dtype=np.float32),
            np.array(self.log_probs, dtype=np.float32),  # shape: [T, num_agents]
            np.array(self.dones, dtype=np.float32)
        )

    def size(self):
        return len(self.states)


class MAPPO:
    def __init__(self, env, num_agents=3, max_episodes=500, max_steps=500):
        # 超参数
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_ratio = 0.1  # 【PPO核心裁剪参数】
        self.actor_lr = 5e-4
        self.critic_lr = 5e-4
        self.update_epochs = 10
        self.mini_batch_size = 64
        self.entropy_coef = 0.001  # 熵系数
        self.value_loss_coef = 0.5
        self.max_grad_norm = 0.5

        # 训练参数
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.update_interval = max_steps

        self.env = env
        self.num_agents = num_agents
        self.state_dim = self.env.n_features
        self.action_dim = self.env.n_actions

        # 动作边界
        self.action_bounds = {
            'uav_height': (0, float(self.env.H_max)),
            'phase': (0, float(2 * np.pi)),
            'power': float(self.env.P_S_max)
        }

        # 历史记录
        self.height_history = []
        self.power_history = []
        self.phase_history = []

        # 创建网络
        self.actors = [self._build_actor() for _ in range(num_agents)]
        self.critic = self._build_critic()

        # 优化器
        self.actor_optimizers = [
            tf.keras.optimizers.Adam(learning_rate=self.actor_lr)
            for _ in range(num_agents)
        ]
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.critic_lr)

        # 使用修复后的RolloutBuffer
        self.rollout_buffer = RolloutBuffer(num_agents)

        # 奖励归一化 - 使用running mean/std
        self.reward_mean = 0
        self.reward_var = 1
        self.reward_count = 0
        self.warmup_steps = 100  # 【新增】预热步数，前N步不进行归一化

    def _build_actor(self):
        """构建Actor网络"""
        inputs = tf.keras.Input(shape=(self.state_dim,))
        x = tf.keras.layers.Dense(256, activation='relu',
                                  kernel_initializer='orthogonal')(inputs)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dense(256, activation='relu',
                                  kernel_initializer='orthogonal')(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dense(128, activation='relu',
                                  kernel_initializer='orthogonal')(x)

        # 输出均值，使用更小的初始化
        mu = tf.keras.layers.Dense(
            self.action_dim,
            activation='tanh',
            kernel_initializer=tf.keras.initializers.Orthogonal(gain=0.01),
            name='mu'
        )(x)

        # log_std层添加硬约束
        mu, log_std = LogStdLayer(
            self.action_dim,
            initial_value=-0.7,  # 对应 std ≈ 0.5，初始探索范围更合理
            min_log_std=-3.0,  # 允许高精度微调
            max_log_std=-0.2
        )(mu)

        return tf.keras.Model(inputs=inputs, outputs=[mu, log_std])

    def _build_critic(self):
        """构建Critic网络"""
        inputs = tf.keras.Input(shape=(self.state_dim * self.num_agents,))
        x = tf.keras.layers.Dense(512, activation='relu',
                                  kernel_initializer='orthogonal')(inputs)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dense(256, activation='relu',
                                  kernel_initializer='orthogonal')(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dense(128, activation='relu',
                                  kernel_initializer='orthogonal')(x)
        value = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0))(x)
        return tf.keras.Model(inputs=inputs, outputs=value)

    def get_action_and_value(self, states, deterministic=False):
        """获取所有智能体的动作、值函数和对数概率"""
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
                # 【修复】先采样，再计算log_prob，最后才clip
                raw_action_unclipped = np.random.normal(mu, std)

                # 在clip之前计算log_prob（这是正确的概率密度）
                log_prob = self._compute_log_prob(raw_action_unclipped, mu, std)

                # 然后才clip动作
                raw_action = np.clip(raw_action_unclipped, -1.0, 1.0)
            else:
                raw_action = mu
                log_prob = 0.0

            all_raw_actions.append(raw_action)
            scaled_action = self._scale_action(raw_action)
            all_scaled_actions.append(scaled_action)
            all_log_probs.append(log_prob)

        # 计算值函数
        global_state = np.concatenate(states).reshape(1, -1).astype(np.float32)
        value = self.critic(global_state).numpy()[0, 0]

        return all_scaled_actions, all_log_probs, value, all_raw_actions

    def _compute_log_prob(self, action, mu, std):
        """计算高斯分布的对数概率"""
        var = std ** 2 + 1e-8
        log_prob = -0.5 * np.sum(
            ((action - mu) ** 2) / var +
            2 * np.log(std + 1e-8) + np.log(2 * np.pi)
        )
        return log_prob

    def _scale_action(self, raw_action):
        """将网络输出缩放到实际动作范围"""
        scaled_action = []

        # UAV高度缩放 [1, H_max]
        height_factor = (raw_action[0] + 1) / 2
        uav_height = 1.0 + height_factor * (self.action_bounds['uav_height'][1] - 1.0)
        scaled_action.append(uav_height)

        # 相位缩放 [0, 2π]
        for i in range(self.env.N):
            phase = ((raw_action[i + 1] + 1) / 2) * 2 * np.pi
            scaled_action.append(phase)

        # 功率缩放 [0.1, P_S_max]
        min_power = 0.1
        power_ratio = (raw_action[-1] + 1) / 2
        power = min_power + power_ratio * (self.action_bounds['power'] - min_power)
        scaled_action.append(power)

        return np.array(scaled_action)

    def update_reward_stats(self, reward):
        """更新奖励统计（Welford算法）"""
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = reward - self.reward_mean
        self.reward_var = ((self.reward_count - 1) * self.reward_var + delta * delta2) / max(self.reward_count, 1)
        self.reward_var = max(self.reward_var, 1e-6)

    def normalize_reward(self, reward):
        """归一化奖励"""
        # 【修复】预热期间不归一化
        if self.reward_count < self.warmup_steps:
            return reward
        return (reward - self.reward_mean) / (np.sqrt(self.reward_var) + 1e-8)

    def compute_gae(self, rewards, values, dones, last_value):
        """计算GAE"""
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
        """PPO更新 - 修复版"""
        (states, raw_actions, scaled_actions, rewards,
         values, old_log_probs, dones) = self.rollout_buffer.get_batch()
        # raw_actions shape: [T, num_agents, action_dim]
        # old_log_probs shape: [T, num_agents]

        # 计算GAE
        advantages, returns = self.compute_gae(rewards, values, dones, last_value)

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 分解全局状态为各智能体状态
        all_agents_states = []
        for i in range(self.num_agents):
            start_idx = i * self.state_dim
            end_idx = (i + 1) * self.state_dim
            all_agents_states.append(states[:, start_idx:end_idx])

        # 多个epoch更新
        n_samples = len(states)
        indices = np.arange(n_samples)

        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        update_count = 0

        for epoch in range(self.update_epochs):
            np.random.shuffle(indices)

            for start in range(0, n_samples, self.mini_batch_size):
                end = min(start + self.mini_batch_size, n_samples)
                mb_indices = indices[start:end]

                mb_states = states[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                # 更新Critic
                with tf.GradientTape() as tape:
                    value_preds = tf.reshape(self.critic(mb_states), [-1])
                    # 使用Huber loss更稳定
                    critic_loss = tf.reduce_mean(tf.keras.losses.huber(mb_returns, value_preds))

                critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
                critic_grads = [tf.clip_by_norm(g, self.max_grad_norm) for g in critic_grads]
                self.critic_optimizer.apply_gradients(
                    zip(critic_grads, self.critic.trainable_variables)
                )
                total_critic_loss += critic_loss.numpy()

                # 更新各Agent的Actor
                for i in range(self.num_agents):
                    mb_agent_states = all_agents_states[i][mb_indices]
                    # 【修复】直接使用存储的raw_actions，无需反向缩放
                    mb_agent_actions = raw_actions[mb_indices, i, :]
                    mb_agent_old_log_probs = old_log_probs[mb_indices, i]

                    mb_agent_states = tf.convert_to_tensor(mb_agent_states, dtype=tf.float32)
                    mb_agent_actions = tf.convert_to_tensor(mb_agent_actions, dtype=tf.float32)

                    with tf.GradientTape() as tape:
                        mu, log_std = self.actors[i](mb_agent_states)
                        std = tf.exp(log_std)
                        var = std ** 2

                        # 新策略的对数概率
                        new_log_probs = -0.5 * tf.reduce_sum(
                            ((mb_agent_actions - mu) ** 2) / (var + 1e-8) +
                            2 * tf.math.log(std + 1e-8) + np.log(2 * np.pi),
                            axis=-1
                        )

                        # 计算ratio
                        mb_agent_old_log_probs_tensor = tf.convert_to_tensor(
                            mb_agent_old_log_probs, dtype=tf.float32
                        )
                        ratio = tf.exp(new_log_probs - mb_agent_old_log_probs_tensor)

                        # 额外clip ratio防止极端值
                        ratio = tf.clip_by_value(ratio, 0.0, 10.0)

                        # 【PPO核心裁剪】
                        mb_advantages_tensor = tf.convert_to_tensor(mb_advantages, dtype=tf.float32)
                        surr1 = ratio * mb_advantages_tensor
                        surr2 = tf.clip_by_value(
                            ratio,
                            1 - self.clip_ratio,
                            1 + self.clip_ratio
                        ) * mb_advantages_tensor

                        # 策略损失（取min实现悲观更新）
                        policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

                        # 熵 (entropy bonus) - 高斯分布熵公式
                        entropy = 0.5 * tf.reduce_mean(
                            tf.reduce_sum(tf.math.log(2 * np.pi * np.e * var + 1e-8), axis=-1)
                        )

                        # 【修复】Actor总损失 = 策略损失 - 熵bonus
                        # 最小化policy_loss，最大化entropy
                        actor_loss = policy_loss - self.entropy_coef * entropy

                    actor_grads = tape.gradient(actor_loss, self.actors[i].trainable_variables)
                    actor_grads = [tf.clip_by_norm(g, self.max_grad_norm) if g is not None else g
                                   for g in actor_grads]

                    grads_and_vars = [
                        (g, v) for g, v in zip(actor_grads, self.actors[i].trainable_variables)
                        if g is not None
                    ]
                    if grads_and_vars:
                        self.actor_optimizers[i].apply_gradients(grads_and_vars)

                    total_actor_loss += actor_loss.numpy()
                    total_entropy += entropy.numpy()

                update_count += 1

        # 清空缓冲区
        self.rollout_buffer.clear()

        return (total_actor_loss / (update_count * self.num_agents),
                total_critic_loss / update_count,
                total_entropy / (update_count * self.num_agents))

    def train(self, print_freq=50):
        """训练主循环 - 已修改为按 Episode 更新"""
        reward_history = []
        rate_sum_history = []

        total_steps = 0
        best_avg_reward = -float('inf')

        for episode in range(self.max_episodes):
            states = self.env.reset()
            episode_rewards = []
            episode_rates = []

            step_heights = []
            step_powers = []
            step_phases = []

            print(f"\n====== 回合 {episode} ======")

            # --- 步骤循环 (收集轨迹) ---
            for step in range(self.max_steps):
                # 1. 获取动作
                all_scaled_actions, all_log_probs, value, all_raw_actions = self.get_action_and_value(states)

                # 2. 组合动作
                uav_heights = [action[0] for action in all_scaled_actions]
                phases = [action[1:-1] for action in all_scaled_actions]
                powers = [action[-1] for action in all_scaled_actions]
                combined_action = np.concatenate((uav_heights, np.concatenate(phases), powers))

                # 3. 环境交互
                next_states, rewards, total_rate, agent_rates, C_P_list, penalty = self.env.step(combined_action)

                reward = rewards[0]
                # 注意：对于时间限制的任务，最后一步通常不算真正的 terminal，但为了代码兼容性保持 done=1
                done = 1.0 if step == self.max_steps - 1 else 0.0

                # 4. 统计与归一化
                self.update_reward_stats(reward)
                normalized_reward = self.normalize_reward(reward)

                # 5. 存入 Buffer
                global_state = np.concatenate(states)
                self.rollout_buffer.add(
                    global_state,
                    all_raw_actions,
                    combined_action,
                    normalized_reward,
                    value,
                    all_log_probs,
                    done
                )

                # 记录数据
                episode_rewards.append(reward)
                episode_rates.append(total_rate)
                step_heights.append(uav_heights)
                step_powers.append(powers)
                step_phases.append(phases)
                total_steps += 1

                # 打印日志
                if step % print_freq == 0:
                    print(f"\n----- 步骤 {step} -----")
                    print(f" PU速率: {C_P_list[0]:.4f}")
                    print(f" SU速率: [{', '.join(f'{rate:.2f}' for rate in agent_rates)}]")
                    print(f" 总速率: {total_rate:.4f}, 奖励: {reward:.2f}")
                    # 显示std监控
                    try:
                        _, log_std = self.actors[0](np.expand_dims(states[0], 0).astype(np.float32))
                        std_mean = np.mean(np.exp(log_std.numpy()))
                        print(f" std均值: {std_mean:.4f}")
                    except:
                        pass

                # 状态流转
                states = next_states

            # --- 步骤循环结束 ---

            # === 【关键修改】在 Episode 结束后统一更新 ===

            # 1. 计算这一回合最终状态的 Value (用于 GAE Bootstrap)
            # 这里的 states 已经是 next_states (即第300步之后的状态)
            global_state = np.concatenate(states).reshape(1, -1).astype(np.float32)
            last_value = self.critic(global_state).numpy()[0, 0]

            # 2. 执行 PPO 更新
            # 这会利用 Buffer 中完整的 300 步数据进行训练
            actor_loss, critic_loss, entropy = self.update(last_value)

            print(
                f"\n>>> 回合结束更新: Actor Loss={actor_loss:.4f}, Critic Loss={critic_loss:.4f}, Entropy={entropy:.4f}")

            # 3. 记录历史数据
            avg_reward = np.mean(episode_rewards)
            avg_rate = np.mean(episode_rates)

            reward_history.append(avg_reward)
            rate_sum_history.append(avg_rate)
            self.height_history.append(np.mean(step_heights, axis=0))
            self.power_history.append(np.mean(step_powers, axis=0))
            self.phase_history.append(np.mean(step_phases, axis=0))

            # 4. 保存最佳模型
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                print(f" 🎉 新最佳奖励: {best_avg_reward:.4f}")

            print("\n" + "=" * 50)
            print(f"第 {episode} 回合完成")
            print(f" 平均奖励: {avg_reward:.4f}")
            print(f" 平均速率: {avg_rate:.4f}")
            print("=" * 50)

        return reward_history, rate_sum_history, self.height_history, self.power_history, self.phase_history


def plot_results(env, mappo, reward_history, rate_history, height_history, power_history, phase_history,
                 save_path='figures'):
    """绘制训练结果"""
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        sns.set_theme()
    except:
        pass

    os.makedirs(save_path, exist_ok=True)

    # 1. 奖励曲线
    plt.figure(figsize=(10, 6))
    plt.plot(reward_history, 'b-', alpha=0.3, label='Episode Reward')
    window = min(50, len(reward_history) // 5) if len(reward_history) > 10 else 1
    if window > 1:
        moving_avg = np.convolve(reward_history, np.ones(window) / window, mode='valid')
        plt.plot(range(window - 1, len(reward_history)), moving_avg, 'b-', linewidth=2, label='Moving Average')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Training Rewards')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f'{save_path}/reward_history.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. 速率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(rate_history, 'g-', alpha=0.3, label='Episode Rate')
    if window > 1:
        moving_avg = np.convolve(rate_history, np.ones(window) / window, mode='valid')
        plt.plot(range(window - 1, len(rate_history)), moving_avg, 'g-', linewidth=2, label='Moving Average')
    plt.xlabel('Episode')
    plt.ylabel('Average Rate (bps/Hz)')
    plt.title('Secondary User Rate Sum')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f'{save_path}/rate_history.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. UAV高度变化
    plt.figure(figsize=(10, 6))
    height_history = np.array(height_history)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i in range(height_history.shape[1]):
        plt.plot(height_history[:, i], color=colors[i], label=f'UAV {i + 1}', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('UAV Height (m)')
    plt.title('UAV Height Changes During Training')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f'{save_path}/height_changes.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 4. 功率变化
    plt.figure(figsize=(10, 6))
    power_history = np.array(power_history)
    for i in range(power_history.shape[1]):
        plt.plot(power_history[:, i], color=colors[i], label=f'Agent {i + 1}', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Power (W)')
    plt.title('Transmit Power Changes During Training')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f'{save_path}/power_changes.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"结果已保存到 {save_path}/")


if __name__ == "__main__":
    from environment import Cognitive_Radio

    # 创建环境和智能体
    env = Cognitive_Radio(N=10)
    mappo = MAPPO(env, max_episodes=500, max_steps=300)

    print("开始训练...")
    print(f"状态维度: {env.n_features}")
    print(f"动作维度: {env.n_actions}")
    print(f"PPO裁剪比率: {mappo.clip_ratio}")

    # 训练
    reward_history, rate_history, height_history, power_history, phase_history = mappo.train(
        print_freq=50
    )

    # 绘制结果
    plot_results(env, mappo, reward_history, rate_history, height_history, power_history, phase_history)