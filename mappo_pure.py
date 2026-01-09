import tensorflow as tf
import numpy as np
from mappo import MAPPO_Reduced


class MAPPO_Pure(MAPPO_Reduced):
    """
    MAPPO-Pure  算法实现

    """

    def __init__(self, env, num_agents=3, max_episodes=500, max_steps=300,
                 phase_mode='linear', fourier_k=2):

        # 1. 初始化基类
        super().__init__(env, num_agents, max_episodes, max_steps, phase_mode, fourier_k)

        # 2. 计算裁剪后的维度
        # 局部状态: [PU(1) + Others(N-1)]
        self.agent_specific_dim = self.state_dim - 1

        # 纯净全局状态: 1个PU状态 + M个智能体独有特征
        self.pruned_global_dim = 1 + self.num_agents * self.agent_specific_dim

        print(f"[MAPPO-Pure] 原始全局维度: {self.state_dim * self.num_agents}")
        print(f"[MAPPO-Pure] 裁剪后全局维度: {self.pruned_global_dim} (移除了 {self.num_agents - 1} 个冗余PU状态)")

        # 3. 重新构建 Critic
        self.critic = self._build_pure_critic()

        # 重新初始化 Critic 优化器
        self.critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.critic_lr, epsilon=1e-5
        )

    def _build_pure_critic(self):
        """构建输入维度为 pruned_global_dim 的 Critic"""
        inputs = tf.keras.Input(shape=(self.pruned_global_dim,))

        x = tf.keras.layers.Dense(512, kernel_initializer='orthogonal')(inputs)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Dense(256, kernel_initializer='orthogonal')(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Dense(128, kernel_initializer='orthogonal')(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        value = tf.keras.layers.Dense(1, kernel_initializer='orthogonal')(x)

        return tf.keras.Model(inputs=inputs, outputs=value)

    def _prune_global_state(self, full_global_state_batch):
        """
        核心裁剪逻辑：从拼接的全局状态中提取纯净状态

        """
        # 转换为 Tensor 方便操作
        x = tf.convert_to_tensor(full_global_state_batch, dtype=tf.float32)
        batch_size = tf.shape(x)[0]

        # 1. 重塑为 (Batch, M, State_Dim)
        x_reshaped = tf.reshape(x, [batch_size, self.num_agents, self.state_dim])

        # 2. 提取唯一的 PU 状态
        # Shape: (Batch, 1)
        pu_state = x_reshaped[:, 0, 0:1]

        # 3. 提取所有智能体的独有特征
        # Shape: (Batch, M, State_Dim-1)
        specific_features = x_reshaped[:, :, 1:]

        # 4. 展平独有特征
        specific_flat = tf.reshape(specific_features, [batch_size, -1])

        # 5. 拼接
        pruned_state = tf.concat([pu_state, specific_flat], axis=1)

        return pruned_state

    def get_action_and_value(self, states, deterministic=False):
        """重写：计算 Value 时需要先裁剪状态"""
        # 1. 获取 Actor 输出
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
            height, phases, power = self._decode_action(raw_action)
            scaled_action = np.concatenate([[height], phases, [power]])
            all_scaled_actions.append(scaled_action)
            all_log_probs.append(log_prob)

        # 2. 计算 Value
        # 先拼接成原始全局状态，再裁剪
        raw_global_state = np.concatenate(states).reshape(1, -1).astype(np.float32)
        pruned_state = self._prune_global_state(raw_global_state)

        value = self.critic(pruned_state).numpy()[0, 0]

        return all_scaled_actions, all_log_probs, value, all_raw_actions

    def update(self, last_value):
        """重写：Critic 更新时使用裁剪后的状态"""
        (states, raw_actions, scaled_actions, rewards,
         values, old_log_probs, dones) = self.rollout_buffer.get_batch()

        # GAE 计算
        advantages, returns = self.compute_gae(rewards, values, dones, last_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 准备 Actor 需要的局部状态
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

                # 取出原始拼接状态
                mb_states_full = states[mb_indices]

                mb_states_pruned = self._prune_global_state(mb_states_full)

                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                # --- Critic 更新  ---
                with tf.GradientTape() as tape:
                    value_preds = tf.reshape(self.critic(mb_states_pruned), [-1])
                    critic_loss = tf.reduce_mean(tf.keras.losses.huber(mb_returns, value_preds))

                critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
                critic_grads = [tf.clip_by_norm(g, self.max_grad_norm) for g in critic_grads]
                self.critic_optimizer.apply_gradients(
                    zip(critic_grads, self.critic.trainable_variables)
                )
                total_critic_loss += critic_loss.numpy()

                # --- Actor 更新  ---
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
                        surr2 = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * mb_advantages

                        policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
                        entropy = 0.5 * tf.reduce_mean(
                            tf.reduce_sum(tf.math.log(2 * np.pi * np.e * var + 1e-8), axis=-1)
                        )
                        actor_loss = policy_loss - self.entropy_coef * entropy

                    actor_grads = tape.gradient(actor_loss, self.actors[i].trainable_variables)
                    actor_grads = [tf.clip_by_norm(g, self.max_grad_norm) if g is not None else g
                                   for g in actor_grads]
                    grads_and_vars = [(g, v) for g, v in zip(actor_grads, self.actors[i].trainable_variables) if
                                      g is not None]
                    if grads_and_vars:
                        self.actor_optimizers[i].apply_gradients(grads_and_vars)

                    total_actor_loss += actor_loss.numpy()
                    total_entropy += entropy.numpy()
                    total_ratio += tf.reduce_mean(ratio).numpy()

                update_count += 1

        self.rollout_buffer.clear()
        n = update_count * self.num_agents
        return total_actor_loss / n, total_critic_loss / update_count, total_entropy / n, total_ratio / n

    def train(self, print_freq=50):
        """
        重写 Train 方法，主要是为了修正 last_value 的计算
        """
        reward_history = []
        rate_history = []

        for episode in range(self.max_episodes):
            states = self.env.reset()
            ep_rewards = []
            ep_rates = []

            step_heights = []
            step_powers = []

            if episode % 10 == 0:
                print(f"\n{'=' * 60}")
                print(f"MAPPO-Pure 回合 {episode}")
                print(f"{'=' * 60}")

            for step in range(self.max_steps):
                all_scaled, all_log_probs, value, all_raw = self.get_action_and_value(states)

                heights = [a[0] for a in all_scaled]
                phases = [a[1:-1] for a in all_scaled]
                powers = [a[-1] for a in all_scaled]
                combined = np.concatenate((heights, np.concatenate(phases), powers))

                next_states, rewards, total_rate, agent_rates, C_P_list, penalty = self.env.step(combined)

                reward = rewards[0]
                done = 1.0 if step == self.max_steps - 1 else 0.0

                self.update_reward_stats(reward)
                norm_reward = self.normalize_reward(reward)

                global_state_full = np.concatenate(states)
                self.rollout_buffer.add(global_state_full, all_raw, combined, norm_reward, value, all_log_probs, done)

                ep_rewards.append(reward)
                ep_rates.append(total_rate)
                step_heights.append(heights)
                step_powers.append(powers)

                if step % print_freq == 0 and episode % 10 == 0:
                    print(f"\n步骤 {step}:")
                    print(f"  PU速率: {C_P_list[0]:.4f} ")
                    print(f"  SU速率: [{', '.join(f'{r:.3f}' for r in agent_rates)}]")
                    print(f"  总速率: {total_rate:.4f}, 奖励: {reward:.3f}")

                states = next_states

            raw_global_state = np.concatenate(states).reshape(1, -1).astype(np.float32)
            # 使用 pruned state 计算 value
            pruned_state = self._prune_global_state(raw_global_state)
            last_value = self.critic(pruned_state).numpy()[0, 0]


            actor_loss, critic_loss, entropy, avg_ratio = self.update(last_value)

            avg_reward = np.mean(ep_rewards)
            avg_rate = np.mean(ep_rates)

            reward_history.append(avg_reward)
            rate_history.append(avg_rate)
            self.height_history.append(np.mean(step_heights, axis=0))
            self.power_history.append(np.mean(step_powers, axis=0))

            if episode % 10 == 0:
                print(f"\n>>> 更新统计:")
                print(f"    Actor Loss: {actor_loss:.4f}")
                print(f"    Critic Loss: {critic_loss:.4f}")
                print(f"    Entropy: {entropy:.4f}")
                print(f"    Avg Ratio: {avg_ratio:.4f}")
                print(f"\n回合 {episode}: 平均奖励={avg_reward:.4f}, 平均速率={avg_rate:.4f}")

        return reward_history, rate_history, self.height_history, self.power_history, self.phase_history


if __name__ == "__main__":
    from environment import Cognitive_Radio
    from mappo import plot_results  # 复用绘图函数

    env = Cognitive_Radio(N=10)
    env.gamma_P = 2.5

    print("正在初始化 MAPPO-Pure...")
    mappo_pure = MAPPO_Pure(
        env,
        max_episodes=1000,
        max_steps=300,
        phase_mode='linear'
    )

    # 开始训练
    print("\n开始训练 MAPPO-Pure...")
    r_hist, rate_hist, _, _, _ = mappo_pure.train(print_freq=100)

    # 绘图
    plot_results(r_hist, rate_hist, save_path='figures_pure')