import numpy as np


class Cognitive_Radio:
    def __init__(self, N=10):
        # 系统参数
        self.N = N  # 每个IRS的反射单元数
        self.num_cells = 3  # 小区数量
        self.K = 10  # 惩罚
        self.xi = 3  # 惩罚系数

        # 信道参数
        self.C = 10  # 环境参数C
        self.D = 0.6  # 环境参数D
        self.fc = 2.4e9  # 载波频率 (2.4GHz)
        self.c = 3e8  # 光速 (m/s)
        self.wavelength = self.c / self.fc
        self.rho = 10 ** (-40 / 10)  # 参考距离处路径损耗
        self.noise_P = 10 ** (-80 / 10) * 0.001  # 主用户处噪声功率-80dBm
        self.noise_S = 10 ** (-80 / 10) * 0.001  # 次要用户处噪声功率-80dBm

        # 功率约束
        self.P_P = 10 ** (40 / 10) * 0.001  # 主用户基站功率40dBm
        self.P_S_max = 10 ** (35 / 10) * 0.001  # 次要用户基站最大功率35dBm

        # UAV参数
        self.H_max = 30  # 无人机最大飞行高度30m

        self.gamma_P = 3  # 主用户速率需求
        self.C_req = 0.25  # 次要用户速率需求

        # PU始终活跃（简化模型）
        self.is_pu_active = True

        self.channel_coherence_steps = 50  # 信道相干步数
        self.current_step = 0  # 当前步数计数
        self.cached_channels = None  # 缓存的信道
        self.channel_seed = None  # 当前信道种子
        self.last_uav_heights = None  # 上次的UAV高度（用于检测高度变化）
        self.height_change_threshold = 2.0  # 高度变化阈值（超过则强制更新信道）

        # 位置设置和空间设置
        self.setup_locations()
        self.setup_spaces()

        self.PU_STATES = {'ACK': 1, 'NACK': 0}

        # 归一化参数
        self.channel_scale = 1e-5  # 典型信道增益量级
        self.height_scale = self.H_max  # 高度归一化系数

    def setup_locations(self):
        # 主用户系统位置
        self.L_PT = np.array([40, 100, 0])  # 主用户基站位置
        self.L_PR = np.array([60, 100, 0])  # 主用户位置

        # 次要用户系统位置
        self.L_ST = [  # 次要用户基站位置
            np.array([80, 20, 0]),
            np.array([140, 100, 0]),
            np.array([80, 200, 0])
        ]
        self.L_SR = [  # 次要用户位置
            np.array([80, 60, 0]),
            np.array([100, 100, 0]),
            np.array([80, 140, 0])
        ]
        self.L_IRS = [  # IRS位置
            np.array([80, 40, 0]),
            np.array([120, 100, 0]),
            np.array([80, 160, 0])
        ]

    def setup_spaces(self):
        # 每个智能体的动作空间：[UAV高度 + IRS相位 + 功率]
        self.n_actions = 1 + self.N + 1

        # 每个智能体的状态空间：
        # [PU状态(1) + 本地UAV高度(1) + 4个信道×N×2(实虚部)(8N) + SU速率状态(1)]
        self.n_features = 3 + 4 * self.N * 2

    def get_local_state(self, agent_id, pu_state, local_uav_height, channels_list, su_rate_state):

        normalized_height = local_uav_height / self.height_scale

        channel_feats = []

        for ch_vec in channels_list:
            ch_arr = np.array(ch_vec)
            norm_ch = ch_arr / self.channel_scale
            channel_feats.extend(np.real(norm_ch))
            channel_feats.extend(np.imag(norm_ch))

        state = np.concatenate([
            [pu_state],
            [normalized_height],
            channel_feats,
            [su_rate_state]
        ])

        return state.astype(np.float32)

    def get_channel(self, l1, l2, height, is_ground_to_ground=False):

        l2_with_height = l2.copy().astype(float)
        l2_with_height[2] = height if not is_ground_to_ground else l2[2]

        d = np.linalg.norm(l2_with_height - l1)
        d = max(d, 1.0)
        PL = np.sqrt(self.rho / (d ** 2))
        K_rician = 10.0

        if is_ground_to_ground:
            h_Los = np.exp(-1j * (2 * np.pi / self.wavelength) * d)
        else:
            theta = np.arcsin(min(height / d, 1.0))
            h_Los = np.exp(-1j * (2 * np.pi / self.wavelength) * d * np.sin(theta))

        h_Los = np.repeat(h_Los, self.N)

        h_NLos = (np.random.randn(self.N) + 1j * np.random.randn(self.N)) / np.sqrt(2)

        h = PL * (
                np.sqrt(K_rician / (K_rician + 1)) * h_Los +
                np.sqrt(1 / (K_rician + 1)) * h_NLos
        )

        return h

    def _should_update_channels(self, uav_heights):
        """
        判断是否需要更新信道

        更新条件：
        1. 缓存为空
        2. 达到信道相干时间
        3. UAV高度变化超过阈值
        """
        if self.cached_channels is None:
            return True

        if self.current_step > 1 and (self.current_step - 1) % self.channel_coherence_steps == 0:
            return True

        # 检查高度变化
        if self.last_uav_heights is not None:
            height_change = np.max(np.abs(np.array(uav_heights) - np.array(self.last_uav_heights)))
            if height_change > self.height_change_threshold:
                return True

        return False

    def _update_channels(self, uav_heights, force_update=False):
        """
        更新信道（带缓存机制）
        """
        should_update = force_update or self._should_update_channels(uav_heights)

        if not should_update:
            return

        seed_offset = self.current_step // self.channel_coherence_steps
        np.random.seed(self.channel_seed + seed_offset)

        # 初始化缓存结构
        self.cached_channels = {
            'h_PR': [],  # PT -> IRS
            'h_SR': [],  # ST -> IRS
            'h_RP': [],  # IRS -> PR
            'h_RS': [],  # IRS -> SR
            'h_PT_PR': None,  # PT -> PR (直接链路)
            'h_PT_SR': [],  # PT -> SR (干扰链路)
            'h_ST_SR': [],  # ST -> SR (直接链路)
            'h_ST_PR': [],  # ST -> PR (干扰链路)
            'h_ST_SR_cross': {}  # ST_j -> SR_m (跨小区干扰)
        }

        # 生成空地链路信道
        for i in range(self.num_cells):
            self.cached_channels['h_PR'].append(
                self.get_channel(self.L_PT, self.L_IRS[i], uav_heights[i], is_ground_to_ground=False)
            )
            self.cached_channels['h_SR'].append(
                self.get_channel(self.L_ST[i], self.L_IRS[i], uav_heights[i], is_ground_to_ground=False)
            )
            self.cached_channels['h_RP'].append(
                self.get_channel(self.L_IRS[i], self.L_PR, uav_heights[i], is_ground_to_ground=False)
            )
            self.cached_channels['h_RS'].append(
                self.get_channel(self.L_IRS[i], self.L_SR[i], uav_heights[i], is_ground_to_ground=False)
            )

        # 生成地对地链路信道
        self.cached_channels['h_PT_PR'] = self.get_channel(
            self.L_PT, self.L_PR, 0, is_ground_to_ground=True
        )

        for i in range(self.num_cells):
            self.cached_channels['h_PT_SR'].append(
                self.get_channel(self.L_PT, self.L_SR[i], 0, is_ground_to_ground=True)
            )
            self.cached_channels['h_ST_SR'].append(
                self.get_channel(self.L_ST[i], self.L_SR[i], 0, is_ground_to_ground=True)
            )
            self.cached_channels['h_ST_PR'].append(
                self.get_channel(self.L_ST[i], self.L_PR, 0, is_ground_to_ground=True)
            )

        # 生成跨小区干扰信道 ST_j -> SR_m
        for j in range(self.num_cells):
            for m in range(self.num_cells):
                if j != m:
                    key = f'{j}_{m}'
                    self.cached_channels['h_ST_SR_cross'][key] = self.get_channel(
                        self.L_ST[j], self.L_SR[m], 0, is_ground_to_ground=True
                    )

        self.last_uav_heights = list(uav_heights)
        np.random.seed(None)

    def step(self, action):

        self.current_step += 1

        # 分解动作
        action_UAV = action[:self.num_cells]  # UAV高度
        action_Phase = action[self.num_cells:self.num_cells + self.N * self.num_cells]  # IRS相位
        action_Rate = action[self.num_cells + self.N * self.num_cells:]  # 发射功率

        self._update_channels(action_UAV)

        # 从缓存获取信道
        h_PR = self.cached_channels['h_PR']
        h_SR = self.cached_channels['h_SR']
        h_RP = self.cached_channels['h_RP']
        h_RS = self.cached_channels['h_RS']
        h_PT_PR = self.cached_channels['h_PT_PR']
        h_PT_SR = self.cached_channels['h_PT_SR']
        h_ST_SR = self.cached_channels['h_ST_SR']
        h_ST_PR = self.cached_channels['h_ST_PR']

        # 计算各用户速率
        C_s_m_list = []  # 次要用户速率列表
        C_P = 0  # 主用户速率

        if self.is_pu_active:
            # 主用户直接接收信号
            direct_signal_P = h_PT_PR

            # 计算主用户接收到的干扰（来自所有次要用户）
            I_ST_at_PR = 0
            for j in range(self.num_cells):
                # 次要用户 j 到主用户的直接干扰路径
                st_pr_direct = h_ST_PR[j]

                # 次要用户 j 通过 RIS 反射到主用户的干扰路径
                st_pr_reflect = np.zeros(1, dtype=complex)
                for l in range(self.num_cells):
                    h_RP_l = h_RP[l][:, np.newaxis]
                    h_SR_jl = h_SR[j][:, np.newaxis]
                    Phi_Mat_l = np.diag(np.exp(1j * action_Phase[l * self.N:(l + 1) * self.N]))

                    st_pr_reflect_l = np.dot(np.dot(h_RP_l.T, Phi_Mat_l), h_SR_jl)
                    st_pr_reflect += st_pr_reflect_l.flatten()

                # 合并直接干扰和反射干扰
                total_interference_j = np.sum(st_pr_direct) + np.sum(st_pr_reflect)

                # 累加干扰功率
                I_ST_at_PR += action_Rate[j] * np.abs(total_interference_j) ** 2

            # 计算主用户的总信号
            total_signal_P = np.sum(direct_signal_P)

            for l in range(self.num_cells):
                h_RP_l = h_RP[l][:, np.newaxis]
                h_PR_l = h_PR[l][:, np.newaxis]
                Phi_Mat_l = np.diag(np.exp(1j * action_Phase[l * self.N:(l + 1) * self.N]))

                reflect_signal_P_l = np.dot(np.dot(h_RP_l.T, Phi_Mat_l), h_PR_l)
                total_signal_P += np.sum(reflect_signal_P_l)

            # 主用户SINR计算
            # 1. 计算 SINR 分子分母
            signal_power = self.P_P * np.abs(total_signal_P) ** 2
            interference_power = I_ST_at_PR + self.noise_P

            # 2. 安全检查：防止除零或 NaN
            if np.isnan(signal_power) or np.isnan(interference_power):
                print(f"[Warning] Step {self.current_step}: Signal/Interference is NaN! Actions might be corrupted.")
                SINR_P = 0.0
            else:
                SINR_P = signal_power / (interference_power + 1e-10)

            # 3. 安全计算速率
            C_P = np.log2(1 + max(SINR_P, 0.0))

        # 计算各次要用户的速率
        for m in range(self.num_cells):
            # 次要用户m的直接接收信号
            direct_signal_S_m = h_ST_SR[m]

            # 次要用户m通过所有RIS的反射信号
            reflect_signal_S_m = np.zeros(1, dtype=complex)
            for l in range(self.num_cells):
                h_RS_l = h_RS[l][:, np.newaxis]
                h_SR_ml = h_SR[m][:, np.newaxis]
                Phi_Mat_l = np.diag(np.exp(1j * action_Phase[l * self.N:(l + 1) * self.N]))

                reflect_signal_S_ml = np.dot(np.dot(h_RS_l.T, Phi_Mat_l), h_SR_ml)
                reflect_signal_S_m += reflect_signal_S_ml.flatten()

            # 合并次要用户m的直接信号和反射信号
            total_signal_S_m = np.sum(direct_signal_S_m) + np.sum(reflect_signal_S_m)

            # 计算次要用户m接收到的干扰
            # 1. 来自主用户PT的干扰
            I_PT_at_SR_m = 0
            if self.is_pu_active:
                pt_sr_direct = h_PT_SR[m]
                pt_sr_reflect = np.zeros(1, dtype=complex)
                for l in range(self.num_cells):
                    h_RS_l = h_RS[l][:, np.newaxis]
                    h_PR_l = h_PR[l][:, np.newaxis]
                    Phi_Mat_l = np.diag(np.exp(1j * action_Phase[l * self.N:(l + 1) * self.N]))

                    pt_sr_reflect_l = np.dot(np.dot(h_RS_l.T, Phi_Mat_l), h_PR_l)
                    pt_sr_reflect += pt_sr_reflect_l.flatten()

                total_pt_interference = np.sum(pt_sr_direct) + np.sum(pt_sr_reflect)
                I_PT_at_SR_m = self.P_P * np.abs(total_pt_interference) ** 2

            # 2. 来自其他次要用户的干扰
            I_ST_at_SR_m = 0
            for j in range(self.num_cells):
                if j != m:
                    key = f'{j}_{m}'
                    st_j_sr_m_direct = self.cached_channels['h_ST_SR_cross'][key]

                    st_j_sr_m_reflect = np.zeros(1, dtype=complex)
                    for l in range(self.num_cells):
                        h_RS_l = h_RS[l][:, np.newaxis]
                        h_SR_jl = h_SR[j][:, np.newaxis]
                        Phi_Mat_l = np.diag(np.exp(1j * action_Phase[l * self.N:(l + 1) * self.N]))

                        st_j_sr_m_reflect_l = np.dot(np.dot(h_RS_l.T, Phi_Mat_l), h_SR_jl)
                        st_j_sr_m_reflect += st_j_sr_m_reflect_l.flatten()

                    total_interference_j = np.sum(st_j_sr_m_direct) + np.sum(st_j_sr_m_reflect)
                    I_ST_at_SR_m += action_Rate[j] * np.abs(total_interference_j) ** 2

            # 计算次要用户m的SINR
            SINR_S_m = action_Rate[m] * np.abs(total_signal_S_m) ** 2 / (
                    I_PT_at_SR_m + I_ST_at_SR_m + self.noise_S + 1e-10)
            C_s_m = np.log2(1 + SINR_S_m)
            C_s_m_list.append(C_s_m)

        # 计算总速率
        C_S = sum(C_s_m_list)
        penalty = sum(min(C_s_m - self.C_req, 0) for C_s_m in C_s_m_list)

        # 奖励函数
        if C_P >= self.gamma_P:
            S_PU = self.PU_STATES['ACK']
            r = C_S + self.xi * penalty
        else:
            S_PU = self.PU_STATES['NACK']
            r = -self.K

        # 构建每个智能体的局部状态
        next_states = []
        for i in range(self.num_cells):
            local_next_state = self.get_local_state(
                i,
                S_PU,
                action_UAV[i],
                [h_PR[i], h_SR[i], h_RP[i], h_RS[i]],
                1 if C_s_m_list[i] >= self.C_req else 0
            )
            next_states.append(local_next_state)

        return next_states, [r] * self.num_cells, C_S, C_s_m_list, [C_P], penalty

    def reset(self):
        """重置环境"""
        # 重置计数器和缓存
        self.current_step = 0
        self.channel_seed = np.random.randint(0, 1000000)
        self.cached_channels = None
        self.last_uav_heights = None

        initial_states = []
        pu_state = self.PU_STATES['ACK']
        uav_heights = [1.0] * self.num_cells  # 初始高度

        # 生成初始信道
        self._update_channels(uav_heights, force_update=True)

        for i in range(self.num_cells):
            h_PR = self.cached_channels['h_PR'][i]
            h_SR = self.cached_channels['h_SR'][i]
            h_RP = self.cached_channels['h_RP'][i]
            h_RS = self.cached_channels['h_RS'][i]

            channel_vectors = [h_PR, h_SR, h_RP, h_RS]

            local_state = self.get_local_state(
                i,
                pu_state,
                uav_heights[i],
                channel_vectors,
                0
            )

            initial_states.append(local_state)

        return initial_states


def monte_carlo_rate_estimation(env, num_samples=1000):
    """
    通过蒙特卡洛仿真估计系统的可行速率范围
    """
    pu_rates = []
    su_rates = []

    for _ in range(num_samples):
        # 重置环境以获取新的信道实现
        env.reset()

        # 随机采样动作
        action = np.concatenate([
            np.random.uniform(1, env.H_max, env.num_cells),
            np.random.uniform(0, 2 * np.pi, env.N * env.num_cells),
            np.random.uniform(0.1, env.P_S_max, env.num_cells)
        ])

        _, _, total_rate, agent_rates, C_P_list, _ = env.step(action)

        pu_rates.append(C_P_list[0])
        su_rates.extend(agent_rates)

    return {
        'PU_rate': {
            'mean': np.mean(pu_rates),
            'std': np.std(pu_rates),
            'min': np.min(pu_rates),
            'max': np.max(pu_rates),
            'percentile_10': np.percentile(pu_rates, 10),
            'percentile_50': np.percentile(pu_rates, 50),
            'percentile_90': np.percentile(pu_rates, 90)
        },
        'SU_rate': {
            'mean': np.mean(su_rates),
            'std': np.std(su_rates),
            'min': np.min(su_rates),
            'max': np.max(su_rates),
            'percentile_10': np.percentile(su_rates, 10),
            'percentile_50': np.percentile(su_rates, 50),
            'percentile_90': np.percentile(su_rates, 90)
        }
    }


# 测试代码
if __name__ == "__main__":
    env = Cognitive_Radio(N=10)

    print("=" * 60)
    print("环境测试 (带信道缓存)")
    print("=" * 60)
    print(f"状态维度: {env.n_features}")
    print(f"动作维度: {env.n_actions}")
    print(f"信道相干步数: {env.channel_coherence_steps}")
    print(f"PU速率要求: {env.gamma_P}")
    print(f"SU速率要求: {env.C_req}")

    # 测试reset
    states = env.reset()
    print(f"\n初始状态形状: {[s.shape for s in states]}")
    print(f"初始状态[0]前10个元素: {states[0][:10]}")
    print(f"初始状态[0]范围: [{states[0].min():.4f}, {states[0].max():.4f}]")

