import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim


st.set_page_config(
    page_title="ResourceRL",
    page_icon="⚙️",
    layout="wide"
)


SEED = 42
MIN_RESOURCES = 20
MAX_RESOURCES = 100
SLA_LIMIT = 800

ACTION_SPACE = [-20, -10, 0, 10, 20]

ACTION_NAMES = {
    -20: "Scale Down 20",
    -10: "Scale Down 10",
    0: "Maintain",
    10: "Scale Up 10",
    20: "Scale Up 20"
}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed()


class WorkloadGenerator:
    def __init__(self, difficulty="Medium", seed=42):
        self.difficulty = difficulty
        self.rng = np.random.default_rng(seed)

        self.settings = {
            "Low": {
                "base": 45,
                "noise": 2.5,
                "spike_probability": 0.04,
                "spike_strength": 12
            },
            "Medium": {
                "base": 52,
                "noise": 4.5,
                "spike_probability": 0.08,
                "spike_strength": 20
            },
            "High": {
                "base": 60,
                "noise": 6.5,
                "spike_probability": 0.12,
                "spike_strength": 28
            }
        }

        self.current = self.settings[difficulty]["base"]

    def reset(self, value=None):
        if value is None:
            self.current = self.settings[
                self.difficulty
            ]["base"]
        else:
            self.current = float(value)

        return self.current

    def step(self):
        config = self.settings[self.difficulty]

        random_change = self.rng.normal(
            0,
            config["noise"]
        )

        if self.rng.random() < config["spike_probability"]:
            direction = self.rng.choice(
                [-1, 1]
            )

            random_change += (
                direction
                * self.rng.uniform(
                    config["spike_strength"] * 0.5,
                    config["spike_strength"]
                )
            )

        target = (
            config["base"]
            + self.rng.normal(
                0,
                config["noise"]
            )
        )

        self.current += (
            0.18 * (target - self.current)
            + random_change
        )

        self.current = float(
            np.clip(
                self.current,
                20,
                95
            )
        )

        return self.current


class ResourceEnvironment:
    def __init__(
        self,
        difficulty="Medium",
        seed=42
    ):
        self.difficulty = difficulty
        self.seed = seed

        self.generator = WorkloadGenerator(
            difficulty,
            seed
        )

        self.min_resources = MIN_RESOURCES
        self.max_resources = MAX_RESOURCES
        self.sla_limit = SLA_LIMIT

        self.max_queue = 100

        self.reset()

    def reset(
        self,
        initial_workload=None,
        initial_resources=60
    ):
        if initial_workload is None:
            self.workload = self.generator.reset()
        else:
            self.workload = float(
                initial_workload
            )
            self.generator.reset(
                self.workload
            )

        self.previous_workload = self.workload

        self.resources = float(
            np.clip(
                initial_resources,
                self.min_resources,
                self.max_resources
            )
        )

        self.queue = 0.0
        self.latency = 0.0
        self.utilization = 0.0
        self.previous_action = 0

        self._update_system()

        return self.get_state()

    def apply_action(self, action):
        self.resources += action

        self.resources = float(
            np.clip(
                self.resources,
                self.min_resources,
                self.max_resources
            )
        )

    def _update_system(self):
        self.utilization = (
            self.workload
            / max(self.resources, 1)
        ) * 100

        capacity = self.resources

        demand_gap = (
            self.workload
            - capacity
        )

        if demand_gap > 0:
            queue_growth = (
                demand_gap * 0.10
            )
        else:
            queue_growth = (
                demand_gap * 0.07
            )

        self.queue += queue_growth

        self.queue = float(
            np.clip(
                self.queue,
                0,
                self.max_queue
            )
        )

        utilization_pressure = max(
            self.utilization - 55,
            0
        )

        high_pressure = max(
            self.utilization - 75,
            0
        )

        underutilization = max(
            45 - self.utilization,
            0
        )

        self.latency = (
            120
            + 1.7 * utilization_pressure
            + 0.72 * (
                high_pressure ** 1.6
            )
            + 0.45 * underutilization
            + 4.0 * self.queue
        )

        self.latency = float(
            np.clip(
                self.latency,
                100,
                2500
            )
        )

    def calculate_reward(self, action):
        reward = 0.0

        if self.latency <= 180:
            reward += 12
        elif self.latency <= 250:
            reward += 9
        elif self.latency <= 400:
            reward += 5
        elif self.latency <= 600:
            reward += 1
        elif self.latency <= self.sla_limit:
            reward -= 4
        else:
            excess = (
                self.latency
                - self.sla_limit
            )

            reward -= 12
            reward -= min(
                excess / 60,
                25
            )

        if self.queue <= 2:
            reward += 6
        elif self.queue <= 8:
            reward += 4
        elif self.queue <= 18:
            reward += 1
        else:
            reward -= min(
                self.queue / 7,
                18
            )

        if 55 <= self.utilization <= 80:
            reward += 6
        elif 45 <= self.utilization <= 90:
            reward += 2
        else:
            reward -= 3

        desired_resources = np.clip(
            self.workload / 0.70,
            self.min_resources,
            self.max_resources
        )

        resource_ratio = (
            self.resources
            / max(
                desired_resources,
                1
            )
        )

        if resource_ratio > 1.10:
            reward -= (
                resource_ratio - 1.10
            ) * 10

        if resource_ratio < 0.75:
            reward -= (
                0.75 - resource_ratio
            ) * 8

        resource_cost = (
            self.resources
            / self.max_resources
        )

        reward -= (
            resource_cost * 1.8
        )

        if action != self.previous_action:
            reward -= 0.20

        if abs(action) >= 20:
            reward -= 0.10

        if self.latency > self.sla_limit:
            reward -= 12

        return float(reward)

    def step(
        self,
        action,
        workload=None
    ):
        self.previous_workload = (
            self.workload
        )

        self.apply_action(action)

        if workload is None:
            workload = self.generator.step()

        self.workload = float(
            np.clip(
                workload,
                20,
                95
            )
        )

        self._update_system()

        reward = self.calculate_reward(
            action
        )

        self.previous_action = action

        info = {
            "workload": self.workload,
            "resources": self.resources,
            "queue": self.queue,
            "latency": self.latency,
            "utilization": self.utilization,
            "sla_violation": (
                self.latency > self.sla_limit
            ),
            "reward": reward
        }

        return (
            self.get_state(),
            reward,
            info
        )

    def get_state(self):
        workload_norm = (
            self.workload / 100
        )

        queue_norm = (
            self.queue
            / self.max_queue
        )

        resources_norm = (
            self.resources
            / self.max_resources
        )

        latency_norm = min(
            self.latency / 2500,
            1
        )

        utilization_norm = min(
            self.utilization / 150,
            1
        )

        workload_trend = np.clip(
            (
                self.workload
                - self.previous_workload
            ) / 30,
            -1,
            1
        )

        resource_pressure = np.clip(
            (
                self.workload
                - self.resources
            ) / 50,
            -1,
            1
        )

        return np.array(
            [
                workload_norm,
                queue_norm,
                resources_norm,
                latency_norm,
                utilization_norm,
                workload_trend,
                resource_pressure
            ],
            dtype=np.float32
        )


class QLearningAgent:
    def __init__(
        self,
        alpha=0.12,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.985,
        epsilon_min=0.05
    ):
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_table = {}

    def state_key(self, state):
        workload = int(
            np.digitize(
                state[0],
                [
                    0.30,
                    0.45,
                    0.60,
                    0.75,
                    0.90
                ]
            )
        )

        queue = int(
            np.digitize(
                state[1],
                [
                    0.05,
                    0.15,
                    0.30,
                    0.60,
                    0.85
                ]
            )
        )

        resources = int(
            np.digitize(
                state[2],
                [
                    0.30,
                    0.50,
                    0.70,
                    0.85
                ]
            )
        )

        utilization = int(
            np.digitize(
                state[4],
                [
                    0.40,
                    0.55,
                    0.70,
                    0.85,
                    1.00
                ]
            )
        )

        trend = int(
            np.digitize(
                state[5],
                [
                    -0.25,
                    -0.08,
                    0.08,
                    0.25
                ]
            )
        )

        pressure = int(
            np.digitize(
                state[6],
                [
                    -0.30,
                    -0.05,
                    0.15,
                    0.35
                ]
            )
        )

        return (
            workload,
            queue,
            resources,
            utilization,
            trend,
            pressure
        )

    def ensure_state(self, key):
        if key not in self.q_table:
            self.q_table[key] = np.zeros(
                len(ACTION_SPACE),
                dtype=np.float32
            )

    def choose_action(
        self,
        state,
        training=True
    ):
        key = self.state_key(state)

        self.ensure_state(key)

        if (
            training
            and random.random()
            < self.epsilon
        ):
            return random.choice(
                ACTION_SPACE
            )

        index = int(
            np.argmax(
                self.q_table[key]
            )
        )

        return ACTION_SPACE[index]

    def update(
        self,
        state,
        action,
        reward,
        next_state,
        done=False
    ):
        state_key = self.state_key(
            state
        )

        next_key = self.state_key(
            next_state
        )

        self.ensure_state(
            state_key
        )

        self.ensure_state(
            next_key
        )

        action_index = ACTION_SPACE.index(
            action
        )

        target = reward

        if not done:
            target += (
                self.gamma
                * np.max(
                    self.q_table[next_key]
                )
            )

        current_value = (
            self.q_table[state_key][
                action_index
            ]
        )

        self.q_table[state_key][
            action_index
        ] += (
            self.alpha
            * (
                target
                - current_value
            )
        )

    def finish_episode(self):
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon
            * self.epsilon_decay
        )


class ReplayBuffer:
    def __init__(
        self,
        capacity=30000
    ):
        self.buffer = deque(
            maxlen=capacity
        )

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        self.buffer.append(
            (
                state,
                action,
                reward,
                next_state,
                done
            )
        )

    def sample(
        self,
        batch_size
    ):
        batch = random.sample(
            self.buffer,
            batch_size
        )

        (
            states,
            actions,
            rewards,
            next_states,
            dones
        ) = zip(*batch)

        return (
            np.asarray(
                states,
                dtype=np.float32
            ),
            np.asarray(
                actions,
                dtype=np.int64
            ),
            np.asarray(
                rewards,
                dtype=np.float32
            ),
            np.asarray(
                next_states,
                dtype=np.float32
            ),
            np.asarray(
                dones,
                dtype=np.float32
            )
        )

    def __len__(self):
        return len(self.buffer)


class DQNNetwork(nn.Module):
    def __init__(
        self,
        input_size=7,
        output_size=5
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_size,
                64
            ),
            nn.ReLU(),
            nn.Linear(
                64,
                64
            ),
            nn.ReLU(),
            nn.Linear(
                64,
                output_size
            )
        )

    def forward(self, x):
        return self.network(x)


class DQNAgent:
    def __init__(
        self,
        state_size=7,
        gamma=0.95,
        learning_rate=0.001,
        epsilon=1.0,
        epsilon_decay=0.985,
        epsilon_min=0.05
    ):
        self.state_size = state_size
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.device = torch.device(
            "cpu"
        )

        self.policy_net = DQNNetwork(
            state_size,
            len(ACTION_SPACE)
        ).to(self.device)

        self.target_net = DQNNetwork(
            state_size,
            len(ACTION_SPACE)
        ).to(self.device)

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate
        )

        self.loss_function = (
            nn.SmoothL1Loss()
        )

        self.memory = ReplayBuffer()

        self.batch_size = 32

        self.target_update_frequency = 100

        self.train_steps = 0

    def choose_action(
        self,
        state,
        training=True
    ):
        if (
            training
            and random.random()
            < self.epsilon
        ):
            return random.choice(
                ACTION_SPACE
            )

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(
                state_tensor
            )

        action_index = int(
            torch.argmax(
                q_values,
                dim=1
            ).item()
        )

        return ACTION_SPACE[
            action_index
        ]

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        action_index = ACTION_SPACE.index(
            action
        )

        self.memory.push(
            state,
            action_index,
            reward,
            next_state,
            done
        )

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones
        ) = self.memory.sample(
            self.batch_size
        )

        states = torch.tensor(
            states,
            dtype=torch.float32,
            device=self.device
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=self.device
        ).unsqueeze(1)

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device
        )

        next_states = torch.tensor(
            next_states,
            dtype=torch.float32,
            device=self.device
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=self.device
        )

        current_q = (
            self.policy_net(
                states
            )
            .gather(
                1,
                actions
            )
            .squeeze(1)
        )

        with torch.no_grad():
            next_q = (
                self.target_net(
                    next_states
                )
                .max(
                    1
                )[0]
            )

            target_q = (
                rewards
                + (
                    1 - dones
                )
                * self.gamma
                * next_q
            )

        loss = self.loss_function(
            current_q,
            target_q
        )

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            1.0
        )

        self.optimizer.step()

        self.train_steps += 1

        if (
            self.train_steps
            % self.target_update_frequency
            == 0
        ):
            self.target_net.load_state_dict(
                self.policy_net.state_dict()
            )

        return float(
            loss.item()
        )

    def finish_episode(self):
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon
            * self.epsilon_decay
        )


def train_q_learning(
    episodes,
    steps,
    difficulty,
    progress_callback=None
):
    agent = QLearningAgent()

    rewards_history = []

    for episode in range(
        episodes
    ):
        env = ResourceEnvironment(
            difficulty=difficulty,
            seed=SEED + episode
        )

        state = env.reset()

        total_reward = 0

        for step in range(
            steps
        ):
            action = agent.choose_action(
                state,
                training=True
            )

            (
                next_state,
                reward,
                _
            ) = env.step(
                action
            )

            agent.update(
                state,
                action,
                reward,
                next_state,
                step == steps - 1
            )

            state = next_state

            total_reward += reward

        agent.finish_episode()

        rewards_history.append(
            total_reward
        )

        if progress_callback:
            progress_callback(
                (episode + 1)
                / episodes,
                episode + 1,
                total_reward
            )

    return (
        agent,
        rewards_history
    )


def train_dqn(
    episodes,
    steps,
    difficulty,
    progress_callback=None
):
    agent = DQNAgent()

    rewards_history = []

    losses = []

    for episode in range(
        episodes
    ):
        env = ResourceEnvironment(
            difficulty=difficulty,
            seed=SEED + episode
        )

        state = env.reset()

        total_reward = 0

        for step in range(
            steps
        ):
            action = agent.choose_action(
                state,
                training=True
            )

            (
                next_state,
                reward,
                _
            ) = env.step(
                action
            )

            done = (
                step
                == steps - 1
            )

            agent.remember(
                state,
                action,
                reward,
                next_state,
                done
            )

            loss = agent.replay()

            if loss is not None:
                losses.append(
                    loss
                )

            state = next_state

            total_reward += reward

        agent.finish_episode()

        rewards_history.append(
            total_reward
        )

        if progress_callback:
            progress_callback(
                (episode + 1)
                / episodes,
                episode + 1,
                total_reward
            )

    return (
        agent,
        rewards_history,
        losses
    )


def generate_evaluation_traces(
    num_traces,
    steps,
    difficulty
):
    traces = []

    for trace_id in range(
        num_traces
    ):
        generator = WorkloadGenerator(
            difficulty=difficulty,
            seed=10000 + trace_id
        )

        current = generator.reset()

        trace = [current]

        for _ in range(
            steps
        ):
            trace.append(
                generator.step()
            )

        traces.append(
            trace
        )

    return traces


def generate_live_trace(
    steps,
    seed=2026
):
    rng = np.random.default_rng(
        seed
    )

    targets = [
        40,
        46,
        55,
        72,
        88,
        74,
        58,
        42,
        32,
        48
    ]

    phase_length = max(
        1,
        steps // len(targets)
    )

    values = []

    current = targets[0]

    for step in range(
        steps
    ):
        phase = min(
            step // phase_length,
            len(targets) - 1
        )

        target = targets[phase]

        current += (
            0.32
            * (target - current)
            + rng.normal(
                0,
                2.0
            )
        )

        if step == int(
            steps * 0.28
        ):
            current += 12

        if step == int(
            steps * 0.55
        ):
            current -= 14

        values.append(
            float(
                np.clip(
                    current,
                    25,
                    95
                )
            )
        )

    return values


def rule_based_action(env):
    utilization = env.utilization
    queue = env.queue
    latency = env.latency

    if (
        latency > 650
        or queue > 15
        or utilization > 88
    ):
        return 20

    if (
        latency > 450
        or queue > 8
        or utilization > 78
    ):
        return 10

    if (
        utilization < 42
        and queue < 2
        and env.resources > 40
    ):
        return -20

    if (
        utilization < 50
        and queue < 3
        and env.resources > 30
    ):
        return -10

    return 0


def static_action(env):
    target = 60

    if env.resources < target:
        return min(
            10,
            target - env.resources
        )

    if env.resources > target:
        return max(
            -10,
            target - env.resources
        )

    return 0


def evaluate_strategy(
    strategy_name,
    agent,
    traces,
    difficulty
):
    records = []

    for trace_id, trace in enumerate(
        traces
    ):
        env = ResourceEnvironment(
            difficulty=difficulty,
            seed=50000 + trace_id
        )

        env.reset(
            initial_workload=trace[0]
        )

        for step in range(
            1,
            len(trace)
        ):
            workload = trace[step]

            resources_before = (
                env.resources
            )

            if strategy_name in [
                "DQN",
                "Q-Learning"
            ]:
                action = agent.choose_action(
                    env.get_state(),
                    training=False
                )

            elif strategy_name == "Rule-Based":
                action = rule_based_action(
                    env
                )

            else:
                action = static_action(
                    env
                )

            (
                _,
                reward,
                info
            ) = env.step(
                action,
                workload=workload
            )

            records.append(
                {
                    "Strategy": strategy_name,
                    "Trace": trace_id + 1,
                    "Step": step,
                    "Workload": info[
                        "workload"
                    ],
                    "Resources": info[
                        "resources"
                    ],
                    "Resources_Before": resources_before,
                    "Action": ACTION_NAMES[
                        action
                    ],
                    "Action_Value": action,
                    "Queue": info[
                        "queue"
                    ],
                    "Latency": info[
                        "latency"
                    ],
                    "Utilization": info[
                        "utilization"
                    ],
                    "Reward": reward,
                    "SLA_Violation": info[
                        "sla_violation"
                    ]
                }
            )

    return pd.DataFrame(
        records
    )


def summarize_results(df):
    summaries = []

    for strategy, group in df.groupby(
        "Strategy"
    ):
        total_steps = len(group)

        violations = int(
            group[
                "SLA_Violation"
            ].sum()
        )

        summaries.append(
            {
                "Strategy": strategy,
                "Avg Latency (ms)": (
                    group[
                        "Latency"
                    ].mean()
                ),
                "Avg Queue": (
                    group[
                        "Queue"
                    ].mean()
                ),
                "Avg Resources": (
                    group[
                        "Resources"
                    ].mean()
                ),
                "Avg Utilization (%)": (
                    group[
                        "Utilization"
                    ].mean()
                ),
                "SLA Violations": violations,
                "SLA Compliance (%)": (
                    (
                        1
                        - violations
                        / total_steps
                    )
                    * 100
                ),
                "Avg Reward": (
                    group[
                        "Reward"
                    ].mean()
                ),
                "Total Reward": (
                    group[
                        "Reward"
                    ].sum()
                )
            }
        )

    return pd.DataFrame(
        summaries
    )


def improvement(
    baseline,
    value
):
    if baseline == 0:
        return None

    return (
        (
            baseline
            - value
        )
        / abs(baseline)
    ) * 100


def build_comparison_table(
    summary
):
    static_rows = summary[
        summary[
            "Strategy"
        ] == "Static"
    ]

    if static_rows.empty:
        return pd.DataFrame()

    static = static_rows.iloc[0]

    rows = []

    for _, row in summary.iterrows():
        latency_change = improvement(
            static[
                "Avg Latency (ms)"
            ],
            row[
                "Avg Latency (ms)"
            ]
        )

        queue_change = improvement(
            static[
                "Avg Queue"
            ],
            row[
                "Avg Queue"
            ]
        )

        sla_change = improvement(
            static[
                "SLA Violations"
            ],
            row[
                "SLA Violations"
            ]
        )

        resource_change = (
            (
                row[
                    "Avg Resources"
                ]
                - static[
                    "Avg Resources"
                ]
            )
            / static[
                "Avg Resources"
            ]
        ) * 100

        rows.append(
            {
                "Strategy": row[
                    "Strategy"
                ],
                "Avg Latency (ms)": round(
                    row[
                        "Avg Latency (ms)"
                    ],
                    2
                ),
                "Latency vs Static": (
                    "N/A"
                    if latency_change
                    is None
                    else f"{latency_change:.1f}%"
                ),
                "Avg Queue": round(
                    row[
                        "Avg Queue"
                    ],
                    2
                ),
                "Queue vs Static": (
                    "N/A"
                    if queue_change
                    is None
                    else f"{queue_change:.1f}%"
                ),
                "Avg Resources": round(
                    row[
                        "Avg Resources"
                    ],
                    2
                ),
                "Resource Change": (
                    f"{resource_change:+.1f}%"
                ),
                "Avg Utilization (%)": round(
                    row[
                        "Avg Utilization (%)"
                    ],
                    2
                ),
                "SLA Violations": int(
                    row[
                        "SLA Violations"
                    ]
                ),
                "SLA Compliance (%)": round(
                    row[
                        "SLA Compliance (%)"
                    ],
                    2
                ),
                "Avg Reward": round(
                    row[
                        "Avg Reward"
                    ],
                    2
                )
            }
        )

    return pd.DataFrame(
        rows
    )


def plot_training_rewards(
    rewards,
    title
):
    fig, ax = plt.subplots(
        figsize=(10, 4)
    )

    ax.plot(
        rewards,
        linewidth=1.2,
        label="Episode Reward"
    )

    if len(rewards) >= 10:
        window = min(
            20,
            max(
                5,
                len(rewards) // 10
            )
        )

        rolling = (
            pd.Series(
                rewards
            )
            .rolling(
                window
            )
            .mean()
        )

        ax.plot(
            rolling,
            linewidth=2,
            label="Moving Average"
        )

    ax.set_title(
        title
    )

    ax.set_xlabel(
        "Episode"
    )

    ax.set_ylabel(
        "Reward"
    )

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    return fig


def run_live_simulation(
    agent,
    algorithm,
    difficulty,
    steps
):
    trace = generate_live_trace(
        steps
    )

    env = ResourceEnvironment(
        difficulty=difficulty,
        seed=90000
    )

    env.reset(
        initial_workload=trace[0]
    )

    records = []

    for step in range(
        1,
        len(trace)
    ):
        workload = trace[step]

        resources_before = (
            env.resources
        )

        action = agent.choose_action(
            env.get_state(),
            training=False
        )

        (
            _,
            reward,
            info
        ) = env.step(
            action,
            workload=workload
        )

        records.append(
            {
                "Step": step,
                "Workload": info[
                    "workload"
                ],
                "Resources Before": (
                    resources_before
                ),
                "Action": ACTION_NAMES[
                    action
                ],
                "Action Value": action,
                "Resources After": (
                    info[
                        "resources"
                    ]
                ),
                "Queue": info[
                    "queue"
                ],
                "Latency": info[
                    "latency"
                ],
                "Utilization": info[
                    "utilization"
                ],
                "Reward": reward,
                "SLA": (
                    "Violation"
                    if info[
                        "sla_violation"
                    ]
                    else "Within Limit"
                )
            }
        )

    return pd.DataFrame(
        records
    )


def initialize_session():
    defaults = {
        "agent": None,
        "algorithm": None,
        "training_rewards": [],
        "training_losses": [],
        "comparison_dqn": None,
        "comparison_qlearning": None,
        "comparison_results": None,
        "comparison_summary": None,
        "live_results": None
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[
                key
            ] = value


initialize_session()


st.title(
    "⚙️ ResourceRL"
)

st.subheader(
    "Adaptive Resource Allocation using Reinforcement Learning"
)

st.write(
    "ResourceRL dynamically allocates computational resources "
    "according to changing workload conditions while balancing "
    "latency, queue pressure, SLA compliance and resource efficiency."
)


with st.sidebar:
    st.header(
        "Configuration"
    )

    algorithm = st.selectbox(
        "Primary RL Algorithm",
        [
            "DQN",
            "Q-Learning"
        ]
    )

    difficulty = st.selectbox(
        "Workload Difficulty",
        [
            "Low",
            "Medium",
            "High"
        ],
        index=1
    )

    episodes = st.slider(
        "Training Episodes",
        min_value=50,
        max_value=600,
        value=200,
        step=50
    )

    steps = st.slider(
        "Steps per Episode",
        min_value=30,
        max_value=120,
        value=60,
        step=10
    )

    evaluation_traces = st.slider(
        "Evaluation Traces",
        min_value=3,
        max_value=15,
        value=5,
        step=1
    )

    evaluation_steps = st.slider(
        "Evaluation Steps",
        min_value=30,
        max_value=100,
        value=60,
        step=10
    )

    live_steps = st.slider(
        "Live Simulation Steps",
        min_value=15,
        max_value=50,
        value=25,
        step=5
    )


st.markdown(
    "---"
)

metric1, metric2, metric3 = st.columns(
    3
)

with metric1:
    st.metric(
        "Resource Range",
        f"{MIN_RESOURCES}–{MAX_RESOURCES}"
    )

with metric2:
    st.metric(
        "Adaptive Actions",
        "5"
    )

with metric3:
    st.metric(
        "SLA Latency Limit",
        f"{SLA_LIMIT} ms"
    )


st.caption(
    "The agent can dynamically scale resources by "
    "−20, −10, 0, +10 or +20 units at each decision step."
)


st.markdown(
    "---"
)


train_col, eval_col, live_col = st.columns(
    3
)

with train_col:
    train_clicked = st.button(
        "🚀 Train Agent",
        type="primary",
        use_container_width=True
    )

with eval_col:
    evaluate_clicked = st.button(
        "📊 Evaluate Strategies",
        use_container_width=True
    )

with live_col:
    live_clicked = st.button(
        "▶️ Run Live Simulation",
        use_container_width=True
    )


if train_clicked:
    set_seed(
        SEED
    )

    progress = st.progress(
        0
    )

    status = st.empty()

    if algorithm == "DQN":
        status.info(
            f"Training DQN for {episodes} episodes..."
        )

        def callback(
            progress_value,
            episode,
            reward
        ):
            progress.progress(
                min(
                    int(
                        progress_value
                        * 100
                    ),
                    100
                )
            )

            status.info(
                f"DQN training: "
                f"Episode {episode}/{episodes} | "
                f"Latest reward: {reward:.1f}"
            )

        (
            agent,
            rewards,
            losses
        ) = train_dqn(
            episodes=episodes,
            steps=steps,
            difficulty=difficulty,
            progress_callback=callback
        )

    else:
        status.info(
            f"Training Q-Learning for {episodes} episodes..."
        )

        def callback(
            progress_value,
            episode,
            reward
        ):
            progress.progress(
                min(
                    int(
                        progress_value
                        * 100
                    ),
                    100
                )
            )

            status.info(
                f"Q-Learning training: "
                f"Episode {episode}/{episodes} | "
                f"Latest reward: {reward:.1f}"
            )

        (
            agent,
            rewards
        ) = train_q_learning(
            episodes=episodes,
            steps=steps,
            difficulty=difficulty,
            progress_callback=callback
        )

        losses = []

    progress.progress(
        100
    )

    status.success(
        f"{algorithm} training completed."
    )

    st.session_state.agent = agent
    st.session_state.algorithm = algorithm
    st.session_state.training_rewards = rewards
    st.session_state.training_losses = losses
    st.session_state.comparison_results = None
    st.session_state.comparison_summary = None
    st.session_state.live_results = None


if st.session_state.agent is not None:
    st.markdown(
        "---"
    )

    st.header(
        "📈 Training Results"
    )

    rewards = (
        st.session_state
        .training_rewards
    )

    metric1, metric2, metric3, metric4 = (
        st.columns(4)
    )

    recent_count = min(
        20,
        len(rewards)
    )

    recent_reward = np.mean(
        rewards[
            -recent_count:
        ]
    )

    best_reward = np.max(
        rewards
    )

    with metric1:
        st.metric(
            "Recent Reward",
            f"{recent_reward:.1f}"
        )

    with metric2:
        st.metric(
            "Best Reward",
            f"{best_reward:.1f}"
        )

    with metric3:
        st.metric(
            "Training Episodes",
            len(rewards)
        )

    with metric4:
        st.metric(
            "Algorithm",
            st.session_state.algorithm
        )

    st.pyplot(
        plot_training_rewards(
            rewards,
            f"{st.session_state.algorithm} Training Reward"
        )
    )

    if (
        st.session_state.algorithm
        == "DQN"
        and st.session_state.training_losses
    ):
        st.subheader(
            "DQN Training Loss"
        )

        loss_fig, loss_ax = plt.subplots(
            figsize=(10, 3.5)
        )

        loss_ax.plot(
            st.session_state.training_losses,
            linewidth=1
        )

        loss_ax.set_xlabel(
            "Optimization Step"
        )

        loss_ax.set_ylabel(
            "Loss"
        )

        loss_ax.set_title(
            "DQN Loss"
        )

        loss_ax.grid(
            alpha=0.25
        )

        loss_fig.tight_layout()

        st.pyplot(
            loss_fig
        )


if evaluate_clicked:
    if st.session_state.agent is None:
        st.warning(
            "Train an RL agent first."
        )

    else:
        st.markdown(
            "---"
        )

        st.header(
            "📊 Strategy Evaluation"
        )

        eval_status = st.empty()

        traces = (
            generate_evaluation_traces(
                num_traces=evaluation_traces,
                steps=evaluation_steps,
                difficulty=difficulty
            )
        )

        primary_agent = (
            st.session_state.agent
        )

        primary_algorithm = (
            st.session_state.algorithm
        )

        all_results = []

        eval_status.info(
            "Evaluating the trained RL agent..."
        )

        all_results.append(
            evaluate_strategy(
                primary_algorithm,
                primary_agent,
                traces,
                difficulty
            )
        )

        secondary_algorithm = (
            "Q-Learning"
            if primary_algorithm
            == "DQN"
            else "DQN"
        )

        if secondary_algorithm == "DQN":
            secondary_agent = (
                st.session_state
                .comparison_dqn
            )

        else:
            secondary_agent = (
                st.session_state
                .comparison_qlearning
            )

        if secondary_agent is None:
            eval_status.info(
                f"Training {secondary_algorithm} "
                "as the secondary RL comparison..."
            )

            if secondary_algorithm == "DQN":
                (
                    secondary_agent,
                    _,
                    _
                ) = train_dqn(
                    episodes=episodes,
                    steps=steps,
                    difficulty=difficulty
                )

                st.session_state.comparison_dqn = (
                    secondary_agent
                )

            else:
                (
                    secondary_agent,
                    _
                ) = train_q_learning(
                    episodes=episodes,
                    steps=steps,
                    difficulty=difficulty
                )

                st.session_state.comparison_qlearning = (
                    secondary_agent
                )

        all_results.append(
            evaluate_strategy(
                secondary_algorithm,
                secondary_agent,
                traces,
                difficulty
            )
        )

        eval_status.info(
            "Evaluating Rule-Based and Static baselines..."
        )

        all_results.append(
            evaluate_strategy(
                "Rule-Based",
                None,
                traces,
                difficulty
            )
        )

        all_results.append(
            evaluate_strategy(
                "Static",
                None,
                traces,
                difficulty
            )
        )

        combined_results = pd.concat(
            all_results,
            ignore_index=True
        )

        summary = summarize_results(
            combined_results
        )

        comparison = (
            build_comparison_table(
                summary
            )
        )

        st.session_state.comparison_results = (
            combined_results
        )

        st.session_state.comparison_summary = (
            summary
        )

        eval_status.success(
            "Evaluation completed using identical workload traces."
        )

        st.subheader(
            "Strategy Comparison"
        )

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Latency and queue improvements are measured against "
            "Static allocation. Resource Change is reported "
            "separately because lower resource usage is only "
            "valuable when service quality remains acceptable."
        )

        st.subheader(
            "Detailed Metrics"
        )

        display_summary = (
            summary.copy()
        )

        numeric_columns = [
            "Avg Latency (ms)",
            "Avg Queue",
            "Avg Resources",
            "Avg Utilization (%)",
            "Avg Reward",
            "Total Reward"
        ]

        for column in numeric_columns:
            display_summary[
                column
            ] = display_summary[
                column
            ].round(2)

        display_summary[
            "SLA Compliance (%)"
        ] = display_summary[
            "SLA Compliance (%)"
        ].round(2)

        st.dataframe(
            display_summary,
            use_container_width=True,
            hide_index=True
        )

        chart_col1, chart_col2 = (
            st.columns(2)
        )

        with chart_col1:
            latency_fig, latency_ax = (
                plt.subplots(
                    figsize=(7, 4)
                )
            )

            for strategy in (
                combined_results[
                    "Strategy"
                ].unique()
            ):
                data = (
                    combined_results[
                        combined_results[
                            "Strategy"
                        ] == strategy
                    ]
                )

                grouped = (
                    data.groupby(
                        "Step"
                    )[
                        "Latency"
                    ].mean()
                )

                latency_ax.plot(
                    grouped.index,
                    grouped.values,
                    label=strategy
                )

            latency_ax.axhline(
                SLA_LIMIT,
                linestyle="--",
                linewidth=1.5,
                label="SLA Limit"
            )

            latency_ax.set_title(
                "Average Latency Across Evaluation"
            )

            latency_ax.set_xlabel(
                "Step"
            )

            latency_ax.set_ylabel(
                "Latency (ms)"
            )

            latency_ax.legend()

            latency_ax.grid(
                alpha=0.25
            )

            latency_fig.tight_layout()

            st.pyplot(
                latency_fig
            )

        with chart_col2:
            resource_fig, resource_ax = (
                plt.subplots(
                    figsize=(7, 4)
                )
            )

            for strategy in (
                combined_results[
                    "Strategy"
                ].unique()
            ):
                data = (
                    combined_results[
                        combined_results[
                            "Strategy"
                        ] == strategy
                    ]
                )

                grouped = (
                    data.groupby(
                        "Step"
                    )[
                        "Resources"
                    ].mean()
                )

                resource_ax.plot(
                    grouped.index,
                    grouped.values,
                    label=strategy
                )

            resource_ax.set_title(
                "Dynamic Resource Allocation"
            )

            resource_ax.set_xlabel(
                "Step"
            )

            resource_ax.set_ylabel(
                "Resources"
            )

            resource_ax.legend()

            resource_ax.grid(
                alpha=0.25
            )

            resource_fig.tight_layout()

            st.pyplot(
                resource_fig
            )

        st.subheader(
            "Workload vs Resource Response"
        )

        response_fig, response_ax = (
            plt.subplots(
                figsize=(11, 4)
            )
        )

        primary_data = (
            combined_results[
                combined_results[
                    "Strategy"
                ]
                == primary_algorithm
            ]
        )

        average_primary = (
            primary_data
            .groupby("Step")
            .agg(
                {
                    "Workload": "mean",
                    "Resources": "mean"
                }
            )
        )

        response_ax.plot(
            average_primary.index,
            average_primary[
                "Workload"
            ],
            label="Workload",
            linewidth=2
        )

        response_ax.plot(
            average_primary.index,
            average_primary[
                "Resources"
            ],
            label=f"{primary_algorithm} Resources",
            linewidth=2
        )

        response_ax.set_title(
            f"{primary_algorithm}: Workload vs Resource Response"
        )

        response_ax.set_xlabel(
            "Step"
        )

        response_ax.set_ylabel(
            "Level"
        )

        response_ax.legend()

        response_ax.grid(
            alpha=0.25
        )

        response_fig.tight_layout()

        st.pyplot(
            response_fig
        )


if live_clicked:
    if st.session_state.agent is None:
        st.warning(
            "Train an RL agent first."
        )

    else:
        st.markdown(
            "---"
        )

        st.header(
            "▶️ Live Adaptive Simulation"
        )

        live_results = (
            run_live_simulation(
                agent=st.session_state.agent,
                algorithm=st.session_state.algorithm,
                difficulty=difficulty,
                steps=live_steps
            )
        )

        st.session_state.live_results = (
            live_results
        )

        last_row = (
            live_results.iloc[-1]
        )

        metric1, metric2, metric3, metric4 = (
            st.columns(4)
        )

        with metric1:
            st.metric(
                "Current Workload",
                f"{last_row['Workload']:.1f}"
            )

        with metric2:
            st.metric(
                "Current Resources",
                f"{last_row['Resources After']:.0f}"
            )

        with metric3:
            st.metric(
                "Current Queue",
                f"{last_row['Queue']:.1f}"
            )

        with metric4:
            st.metric(
                "Current Latency",
                f"{last_row['Latency']:.0f} ms"
            )

        plot_col1, plot_col2 = (
            st.columns(2)
        )

        with plot_col1:
            fig1, ax1 = plt.subplots(
                figsize=(7, 4)
            )

            ax1.plot(
                live_results[
                    "Step"
                ],
                live_results[
                    "Workload"
                ],
                label="Workload",
                linewidth=2
            )

            ax1.plot(
                live_results[
                    "Step"
                ],
                live_results[
                    "Resources After"
                ],
                label="Resources",
                linewidth=2
            )

            ax1.set_title(
                "Workload vs Dynamic Resource Allocation"
            )

            ax1.set_xlabel(
                "Step"
            )

            ax1.set_ylabel(
                "Level"
            )

            ax1.legend()

            ax1.grid(
                alpha=0.25
            )

            fig1.tight_layout()

            st.pyplot(
                fig1
            )

        with plot_col2:
            fig2, ax2 = plt.subplots(
                figsize=(7, 4)
            )

            ax2.plot(
                live_results[
                    "Step"
                ],
                live_results[
                    "Latency"
                ],
                linewidth=2,
                label="Latency"
            )

            ax2.axhline(
                SLA_LIMIT,
                linestyle="--",
                linewidth=1.5,
                label="SLA Limit"
            )

            ax2.set_title(
                "Latency and SLA Compliance"
            )

            ax2.set_xlabel(
                "Step"
            )

            ax2.set_ylabel(
                "Latency (ms)"
            )

            ax2.legend()

            ax2.grid(
                alpha=0.25
            )

            fig2.tight_layout()

            st.pyplot(
                fig2
            )

        fig3, ax3 = plt.subplots(
            figsize=(10, 4)
        )

        ax3.plot(
            live_results[
                "Step"
            ],
            live_results[
                "Queue"
            ],
            linewidth=2
        )

        ax3.set_title(
            "Live Queue Pressure"
        )

        ax3.set_xlabel(
            "Step"
        )

        ax3.set_ylabel(
            "Queue"
        )

        ax3.grid(
            alpha=0.25
        )

        fig3.tight_layout()

        st.pyplot(
            fig3
        )

        st.subheader(
            "Live Decision Timeline"
        )

        display_live = (
            live_results.copy()
        )

        display_live[
            "Workload"
        ] = display_live[
            "Workload"
        ].round(1)

        display_live[
            "Queue"
        ] = display_live[
            "Queue"
        ].round(1)

        display_live[
            "Latency"
        ] = display_live[
            "Latency"
        ].round(0)

        display_live[
            "Utilization"
        ] = display_live[
            "Utilization"
        ].round(1)

        display_live[
            "Resources Before"
        ] = display_live[
            "Resources Before"
        ].round(0)

        display_live[
            "Resources After"
        ] = display_live[
            "Resources After"
        ].round(0)

        st.dataframe(
            display_live,
            use_container_width=True,
            hide_index=True
        )

        increases = int(
            (
                live_results[
                    "Action Value"
                ] > 0
            ).sum()
        )

        decreases = int(
            (
                live_results[
                    "Action Value"
                ] < 0
            ).sum()
        )

        maintains = int(
            (
                live_results[
                    "Action Value"
                ] == 0
            ).sum()
        )

        strong_scaling = int(
            (
                live_results[
                    "Action Value"
                ].abs()
                >= 20
            ).sum()
        )

        st.subheader(
            "Adaptive Behavior"
        )

        behavior1, behavior2, behavior3, behavior4 = (
            st.columns(4)
        )

        with behavior1:
            st.metric(
                "Scale Up",
                increases
            )

        with behavior2:
            st.metric(
                "Scale Down",
                decreases
            )

        with behavior3:
            st.metric(
                "Maintain",
                maintains
            )

        with behavior4:
            st.metric(
                "Large Adjustments",
                strong_scaling
            )

        st.info(
            "The workload intentionally changes over time, including "
            "both demand increases and decreases. The trained agent "
            "can dynamically select the magnitude and direction of "
            "resource allocation at every decision step."
        )


st.markdown(
    "---"
)

st.header(
    "🧠 How ResourceRL Works"
)

overview_col1, overview_col2 = (
    st.columns(2)
)

with overview_col1:
    st.markdown(
        """
**Environment**

- Simulates changing computational workload
- Tracks latency, queue, utilization and resources
- Applies an SLA latency constraint
- Allows workload to rise and fall naturally

**Available Actions**

- Scale down by 20
- Scale down by 10
- Maintain resources
- Scale up by 10
- Scale up by 20

**State**

- Workload
- Queue pressure
- Current resources
- Latency
- Utilization
- Workload trend
- Resource pressure
"""
    )

with overview_col2:
    st.markdown(
        """
**Reinforcement Learning**

- DQN learns a neural-network-based allocation policy
- Q-Learning provides a classical tabular RL baseline
- The agent receives rewards for efficient and reliable allocation

**Baselines**

- Static allocation
- Rule-Based allocation

**Objective**

Minimize latency and queue pressure while maintaining SLA
compliance and avoiding unnecessary resource allocation.

**Evaluation**

All strategies receive the same workload traces so their
performance can be compared fairly.
"""
    )


st.markdown(
    "---"
)

st.subheader(
    "ResourceRL Objective"
)

st.write(
    "The system does not attempt to keep latency or resource "
    "allocation constant. Instead, it dynamically adapts resource "
    "capacity to changing workload conditions. Higher workloads "
    "may require more resources, while lower workloads can allow "
    "the agent to scale resources down and improve efficiency."
)


st.caption(
    "ResourceRL — Adaptive Resource Allocation using Reinforcement Learning"
)