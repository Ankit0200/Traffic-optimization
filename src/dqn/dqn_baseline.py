"""
DQN Baseline for Adaptive Traffic Signal Control
=================================================
This script implements a Deep Q-Network (DQN) reinforcement learning agent
that learns to control a traffic signal at a single intersection in SUMO.

HOW IT WORKS:
    1. The agent OBSERVES the intersection (queue lengths, speeds, current phase)
    2. The agent DECIDES to either keep the current signal or switch it
    3. SUMO simulates 5 seconds of traffic with that decision
    4. The agent gets a REWARD: negative of current total waiting time (less waiting = better)
    5. The agent stores this experience and learns from it (neural network training)
    6. Repeat until the simulation ends, then start a new episode

The trained DQN is compared against:
    - Fixed timer (42s NS / 42s EW, no adaptation)
    - Our adaptive controller (queue + arrival rate + LSTM, Optuna-optimized)

Usage:
    python dqn_baseline.py train                          # train on all scenarios
    python dqn_baseline.py train --versions v3 v5         # train on specific versions
    python dqn_baseline.py train --timesteps 200000       # more training steps
    python dqn_baseline.py train --mode global            # one brain for all scenarios
    python dqn_baseline.py eval                           # evaluate trained model
    python dqn_baseline.py eval --versions v3             # evaluate on specific version
"""

import os
import sys
import json
import shutil
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
import uuid

# TraCI is SUMO's Python API — it lets us control the simulation via TCP socket.
# We can query vehicle positions, change traffic lights, step the simulation, etc.
import traci

# Gymnasium (formerly OpenAI Gym) provides a standard interface for RL environments.
# Any RL library (stable-baselines3, RLlib, etc.) can train on a Gym environment.
import gymnasium as gym
from gymnasium import spaces

# stable-baselines3 provides ready-to-use RL algorithms (DQN, PPO, A2C, etc.)
# DQN = Deep Q-Network: learns a Q-value function Q(state, action) that estimates
# the expected future reward for taking an action in a given state.
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

# ── Project paths ────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent       # src/dqn/
PROJECT_ROOT = SCRIPT_DIR.parents[1]               # Research_zhou/
SUMO_DIR = PROJECT_ROOT / "data" / "sumo_simulation"  # SUMO configs live here

# ── Simulation Constants ────────────────────────────────────────────────
# SUMO advances in small time steps. 1/30 second per step = 30 steps per second
# This matches video FPS so each simulation step = one video frame
STEP_LENGTH = 1.0 / 30

# The DQN agent makes a decision every 5 real seconds (not every step).
# This means 150 SUMO steps pass between each decision.
# Why? Traffic lights shouldn't change every 1/30th of a second — that would
# cause dangerous rapid flickering. 5 seconds gives drivers time to react.
DECISION_INTERVAL = 5.0
STEPS_PER_DECISION = int(DECISION_INTERVAL / STEP_LENGTH)  # = 150 steps

# Safety constraints for signal timing
MIN_GREEN = 10.0   # Minimum green time before allowing a switch (seconds)
YELLOW_DUR = 3.0   # Yellow phase duration before switching to other direction

# ── Traffic Light State Strings ─────────────────────────────────────────
# Each character represents one lane's signal: G=green, g=protected green,
# r=red, y=yellow. The 20 characters cover all lanes at intersection "C".
# First 5 chars = northbound lanes, next 5 = eastbound, next 5 = southbound, last 5 = westbound
NS_GREEN_STATE  = "GGGggrrrrrGGGggrrrrr"  # North-South has green, East-West has red
NS_YELLOW_STATE = "yyyyyrrrrryyyyyrrrrr"  # North-South transitioning to red
EW_GREEN_STATE  = "rrrrrGGGggrrrrrGGGgg"  # East-West has green, North-South has red
EW_YELLOW_STATE = "rrrrryyyyyrrrrryyyyy"  # East-West transitioning to red
ALL_RED_STATE    = "rrrrrrrrrrrrrrrrrrrr"  # All red (not used in normal operation)

# The four approach roads leading to the center intersection "C"
EDGE_IDS = ["N_to_C", "S_to_C", "E_to_C", "W_to_C"]


# ═══════════════════════════════════════════════════════════════════════════
# PART 1: THE ENVIRONMENT
# This is the core of the DQN system. It wraps SUMO into a standard Gym
# environment so that any RL algorithm can interact with it.
#
# The RL loop:  Agent  ──action──>  Environment (SUMO)  ──obs, reward──>  Agent
# ═══════════════════════════════════════════════════════════════════════════

class SumoIntersectionEnv(gym.Env):
    """
    Gymnasium environment wrapping a single SUMO intersection.

    This class translates between the RL world (observations, actions, rewards)
    and the SUMO simulation world (vehicles, traffic lights, roads).

    Observation (10 values):
        [0-3] Queue length per edge: how many cars are stopped (speed < 0.1 m/s)
              on N_to_C, S_to_C, E_to_C, W_to_C
        [4-7] Mean speed per edge: average speed of vehicles on each approach road
        [8]   Current phase: 0 = NS has green, 1 = EW has green
        [9]   Time in current phase / 60.0 (normalized to 0-1 range)

    Actions:
        0 = keep current phase (don't change the light)
        1 = switch to other phase (NS→EW or EW→NS, with yellow transition)

    Reward:
        R = -current_total_wait
        Every decision step, the agent pays a penalty equal to how much waiting
        is happening right now. More waiting = more negative reward.
        This teaches the agent: "minimize overall waiting across the episode"
    """

    metadata = {"render_modes": []}

    def __init__(self, version="v3", sumo_cfg_dir="adaptive_timer", track_vehicles=False):
        super().__init__()

        # Which traffic scenario to simulate (v2=Demand Flip, v3=CBD Rush, etc.)
        self.version = version

        # Vehicle tracking: only enabled during evaluation (not training — too slow)
        # When True, records per-vehicle wait times every frame for accurate measurement
        self.track_vehicles = track_vehicles

        # Path to the SUMO config file that defines the road network and traffic demand
        self.cfg_path = str(SUMO_DIR / f"{sumo_cfg_dir}/{version}/intersection.sumocfg")
        assert os.path.exists(self.cfg_path), f"Config not found: {self.cfg_path}"

        # Find the SUMO binary (headless, no GUI — faster for training)
        self.sumo_bin = shutil.which("sumo")
        assert self.sumo_bin, "sumo binary not found on PATH"

        # Each environment instance gets a unique label for its TraCI connection.
        # This prevents conflicts when multiple environments run simultaneously.
        self._conn_label = f"dqn_{uuid.uuid4().hex[:8]}"

        # Define what the agent can SEE (observation space)
        # Box = continuous values within a range
        # [queue_N, queue_S, queue_E, queue_W, speed_N, speed_S, speed_E, speed_W, phase, timer]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([50, 50, 50, 50, 15, 15, 15, 15, 1, 1], dtype=np.float32),
        )

        # Define what the agent can DO (action space)
        # Discrete(2) means two choices: 0 or 1
        self.action_space = spaces.Discrete(2)  # 0=keep, 1=switch

        # Internal state tracking
        self._sumo_running = False   # Is SUMO currently running?
        self._conn = None            # TraCI connection object
        self.current_phase = 0       # 0 = NS green, 1 = EW green
        self.phase_timer = 0.0       # How long current phase has been active
        self.yellow_active = False   # Is the light currently yellow (transitioning)?
        self.yellow_timer = 0.0      # How long yellow has been active

        # Vehicle tracking state (only used when track_vehicles=True)
        self.veh_last_wait = {}    # vehicle_id → latest accumulated wait
        self.veh_total_wait = []   # final wait for departed vehicles
        self.arrived = 0           # count of vehicles that completed their trip

    def _start_sumo(self):
        """Launch a fresh SUMO simulation process and connect via TraCI."""
        # If SUMO is already running from a previous episode, shut it down first
        if self._sumo_running:
            try:
                self._conn.close()
            except:
                pass
            self._sumo_running = False
            self._conn = None

        # Generate a new unique label for this connection
        # (old label was freed by close() above)
        self._conn_label = f"dqn_{uuid.uuid4().hex[:8]}"

        # Build the SUMO command — runs headless (no GUI window)
        sumo_cmd = [
            self.sumo_bin, "-c", self.cfg_path,
            "--step-length", str(STEP_LENGTH),  # Time per simulation step
            "--no-warnings",                     # Suppress SUMO warnings
            "--no-step-log",                     # Don't log every step
            "--waiting-time-memory", "3600",     # Remember up to 1 hour of waiting per car
                                                 # (default is 100s, which hides worst cases)
        ]

        # traci.start() launches SUMO as a subprocess and connects via TCP.
        # numRetries=60 means it will try for up to 60 seconds to connect
        # (SUMO needs time to load the network and route files).
        # We suppress stdout because traci prints "Retrying in 1 seconds" messages.
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            traci.start(sumo_cmd, label=self._conn_label, numRetries=60)

        # Get a connection handle — all subsequent SUMO commands go through this
        self._conn = traci.getConnection(self._conn_label)
        self._sumo_running = True

        # Let a few simulation steps run so vehicles can spawn and enter the network
        for _ in range(15):
            self._conn.simulationStep()

        # Take over the traffic light controller at intersection "C".
        # By default, SUMO uses its own signal plan. This overrides it
        # so our DQN agent has full control. Start with NS green.
        self._conn.trafficlight.setRedYellowGreenState("C", NS_GREEN_STATE)

    def _get_obs(self):
        """
        Query SUMO for current intersection state and return as a numpy array.

        This is what the DQN "sees" every 5 seconds when making a decision.
        Note: Unlike our adaptive controller, the DQN does NOT see:
          - Arrival rates (how fast new cars are approaching)
          - LSTM predictions (where vehicles will exit)
          - Historical patterns
        It only sees the current snapshot of queues and speeds.
        """
        queues = []
        speeds = []
        for edge in EDGE_IDS:
            # How many vehicles are stopped (speed < 0.1 m/s) on this road?
            queues.append(float(self._conn.edge.getLastStepHaltingNumber(edge)))
            # What's the average speed of all vehicles on this road?
            speeds.append(float(self._conn.edge.getLastStepMeanSpeed(edge)))

        # Combine into a single observation vector:
        # [N_queue, S_queue, E_queue, W_queue, N_speed, S_speed, E_speed, W_speed, phase, timer]
        obs = np.array(
            queues + speeds + [float(self.current_phase), self.phase_timer / 60.0],
            dtype=np.float32,
        )
        return obs

    def _get_total_wait(self):
        """
        Sum up the accumulated waiting time of ALL vehicles currently in the network.

        Accumulated waiting time = total seconds a vehicle has spent at speed < 0.1 m/s.
        With --waiting-time-memory 3600, SUMO remembers up to 1 hour of waiting per car
        (default is only 100s). Used to compute the reward signal.
        """
        total = 0.0
        for vid in self._conn.vehicle.getIDList():
            total += self._conn.vehicle.getAccumulatedWaitingTime(vid)
        return total

    def _record_vehicles(self):
        """
        Record each car's waiting time, and save it when the car leaves.

        Called every frame during evaluation (when track_vehicles=True).
        This is the same counting logic the fixed-timer evaluation uses,
        ensuring all three controllers are measured the same way.

        Why per-frame? getArrivedIDList() only returns cars that departed
        on the MOST RECENT frame. If you only check once per decision
        (every 150 frames), you miss 149/150 departures.
        """
        for vid in self._conn.vehicle.getIDList():
            self.veh_last_wait[vid] = self._conn.vehicle.getAccumulatedWaitingTime(vid)
        for vid in self._conn.simulation.getArrivedIDList():
            if vid in self.veh_last_wait:
                self.veh_total_wait.append(self.veh_last_wait.pop(vid))
        self.arrived += self._conn.simulation.getArrivedNumber()

    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode.

        Called at the start of training and after each episode ends.
        Launches a fresh SUMO simulation and resets all internal state.
        Returns the initial observation.
        """
        super().reset(seed=seed)
        self._start_sumo()             # Launch fresh SUMO
        self.current_phase = 0         # Start with NS green
        self.phase_timer = 0.0         # Reset phase timer
        self.yellow_active = False     # No yellow transition in progress
        self.yellow_timer = 0.0
        self.veh_last_wait = {}        # Reset vehicle tracking for new episode
        self.veh_total_wait = []
        self.arrived = 0
        return self._get_obs(), {}     # Return initial observation

    def step(self, action):
        """
        Execute one agent decision and advance the simulation by 5 seconds.

        This is the core RL step:
        1. Agent picks action (0=keep, 1=switch)
        2. If switching: set yellow, then after 3s switch to other phase
        3. Simulate 150 SUMO steps (= 5 real seconds)
        4. Compute reward = -current_wait / 100 (less waiting = closer to 0 = better)
        5. Return new observation, reward, and whether episode is done

        Args:
            action: 0 (keep current phase) or 1 (switch to other phase)

        Returns:
            obs:        new observation (10 values)
            reward:     float — negative, closer to 0 means less waiting
            terminated: True if all vehicles have left the network
            truncated:  always False (no time limit truncation)
            info:       empty dict (could be used for debugging)
        """
        # ── Step 1: Process the agent's action ──
        # Only allow switching if:
        #   - Agent chose action=1 (switch)
        #   - Not already in a yellow transition
        #   - Current phase has been green for at least MIN_GREEN (10 seconds)
        #     This prevents dangerous rapid switching
        if action == 1 and not self.yellow_active and self.phase_timer >= MIN_GREEN:
            # Start yellow transition — can't go directly from green to green
            if self.current_phase == 0:  # Currently NS green
                self._conn.trafficlight.setRedYellowGreenState("C", NS_YELLOW_STATE)
            else:  # Currently EW green
                self._conn.trafficlight.setRedYellowGreenState("C", EW_YELLOW_STATE)
            self.yellow_active = True
            self.yellow_timer = 0.0

        # ── Step 2: Advance SUMO by 5 seconds (150 substeps) ──
        for _ in range(STEPS_PER_DECISION):
            # Stop if all vehicles have left the network (simulation done)
            if self._conn.simulation.getMinExpectedNumber() <= 0:
                break

            # Advance SUMO by one timestep (1/30th of a second)
            # Cars move, accelerate, brake, enter/exit the network
            self._conn.simulationStep()
            self.phase_timer += STEP_LENGTH

            # During evaluation, track every departing car's wait time
            if self.track_vehicles:
                self._record_vehicles()

            # Handle yellow-to-green transition
            if self.yellow_active:
                self.yellow_timer += STEP_LENGTH
                if self.yellow_timer >= YELLOW_DUR:  # Yellow lasted 3 seconds
                    # Switch to the OTHER phase
                    self.current_phase = 1 - self.current_phase  # 0→1 or 1→0
                    if self.current_phase == 0:
                        self._conn.trafficlight.setRedYellowGreenState("C", NS_GREEN_STATE)
                    else:
                        self._conn.trafficlight.setRedYellowGreenState("C", EW_GREEN_STATE)
                    self.yellow_active = False
                    self.phase_timer = 0.0  # Reset: new phase just started

        # ── Step 3: Compute reward ──
        # Reward = negative of current total waiting time
        # More cars waiting = more negative reward (bad!)
        # Few/no cars waiting = reward near 0 (good!)
        # Unlike the old "change in wait" reward, this doesn't cancel out
        # over an episode, so episode totals reflect actual performance.
        current_wait = self._get_total_wait()
        reward = -current_wait / 100.0

        # ── Step 4: Check if episode is done ──
        # Episode ends when no more vehicles are expected (all departed + none waiting to enter)
        terminated = self._conn.simulation.getMinExpectedNumber() <= 0
        truncated = False

        # Get new observation for the agent's next decision
        obs = self._get_obs() if not terminated else np.zeros(10, dtype=np.float32)

        return obs, reward, terminated, truncated, {}

    def close(self):
        """Shut down the SUMO simulation and close the TraCI connection."""
        if self._sumo_running:
            try:
                self._conn.close()
            except:
                pass
            self._sumo_running = False


# ═══════════════════════════════════════════════════════════════════════════
# PART 2: MULTI-VERSION TRAINING WRAPPER
# For "global" mode: trains one DQN across all traffic scenarios.
# Each episode randomly picks a different scenario so the agent
# learns a general policy that works for any traffic pattern.
# ═══════════════════════════════════════════════════════════════════════════

class MultiVersionEnv(gym.Env):
    """
    Wraps multiple SUMO scenarios into one environment.

    Each time the environment resets (new episode), it randomly picks
    one of the available scenarios (v2, v3, v4, etc.). This forces
    the DQN to learn a policy that works across all traffic patterns,
    rather than memorizing one specific scenario.
    """

    metadata = {"render_modes": []}

    def __init__(self, versions=None):
        super().__init__()
        self.versions = versions or ["v2", "v3", "v4", "v5", "v6", "v7"]
        # Pre-create an environment for each version
        self._envs = {v: SumoIntersectionEnv(version=v) for v in self.versions}
        self._current_env = None      # Which env is active this episode
        self._current_version = None  # Which version was picked

        # All versions share the same observation/action spaces
        ref = self._envs[self.versions[0]]
        self.observation_space = ref.observation_space
        self.action_space = ref.action_space

    def reset(self, seed=None, options=None):
        """Pick a random scenario and start a new episode."""
        super().reset(seed=seed)
        # Randomly select which traffic scenario to use this episode
        self._current_version = self.np_random.choice(self.versions)
        self._current_env = self._envs[self._current_version]
        return self._current_env.reset(seed=seed)

    def step(self, action):
        """Forward the action to whichever scenario is currently active."""
        return self._current_env.step(action)

    def close(self):
        """Shut down all SUMO instances."""
        for env in self._envs.values():
            env.close()


# ═══════════════════════════════════════════════════════════════════════════
# PART 3: TRAINING CALLBACK
# This monitors the training process — tracks rewards, episode lengths,
# and prints progress updates. It's called by stable-baselines3 after
# every training step.
# ═══════════════════════════════════════════════════════════════════════════

class ProgressCallback(BaseCallback):
    """
    Monitors DQN training progress.

    Called automatically by stable-baselines3 after every timestep.
    Tracks:
      - Total reward per episode (did the agent improve overall?)
      - Episode length (how many decisions before all cars left?)
      - Which scenario was played
      - Exploration rate (how random vs learned are the agent's actions?)
    """
    def __init__(self, total_timesteps, print_freq=5000):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.print_freq = print_freq  # Print status every N timesteps

        # Completed episode tracking
        self.episode_rewards = []    # Total reward for each completed episode
        self.episode_lengths = []    # Number of decisions in each episode
        self.episode_versions = []   # Which scenario (v2, v3, etc.) each episode used

        # Current episode accumulators (reset when episode ends)
        self._ep_reward = 0.0
        self._ep_length = 0
        self._ep_actions = []

        # Periodic snapshots for the training log
        self.history = []

    def _on_step(self):
        """Called by stable-baselines3 after every single timestep."""
        # Get the reward and action from this timestep
        reward = self.locals.get("rewards", [0])[0]
        action = self.locals.get("actions", [0])[0]
        self._ep_reward += reward
        self._ep_length += 1
        self._ep_actions.append(int(action))

        # Check if this episode just ended
        dones = self.locals.get("dones", [False])
        if dones[0]:
            # Save completed episode stats
            self.episode_rewards.append(self._ep_reward)
            self.episode_lengths.append(self._ep_length)

            # Figure out which scenario was played
            env = self.training_env.envs[0]
            if hasattr(env, 'env'):
                env = env.env  # Unwrap the Monitor wrapper
            version = getattr(env, '_current_version', 'unknown')
            self.episode_versions.append(version)

            # Reset accumulators for next episode
            self._ep_reward = 0.0
            self._ep_length = 0
            self._ep_actions = []

        # Print progress periodically
        if self.num_timesteps % self.print_freq == 0:
            pct = self.num_timesteps / self.total_timesteps * 100
            n_eps = len(self.episode_rewards)
            avg_r = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
            avg_len = np.mean(self.episode_lengths[-10:]) if self.episode_lengths else 0

            # Save a snapshot for the training log
            snapshot = {
                "timestep": self.num_timesteps,
                "episodes_completed": n_eps,
                "avg_reward_last10": round(float(avg_r), 2),
                "avg_length_last10": round(float(avg_len), 1),
                # exploration_rate: probability of taking a random action
                # Starts at 1.0 (100% random) and decays to 0.05 (5% random)
                "exploration_rate": round(self.model.exploration_rate, 4),
            }
            self.history.append(snapshot)

            print(f"  Training: {self.num_timesteps}/{self.total_timesteps} ({pct:.0f}%) "
                  f"| eps={n_eps} | avg_reward={avg_r:.1f} | explore={self.model.exploration_rate:.3f}")
            sys.stdout.flush()

        return True  # Return True to continue training

    def get_training_log(self):
        """Return the full training history as a JSON-serializable dict."""
        return {
            "snapshots": self.history,
            "episodes": [
                {
                    "episode": i,
                    "version": self.episode_versions[i] if i < len(self.episode_versions) else "unknown",
                    "total_reward": round(float(self.episode_rewards[i]), 2),
                    "length_steps": self.episode_lengths[i],
                }
                for i in range(len(self.episode_rewards))
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════
# PART 4: EVALUATION FUNCTIONS
# These run each control method (DQN, fixed timer, adaptive) on a scenario
# and measure the average waiting time per vehicle.
# ═══════════════════════════════════════════════════════════════════════════

def run_dqn_evaluation(model, version):
    """
    Run a trained DQN model on a single scenario and measure performance.

    Uses track_vehicles=True so the environment records every car's wait
    time on every frame (not just once per decision). This matches how
    the fixed-timer evaluation counts, ensuring a fair comparison.

    Returns: average waiting time per vehicle (seconds)
    """
    env = SumoIntersectionEnv(version=version, track_vehicles=True)
    obs, _ = env.reset()

    while True:
        # deterministic=True = always pick the best action (no exploration)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    # Include vehicles still in network when sim ended (survivorship bias fix)
    env.veh_total_wait.extend(env.veh_last_wait.values())
    env.arrived += len(env.veh_last_wait)
    total_wait = sum(env.veh_total_wait)
    arrived = env.arrived
    env.close()
    return total_wait / max(arrived, 1)


def run_fixed_evaluation(version):
    """
    Run a fixed-timer baseline on a scenario.

    The simplest possible controller: alternate between NS green (42s)
    and EW green (42s) with 3s yellow transitions. No adaptation at all.
    This is what most real intersections without sensors use.

    Returns: average waiting time per vehicle (seconds)
    """
    sumo_bin = shutil.which("sumo")
    cfg = str(SUMO_DIR / f"adaptive_timer/{version}/intersection.sumocfg")
    sumo_cmd = [sumo_bin, "-c", cfg, "--step-length", str(STEP_LENGTH),
                "--no-warnings", "--no-step-log",
                "--waiting-time-memory", "3600"]
    traci.start(sumo_cmd)

    # Warm up
    for _ in range(15):
        traci.simulationStep()

    # Start with NS green, 42-second fixed cycle per direction
    traci.trafficlight.setRedYellowGreenState("C", NS_GREEN_STATE)
    phase = "NS_GREEN"
    timer = 0.0

    veh_last_wait = {}
    veh_total_wait = []
    arrived = 0

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        timer += STEP_LENGTH

        # Track per-vehicle wait times
        for vid in traci.vehicle.getIDList():
            veh_last_wait[vid] = traci.vehicle.getAccumulatedWaitingTime(vid)
        for vid in traci.simulation.getArrivedIDList():
            if vid in veh_last_wait:
                veh_total_wait.append(veh_last_wait.pop(vid))
        arrived += traci.simulation.getArrivedNumber()

        # Fixed cycle: NS green 42s → NS yellow 3s → EW green 42s → EW yellow 3s → repeat
        if phase == "NS_GREEN" and timer >= 42.0:
            traci.trafficlight.setRedYellowGreenState("C", NS_YELLOW_STATE)
            phase = "NS_YELLOW"
            timer = 0.0
        elif phase == "NS_YELLOW" and timer >= YELLOW_DUR:
            traci.trafficlight.setRedYellowGreenState("C", EW_GREEN_STATE)
            phase = "EW_GREEN"
            timer = 0.0
        elif phase == "EW_GREEN" and timer >= 42.0:
            traci.trafficlight.setRedYellowGreenState("C", EW_YELLOW_STATE)
            phase = "EW_YELLOW"
            timer = 0.0
        elif phase == "EW_YELLOW" and timer >= YELLOW_DUR:
            traci.trafficlight.setRedYellowGreenState("C", NS_GREEN_STATE)
            phase = "NS_GREEN"
            timer = 0.0

    traci.close()
    # Include vehicles still in network when sim ended (survivorship bias fix)
    veh_total_wait.extend(veh_last_wait.values())
    arrived += len(veh_last_wait)
    total_wait = sum(veh_total_wait) if veh_total_wait else 0
    return total_wait / max(arrived, 1)


def run_adaptive_evaluation(version):
    """
    Run our adaptive controller (Queue + Arrival Rate + LSTM) with
    Optuna-optimized weights on a scenario.

    This is the controller we developed — it uses three signals:
      1. Queue count from vision-based detection
      2. Arrival rate estimation (how fast cars are coming)
      3. LSTM neural network prediction of through-traffic direction

    These are combined into a demand score formula with per-scenario
    optimized weights to compute green time allocation.

    Returns: average waiting time per vehicle (seconds)
    """
    sys.path.insert(0, str(SUMO_DIR))
    from optimize_weights import run_headless_adaptive

    # Load the Optuna-optimized parameters for this scenario
    opt_path = SUMO_DIR / "optimized_params.json"
    with open(str(opt_path)) as f:
        opt = json.load(f)

    # Use per-version optimized params if available
    if opt.get("mode") == "per_version" and version in opt.get("per_version", {}):
        params = opt["per_version"][version]["best_params"]
    elif "best_params" in opt:
        params = opt["best_params"]
    else:
        # Fallback to hand-tuned defaults
        params = {"w_queue": 1.0, "w_arrival": 3.0, "w_through": 0.3, "cycle_total": 90.0}

    return run_headless_adaptive(version, params)


# ═══════════════════════════════════════════════════════════════════════════
# PART 5: TRAINING
# Creates and trains DQN models. Two modes:
#   - per_version: one specialized DQN per scenario (fairer comparison)
#   - global: one DQN trained across all scenarios (tests generalization)
# ═══════════════════════════════════════════════════════════════════════════

def train_single_version(version, total_timesteps):
    """
    Train a dedicated DQN for a single traffic scenario.

    The DQN only sees one type of traffic pattern, so it can specialize.
    This is analogous to our adaptive controller having per-version
    Optuna-optimized weights.
    """
    model_path = str(SCRIPT_DIR / f"dqn_models/dqn_{version}")

    # Create environment for this specific scenario
    env = SumoIntersectionEnv(version=version)

    # Create the DQN agent
    model = DQN(
        "MlpPolicy",               # Neural network: Multi-Layer Perceptron (fully connected)
                                    # Architecture: 10 inputs → 64 neurons → 64 neurons → 2 outputs
                                    # Outputs are Q-values: Q(state, keep) and Q(state, switch)
        env,
        learning_rate=1e-3,         # How fast the network updates its weights (0.001)
        buffer_size=50000,          # Replay buffer: stores 50k past experiences to learn from
                                    # (not just the latest — random sampling stabilizes learning)
        learning_starts=1000,       # Don't start training until 1000 steps of random exploration
                                    # (need enough experiences in the buffer first)
        batch_size=64,              # Train on 64 random experiences at a time
        gamma=0.99,                 # Discount factor: how much to value future rewards vs immediate
                                    # 0.99 = future rewards almost as important as immediate ones
        target_update_interval=500, # Update the target network every 500 steps
                                    # (DQN uses two networks for stability)
        exploration_fraction=0.3,   # Spend first 30% of training exploring randomly
        exploration_final_eps=0.05, # After exploration phase, still take random action 5% of the time
        verbose=0,                  # Don't print stable-baselines3 logs
    )

    # Train the model — this runs thousands of episodes of SUMO simulation
    callback = ProgressCallback(total_timesteps, print_freq=5000)
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save the trained model (weights + config) as a zip file
    model.save(model_path)
    env.close()

    # Save training history for analysis
    training_log = callback.get_training_log()
    training_log["config"] = {
        "version": version,
        "total_timesteps": total_timesteps,
        "mode": "per_version",
    }
    log_path = SCRIPT_DIR / f"dqn_models/dqn_{version}_log.json"
    with open(str(log_path), "w") as f:
        json.dump(training_log, f, indent=2)

    return model, training_log


def train(args):
    """
    Main training function. Supports two modes:

    per_version (default): Train a separate DQN for each scenario.
        Each DQN specializes on one traffic pattern. This is a fairer
        comparison against our adaptive controller which also has
        per-scenario optimized parameters.

    global: Train one DQN across all scenarios.
        The DQN must learn a general policy. Each episode randomly
        picks a different scenario. Tests whether the DQN can generalize.
    """
    versions = args.versions
    total_timesteps = args.timesteps
    mode = args.mode

    print(f"\n{'='*60}")
    print(f"  DQN TRAINING — mode: {mode}")
    print(f"  Scenarios: {versions}")
    print(f"  Timesteps: {total_timesteps}" + (f" per version" if mode == "per_version" else ""))
    print(f"{'='*60}\n")
    sys.stdout.flush()

    if mode == "per_version":
        # ── Per-version training: one DQN per scenario ──
        os.makedirs(str(SCRIPT_DIR / "dqn_models"), exist_ok=True)

        models = {}
        for v in versions:
            print(f"\n  ── Training DQN for {v} ──")
            sys.stdout.flush()
            model, log = train_single_version(v, total_timesteps)
            models[v] = model
            print(f"  ✓ {v} done — {len(log['episodes'])} episodes")
            sys.stdout.flush()

        print(f"\n  All per-version models saved to: dqn_models/")
        sys.stdout.flush()

        # Run evaluation immediately after training
        print(f"\n  Running post-training evaluation...")
        evaluate(args, models=models)

    else:
        # ── Global training: one DQN for all scenarios ──
        model_path = str(SCRIPT_DIR / "dqn_traffic_model")

        # MultiVersionEnv randomly picks a scenario each episode
        env = MultiVersionEnv(versions=versions)

        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=1e-3,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=64,
            gamma=0.99,
            target_update_interval=500,
            exploration_fraction=0.3,
            exploration_final_eps=0.05,
            verbose=0,
        )

        callback = ProgressCallback(total_timesteps, print_freq=5000)
        model.learn(total_timesteps=total_timesteps, callback=callback)

        model.save(model_path)
        print(f"\n  Model saved to: {model_path}.zip")
        sys.stdout.flush()

        # Save training log
        training_log = callback.get_training_log()
        training_log["config"] = {
            "total_timesteps": total_timesteps,
            "versions": versions,
            "mode": "global",
        }
        log_path = SCRIPT_DIR / "dqn_training_log.json"
        with open(str(log_path), "w") as f:
            json.dump(training_log, f, indent=2)
        print(f"  Training log saved to: {log_path}")
        print(f"  Episodes completed: {len(training_log['episodes'])}")
        sys.stdout.flush()

        env.close()

        print(f"\n  Running post-training evaluation...")
        evaluate(args, model=model)


# ═══════════════════════════════════════════════════════════════════════════
# PART 6: EVALUATION
# Compares all three methods (Fixed, DQN, Adaptive) on each scenario.
# Prints a results table and saves to dqn_comparison.json.
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(args, model=None, models=None):
    """
    Evaluate all three control methods on each scenario and compare.

    Supports three ways to provide the DQN model:
      - models={v2: model, v3: model, ...}  → per-version DQN
      - model=single_model                   → global DQN (one brain for all)
      - neither                              → loads saved models from disk
    """
    versions = args.versions
    mode = args.mode

    # Load models from disk if not passed directly (e.g., running eval standalone)
    if models is None and model is None:
        if mode == "per_version":
            models = {}
            for v in versions:
                mp = str(SCRIPT_DIR / f"dqn_models/dqn_{v}.zip")
                if not os.path.exists(mp):
                    print(f"  No per-version model for {v} at {mp}")
                    print(f"  Run 'python dqn_baseline.py train --mode per_version' first.")
                    return
                models[v] = DQN.load(mp)
        else:
            model_path = str(SCRIPT_DIR / "dqn_traffic_model")
            if not os.path.exists(model_path + ".zip"):
                print(f"  No trained model found at {model_path}.zip")
                print(f"  Run 'python dqn_baseline.py train' first.")
                return
            model = DQN.load(model_path)

    dqn_label = "DQN-each" if (models is not None) else "DQN-glob"

    print(f"\n{'='*60}")
    print(f"  EVALUATION — ALL METHODS ({dqn_label})")
    print(f"{'='*60}")
    print(f"\n  {'Version':<10} {'Fixed':>10} {dqn_label:>10} {'Adaptive':>10}")
    print(f"  {'-'*45}")
    sys.stdout.flush()

    results = {}
    for v in versions:
        print(f"  Running {v}...", end=" ", flush=True)

        # Run all three methods on this scenario
        fixed_wait = run_fixed_evaluation(v)

        # Use per-version model if available, else the single global model
        dqn_model = models[v] if models else model
        dqn_wait = run_dqn_evaluation(dqn_model, v)

        adaptive_wait = run_adaptive_evaluation(v)

        results[v] = {
            "fixed": round(fixed_wait, 2),
            "dqn": round(dqn_wait, 2),
            "adaptive": round(adaptive_wait, 2),
        }
        print(f"done")
        print(f"  {v:<10} {fixed_wait:>9.2f}s {dqn_wait:>9.2f}s {adaptive_wait:>9.2f}s")
        sys.stdout.flush()

    # ── Compute averages across all scenarios ──
    avg_fixed = np.mean([r["fixed"] for r in results.values()])
    avg_dqn = np.mean([r["dqn"] for r in results.values()])
    avg_adaptive = np.mean([r["adaptive"] for r in results.values()])

    print(f"  {'-'*45}")
    print(f"  {'Average':<10} {avg_fixed:>9.2f}s {avg_dqn:>9.2f}s {avg_adaptive:>9.2f}s")

    # ── Compute improvement percentages ──
    dqn_vs_fixed = (avg_fixed - avg_dqn) / avg_fixed * 100
    adp_vs_fixed = (avg_fixed - avg_adaptive) / avg_fixed * 100
    adp_vs_dqn = (avg_dqn - avg_adaptive) / avg_dqn * 100

    print(f"\n  {dqn_label} vs Fixed: {dqn_vs_fixed:+.1f}%")
    print(f"  Adaptive vs Fixed: {adp_vs_fixed:+.1f}%")
    print(f"  Adaptive vs {dqn_label}: {adp_vs_dqn:+.1f}%")
    print(f"{'='*60}")

    # ── Save results to JSON ──
    out = {
        "mode": mode,
        "per_version": results,
        "averages": {
            "fixed": round(avg_fixed, 2),
            "dqn": round(avg_dqn, 2),
            "adaptive": round(avg_adaptive, 2),
        },
        "improvements": {
            "dqn_vs_fixed_pct": round(dqn_vs_fixed, 2),
            "adaptive_vs_fixed_pct": round(adp_vs_fixed, 2),
            "adaptive_vs_dqn_pct": round(adp_vs_dqn, 2),
        },
    }
    out_path = SCRIPT_DIR / "dqn_comparison.json"
    with open(str(out_path), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 7: COMMAND LINE INTERFACE
# ═══════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN baseline for traffic signal control")
    parser.add_argument("command", choices=["train", "eval"], help="train or evaluate")
    parser.add_argument("--mode", choices=["global", "per_version"], default="per_version",
                        help="global = one DQN for all, per_version = one DQN per scenario (default: per_version)")
    parser.add_argument("--versions", nargs="+", default=["v2", "v3", "v4", "v5", "v6", "v7"],
                        help="Scenario versions (default: v2-v7)")
    parser.add_argument("--timesteps", type=int, default=100000,
                        help="Total training timesteps (default: 100000)")
    args = parser.parse_args()

    if args.command == "train":
        train(args)
    else:
        evaluate(args)
