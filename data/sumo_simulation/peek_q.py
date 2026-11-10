import numpy as np
import torch
from stable_baselines3 import DQN

model = DQN.load("dqn_models/dqn_v3.zip")

# Print the network's layers
print(model.q_net)

# A made-up observation: NS green 21 s, long N/S queues, E/W empty
obs = np.array([8, 6, 1, 0, 0.5, 1.2, 9.8, 13.9, 0, 0.35], dtype=np.float32)

obs_tensor, _ = model.policy.obs_to_tensor(obs)
with torch.no_grad():
    q_values = model.q_net(obs_tensor)

print("Q(keep), Q(switch):", q_values.numpy()[0])
print("Chosen action:", int(q_values.argmax()))