import torch

import numpy as np
import envwrapper

import agent

torch.manual_seed(0)
np.random.seed(0)


# Create environment
env = envwrapper.Env(no_graphics=True)
nA, nS, num_agents = env.return_sizes()


# ACHIEVES MORE THAN 30 POINTS IN THE 70th ITERATION

hn_actor = [256, 128]
hn_critic = [256, 128]

actor_dict = {"input_size": nS,
              "output_size": nA,
              "hn": hn_actor,
              "batch_norm": False}

critic_dict = {"state_size": nS,
               "action_size": nA,
               "hn": hn_critic,
               "concat_stage": 1,
               "batch_norm": False}

agent_dict = {"num_episodes": 200,
              "num_replays": 1,
              "memory_size": 2**14,
              "batchsize": 128,
              "gamma": 0.99,
              "tau": 0.001,
              "learning_rate_actor": 1E-4,
              "learning_rate_critic": 1E-4,
              "save_after": 80}


# Create agent
agent = agent.Agent(agent_dict = agent_dict, actor_dict = actor_dict, critic_dict = critic_dict)

# Train agent
agent.run(env)

