#!/usr/bin/env python3
"""Module play.py
This script allows a trained agent
to play Atari's Breakout.
"""

import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory

# Initialize Gym environment
ENV_NAME = 'BreakoutDeterministic-v4'
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)

# Number of actions
nb_actions = env.action_space.n

# Define the model architecture
input_shape = (84, 84)
window_length = 4

model = Sequential()
model.add(Permute((2, 3, 1), input_shape=(window_length,) + input_shape))
model.add(Convolution2D(32, (8, 8), strides=(4, 4)))
model.add(Activation('relu'))
model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))

# Configure and compile the agent
memory = SequentialMemory(limit=1000000, window_length=window_length)
policy = GreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=50000,
               target_model_update=10000, policy=policy)
dqn.compile(Adam(lr=0.00025), metrics=['mae'])

# Load the trained weights
dqn.load_weights('policy.h5')

# Evaluate the agent
dqn.test(env, nb_episodes=10, visualize=True)
