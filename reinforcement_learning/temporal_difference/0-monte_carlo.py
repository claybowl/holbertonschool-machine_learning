#!/usr/bin/env python3
"""
This module contains the function for the Monte Carlo algorithm.
"""

import numpy as np

def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm to update the value function based on the policy.

    Parameters:
    env: The environment.
    V: The value function.
    policy: The policy.
    episodes: The number of episodes.
    max_steps: The maximum number of steps per episode.
    alpha: The learning rate.
    gamma: The discount factor.

    Returns:
    The updated value function.
    """
    # Creates dictionary with 'n' keys corresponding to a state. Value
    # for each key will be empty list, used later for storing returns from
    # Monte Carlo algorithm.
    returns = {s: [] for s in range(env.observation_space.n)}

    # Resets environment at beginning of each episode and initializes and empty
    # list to store episode-related data
    for episode in range(episodes):
        state = env.reset()
        current_episode = []

        # Executes one episode in environment, following policy to sle
        for step in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            current_episode.append((state, action, reward))
            if done:
                break
            state = next_state

        G = 0
        for state, _, reward in reversed(current_episode):
            G = gamma * G + reward
            returns[state].append(G)
            V[state] = (1 - alpha) * V[state] + alpha * G

    return V
