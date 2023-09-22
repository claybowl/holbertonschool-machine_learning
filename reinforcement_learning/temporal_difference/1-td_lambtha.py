#!/usr/bin/env python3
"""
This module contains the function for the TD(λ) algorithm.
"""

import numpy as np

def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the TD(λ) algorithm to update the value function based on the policy.

    Parameters:
    env: The environment.
    V: The value function.
    policy: The policy.
    lambtha: The trace decay parameter.
    episodes: The number of episodes.
    max_steps: The maximum number of steps per episode.
    alpha: The learning rate.
    gamma: The discount factor.

    Returns:
    The updated value function.
    """
    # Initialize eligibility traces
    E = np.zeros_like(V)

    for episode in range(episodes):
        state = env.reset()

        for step in range(max_steps):
            # Take an action, observe the reward and next state
            action = policy[state]
            next_state, reward, done, _ = env.step(action)

            # Compute the TD error
            td_error = reward + gamma * V[next_state] - V[state]

            # Update eligibility trace
            E[state] += 1

            # Update V and E
            V += alpha * td_error * E
            E *= gamma * lambtha

            if done:
                break

            # Update state
            state = next_state

    return V
