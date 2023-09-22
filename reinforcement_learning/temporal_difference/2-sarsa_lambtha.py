#!/usr/bin/env python3
"""
This module contains the function for the SARSA(λ) algorithm.
"""

import numpy as np

def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs the SARSA(λ) algorithm to update the action-value function based on the policy.

    Parameters:
    env: The environment.
    Q: The action-value function.
    lambtha: The trace decay parameter.
    episodes: The number of episodes.
    max_steps: The maximum number of steps per episode.
    alpha: The learning rate.
    gamma: The discount factor.
    epsilon: The initial epsilon for the epsilon-greedy policy.
    min_epsilon: The minimum epsilon.
    epsilon_decay: The decay rate for epsilon.

    Returns:
    The updated action-value function.
    """
    # Initialize eligibitility traces
    E = np.zeros_like(Q)

    for episode in range(episodes):
        state = env.reset()
        action = np.argmax(Q[state] + np.random.randn(1, env.action_space.n) * (1. / (episode + 1)))

        for step in range(max_steps):
            # Take an action, observe the reward and next state
            next_state, reward, done, _ = env.step(action)

            # Choose next action using epsilon-greedy policy
            next_action = np.argmax(Q[next_state] + np.random.randn(1, env.action_space.n) * (epsilon))

            # Compute the TD error
            td_error = reward + gamma * Q[next_state, next_action] - Q[state, action]

            # Update eligibility trace
            E[state, action] += 1

            # Update Q and E
            Q += alpha * td_error * E
            E *= gamma * lambtha

            # Update state and action
            state = next_state
            action = next_action

        # Decay epsilon
        epsilon = max(min_epsilon, min(epsilon, 1.0 - epsilon_decay * (episode / episodes)))

    return Q
