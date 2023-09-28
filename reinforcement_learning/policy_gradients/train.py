#!/usr/bin/env python3
"""Module train
Function that implements a full training.
"""
import numpy as np
import gym


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """Implements a full training"""
    np.random.seed(1)
    weight = np.random.rand(4, 2)
    scores = []

    for episode in range(1, nb_episodes + 1):
        state = env.reset()[None, :]
        grads = []
        rewards = []
        score = 0
    
        while True:
            action, grad = policy_gradient(state, weight)
            state, reward, done, _ = env.step(action)
            state = state[None, :]

            grads.append(grad)
            rewards.append(reward)
            score += reward

            if done:
                break

        for i in range(len(grads)):
            for j in range(i, len(grads)):
                grads[i] += gamma ** (j - i) * rewards[j]

        for grad in grads:
            weight += alpha * grad
        
        scores.append(score)
        print(f"Episode: {episode} - Score: {score}", end="\r", flush=False)
    
    return scores
