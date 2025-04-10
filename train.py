import gymnasium as gym
import numpy as np 
from agent import Agent

def training():
    env = gym.make("FrozenLake-v1")

    agent = Agent(env)

    iter = 0
    while True:
        iter += 1
        agent.play_random(num_episodes=100)
        agent.value_update()
        print(f'Iterations: {iter}')
        if agent.get_test_reward(num_episodes=30) > 28/30:
            break

    agent.save_table('model/frozen_qtable.npy')
    print('Training successfully!')

if __name__ == "__main__":
    training()
    

