import gymnasium as gym
from agent import Agent

def play():
    env = gym.make("FrozenLake-v1", render_mode='human')
    state = env.reset()[0]
    agent = Agent(env=env)
    
    done = False
    ep_reward = 0
    agent.load_table('model/frozen_qtable.npy')
    print('Playing game...')
    while not done:
        action = agent.get_best_value_action(state)[0]
        next_state, reward, done, truncate, info = env.step(action)
        ep_reward += reward

        if done or truncate:
            break
        
        state = next_state
    
    print(f'Total Reward: {ep_reward}')

if __name__ == "__main__":
    play()
