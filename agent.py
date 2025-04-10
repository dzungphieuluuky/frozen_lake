import collections
import random
import numpy as np

class Agent:
    def __init__(self, env):
        self.env = env
        self.gamma = 0.95
        self.q_table = []
        for state in range(env.observation_space.n):
            self.q_table.append([])
            for action in range(env.action_space.n):
                self.q_table[state].append(0)

        self.reward_table = collections.defaultdict(float)
        self.transition_table = collections.defaultdict(collections.Counter)

    def load_table(self, path):
        self.q_table = np.load(path)
    
    def save_table(self, path):
        np.save(path, self.q_table)

    def play_random(self, num_episodes):
        for ep in range(num_episodes):
            done = False
            state = self.env.reset()[0]
            while not done:
                action = random.randint(0, self.env.action_space.n - 1)
                next_state, reward, done, truncate, info = self.env.step(action)
                self.reward_table[(state, action, next_state)] = reward
                self.transition_table[(state, action)][next_state] += 1
                if done or truncate:
                    break
                state = next_state

    def value_update(self):
        for state in range(self.env.observation_space.n):
            for action in range(self.env.action_space.n):
                action_value = 0
                target_counts = self.transition_table[(state, action)]
                total_transit = sum(target_counts.values())
                for next_state, count in target_counts.items():
                    key = (state, action, next_state)
                    next_state_val = self.reward_table[key] + self.gamma * self.get_best_value_action(next_state)[1]
                    action_value += (count / total_transit) * next_state_val
                self.q_table[state][action] = action_value
    
    def get_best_value_action(self, state):
        best_value = None
        best_action = None
        for act in range(self.env.action_space.n):
            if best_value is None or self.q_table[state][act] > best_value:
                best_value = self.q_table[state][act]
                best_action = act
        return [best_action, best_value]

    def get_test_reward(self, num_episodes):
        total_reward = 0
        for ep in range(num_episodes):
            state = self.env.reset()[0]
            done = False
            ep_reward = 0
            
            while not done:
                action = self.get_best_value_action(state)[0]
                next_state, reward, done, truncate, info = self.env.step(action)
                ep_reward += reward
                
                if done or truncate:
                    break
                
                state = next_state
            
            total_reward += ep_reward
        return total_reward/num_episodes