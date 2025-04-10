# 🏆 RL Agent for FrozenLake-v1

## 🎯 Overview
This project implements a **Reinforcement Learning (DRL) agent** using **Value Iteration** to solve **FrozenLake-v1** from Gymnasium.

The goal of the agent is to reach the target on a 4x4 gridworld. At any time, the agent can oly know its current position, not the location of the target and other obstacles.

## 🚀 Key Feature
- Training using value iteration (dynamic programming).
- Update value after each time playing 100 episodes.
- Simple Q-table with size 16x4 to hole the value of each state-action pair.

## 🎖️ Architecture
- Environment: FrozenLake-v1.
- Update method: Value iteration.
- Datastructure: Q-table.
- Training termination: get 28/30 average score in 30 random episodes.

## 🌹 Dependencies (Windows)
```bash=
pip install -r requirements.txt
```

## 🐼 References
**Deep Reinforcement Learning Hands-On** by Maxim Lapan.

## 🐧 Usage
- Install all dependencies.
- Run `train.py` to train the agent.
- Model file is saved in `model` folder.
- Run `play.py` to watch the agent play.
- Enjoy!
