from agents.agent import Agent
from agents.root_agent import RootAgent
from monitor import interact
import gym
import numpy as np

env = gym.make('Taxi-v2')
env.render()
agent = Agent()




avg_rewards, best_avg_reward = interact(env, agent)
