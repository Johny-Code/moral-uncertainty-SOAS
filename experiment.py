import gym
import numpy as np

from agent import Agent
from env import TrolleyEnv


def run_experiment(env, agent, num_episodes_train, num_episodes_test):
    
    print("Training the agent...")
    agent.train(num_episodes_train)
    
    print("Testing the agent...")
    num_actions = agent.test(num_episodes_test)

    print("Number of bystanders:", env.num_people_on_track)
    print("Credence in deontology:", env.credence)
    print()
    print(f"Number of agents who did nothing: {num_actions[0]}")
    print(f"Number of agents who switched: {num_actions[1]}")


def simple_experiment():
    credence = 0.1
    num_bystanders = 10
    num_episodes_train = 1000
    num_episodes_test = 100
    
    env = TrolleyEnv(num_bystanders=num_bystanders, credence=credence)

    #agent learning parameters 
    alpha = 0.1  # Learning rate
    gamma = 0.9  # Discount factor
    epsilon = 0.1  # Exploration-exploitation trade-off parameter

    agent = Agent(env, credence=credence, alpha=alpha, gamma=gamma, epsilon=epsilon)

    run_experiment(env, agent, num_episodes_train, num_episodes_test)


def MEC_experiment():
    num_episodes_train = 10
    num_episodes_test = 10

    #agent learning parameters 
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.1

    for credence in range(0.0, 1.1, 0.1):
        for num_bystanders in range(1, 11):
            env = TrolleyEnv(num_bystanders=num_bystanders, credence=credence)
            agent = Agent(env, credence=credence, alpha=alpha, gamma=gamma, epsilon=epsilon)

if __name__ == "__main__":
    
    simple_experiment()

    MEC_experiment() 
    
    


            