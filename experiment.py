import time
import numpy as np

from agent import Agent
from env import TrolleyEnv
from utils import plot_output


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
    num_episodes_train = 100
    num_episodes_test = 100
    
    env = TrolleyEnv(num_bystanders=num_bystanders, credence=credence)

    #agent learning parameters 
    alpha = 0.1  # Learning rate
    gamma = 0.9  # Discount factor
    epsilon = 0.1  # Exploration-exploitation trade-off parameter

    agent = Agent(env, credence=credence, alpha=alpha, gamma=gamma, epsilon=epsilon)

    run_experiment(env, agent, num_episodes_train, num_episodes_test)


def MEC_experiment():
    
    num_episodes_train = 100000
    num_episodes_test = 100000

    #agent learning parameters 
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.1

    output = np.zeros((10, 11))
    experiment_time = []
    start = time.time()

    for x, credence in enumerate(np.arange(0.0, 1.1, 0.1)):
        for y, num_bystanders in enumerate(range(1, 11)):
            start_experiment = time.time()
            env = TrolleyEnv(num_bystanders=num_bystanders, credence=credence)
            agent = Agent(env, credence=credence, alpha=alpha, gamma=gamma, epsilon=epsilon)
            agent.train(num_episodes_train)

            num_actions = agent.test(num_episodes_test)
            
            if num_actions[0] > num_actions[1]:
                output[y, x] = 0 # the majority decided - do nothing
            else:
                output[y, x] = 1 # the majority decided - to switch
            
            end_experiment = time.time()
            experiment_time.append(end_experiment - start_experiment)

    end = time.time()
    print(f"Time elapsed: {round(end - start, 3)} seconds")
    print(f"Average time per experiment: {round(np.mean(experiment_time), 3)} seconds")

    plot_output(output)

if __name__ == "__main__":
    
    simple_experiment()

    MEC_experiment() 
    
    


            