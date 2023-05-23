import gym
import numpy as np

class Agent:
    def __init__(self, env, credence=0.5, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.credence = credence

        # agent learning parameters
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration-exploitation trade-off parameter

        self.q_table = np.zeros((3, 2))

    def choose_action(self, observation):
        state = observation[0]
        actions = self.q_table[state]

        if np.random.uniform(0, 1) < self.epsilon:
            # Exploration -> random action
            return self.env.action_space.sample()
        else:
            # Exploitation -> greedy action (choose the action with the highest Q-value)
            return np.argmax(actions)

    def update_q_table(self, state, action, reward, next_state):
        current_q_value = self.q_table[state][action]
        next_max_q_value = np.max(self.q_table[next_state])
        td_error = reward + self.gamma * next_max_q_value - current_q_value
        updated_q_table = current_q_value + self.alpha * td_error
        self.q_table[state][action] = updated_q_table

    def train(self, num_episodes):
        for episode in range(num_episodes):
            observation = self.env.reset()
            done = False

            while not done:
                action = self.choose_action(observation)
                next_observation, reward, done = self.env.step(action)
                self.update_q_table(observation[0], action, reward, next_observation[0])
                observation = next_observation
            
            if (episode + 1) % 100 == 0:
                print("Episode:", episode + 1)

        print("Training completed.")
        
    def test(self, num_episodes):

        num_actions = {
            0: 0, #do nothing
            1: 0  #switch
        }

        for episode in range(num_episodes):
            observation = self.env.reset()
            done = False

            while not done:
                action = self.choose_action(observation)
                next_observation, _, done = self.env.step(action)
                observation = next_observation

            # print("Episode:", episode + 1, "Final state:", observation)
            
            num_actions[observation[1]] += 1
        
        print("Testing completed.")

        return num_actions

