import gym
from gym import spaces
import numpy as np

class TrolleyEnv(gym.Env):
    def __init__(self, num_bystanders=10, credence=0.5):
        super(TrolleyEnv, self).__init__()
        #number of people on the track
        self.num_people_on_track = num_bystanders
        
        #credence of the agent in deontology
        # credence in utilitarianism = (1 - credence in deontology)
        self.credence = credence

        # Actions: 
        # 0 (do nothing) -> crash into (X) bystanders
        # 1 (switch trolley's direction) -> crash into 1 bystander
        self.action_space = spaces.Discrete(2)  

        # Observations:
        # 0 -> beginning position
        # 1 -> on the switch
        # 2 -> end position (crash)

        #switch position
        # 0 -> crash into (X) bystanders
        # 1 -> crash into 1 bystander

        # number of bystanders (X)

        self.observation_space = spaces.Tuple((
            spaces.Discrete(3), 
            spaces.Discrete(2), 
            spaces.Discrete(10) # crash into -> consequence of switch action 
            ))
        
    def reset(self):
        self.trolley_pos = 0
        self.agent_switch_decision = 0
        self.num_people_on_track = self.num_people_on_track
        return self._get_observation()
        
    def _get_observation(self):
        return self.trolley_pos, self.agent_switch_decision, self.num_people_on_track
        
    def step(self, action):
        self.trolley_pos += 1

        if action == 1: #switch action
            self.agent_switch_decision = 1
            self.num_people_on_track = 1
        
        if action == 0: #do nothing action
            self.agent_switch_decision = 0
            self.num_people_on_track = self.num_people_on_track
        
        if self.trolley_pos == 2:
            
            reward_denotology = self._get_deontology_reward(self.num_people_on_track)
            reward_utilitarianism = self._get_utilitarianism_reward(self.num_people_on_track)

            reward = (self.credence * reward_denotology) + ((1 - self.credence) * reward_utilitarianism)

            done = True
        else:
            reward = 0
            done = False

        return self._get_observation(), reward, done

    def _get_deontology_reward(self, num_people_on_track):
        if num_people_on_track == 1:
            return -1
        else:
            return 0
    
    def _get_utilitarianism_reward(self, num_people_on_track):
        return -num_people_on_track