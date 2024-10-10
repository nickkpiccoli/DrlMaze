from typing import Optional, Any, SupportsFloat
import numpy as np
import gymnasium as gym
from gymnasium.core import ObsType, ActType


class MazeWorldEnv(gym.Env):

    def __init__(self, size: int = 5):
        #set the maze size
        self.size = size
        #set the positions of agent and objective
        self._agent_pos = np.array([-1,-1], dtype=np.int32)
        self._objective_pos = np.array([-1,-1], dtype=np.int32)

        '''set the observation space, values that environment returns
           as observation of the current state
           in this case the observation space is a Dict containing two couples
           agent and target, that are two Box(matrices) in this case representing
           a vector of [x,y] where x and y can assume values between 0,size-1'''
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size-1, shape=(2,), dtype=int),
                "objective": gym.spaces.Box(0, size-1, shape= (2,), dtype=int)
            }
        )


        '''
        here we define the possible action that the agent can make
        Discrete(4) because our agent can only go up,down,right,left
        '''
        self.action_space = gym.spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1,0]), #moving right
            1: np.array([0,1]), #moving up
            2: np.array([-1,0]), #moving left
            3: np.array([0,-1]) #moving down
        }

    #get the observation about the actual environment state
    def _get_obs(self):
        return {"agent": self._agent_pos, "target": self._objective_pos}

    #auxiliary information, return the manhattan distance between agent and objective
    def _get_info(self):
        return {
            "distance": np.lingal.norm(
                self._agent_pos - self._objective_pos, ord=1
            )
        }

    '''
    The pourpose of this function is to initiate a new episode for an env,
    two principal parameters:
    - seed: can be used to initialize the random number generator to a deterministic state
    - options: can be used to specify values used within reset
    '''
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        '''
        we could set the agent and objective position randomly at each episode
        but in our case we are interested in a static position of agent and objective
        '''
        self._agent_pos = np.array([0, 0], dtype=np.int32)
        self._objective_pos = np.array([self.size - 1, self.size - 1], dtype=np.int32)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    '''
    contains the logic of the environment
    '''
    def step(self, action: ActType):

if __name__ == '__main__':
