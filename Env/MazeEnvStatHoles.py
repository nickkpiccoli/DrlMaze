from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium.core import ActType

class MazeWorldEnv(gym.Env):

    def __init__(self, size: int = 5):
        #set the maze size
        self.size = size
        #set the positions of agent and objective
        self._agent_pos = np.array([-1,-1], dtype=np.int32)
        self._objective_pos = np.array([-1,-1], dtype=np.int32)
        self.hole_positions = np.array([[1, 1], [2, 2], [3, 3], [1, 3], [3, 1]], dtype=np.int32)
        self._isfallen = False


        '''set the observation space, values that environment returns
           as observation of the current state
           in this case the observation space is a Dict containing two couples
           agent and target, that are two Box(matrices) in this case representing
           a vector of [x,y] where x and y can assume values between 0,size-1'''
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size-1, shape=(2,), dtype=int),
                "objective": gym.spaces.Box(0, size-1, shape= (2,), dtype=int),
                "holes": gym.spaces.Box(0, size - 1, shape=(len(self.hole_positions), 2), dtype=int)
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
        return {"agent": self._agent_pos, "objective": self._objective_pos, "holes": self.hole_positions}

    #auxiliary information, return the manhattan distance between agent and objective
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_pos - self._objective_pos, ord=1
            ),
            "fallen": self._isfallen
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
        self._isfallen = False

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    '''
    contains the logic of the environment, take an action as input and
    compute the state of the environment after the action is performed
    returning a tuple containing
    - the next observation
    - the resulting reward
    - is environ terminated?
    - is environ truncated?
    - aux info
    '''
    def step(self, action: ActType):
        #first we convert the action to direction
        direction = self._action_to_direction[action]
        '''
        using clip just to be sure not to leave the grid bounds we calculate the
        new agent position
        '''
        self._agent_pos = np.clip(
            self._agent_pos + direction, 0, self.size - 1
        )

        #environment terminate if the target and the objective has the same position or target get in a hole(negative reward)
        terminated = np.array_equal(self._agent_pos, self._objective_pos)
        fall = any(np.array_equal(self._agent_pos, hole) for hole in self.hole_positions)
        truncated = False

        if terminated:
            reward = 1  # Positive reward for reaching the objective
        elif fall:
            reward = -10  # Negative reward for falling into a hole
            self._isfallen = True
            terminated = True
        else:
            reward = -0.1  # Small positive reward for a valid move

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
