import gymnasium as gym
from stable_baselines3 import DQN


#registering model to use it later with gym.make()
gym.register(
    id="MazeWorld-v0",
    entry_point="Env.MazeEnvNoHoles:MazeWorldEnv",
)

#trying the model with a simple simulation
model = DQN.load("models/dqn_maze_world")

env = gym.make("MazeWorld-v0")

obs, info = env.reset()

terminated = False
steps = 0

print("Agent starting to move:")

while not terminated:
    action, _states = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(int(action))

    agent_pos = obs["agent"]
    print(f"Step {steps}: Agent move to {agent_pos}")

    if terminated:
        print("BullsEye!")
    elif truncated:
        obs, info = env.reset()
        print("Fail")

    steps += 1
env.close()


