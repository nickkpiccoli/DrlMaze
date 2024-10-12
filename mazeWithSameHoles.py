import gymnasium as gym
from stable_baselines3 import PPO

gym.register(
    id="MazeWorld-v1",
    entry_point="Env.MazeEnvStatHoles:MazeWorldEnv",
)

model = PPO.load("models/ppo_maze_world_with_same_holes")

env = gym.make("MazeWorld-v1")

obs, info = env.reset()

done = False
steps = 0

print("Agent starting to move:")

while not done:
    action, _states = model.predict(obs, deterministic=True)

    obs, reward, done, truncated, info = env.step(int(action))

    agent_pos = obs["agent"]
    print(f"Step {steps}: Agent move to {agent_pos}")

    if done and not info["fallen"]:
        print("BullsEye!")
    elif info["fallen"]:
        env.reset()
        done = False
        steps=0
        print("Fail")

    steps += 1
env.close()


