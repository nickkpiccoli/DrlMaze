import gymnasium as gym
from stable_baselines3 import PPO

gym.register(
    id="MazeWorld-v1",
    entry_point="Env.MazeEnvStatHoles:MazeWorldEnv",
)

def train_maze_model():
    env = gym.make("MazeWorld-v1")
    print("Starting to train model...")
    model = PPO("MultiInputPolicy", "MazeWorld-v1", verbose=1).learn(50000)
    model.save("models/ppo_maze_world_with_same_holes")
    print("Model trained!")

#train_maze_model()


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


