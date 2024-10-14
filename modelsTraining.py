import gymnasium as gym
from stable_baselines3 import PPO, DQN

gym.register(
    id="MazeWorld-v0",
    entry_point="Env.MazeEnvNoHoles:MazeWorldEnv",
)
gym.register(
    id="MazeWorld-v1",
    entry_point="Env.MazeEnvStatHoles:MazeWorldEnv",
)
gym.register(
    id="MazeWorld-v2",
    entry_point="Env.MazeEnvDynHoles:MazeWorldEnv",
)

def train_DQN_model(name, envname):
    env = gym.make(envname)
    print("Starting to train DQN model...")
    model = DQN("MultiInputPolicy", envname, verbose=1).learn(50000)
    model.save(f"models/dqn_{name}")
    print("model trained")

def train_DQN_model_cyclic(name, envname):
    env = gym.make(envname)
    l = 1
    h = 10
    for i in range(1,10):
        print(f"Training DQN model phase {i}/{h}...")
        model = DQN("MultiInputPolicy", envname, verbose=1, tensorboard_log="log",ent_coef=0.01).learn(50000*i, reset_num_timesteps=False)
        model.save(f"models/dqn_{name}")
        print(f"phase {i}/{h} concluded")

def train_PPO_model(name, envname):
    env = gym.make(envname)
    print("Starting to train PPO model...")
    model = PPO("MultiInputPolicy", envname, verbose=1).learn(500)
    model.save(f"models/ppo_{name}")
    print("model trained")

def train_PPO_model_cyclic(name,envname):
    env = gym.make(envname)
    l = 1
    h = 10
    for i in range(l,h):
        print(f"Training PPO model, phase {i}/{h}...")
        model = PPO("MultiInputPolicy", envname, verbose=1, tensorboard_log="log",ent_coef=0.01).learn(50000*i, reset_num_timesteps=False)
        model.save(f"models/ppo_{name}")
        print(f"phase {i}/{h} concluded")

environments = {
    "v0": "MazeWorld-v0",
    "v1": "MazeWorld-v1",
    "v2": "MazeWorld-v2",
}

train_PPO_model("test", environments["v0"])
