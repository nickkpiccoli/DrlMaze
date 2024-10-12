import tkinter as tk
import gymnasium as gym
from stable_baselines3 import PPO
gym.register(
    id="MazeWorld-v2",
    entry_point="Env.MazeEnvDynHoles:MazeWorldEnv",
)

#Function to update grid position of agent and holes
def update_buttons(obs):
    #ripristinate buttons
    for i in range(env.size):
        for j in range(env.size):
            buttons[i][j].config(bg="SystemButtonFace")  # Ripristina il colore normale

    #holes
    for hole in obs["holes"]:
        buttons[hole[0]][hole[1]].config(text="H", bg="red")

    #agent
    agent_pos = obs["agent"]
    buttons[agent_pos[0]][agent_pos[1]].config(text="A", bg="black", fg="white")

    #objective
    objective_pos = obs["objective"]
    buttons[objective_pos[0]][objective_pos[1]].config(text="O", bg="green")


def update_agent():
    obs, info = env.reset()

    #update grid at episode beginning
    update_buttons(obs)

    done = False
    steps = 0

    print("Agent starting to move:")

    def step_agent():
        nonlocal done, steps, obs

        if not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))

            #update positions of elements in grid
            update_buttons(obs)

            if done and not info["fallen"]:
                print("BullsEye!")
            if done and info["fallen"]:
                obs, info = env.reset()
                update_buttons(obs)
                done = False
                steps = 0
                print("Fail")
            if truncated:
                obs, info = env.reset()
                update_buttons(obs)
                done = False
                steps = 0
                print("truncated, too many steps in same cell")

            steps += 1
            #next step will happen in 1500 ms
            master.after(1500, step_agent)

    step_agent()


master = tk.Tk()

buttons = []
button_size = 10

model = PPO.load("models/ppo_maze_world_with_dyn_holes_repstep_v3")

env = gym.make("MazeWorld-v2")

for i in range(env.size):
    row_buttons = []
    for j in range(env.size):
        btn = tk.Button(master, height=5, width=20)
        btn.grid(row=i, column=j, padx=2, pady=2)
        row_buttons.append(btn)
    buttons.append(row_buttons)

master.after(100, update_agent)

tk.mainloop()