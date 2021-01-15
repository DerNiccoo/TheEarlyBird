import gym
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
    
    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return F.softmax(y_pred, dim=0)
    
def train(network, optimizer, criterion, top20):
    network.train()
    
    for episode in top20:
        for state, target in episode:
            optimizer.zero_grad()
            output = network.forward(torch.from_numpy(np.array(state)))
            loss = criterion(output.float(), torch.tensor(target).float())
            loss.backward()
            optimizer.step()
        

def main():
    env = gym.make("LunarLander-v2")
    #env = gym.wrappers.Monitor(env, "recording_lunar", force=True)

    no_of_actions = env.action_space.n
    total_reward = 0
    state = env.reset()
    done = False
    
    num_episodes = 100
    num_steps = 500
    episodes = []
    plot_data = []
    
    network = Net(8, 100, 4)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(network.parameters(), momentum=0.9, lr=0.1)
    
    #while(len(episodes) == 0 or episodes[:,1].mean() <= 100):
    for i in range(100):
        episodes = []
        for e in range(num_episodes):
            episode = []
            reward_e = 0
            for s in range(num_steps):
                # probabilities for action
                output = network.forward(torch.from_numpy(np.array(state)))
                action = np.argmax(output.detach().numpy(), axis=0)
                savedAction = np.zeros(4)
                savedAction[action] = 1
                episode.append((state, savedAction))
                state, reward, done, _ = env.step(action)
                reward_e += reward

                if done:
                    break

            episodes.append((episode, reward_e))
            #while(not done):
            #    time.sleep(0.001)
            state = env.reset()

        episodes = np.array(episodes)
        mean = episodes[:,1].mean()
        plot_data.append(mean)
        print(mean)

        episodes_sorted = episodes[episodes[:,1].argsort()]
        top20 = episodes_sorted[80:]
        train(network, optimizer, criterion, top20[:,0])
    
    plt.figure()
    plt.xlabel("episode")
    plt.ylabel("mean reward")
    plt.plot(plot_data, label="100 Hidden layer, lr=0.1")
    plt.legend()
    plt.show()

main()
    