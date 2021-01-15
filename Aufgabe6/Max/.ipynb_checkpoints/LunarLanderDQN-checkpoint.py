import gym
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time

class Net(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
    
    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return F.softmax(y_pred, dim=0)
    
def train(network, top20, learningRate):
    network.train()
    criterion = nn.MSELoss()
    #criterion = nn.L1Loss()
    optimizer = optim.SGD(network.parameters(), momentum=0.5, lr=learningRate)
    
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
    
    learningRates = [0.1, 0.01, 0.001, 0.0001]
    layers = [100, 200, 300, 400, 500]
    iterations = 100
    num_episodes = 100
    num_steps = 500
    episodes = []
    
    plotData = []
    
    
    
    #while(len(episodes) == 0 or episodes[:,1].mean() <= 100):
    
            network = Net(8, l, 4)
            for i in range(iterations):
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

                    done = True    
                    episodes.append((episode, reward_e))
                    state = env.reset()

                print("MeanReward:")
                episodes = np.array(episodes)
                print(episodes[:,1].mean())
                
                episodes_sorted = episodes[episodes[:,1].argsort()]
                top20 = episodes_sorted[80:]
                print(episodes_sorted[80:,1])

                train(network, top20[:,0], lr)
                
            plotData.append((episodes[:,1].mean(), lr, l))
    
    plotData = plotData[plotData[:,0].argsort()]
    print(plotData)
    
main()
    