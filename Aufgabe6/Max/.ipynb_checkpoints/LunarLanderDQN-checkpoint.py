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
        #return F.softmax(y_pred, dim=0)
        return y_pred
    
def train(network, optimizer, criterion, top20):
    network.train()
    counter = 0
    
    for episode in top20:
        for state, target in episode:
            optimizer.zero_grad()
            target = torch.tensor([target])
            output = network(torch.from_numpy(np.array(state))).unsqueeze(0)
            #if counter < 5:
            #    print(F.softmax(output), target)
            #    counter += 1
            #print(output.shape)
            #print(torch.tensor([target]))
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            
def choose_action(output, epsilon):
    if random.random() > epsilon:
        return np.argmax(output.detach().numpy(), axis=0)
    else:
        return random.randint(0, 3)

        

def main():
    env = gym.make("LunarLander-v2")
    is_monitor_active = False
    

    no_of_actions = env.action_space.n
    total_reward = 0
    state = env.reset()
    done = False
    
    num_episodes = 100
    num_steps = 500
    episodes = []
    plot_data = []
    
    network = Net(8, 256, 4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), momentum=0.9, lr=0.00015)
    
    while(len(episodes) == 0 or episodes[:,1].mean() <= 100):
    #for i in range(100):
        episodes = []
        for e in range(num_episodes):
            episode = []
            reward_e = 0
            for s in range(num_steps):
                
            #while (not done):
                # probabilities for action
                output = F.softmax(network(torch.from_numpy(np.array(state))))
                action =  random.choices([0,1,2,3], output, k=1)[0]
                episode.append((state, action))
                state, reward, done, _ = env.step(action)
                reward_e += reward
                last_reward = reward

                if done:
                    #reward_e /= s
                    break

            episodes.append((episode, reward_e, last_reward))
            if is_monitor_active:
                time.sleep(0.1)
            state = env.reset()
            done = False
            

        episodes = np.array(episodes)
        mean = episodes[:,1].mean()
        #if (mean > 50) and (not is_monitor_active):
        #    env = gym.wrappers.Monitor(env, "recording_lunar", force=True)
        #    time.sleep(1)
        #    state = env.reset()
        #    is_monitor_active = True
            
        plot_data.append(mean)
        print(mean)

        episodes_sorted = episodes[episodes[:,1].argsort()]
        top20 = episodes_sorted[80:]
        #print(top20[:,1])
        train(network, optimizer, criterion, top20[:,0])
    
    plt.figure()
    plt.xlabel("episode")
    plt.ylabel("mean reward")
    plt.plot(plot_data, label="100 Hidden layer, lr=0.1")
    plt.legend()
    plt.show()

main()
    