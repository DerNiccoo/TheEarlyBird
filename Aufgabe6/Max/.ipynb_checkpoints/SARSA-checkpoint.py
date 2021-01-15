import random
import gym
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("FrozenLake-v0")
random.seed(0)
np.random.seed(0)
#env.seed(0)

print("## Frozen Lake ##")
print("Start state:")
env.render()


def init_q():
    q_values = {}
    for s in range(0, env.observation_space.n):
        for a in range(0, env.action_space.n):
            q_values[(s, a)] = 0
    return q_values


# TODO update Q-values with TD prediction after every step
# TODO update policy after every step
def play_episode(q_values, epsilon):
    # current state we are in
    state = env.reset()
    done = False
    r_s = [] # rewards after each step of episode
    s_a = [] # state|action pairs of episode
    
    # choose first action
    action = choose_action(q_values, state, epsilon)
    
    while not done:
        s_a.append((state, action))
        state_bar, reward, done, _ = env.step(action)
        r_s.append(reward)
        
        # get A' for S' to calculate q_values
        action_bar = choose_action(q_values, state_bar, epsilon)
        
        # update q-values
        q_values[(state, action)] += 0.3 * (reward + q_values[(state_bar, action_bar)] - q_values[(state, action)])
        
        state = state_bar
        action = action_bar
        
    return s_a, r_s


# pick highest q value action or if you slip, pick random action
def choose_action(q_values, state, epsilon):
    if random.random() > epsilon:
        relevant_qs = [q_values[(state, a)] for a in range(0, env.action_space.n)]
        # there can be more than one best action
        best_actions_indexes = [i for i, v in enumerate(relevant_qs) if v == max(relevant_qs)]
        # in this case randomly choose one of them
        return random.choice(best_actions_indexes)
    else:
        return random.randint(0, 3)


def main():
    no_executions = 100
    no_episodes = 1000
    
    epsilons = [0.01, 0.1, 0.5, 1.0]

    plot_data = []
    for e in epsilons:
        rewards = np.zeros(no_executions * no_episodes)
        rewards = rewards.reshape(no_executions, no_episodes)
        for j in range(0, no_executions):
            q_values = init_q()
            for i in range(0, no_episodes):
                s, r = play_episode(q_values, epsilon=e)
                rewards[j][i] = sum(r)
        
        rewards = rewards.mean(axis=0)
        plot_data.append(np.cumsum(rewards))

    plt.figure()
    plt.xlabel("No. of episodes")
    plt.ylabel("Sum of rewards")
    for i, eps in enumerate(epsilons):
        plt.plot(range(0, no_episodes), plot_data[i], label="e=" + str(eps))
    plt.legend()
    plt.show()


main()
