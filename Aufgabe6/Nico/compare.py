import random
import gym
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("FrozenLake-v0")

gamma = 0.6

print("## Frozen Lake ##")
print("Start state:")
env.render()


def init_q():
  q_values = {}
  for s in range(0, env.observation_space.n):
    for a in range(0, env.action_space.n):
      q_values[(s, a)] = 0
  return q_values


def play_episode(q_values, epsilon):

  state = env.reset()
  done = False
  r_s = []
  s_a = []
  next_state = []
  while not done:
    action = choose_action(q_values, state, epsilon)

    s_a.append((state, action))
    state, reward, done, _ = env.step(action)
    next_state.append(state)
    r_s.append(reward)
  return s_a, r_s, next_state


def choose_action(q_values, state, epsilon):
  if random.random() > epsilon:
    relevant_qs = [q_values[(state, a)] for a in range(0, env.action_space.n)]
    # there can be more than one best action
    best_actions_indexes = [i for i, v in enumerate(relevant_qs) if v == max(relevant_qs)]
    # in this case randomly choose on of them
    return random.choice(best_actions_indexes)
  else:
    return random.randint(0, 3)


def make_mean_plot_data(execution_plot_data):
  result = []

  for eps_data in execution_plot_data:
    mean_plot = []
    time_dict = {}

    for iteration in eps_data:
      for slot, data in enumerate(iteration):
        if slot in time_dict:
          time_dict[slot].append(data)
        else:
          time_dict[slot] = []
          time_dict[slot].append(data)

    for ts, reward in time_dict.items():
      mean_plot.append(sum(reward) / len(reward))

    result.append(mean_plot)

  return result


def main():
  no_episodes = 1000
  epsilons = [0.01, 0.1, 0.5, 1.0]
  algos = ['SARSA', 'Q-Learning', 'MC-Control']

  result_plot = []

  for algo in algos:
    run_plot = []
    for e in epsilons:
      plot_data = []
      for run in range(100):
        rewards = []
        q_values = init_q()
        for i in range(0, no_episodes):
          s, r, n_s = play_episode(q_values, epsilon=e)
          rewards.append(sum(r))


          if algo == 'SARSA': # SARSA
            # update q-values
            for state, q in enumerate(s):
              q_next = 0
              if (state + 1 < len(s)):
                q_next = s[state + 1][1]

              reward = r[state]
              q_values[q] += 0.3 * (reward + gamma * q_next - q_values[q])
          elif algo == 'Q-Learning': # Q-Learning
            # update q-values
            for state, q in enumerate(s):
                reward = r[state]
                q_values[q] += 0.3 * (reward + gamma * choose_action(q_values, n_s[state], 0) - q_values[q])
          elif algo == 'MC-Control': # MC-Control
            # update q-values
            for i2, q in enumerate(s):
                return_i = sum(r[i2:])
                q_values[q] += 0.3 * (return_i - q_values[q])

        plot_data.append(np.cumsum(rewards))
      run_plot.append(plot_data)

    mean_plot_data = make_mean_plot_data(run_plot)

    best_reward = 0
    best_data = []
    best_eps = 0
    plot_data = []
    for eps, data in enumerate(mean_plot_data):
      if data[-1] > best_reward:
        best_reward = data[-1]
        best_data = data
        best_eps = epsilons[eps]

    result_plot.append((algo, best_eps, best_data))

  plt.figure()
  plt.xlabel("No. of episodes")
  plt.ylabel("Sum of rewards")
  for algo, eps, data in result_plot:
    plt.plot(range(0, no_episodes), data, label=algo + " e=" + str(eps))
  plt.legend()
  plt.show()


main()
