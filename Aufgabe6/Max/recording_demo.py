import gym
import random

env = gym.make("LunarLander-v2")
env = gym.wrappers.Monitor(env, "recording_lunar", force=True)

no_of_actions = env.action_space.n
total_reward = 0
state = env.reset()
done = False

while not done:
    action = random.randint(0, env.action_space.n-1)  # choose a random action
    state, reward, done, info = env.step(action)
    # state (x, y ,v(x) ,v(y) ,angle ,v(angle) ,leg1contact ,leg1contact)
    print(state)
    total_reward += reward


print("\ndone!")
print(f"Total reward: {total_reward}")
