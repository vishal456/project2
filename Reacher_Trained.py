from unityagents import UnityEnvironment
import torch
import numpy as np


from ddpg_agent import Agent

env = UnityEnvironment(file_name='Reacher_Linux/Reacher.x86_64',seed=0)

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]

num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


agent = Agent(state_size,action_size,random_seed=0)


agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))
scores_total = 0
NUM_GAMES = 50
max_time = 1000

for _ in range(NUM_GAMES):
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations
    score = np.zeros(num_agents)
    for t in range(max_time):
        actions = []
        for j in range(num_agents):
            actions.append(agent.act(np.array([state[j]]), add_noise=False))
        env_info = env.step(actions)[brain_name]
        state = env_info.vector_observations
        reward = env_info.rewards
        done = env_info.local_done
        score += reward
        if t == max_time - 1:
            scores_total += np.mean(score)
            print(np.mean(score))
            break

print("final Scores :")
print(scores_total/NUM_GAMES)