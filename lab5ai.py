import gym
import numpy as np
from IPython.display import clear_output
import random

env = gym.make('Taxi-v3')

actions = env.action_space.n
state = env.observation_space.n

q_table = np.zeros((state,actions))


num_episodes = 50000
max_steps_per_episode =99
learning_rate=0.7
discount_rate = 0.6
exploration_rate=1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate= 0.01

rewards_all_episodes = []


for episode in range(num_episodes):
    state = env.reset()
    done = False
    reward_current_episode = 0
    
    for step in range(max_steps_per_episode):
        
        exploration_threshold = random.uniform(0,1)
        if exploration_threshold > exploration_rate:
            action = np.argmax(q_table[state,:])
        else:
            action = env.action_space.sample()
        new_state,reward,done,info = env.step(action)
        
        
        q_table[state,action] = q_table[state,action]*(1-learning_rate)+ learning_rate*(reward + discount_rate * np.max(q_table[new_state, :]))
        state=new_state
        reward_current_episode += reward
        
        if done== True:
            break
    exploration_rate = min_exploration_rate + \
        (max_exploration_rate- min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
    rewards_all_episodes.append(reward_current_episode)
print("Entrenamiento completo")

q_table


rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes/1000)
count = 1000

print("Promedio de 1000 episodios")

for r in rewards_per_thousand_episodes:
    print(count, ":", str(sum(r/1000)))
    count+=1000
    
import time 
for episode in range(3):
    status = env.reset()
    done = False
    print("El episodio es: "+ str(episode))
    time.sleep(1)
    
    for step in range(max_steps_per_episode):
        clear_output(wait=True)
        env.render()
        time.sleep(.4)
        
        action = np.argmax(q_table[state,:])
        
        new_state, reward, done, info = env.step(action)
        
        if done:
            clear_output(wait=True)
            env.render()
            if reward == 1:
                print("Meta lograda")
                time.sleep(2)
                clear_output(wait=True)
            else:
                print("****Fallo****")
                time.sleep(2)
                clear_output(wait=True)
                
            break
        state=new_state
env.close()