import numpy as np
import pandas as pd
import random
import tensorflow as tf
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import matplotlib.pyplot as plt
import json
import re


# Constants
EDGE_CPU_LIMIT = 0.015
FOG_CPU_LIMIT = 0.15
CPU_USAGE_PER_TASK = 10
high_penalty = 100  # Example high penalty for exceeding capacity

class Environment:
    def __init__(self, data):
        self.data = data
        self.current_step = 0
        self.done = False
        self.tasks_at_edge = 0
        self.tasks_at_fog = 0
        self.tasks_at_cloud = 0
        self.cpu_at_edge = 0  # Current CPU usage at edge
        self.cpu_at_fog = 0   # Current CPU usage at fog

    def reset(self):
        self.current_step = 0
        self.done = False
        self.tasks_at_edge = 0
        self.tasks_at_fog = 0
        self.tasks_at_cloud = 0
        self.cpu_at_edge = 0
        self.cpu_at_fog = 0
        return self._next_observation()

    def _next_observation(self):
        return np.array([self.tasks_at_edge, self.tasks_at_fog, self.tasks_at_cloud])
        
    def step(self, action):
        current_data = self.data.iloc[self.current_step]

        try:
            # Check if the value can be converted to a float, and if so, check if it's NaN
            if(np.isnan(current_data)):
                self.tasks_at_edge += 1
                reward = 1  # Reward for processing at edge
        except (ValueError, TypeError):
            tasks = json.loads(current_data.replace("'", '"'))
            
            if action == 0:  # Process at edge
                if self.cpu_at_edge + tasks['cpus'] > EDGE_CPU_LIMIT:
                    reward = -high_penalty  # High penalty for overloading CPU at edge
                else:
                    self.tasks_at_edge += 1
                    self.cpu_at_edge += tasks['cpus']
                    reward = 1  # Reward for processing at edge
            elif action == 1:  # Offload to fog
                if self.cpu_at_fog + tasks['cpus'] > FOG_CPU_LIMIT:
                    reward = -high_penalty  # High penalty for overloading CPU at fog
                else:
                    self.tasks_at_fog += 1
                    self.cpu_at_fog += tasks['cpus']
                    reward = -0.1  # Lesser penalty for offloading to fog
            else:  # Offload to cloud
                self.tasks_at_cloud += 1
                reward = -0.2  # Penalty for offloading to cloud

        self.current_step += 1
        self.done = self.current_step >= len(self.data)
        next_state = self._next_observation()
        return next_state, reward, self.done

class DQNLSTM:
    def __init__(self, state_size, action_size, memory_size=1000, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma    # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(64, return_sequences=True))
        model.add(tf.keras.layers.LSTM(64))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(np.expand_dims(state, axis=0))
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([x[0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch])
        dones = np.array([x[4] for x in minibatch])

        targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - dones)

        target_f = self.model.predict_on_batch(states)
        target_f[np.arange(batch_size), actions] = targets
        self.model.fit(states, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def load_data(filename):
    data = pd.read_csv(filename)
    
    return data['resource_request']


data = load_data('../borg_traces_data.csv')

for x in range(10000):

    env = Environment(data)
    agent = DQNLSTM(state_size=3, action_size=3)

    episodes = x
    results = []
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, 3])
        total_reward = 0
        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, 3])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                results.append(total_reward)
                print(f"Episode: {e+1}/{episodes}, Reward: {total_reward}, Epsilon: {agent.epsilon}")
                break
        if len(agent.memory) > 32:
            agent.replay(32)

    # Plotting the results to show improvements
    plt.clf()
    plt.plot(results)
    plt.title('Improvement in Offloading with DQNLSTM')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig('lstmdqn-'+str(x)+'.png')  # Saves the plot as a PNG file
    #plt.show()

