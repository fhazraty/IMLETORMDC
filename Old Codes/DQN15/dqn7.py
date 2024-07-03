import numpy as np
import pandas as pd
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import json
import re


class Environment:
    def __init__(self, data):
        self.data = data
        self.current_step = 0
        self.done = False
        self.tasks_at_edge = 0
        self.tasks_at_fog = 0
        self.tasks_at_cloud = 0

    def reset(self):
        self.current_step = 0
        self.done = False
        self.tasks_at_edge = 0
        self.tasks_at_fog = 0
        self.tasks_at_cloud = 0
        return self._next_observation()

    def _next_observation(self):
        return np.array([self.tasks_at_edge, self.tasks_at_fog, self.tasks_at_cloud])

    def step(self, action):
        # Example of adjusting data extraction
        current_data = self.data.iloc[self.current_step]
        task_type = current_data['event']  # For example, using 'event' as task type
        tasks = current_data['Tasks']  # Assuming 'Tasks' is already calculated as we discussed before

        # Example processing logic (adjust based on actual requirements)
        if action == 0:  # Process at edge
            processed = min(tasks, self.tasks_at_edge)
            self.tasks_at_edge -= processed
            reward = processed
        elif action == 1:  # Offload to fog
            self.tasks_at_fog += tasks
            reward = -tasks * 0.1
        else:  # Offload to cloud
            self.tasks_at_cloud += tasks
            reward = -tasks * 0.2

        self.current_step += 1
        self.done = self.current_step >= len(self.data)
        next_state = self._next_observation()
        return next_state, reward, self.done


class DQNAgent:
    def __init__(self, state_size):
        self.state_size = state_size
        self.action_size = 3  # Edge, Fog, Cloud
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        # Add the experience tuple to the memory buffer
        self.memory.append((state, action, reward, next_state, done))

    # Other methods of the DQNAgent class
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward if done else reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def load_data(filename):
    data = pd.read_csv(filename)
    # Convert the space-separated string of numbers to a Python list
    def custom_parser(x):
        # Replace spaces with commas to create a proper list format
        x = re.sub(r'\s+', ',', x.strip('[]'))
        return len(eval(f'[{x}]'))

    data['Tasks'] = data['cpu_usage_distribution'].apply(custom_parser)
    return data

data = load_data('borg_traces_data.csv')

env = Environment(data)
agent = DQNAgent(state_size=3)

episodes = 1000
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
plt.plot(results)
plt.title('Improvement in Offloading with DQN')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()
