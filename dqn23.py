import threading
import time
import numpy as np
import csv
import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = np.random.choice(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class EdgeNode:
    def __init__(self, cpu_capacity, memory_capacity):
        self.cpu_capacity = cpu_capacity
        self.memory_capacity = memory_capacity
        self.queue = Queue(max_size=10)

    def load(self):
        return self.queue.size()

class FogNode:
    def __init__(self, cpu_capacity, memory_capacity):
        self.cpu_capacity = cpu_capacity
        self.memory_capacity = memory_capacity
        self.queue = Queue(max_size=20)

    def load(self):
        return self.queue.size()

class CloudNode:
    def __init__(self, cpu_capacity, memory_capacity):
        self.cpu_capacity = cpu_capacity
        self.memory_capacity = memory_capacity
        self.queue = Queue(max_size=100)

    def load(self):
        return self.queue.size()

class Queue:
    def __init__(self, max_size):
        self.items = []
        self.max_size = max_size
        self.lock = threading.Lock()

    def enqueue(self, item):
        with self.lock:
            if len(self.items) < self.max_size:
                self.items.append(item)
            else:
                print("Queue overflow: unable to enqueue task.")

    def dequeue(self):
        with self.lock:
            if self.items:
                return self.items.pop(0)
            else:
                return None

    def is_empty(self):
        with self.lock:
            return len(self.items) == 0

    def size(self):
        with self.lock:
            return len(self.items)

class Task:
    def __init__(self, cpu_usage, memory_usage):
        self.cpu_usage = cpu_usage
        self.memory_usage = memory_usage

class EdgeLayer:
    def __init__(self, num_nodes, max_queue_size):
        self.nodes = [EdgeNode(cpu_capacity=0.015, memory_capacity=0.2) for _ in range(num_nodes)]

    def process_task(self, task):
        for node in self.nodes:
            if node.cpu_capacity >= task.cpu_usage and node.memory_capacity >= task.memory_usage:
                node.queue.enqueue(task)
                return True
        return False

class FogLayer:
    def __init__(self, num_nodes, max_queue_size):
        self.nodes = [FogNode(cpu_capacity=0.03, memory_capacity=0.4) for _ in range(num_nodes)]

    def process_task(self, task):
        for node in self.nodes:
            if node.cpu_capacity >= task.cpu_usage and node.memory_capacity >= task.memory_usage:
                node.queue.enqueue(task)
                return True
        return False

class CloudLayer:
    def __init__(self, num_nodes, max_queue_size):
        self.nodes = [CloudNode(cpu_capacity=0.06, memory_capacity=0.8) for _ in range(num_nodes)]

    def process_task(self, task):
        for node in self.nodes:
            if node.cpu_capacity >= task.cpu_usage and node.memory_capacity >= task.memory_usage:
                node.queue.enqueue(task)
                return True
        return False

performance_metrics = {
    'tasks_processed': 0,
    'total_latency': 0,
    'cpu_utilization': 0,
    'memory_utilization': 0
}

def update_metrics(task, start_time, end_time, layers):
    performance_metrics['tasks_processed'] += 1
    performance_metrics['total_latency'] += (end_time - start_time)
    performance_metrics['cpu_utilization'] += sum([node.cpu_capacity - node.queue.size() for layer in layers for node in layer.nodes])
    performance_metrics['memory_utilization'] += sum([node.memory_capacity - node.queue.size() for layer in layers for node in layer.nodes])

def calculate_average_metrics():
    num_tasks = performance_metrics['tasks_processed']
    return {
        'average_latency': performance_metrics['total_latency'] / num_tasks,
        'average_cpu_utilization': performance_metrics['cpu_utilization'] / num_tasks,
        'average_memory_utilization': performance_metrics['memory_utilization'] / num_tasks
    }

def offload_task_baseline(task, edge_layer, fog_layer, cloud_layer):
    for layer in [edge_layer, fog_layer, cloud_layer]:
        if layer.process_task(task):
            return

def simulate_baseline():
    with open('/content/drive/MyDrive/Colab Notebooks/data.csv', 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)

        for line_number, row in enumerate(csv_reader, start=2):
            try:
                task_cpu_usage = float(row[0])
                task_memory_usage = float(row[1])
                task = Task(cpu_usage=task_cpu_usage, memory_usage=task_memory_usage)
                start_time = time.time()
                offload_task_baseline(task, edge_layer, fog_layer, cloud_layer)
                end_time = time.time()
                update_metrics(task, start_time, end_time, [edge_layer, fog_layer, cloud_layer])
            except (ValueError, IndexError, KeyError) as e:
                print(f"Error: Unable to extract CPU and memory usage from row {line_number}. Skipping row. Error: {e}")
                continue

def offload_task_dqn(task, edge_layer, fog_layer, cloud_layer, dqn, state):
    action = dqn.act(state)
    start_time = time.time()

    if action == 0:
        edge_layer.process_task(task)
    elif action == 1:
        fog_layer.process_task(task)
    else:
        cloud_layer.process_task(task)

    end_time = time.time()
    update_metrics(task, start_time, end_time, [edge_layer, fog_layer, cloud_layer])

    next_state = get_state(edge_layer, fog_layer, cloud_layer)
    reward = calculate_reward(edge_layer, fog_layer, cloud_layer)
    done = check_done_condition()

    dqn.remember(state, action, reward, next_state, done)
    state = next_state

def get_state(edge_layer, fog_layer, cloud_layer):
    state = []
    state.extend([node.load() for node in edge_layer.nodes])
    state.extend([node.load() for node in fog_layer.nodes])
    state.extend([node.load() for node in cloud_layer.nodes])
    return np.array(state).reshape(1, -1)

def calculate_reward(edge_layer, fog_layer, cloud_layer):
    return -sum([node.load() for node in edge_layer.nodes + fog_layer.nodes + cloud_layer.nodes])

def check_done_condition():
    return False

def plot_metrics(baseline_metrics, dqn_metrics):
    metrics = ['average_latency', 'average_cpu_utilization', 'average_memory_utilization']
    baseline_values = [baseline_metrics[metric] for metric in metrics]
    dqn_values = [dqn_metrics[metric] for metric in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, baseline_values, width, label='Baseline')
    ax.bar(x + width/2, dqn_values, width, label='DQN')

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title('Performance Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    plt.show()

    # Simulate Task Arrival for DQN
def simulate_dqn():
    with open('/content/drive/MyDrive/Colab Notebooks/data.csv', 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)

        for line_number, row in enumerate(csv_reader, start=2):
            try:
                task_cpu_usage = float(row[0])
                task_memory_usage = float(row[1])
                task = Task(cpu_usage=task_cpu_usage, memory_usage=task_memory_usage)
                offload_task_dqn(task, edge_layer, fog_layer, cloud_layer, dqn, state)
            except (ValueError, IndexError, KeyError) as e:
                print(f"Error: Unable to extract CPU and memory usage from row {line_number}. Skipping row. Error: {e}")
                continue

# Baseline simulation
edge_layer = EdgeLayer(num_nodes=3, max_queue_size=10)
fog_layer = FogLayer(num_nodes=2, max_queue_size=20)
cloud_layer = CloudLayer(num_nodes=1, max_queue_size=30)
simulate_baseline()
baseline_metrics = calculate_average_metrics()

# DQN simulation
edge_layer = EdgeLayer(num_nodes=3, max_queue_size=10)
fog_layer = FogLayer(num_nodes=2, max_queue_size=20)
cloud_layer = CloudLayer(num_nodes=1, max_queue_size=30)
state_size = 6
action_size = 3
dqn = DQN(state_size, action_size)
state = get_state(edge_layer, fog_layer, cloud_layer)
simulate_dqn()
dqn_metrics = calculate_average_metrics()

# Plot metrics
plot_metrics(baseline_metrics, dqn_metrics)
