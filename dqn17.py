import random
import numpy as np
import csv
import json

class EdgeNode:
    pass  # Placeholder for EdgeNode implementation

class FogNode:
    pass  # Placeholder for FogNode implementation

class CloudNode:
    pass  # Placeholder for CloudNode implementation

class Queue:
    def __init__(self, max_size):
        self.items = []
        self.max_size = max_size

    def enqueue(self, item):
        if len(self.items) < self.max_size:
            self.items.append(item)
        else:
            print("Queue overflow: unable to enqueue task.")

    def dequeue(self):
        if self.items:
            return self.items.pop(0)
        else:
            return None

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)

class Task:
    def __init__(self, cpu_usage, memory_usage):
        self.cpu_usage = cpu_usage
        self.memory_usage = memory_usage

class EdgeLayer:
    def __init__(self, num_nodes, max_queue_size):
        self.queue = Queue(max_size=max_queue_size)
        self.nodes = [EdgeNode() for _ in range(num_nodes)]
        self.network_delay = 0
        self.queue_delay = 0

    def process_task(self, task):
        self.queue.enqueue(task)

    def handle_queue_overflow(self):
        print("Edge Layer: Queue overflow occurred.")

    def update_delays(self, network_delay, queue_delay):
        self.network_delay += network_delay
        self.queue_delay += queue_delay

    # Implement M/M/n model for edge layer
    def mmn_model(self, arrival_rate, service_rate, num_servers):
        rho = arrival_rate / (num_servers * service_rate)
        if rho < 1:
            p0 = 1 / (sum([(num_servers * rho) ** n / np.math.factorial(n) for n in range(num_servers)]) + (num_servers * rho) ** num_servers / (np.math.factorial(num_servers) * (1 - rho)))
            Lq = ((num_servers * rho) ** (num_servers + 1) / (np.math.factorial(num_servers - 1) * (1 - rho) ** 2)) * p0
            Wq = Lq / arrival_rate
            L = arrival_rate * Wq
            W = Wq + (1 / service_rate)
            return L, W
        else:
            print("The system is unstable.")
            return None, None

class FogLayer:
    def __init__(self, num_nodes, max_queue_size):
        self.queue = Queue(max_size=max_queue_size)
        self.nodes = [FogNode() for _ in range(num_nodes)]
        self.network_delay = 0
        self.queue_delay = 0

    def process_task(self, task):
        self.queue.enqueue(task)

    def handle_queue_overflow(self):
        print("Fog Layer: Queue overflow occurred.")

    def update_delays(self, network_delay, queue_delay):
        self.network_delay += network_delay
        self.queue_delay += queue_delay

    # Implement M/M/n model for fog layer
    def mmn_model(self, arrival_rate, service_rate, num_servers):
        rho = arrival_rate / (num_servers * service_rate)
        if rho < 1:
            p0 = 1 / (sum([(num_servers * rho) ** n / np.math.factorial(n) for n in range(num_servers)]) + (num_servers * rho) ** num_servers / (np.math.factorial(num_servers) * (1 - rho)))
            Lq = ((num_servers * rho) ** (num_servers + 1) / (np.math.factorial(num_servers - 1) * (1 - rho) ** 2)) * p0
            Wq = Lq / arrival_rate
            L = arrival_rate * Wq
            W = Wq + (1 / service_rate)
            return L, W
        else:
            print("The system is unstable.")
            return None, None

class CloudLayer:
    def __init__(self, num_nodes, max_queue_size):
        self.queue = Queue(max_size=max_queue_size)
        self.nodes = [CloudNode() for _ in range(num_nodes)]
        self.network_delay = 0
        self.queue_delay = 0

    def process_task(self, task):
        self.queue.enqueue(task)

    def handle_queue_overflow(self):
        print("Cloud Layer: Queue overflow occurred.")

    def update_delays(self, network_delay, queue_delay):
        self.network_delay += network_delay
        self.queue_delay += queue_delay

    # Implement M/M/n model for cloud layer
    def mmn_model(self, arrival_rate, service_rate, num_servers):
        rho = arrival_rate / (num_servers * service_rate)
        if rho < 1:
            p0 = 1 / (sum([(num_servers * rho) ** n / np.math.factorial(n) for n in range(num_servers)]) + (num_servers * rho) ** num_servers / (np.math.factorial(num_servers) * (1 - rho)))
            Lq = ((num_servers * rho) ** (num_servers + 1) / (np.math.factorial(num_servers - 1) * (1 - rho) ** 2)) * p0
            Wq = Lq / arrival_rate
            L = arrival_rate * Wq
            W = Wq + (1 / service_rate)
            return L, W
        else:
            print("The system is unstable.")
            return None, None

# Example usage:
edge_layer = EdgeLayer(num_nodes=5, max_queue_size=10)
fog_layer = FogLayer(num_nodes=3, max_queue_size=20)
cloud_layer = CloudLayer(num_nodes=1, max_queue_size=30)


# Reading data from the CSV file and creating Task instances
with open('/content/drive/MyDrive/Colab Notebooks/slice.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    for line_number, row in enumerate(reader, start=2):  # Start from line 2
        try:
            cpu_usage_str = row[10]  # Assuming CPU usage is at index 24
            memory_usage_str = row[10]  # Assuming memory usage is at index 23
            print(cpu_usage_str)

            # Check if memory usage is 'None' as a string
            if eval(memory_usage_str)['memory'] == 'None':
                memory_usage = None
            else:
                memory_usage = float(eval(memory_usage_str)['memory'])
            


            # Check if memory usage is 'None' as a string
            if eval(cpu_usage_str)['cpus'] == 'None':
                cpu_usage = None
            else:
                cpu_usage = float(eval(cpu_usage_str)['cpus'])
            print(cpu_usage)
            print(memory_usage)
            
            task = Task(cpu_usage=cpu_usage, memory_usage=memory_usage)
            # Process the task based on the layer
            # For example, if CPU usage is more than memory usage, process it in the edge layer
            if cpu_usage > memory_usage:
                edge_layer.process_task(task)
            else:
                # If not, process it in the fog layer
                fog_layer.process_task(task)
        except (ValueError, IndexError, KeyError) as e:
            print(f"Error: Unable to extract CPU and memory usage from row {line_number}. Skipping row. Error: {e}")
            continue

# Example of M/M/n model calculation for Edge Layer
arrival_rate = 10  # tasks per second
service_rate = 5  # tasks per second
num_servers = 2  # number of servers

L, W = edge_layer.mmn_model(arrival_rate, service_rate, num_servers)
if L and W:
    print("Edge Layer:")
    print("Average number of tasks in the system (L):", L)
    print("Average time a task spends in the system (W):", W)

# Similar calculations can be done for Fog and Cloud layers

# Example of M/M/n model calculation for Fog Layer
arrival_rate_fog = 15  # tasks per second
service_rate_fog = 8  # tasks per second
num_servers_fog = 3  # number of servers

L_fog, W_fog = fog_layer.mmn_model(arrival_rate_fog, service_rate_fog, num_servers_fog)
if L_fog and W_fog:
    print("\nFog Layer:")
    print("Average number of tasks in the system (L):", L_fog)
    print("Average time a task spends in the system (W):", W_fog)

# Example of M/M/n model calculation for Cloud Layer
arrival_rate_cloud = 20  # tasks per second
service_rate_cloud = 12  # tasks per second
num_servers_cloud = 1  # number of servers

L_cloud, W_cloud = cloud_layer.mmn_model(arrival_rate_cloud, service_rate_cloud, num_servers_cloud)
if L_cloud and W_cloud:
    print("\nCloud Layer:")
    print("Average number of tasks in the system (L):", L_cloud)
    print("Average time a task spends in the system (W):", W_cloud)
