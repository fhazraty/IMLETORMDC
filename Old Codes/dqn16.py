import random
import numpy as np

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
    def __init__(self, data):
        self.data = data

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

# Simulate task arrival and processing
for _ in range(20):
    task = Task(data=random.randint(1, 100))
    edge_layer.process_task(task)

for _ in range(30):
    task = Task(data=random.randint(1, 100))
    fog_layer.process_task(task)

for _ in range(40):
    task = Task(data=random.randint(1, 100))
    cloud_layer.process_task(task)

# Example of M/M/n model calculation
arrival_rate = 10  # tasks per second
service_rate = 5  # tasks per second
num_servers = 2  # number of servers

L, W = edge_layer.mmn_model(arrival_rate, service_rate, num_servers)
if L and W:
    print("Edge Layer:")
    print("Average number of tasks in the system (L):", L)
    print("Average time a task spends in the system (W):", W)

# Similar calculations can be done for Fog and Cloud layers
