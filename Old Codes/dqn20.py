import threading
import time
import numpy as np
import csv
from datetime import timedelta

class EdgeNode:
    def __init__(self, cpu_capacity, memory_capacity):
        self.cpu_capacity = cpu_capacity
        self.memory_capacity = memory_capacity
        self.queue = Queue(max_size=10)  # Adjust max_size as needed

    def load(self):
        return self.queue.size()

class FogNode:
    def __init__(self, cpu_capacity, memory_capacity):
        self.cpu_capacity = cpu_capacity
        self.memory_capacity = memory_capacity
        self.queue = Queue(max_size=20)  # Adjust max_size as needed

    def load(self):
        return self.queue.size()

class CloudNode:
    def __init__(self, cpu_capacity, memory_capacity):
        self.cpu_capacity = cpu_capacity
        self.memory_capacity = memory_capacity
        self.queue = Queue(max_size=100)  # Adjust max_size as needed

    def load(self):
        return self.queue.size()

class Queue:
    def __init__(self, max_size):
        self.items = []
        self.max_size = max_size
        self.lock = threading.Lock()  # Lock for thread safety

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
        self.queue = Queue(max_size=max_queue_size)
        self.nodes = [EdgeNode(cpu_capacity=0.015,memory_capacity=0.015) for _ in range(num_nodes)]
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
        self.nodes = [FogNode(cpu_capacity=0.15,memory_capacity=0.15) for _ in range(num_nodes)]
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
        self.nodes = [CloudNode(cpu_capacity=100,memory_capacity=100) for _ in range(num_nodes)]
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
def calculate_processing_time(task):
    # Here, you can define your own function to calculate processing time
    # based on CPU and memory usage of the task.
    # For example, you can use a linear combination of CPU and memory usage
    # or any other function that reflects the actual processing behavior.
    # This is just a placeholder implementation.
    processing_time = task.cpu_usage * 0.1 + task.memory_usage * 0.05  # Example formula
    return processing_time

# Function to simulate processing and dequeue
def simulate_processing_and_dequeue(layer):
    if not layer.queue.is_empty():
        task = layer.queue.dequeue()
        print(f"Processing task with CPU usage: {task.cpu_usage}, Memory usage: {task.memory_usage} in {type(layer).__name__} Layer.")
        
        # Calculate processing time based on CPU and memory usage
        processing_time = calculate_processing_time(task)
        
        # Simulate processing time
        time.sleep(processing_time)
    else:
        print(f"No tasks in the queue of {type(layer).__name__} Layer.")
# Function for concurrent processing
def concurrent_processing(layer):
    threads = []
    for node in layer.nodes:
        thread = threading.Thread(target=simulate_processing_and_dequeue, args=(node,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

# Function to offload tasks based on resource availability
def offload_task(task, edge_layer, fog_layer, cloud_layer):
    edge_node_selected = None
    fog_node_selected = None

    # Check if the task can be offloaded to an edge node
    for edge_node in edge_layer.nodes:
        if edge_node.cpu_capacity >= task.cpu_usage and edge_node.memory_capacity >= task.memory_usage:
            edge_node_selected = edge_node
            break

    # Check if the task can be offloaded to a fog node
    for fog_node in fog_layer.nodes:
        if fog_node.cpu_capacity >= task.cpu_usage and fog_node.memory_capacity >= task.memory_usage:
            fog_node_selected = fog_node
            break

    # If both edge and fog nodes are available, choose the one with lower load
    if edge_node_selected and fog_node_selected:
        if edge_node_selected.load() <= fog_node_selected.load():
            edge_layer.process_task(task)
        else:
            fog_layer.process_task(task)
    elif edge_node_selected:
        edge_layer.process_task(task)
    elif fog_node_selected:
        fog_layer.process_task(task)
    else:
        print("No available resources in Edge or Fog layer. Offloading task to Cloud layer.")
        cloud_layer.process_task(task)

# Example usage:
edge_layer = EdgeLayer(num_nodes=5, max_queue_size=10)
fog_layer = FogLayer(num_nodes=3, max_queue_size=20)
cloud_layer = CloudLayer(num_nodes=1, max_queue_size=30)

# Reading data from the CSV file and creating Task instances




with open('/content/drive/MyDrive/Colab Notebooks/RasberryPi.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row

    start_time = None
    

    for line_number, row in enumerate(reader, start=2):  # Start from line 2
        try:
            cpu_usage = float(row[1])
            memory_usage = float(row[1])
            datetime_task = row[0]

            delaytimesleep = 0
            if start_time is None:
                start_time = datetime_task
                delaytimesleep = 0
            else :
                # Define the timestamps in the format 'Day Mon DD HH:MM:SS YYYY'
                # Convert the timestamps to datetime objects
                format_str = '%a %b %d %H:%M:%S %Y'
                dt1 = datetime.strptime(start_time, format_str)
                dt2 = datetime.strptime(datetime_task, format_str)

                # Calculate the difference in seconds
                delaytimesleep = (dt2 - dt1).total_seconds()
                
                #delaytimesleep = datetime_task - start_time
                start_time = datetime_task

            task = Task(cpu_usage=cpu_usage, memory_usage=memory_usage)

            # Offload the task based on resource availability
            time.sleep(delaytimesleep)  # Delay enqueueing


            offload_task(task, edge_layer, fog_layer,cloud_layer)
        except (ValueError, IndexError, KeyError) as e:
            print(f"Error: Unable to extract CPU and memory usage from row {line_number}. Skipping row. Error: {e}")
            continue

# Concurrent processing for Edge Layer
print("\nConcurrent Processing for Edge Layer:")
concurrent_processing(edge_layer)

# Concurrent processing for Fog Layer
print("\nConcurrent Processing for Fog Layer:")
concurrent_processing(fog_layer)

# Concurrent processing for Cloud Layer
print("\nConcurrent Processing for Cloud Layer:")
concurrent_processing(cloud_layer)
