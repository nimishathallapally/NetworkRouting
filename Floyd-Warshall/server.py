import socket
import threading
import heapq
import random
import time
import networkx as nx
import matplotlib.pyplot as plt


def generate_large_sparse_graph(nodes, edges):
    graph = {i: {} for i in range(nodes)}
    edge_count = 0
    current_edge = 0

    
    while edge_count < edges:
        u = current_edge % nodes
        v = (current_edge + 1) % nodes
        weight = (current_edge + 1) % 10 + 1  

        if u != v and v not in graph[u]:
            graph[u][v] = weight
            graph[v][u] = weight 
            edge_count += 1
        
        
        current_edge += 1

       
        if current_edge > nodes * 2: 
            print("Unable to generate enough edges due to constraints.")
            break

    return graph
def floyd_warshall(graph, start_node, failed_node=None):
    # Step 1: Initialize distances matrix
    nodes = list(graph.keys())
    num_nodes = len(nodes)
    distances = {node: {neighbor: float("inf") for neighbor in nodes} for node in nodes}
    previous_nodes = {node: None for node in nodes}

    # Set initial distances based on the graph
    for u in graph:
        distances[u][u] = 0  # Distance to itself is 0
        for v, weight in graph[u].items():
            distances[u][v] = weight

    # Step 2: Modify graph to exclude failed node if specified
    if failed_node is not None:
        distances = {
            node: {
                neighbor: distances[node][neighbor]
                for neighbor in distances[node]
                if neighbor != failed_node
            }
            for node in distances
            if node != failed_node
        }

    # Step 3: Apply Floyd-Warshall algorithm
    for k in distances:  
        for i in distances:  
            for j in distances: 
                if distances[i][k] + distances[k][j] < distances[i][j]:
                    distances[i][j] = distances[i][k] + distances[k][j]

    # Extract the distances from the start_node
    single_source_distances = distances[start_node]
    return single_source_distances, previous_nodes


def plot_graph(graph, failed_nodes=None):
    """Visualizes the graph with optional highlighting of failed nodes."""
    G = nx.Graph()
    for node, neighbors in graph.items():
        for neighbor, weight in neighbors.items():
            G.add_edge(node, neighbor, weight=weight)

    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="skyblue",
        edge_color="gray",
        node_size=700,
        font_size=10,
        font_weight="bold",
    )
    if failed_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=failed_nodes, node_color="red")

    labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    plt.title("Network Graph")
    plt.show()


# Server code for each node
class Server:
    def __init__(self, node_id, graph):
        self.node_id = node_id
        self.graph = graph  # Network topology as graph
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(("localhost", 9000 + node_id))
        self.server_socket.listen(5)
        self.failed_nodes = set()
        self.routing_table = {}  # Stores distances to all nodes
        self.is_active = True  # Indicates if the server is up or down
        print(f"Server {node_id} started on port {9000 + node_id}")
        self.recalculate_routing_table()  # Calculate initial routing table

    def recalculate_routing_table(self):
        """Recalculate the routing table for all nodes."""
        print(f"Recalculating routing table for Server {self.node_id}")
        self.routing_table, _ = floyd_warshall(self.graph, self.node_id)
        print(f"Updated Routing Table for Server {self.node_id}:")
        for node, distance in self.routing_table.items():
            print(f"  To Node {node}: Distance {distance}")

    def handle_client(self, client_socket):
        try:
            if not self.is_active:
                # Notify client that the server is down
                print(f"Server {self.node_id} is down. Rejecting request.")
                client_socket.send(
                    f"Server {self.node_id} is currently down.".encode("utf-8")
                )
                return

            data = client_socket.recv(1024).decode("utf-8")
            if data:
                target_node = int(data)  # Target node ID
                print(
                    f"Server {self.node_id} received request for shortest path to node {target_node}"
                )
                server_start_time = time.time()

                # Retrieve the shortest path from the precomputed routing table
                distance = self.routing_table.get(target_node, float("inf"))
                response = f"Shortest path to node {target_node} from {self.node_id}: {distance}"
                server_end_time = time.time()
                server_computation_time = server_end_time - server_start_time  # Time taken to process the request
                print(f"Server {self.node_id} computation time: {server_computation_time:.6f} seconds")
                # client_socket.send(response.encode("utf-8"))
                client_socket.send(response.encode("utf-8"))
        finally:
            client_socket.close()

    def run(self):
        while True:
            client_socket, _ = self.server_socket.accept()
            client_thread = threading.Thread(
                target=self.handle_client, args=(client_socket,)
            )
            client_thread.start()

    def simulate_failure(self, failed_node):
        print(f"Simulating failure of Node {failed_node} in Server {self.node_id}")

        # If this server is the one that failed, mark it as inactive
        if self.node_id == failed_node:
            self.is_active = False
            print(f"Server {self.node_id} is now marked as down.")
            return

        # Mark the failed node as unreachable and update the graph
        self.failed_nodes.add(failed_node)

        # Update the graph for the failure
        for node in self.graph:
            if failed_node in self.graph[node]:
                del self.graph[node][failed_node]
            if node == failed_node:
                self.graph[node] = {}

        # Recalculate the routing table for all servers after the failure
        print(f"Node {failed_node} failure simulated. Recalculating routing table.")
        self.recalculate_routing_table()
        # plot_graph(self.graph, failed_nodes=self.failed_nodes)



# Start server for each node in the network
def start_servers():
    # graph = {
    #     0: {1: 10},
    #     1: {0: 10, 2: 10, 3: 50},
    #     2: {1: 10, 3: 30},
    #     3: {2: 30, 4: 40, 1: 50},
    #     4: {3: 40},
    # }
    nodes = 100
    edges = 200
    graph=generate_large_sparse_graph(nodes,edges)
    servers = []
    # plot_graph(graph)
    # Start servers
    for node_id in range(len(graph)):
        server = Server(node_id, graph)
        threading.Thread(target=server.run).start()
        servers.append(server)

    # Simulate failure of a node
    failed_node = 2  
    time.sleep(5)  
    for server in servers:
        server.simulate_failure(failed_node)

    
    print("\nSimulating client trying to access a failed server:")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect(("localhost", 8000 + failed_node))
        client_socket.send("3".encode("utf-8"))  
        response = client_socket.recv(1024).decode("utf-8")
        print(f"Response from Server {failed_node}: {response}")
    except ConnectionRefusedError:
        print(f"Server {failed_node} is unavailable.")
    finally:
        client_socket.close()


if __name__ == "__main__":
    start_servers()
