# Shortest Path Algorithms in Dynamic Networks

## Project Overview
This project explores and implements three classical shortest path algorithms — **Dijkstra's Algorithm**, **Bellman-Ford Algorithm**, and **Floyd-Warshall Algorithm** — for dynamic network routing. The algorithms are tested and compared based on their performance, handling of node failures, and adaptability to network changes. The project includes client-server interactions, where clients request the shortest paths to specific nodes in a network.

## Key Features
- **Dijkstra's Algorithm**: Fast and efficient for finding the shortest path from a source node to all other nodes in a graph with non-negative edge weights.
- **Bellman-Ford Algorithm**: Handles graphs with negative weights and detects negative weight cycles.
- **Floyd-Warshall Algorithm**: A comprehensive solution for finding the shortest paths between all pairs of nodes in a graph.

### Additional Features:
- **Node Failure Simulation**: Simulates the failure of network nodes and observes how the algorithms recompute the shortest paths.
- **Client-Server Model**: A multi-server setup where clients request shortest paths from different servers based on the algorithm they choose.

## System Architecture
The system is designed with a client-server architecture where each algorithm runs on a separate server. Clients communicate with the servers to request the shortest paths, and the server processes the request and sends back the results.

- **Client**: Sends requests to the server for the shortest path between a source node and a target node.
- **Server**: Processes the request using one of the shortest path algorithms and sends back the result.
- **Failure Handling**: Node failures are simulated by removing nodes and edges, and the algorithms are tested to recalculate paths accordingly.

## Tools & Technologies
- **Programming Language**: Python
- **Libraries**: `socket`, `time`
- **Networking**: TCP/IP communication between client and server
- **Algorithms Implemented**:
  - Dijkstra's Algorithm
  - Bellman-Ford Algorithm
  - Floyd-Warshall Algorithm

## Setup & Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/shortest-path-algorithms.git
   cd shortest-path-algorithms
   ```

2. **Run the server** (You can run separate servers for each algorithm):
   - For Dijkstra's Server:
     ```bash
     python dijkstra_server.py
     ```
   - For Bellman-Ford Server:
     ```bash
     python bellman_ford_server.py
     ```
   - For Floyd-Warshall Server:
     ```bash
     python floyd_warshall_server.py
     ```

3. **Run the client**:
   - The client will connect to the appropriate server based on the request:
     ```bash
     python client.py
     ```

## Usage
- **Client Requests**: The client sends a target node and receives the shortest path computed by the server. This interaction happens over TCP/IP communication.
- **Simulating Node Failures**: Node failures are simulated within the network topology, and the algorithms adjust the paths accordingly.

### Example:
```python
client_request(server_id=0, target_node=4)
```

## Performance Comparison
| **Algorithm**       | **Time Complexity** | **Suitability**                                | **Limitations**                             |
|---------------------|---------------------|------------------------------------------------|---------------------------------------------|
| **Dijkstra**         | O(V²)               | Best for networks with positive weights        | Limited to graphs with non-negative weights |
| **Bellman-Ford**     | O(VE)               | Works for graphs with negative weights         | Slower compared to Dijkstra                 |
| **Floyd-Warshall**   | O(V³)               | Computes all-pairs shortest paths              | Computationally expensive for large graphs |

## References
1. **Forouzan, B. A.** (2007). *Data Communications and Networking with TCP/IP Protocol Suite* (6th ed.). McGraw-Hill.
2. **Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C.** (2009). *Introduction to Algorithms* (3rd ed.). MIT Press.
