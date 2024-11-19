import socket
import time  

def client_request(server_id, target_node):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('localhost', 9000 + server_id))  
    client_socket.send(str(target_node).encode('utf-8'))  
    start_time = time.time()

    response = client_socket.recv(1024).decode('utf-8')

   
    end_time = time.time()
    round_trip_time = end_time - start_time  

    print(f"Client received: {response}")
    print(f"Round-trip time for request: {round_trip_time:.6f} seconds")

    client_socket.close()

# Simulate clients requesting shortest paths to different target nodes
def run_clients():
    client_request(0, 4)  
    client_request(1, 4)  
    client_request(2, 4)  

if __name__ == "__main__":
    run_clients()
