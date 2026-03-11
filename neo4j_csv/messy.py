import networkx as nx
import matplotlib.pyplot as plt
import random

# 1. Create a random messy graph (Brute Force Simulation)
G = nx.erdos_renyi_graph(n=50, p=0.3) # 50 nodes, 30% chance of connection = MESSY

# 2. Draw it to look "Bad" (Red & Cluttered)
plt.figure(figsize=(10, 10), facecolor='black')
pos = nx.random_layout(G)  # Random layout makes it look messier than spring layout

nx.draw_networkx_nodes(G, pos, node_size=50, node_color='yellow', alpha=0.8)
nx.draw_networkx_edges(G, pos, edge_color='white', alpha=0.3, width=1)

plt.axis('off')
plt.title("Brute Force (O(N²))", color='white', fontsize=20)
plt.show()