import json
import pickle
import os
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch_geometric.data import Data
from sklearn.manifold import TSNE
import torch.nn.functional as F
from torch_geometric.data import Batch

# Load and return the CFG data.
def read_graph_data(graph_addr):
    with open(graph_addr, 'r') as graph_file:
        graph_data = json.load(graph_file)
        return graph_data

# Read a pickle file.
def read_pickle(file_addr):
    with open(file_addr, 'rb') as handle:
        file = pickle.load(handle)
    return file

# Process the graph data and convert to a PyG Data object.
def process_graph_data(graph_data, label, label_transformer):
    if 'edges' in graph_data.keys():
        raw_edges = graph_data['edges']
    else:
        raw_edges = graph_data['edge_list']

    if 'nodes' in graph_data.keys():
        raw_nodes = graph_data['nodes']
    else:
        raw_nodes = graph_data['node_dict']
        
    y = label

    unique_node_idx_counter = 0
    node_mapping = {}
    edges = [[], []]

    for node in raw_nodes:
        node_mapping[str(node)] = unique_node_idx_counter
        unique_node_idx_counter += 1

    for edge in raw_edges:
        edges[0].append(node_mapping[str(edge[0])])
        edges[1].append(node_mapping[str(edge[1])])

    edge_idx = torch.tensor(edges, dtype=torch.long)
    if y != 'adversarial':
        y = torch.tensor(label_transformer.transform([y]), dtype=torch.long)
    x = None
    data = Data(x=x, edge_index=edge_idx, y=y)

    return data

def get_processed_addr(addr):
    base_addr = "datasets/normal_dataset"
    dataset_name, class_name  = None, None

    if 'benign' in addr:
        class_name = 'benign'
    elif 'gafgyt' in addr:
        class_name = 'gafgyt'
    elif 'mirai' in addr:
        class_name = 'mirai'
    elif 'tsunami' in addr:
        class_name = 'tsunami'

    processed_path = os.path.join(base_addr, class_name, 'test', os.path.basename(addr))

    if not os.path.exists(processed_path):
        print(processed_path, ' does not exist.')

    return processed_path, class_name


def plot_histogram(data_list, num_bins, legend_list = None, title = None, x_label = None, y_label = None, save_path = None):
    plt.figure(figsize=(15, 5))
    for data, legend in zip(data_list, legend_list):
        plt.hist(data, bins = num_bins, label = legend)
    plt.legend()
    plt.xlabel(x_label) 
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    plt.show()