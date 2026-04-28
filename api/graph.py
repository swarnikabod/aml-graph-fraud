import networkx as nx
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

def build_graph_features(transactions):
    G = nx.DiGraph()
    for t in transactions:
        G.add_edge(t["src"], t["dst"])

    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    in_degree  = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    pagerank   = nx.pagerank(G, alpha=0.85)
    clustering = nx.clustering(G.to_undirected())

    features = np.array([
        [in_degree.get(n, 0),
         out_degree.get(n, 0),
         pagerank.get(n, 0),
         clustering.get(n, 0)]
        for n in nodes
    ], dtype=np.float32)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    x = torch.tensor(features_scaled, dtype=torch.float)

    edges = [[node_to_idx[t["src"]], node_to_idx[t["dst"]]]
              for t in transactions
              if t["src"] in node_to_idx and t["dst"] in node_to_idx]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index)
    return data, nodes, {"in_degree": in_degree, "out_degree": out_degree, "pagerank": pagerank, "clustering": clustering}
