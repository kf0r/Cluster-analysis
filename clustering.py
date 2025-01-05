import networkx as nx
import community as community_louvain  
from cdlib import algorithms
import numpy as np

def apply_clustering_algorithms(G):
    clusters = {}
    print("Applying Louvain clustering...")
    louvain_partition = community_louvain.best_partition(G)
    clusters['louvain'] = louvain_partition
    print("Louvain clustering done.")
    
    print("Applying Leiden clustering...")
    leiden_communities = algorithms.leiden(G)
    leiden_partition = {node: idx for idx, community in enumerate(leiden_communities.communities) for node in community}
    clusters['leiden'] = leiden_partition
    print("Leiden clustering done.")
    
    print("Applying Label Propagation clustering...")
    label_propagation_communities = algorithms.label_propagation(G)
    label_propagation_partition = {node: idx for idx, community in enumerate(label_propagation_communities.communities) for node in community}
    clusters['label_propagation'] = label_propagation_partition
    print("Label Propagation clustering done.")
    return clusters

def calculate_modularity(G, partition):
    return community_louvain.modularity(partition, G)

def calculate_density(G, community):
    subgraph = G.subgraph(community)
    return nx.density(subgraph)

def analyze_clusters(clusters):
    analysis = {}
    for method, cluster in clusters.items():
        if isinstance(cluster, dict):
            # For Louvain, which returns a dictionary
            sizes = [len([k for k, v in cluster.items() if v == c]) for c in set(cluster.values())]
        elif hasattr(cluster, 'communities'):
            # For NodeClustering objects
            sizes = [len(c) for c in cluster.communities]
        analysis[method] = {
            'mean': np.mean(sizes),
            'variance': np.var(sizes),
            'std_dev': np.std(sizes)
        }
    return analysis

def analyze_centrality(G, amount=10):
    degree_centrality = nx.degree_centrality(G)
    print("Degree centrality calculated.")
    closeness_centrality = nx.closeness_centrality(G)
    print("Closeness centrality calculated.")
    betweenness_centrality = nx.betweenness_centrality(G)
    print("Betweenness centrality calculated.")
    
    results = {
        "Degree Centrality": sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:amount],
        "Betweenness Centrality": sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:amount],
        "Closeness Centrality": sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:amount]
    }
    return results
