import networkx as nx
import community as community_louvain  
from cdlib import algorithms
import numpy as np

def apply_clustering_algorithms(G):
    '''
    Apply clustering algorithms to provided graph G.
    Parameters:
        G (nx.Graph): graph to partition
    Returns: 
        clusters (dict): dictionary where keys are method names, values are partition results.
    '''
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
    '''
    Calculate modularity of given partition of graph G.
    Parameters:
        G (nx.Graph): partitioned graph
        partition (dict): dictionary where keys are nodes, values are community assignments
    Returns:
        modularity (float): modularity of given partition
    '''
    return community_louvain.modularity(partition, G)

def calculate_density(G, community):
    '''
    Calculate density of given community in graph G.
    Density is defined as number of edges divided by number of all possible edges in a clique of the same size.
    Parameters:
        G (nx.Graph): graph containing community
        community (list): list of nodes in community
    Returns:
        density (float): density of community as a subgraph of G
    '''
    subgraph = G.subgraph(community)
    return nx.density(subgraph)

def analyze_clusters(clusters):
    '''
    Analyze clusters in statistical terms.
    Parameters:
        clusters (dict): dictionary where keys are method names, values are partition results.
    Returns:
        analysis (dict): dictionary where keys are method names, values are dictionaries with mean, variance and standard deviation of community sizes.
    '''
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
    '''
    Analyze centrality measures of nodes in graph G.
    Parameters:
        G (nx.Graph): graph to analyze
        amount (int): number of top nodes to return
    Returns:
        results (dict): dictionary where keys are centrality measures, values are lists of tuples (node, centrality value)
    '''
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
