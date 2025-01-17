from clustering import calculate_density, analyze_centrality
import os
import database
import random
import sqlite3
import networkx as nx
import numpy as np

def jaccard_similarity(set1, set2):
    '''
    Calculate Jaccard index of two sets.
    If both sets are empty, return 1.
    Parameters:
        set1 (set): first set
        set2 (set): second set
    Returns:
        jaccard_index (float): Jaccard index of two sets
    '''
    union_of_sets = set1.union(set2)
    intersection_of_sets = set1.intersection(set2)
    if union_of_sets == 0:
        return 1
    return len(intersection_of_sets) / len(union_of_sets)

def normalize_clusters(cluster):
    '''
    Normalize cluster dictionary, starting dictionary has nodes as keys and communities as values
    Returns dictionary where keys are communities identifiers, values are lists of nodes in that community.
    Parameters:
        cluster (dict): dictionary where keys are nodes, values are community assignments
    Returns:
        normalized_cluster (dict): dictionary where keys are community identifiers, values are lists of nodes in that community
    Examples:
        cluster = {1: 0, 2: 0, 3: 1, 4: 1}
        normalize_clusters(cluster) -> {0: [1, 2], 1: [3, 4]}
    '''
    return {c: [k for k, v in cluster.items() if v == c] for c in set(cluster.values())}

def find_dense(G, clusters, num_communities=10):
    '''
    Find densest communities for each method in clusters dictionary.
    Parameters:
        G (nx.Graph): graph to analyze
        clusters (dict): dictionary where keys are method names, values are partition results.
        num_communities (int): number of densest communities to return
    Returns:
        densest_communities (dict): dictionary where keys are method names, values are communities
    '''
    densest_communities = {}
    for method, cluster in clusters.items():
        communities = normalize_clusters(cluster)
        
        community_densities = []
        for idx, community in communities.items():
            density = calculate_density(G, community)
            community_densities.append((community, density))
        
        densest_communities[method] = [community for community, _ in sorted(community_densities, key=lambda x: x[1], reverse=True)[:num_communities]]
    return densest_communities
        
def find_largest(clusters, num_communities=10):
    '''
    Find largest, medium and smallest communities for each method in clusters dictionary.
    Parameters:
        clusters (dict): dictionary where keys are method names, values are partition results.
        num_communities (int): number of communities to return
    Returns:
        largest_communities (dict): dictionary where keys are method names, values are lists of largest communities
        smallest_communities (dict): dictionary where keys are method names, values are lists of smallest communities
        medium_communities (dict): dictionary where keys are method names, values are lists of medium communities
    '''
    largest_communities = {}
    smallest_communities = {}
    medium_communities = {}
    for method, cluster in clusters.items():
        communities = normalize_clusters(cluster)
        
        community_sizes = []
        for idx, community in communities.items():
            community_sizes.append(community)

        sorted_by_size = sorted(community_sizes, key=lambda x: len(x))
        
        largest_communities[method] = sorted_by_size[-num_communities:]
        smallest_communities[method] = sorted_by_size[:num_communities]
        medium_communities[method] = sorted_by_size[num_communities//2:num_communities//2+num_communities]
    return largest_communities, smallest_communities, medium_communities

def find_random(clusters, num_communities=10):
    '''
    Find random communities for each method in clusters dictionary.
    Parameters:
        clusters (dict): dictionary where keys are method names, values are partition results.
        num_communities (int): number of communities to return
    Returns:
        random_communities (dict): dictionary where keys are method names, values are lists of random communities
    '''
    random_communities = {}
    for method, cluster in clusters.items():
        communities = normalize_clusters(cluster)
        community_list = list(communities.values())
        random_communities[method] = random.sample(community_list, min(num_communities, len(community_list)))
    return random_communities

def save_communities(communities, db_path, prefix):
    '''
    Save product metadata of communities to files.
    Parameters:
        communities (dict): dictionary where keys are method names, values are lists of communities
        db_path (str): path to SQLite database
        prefix (str): prefix for output directory
    Returns:
        None'''
    for method, community_list in communities.items():
        for i, community in enumerate(community_list):
            directory = f"../output/{method}/{prefix}"
            os.makedirs(directory, exist_ok=True)
            
            with open(f"../output/{method}/{prefix}/community_{i}.txt", "w") as f:
                f.write(f"Size: {len(community)}\n")
                f.write("\n")
                for product_id in community:
                    #print(f"{type(product_id)}")
                    try:
                        product_metadata = database.get_metadata(product_id, db_path)
                        if product_metadata:
                            f.write(f"Product ID: {product_id}\n")
                            f.write(f"Title: {product_metadata.get('title', 'N/A')}\n")
                            f.write(f"Subtitle: {product_metadata.get('subtitle', 'N/A')}\n")
                            f.write(f"Category: {product_metadata.get('main_category', 'N/A')}\n")
                            f.write(f"Subcategories: {product_metadata.get('categories', 'N/A')}\n")
                            f.write(f"Average Rating: {product_metadata.get('average_rating', 'N/A')}\n")
                            f.write(f"Rating Number: {product_metadata.get('rating_number', 'N/A')}\n")
                            f.write(f"Author: {product_metadata.get('author', 'N/A')}\n")
                            f.write(f"Bought from: {product_metadata.get('store', 'N/A')}\n")
                            f.write("\n")
                        else:
                            f.write(f"Product ID: {product_id} not found in {db_path}\n")
                            f.write("\n")
                    except sqlite3.Error as e:
                        f.write(f"Error while fetching {product_id} in {db_path}: {e}\n")
                        f.write("\n")
                else:
                    f.write(f"Product ID: {product_id}\n")
                    f.write("Metadata not found\n\n")
            print(f"Saved {prefix} {method} community {i} to {prefix}/{method}/community_{i}.txt")

def save_central_nodes(G, db_path, amount=10, output_dir="../output/centralities"):
    '''
    Save most central nodes in graph to files.
    Finds most central nodes using degree, closeness and betweenness centrality and saves them to files.
    Parameters:
        G (nx.Graph): graph to analyze
        db_path (str): path to SQLite database
    Returns:
        None
    '''
    results = analyze_centrality(G, amount)
    os.makedirs(output_dir, exist_ok=True)
    for measure, nodes in results.items():
        filename = os.path.join(output_dir, f"{measure.replace(' ', '_').lower()}_centrality.txt")
        with open(filename, 'w') as f:
            mean_rating = 0
            for node, value in nodes:
                product_metadata = database.get_metadata(node, db_path)
                r_number = product_metadata.get('rating_number', 'N/A')
                subcategories = product_metadata.get('categories', 'N/A')
                f.write(f"Centrality: {value}\n")
                f.write(f"Product ID: {node}\n")
                f.write(f"Title: {product_metadata.get('title', 'N/A')}\n")
                f.write(f"Category: {product_metadata.get('main_category', 'N/A')}\n")
                f.write(f"Subcategories, length: {len(subcategories)}: {subcategories}\n")
                f.write(f"Average Rating: {product_metadata.get('average_rating', 'N/A')}\n")
                f.write(f"Rating Number: {r_number}\n")
                f.write(f"Author: {product_metadata.get('author', 'N/A')}\n")
                f.write(f"Bought from: {product_metadata.get('store', 'N/A')}\n")
                f.write("\n")
                mean_rating+=r_number
            mean_rating/=amount
        print(f"{measure} saved to {filename}. Mean rating: {mean_rating}")
    compare_centralities(results)
    mean_revs_amount(G)

def mean_revs_amount(graph, amount = 1000, db_path = '../data/metadata.db'):
    '''
    Use Monte Carlo technique for finding mean rating number
    Used for comparing mean node with most central nodes
    Parameters:
        graph (nx.Graph): analyzed graph
        amount (int): amount of nodes to be randomly chosen
        db_path (str): path to metadata database
    Returns:
        None
    '''
    random_nodes = random.sample(list(graph.nodes), amount)
    rev_amount= 0
    for node in random_nodes:
        product_metadata = database.get_metadata(node, db_path)
        rev_amount += product_metadata.get('rating_number')
    rev_amount/=amount
    print(f"Mean rating number: {rev_amount}")


def compare_centralities(results):
    '''
    Compare central nodes found using different centrality measures by calculating Jaccard similarity.
    Jaccard similarity is calculated as intersection of nodes found by two measures divided by union of nodes found by two measures.
    It is used to print out similarity between two centralities, rather than saving it, because its only one line. 
    Parameters:
        results (dict): dictionary where keys are centrality measures, values are lists of tuples (node, centrality value)
    Returns:
        None
    '''
    metrics = list(results.keys())
    for i in range(len(metrics)):
        for j in range(i+1, len(metrics)):
            metric1 = metrics[i]
            metric2 = metrics[j]
            nodes1 = set([node for node, _ in results[metric1]])
            nodes2 = set([node for node, _ in results[metric2]])
            similarity = jaccard_similarity(nodes1, nodes2)
            print(f"Similarity between {metric1} and {metric2}: {similarity}")
        

def get_moderate_community(cluster, min_size=50, max_size=500):
    '''
    Get moderate community from cluster.
    Looks for communities with size good for representation
    Parameters:
        cluster (dict): dictionary where keys are nodes, values are community assignments   
    Returns:
        community (list): list of nodes in community
    '''
    communities = normalize_clusters(cluster)
    #print(communities)
    #sizes = [len(community) for community in communities.values()]
    # mean_size = np.mean(sizes)
    # std_dev = np.std(sizes)
    # biggest_yet = 0
    # index = 0
    # for idx, community in communities.items():
    #     if len(community) > min_size and len(community) < max_size:
    #         return community
    #return None
    return next(
        filter(lambda community: print(f"Checking community: if {min_size}<{len(community)}<{max_size} ") or min_size < len(community) < max_size, communities.values()),
        None
    )
   
def save_basic_stats(graph, filepath = '../output/basic_stats.txt'):
    '''
    Finds graphs statistics and saves it in LaTeX friendly format
    Parameters:
        graph (nx.Graph): examinated graph
        filepath (str): path to file where results are saved
    Returns:
        None
    '''
    with open(filepath, 'w') as f:
        num_nodes = graph.number_of_nodes()
        f.write(f"Liczba wierzchołków & {num_nodes} \\\\ \\hline \n")
        num_edges = graph.number_of_edges()
        f.write(f"Liczba krawędzi & {num_edges} \\\\ \\hline \n")
        avg_degree = sum(dict(graph.degree()).values()) / num_nodes
        f.write(f"Średni stopień wierzchołków & {avg_degree} \\\\ \\hline \n")
        num_components = nx.number_connected_components(graph)
        f.write(f"Liczba spójnych składowych & {num_components} \\\\ \\hline \n")
        # largest_cc = max(nx.connected_components(review_graph), key=len)
        # subgraph = review_graph.subgraph(largest_cc).copy()
        # avg_path_length = nx.average_shortest_path_length(subgraph)

        density = nx.density(graph)
        f.write(f"Gęstość grafu & {density} \n")
        # sample_nodes = random.sample(graph.nodes(), k=10000)
        # clustering_coeff = nx.average_clustering(graph, nodes=sample_nodes)
        # f.write(f"Średnia klastrowalność grafu & {clustering_coeff}")
