from clustering import calculate_density, analyze_centrality
import os
import database
import random
import numpy as np

def jaccard_similarity(set1, set2):
    '''
    Calculate Jaccard index of two sets.
    If both sets are empty, return 0.
    Parameters:
        set1 (set): first set
        set2 (set): second set
    Returns:
        jaccard_index (float): Jaccard index of two sets
    '''
    if len(set1.union(set2)) == 0:
        return 0
    return len(set1.intersection(set2)) / len(set1.union(set2))

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
    Find densest communities
    '''
    densest_communities = {}
    for method, cluster in clusters.items():
        communities = normalize_clusters(cluster)
        
        # Calculate density for each community
        community_densities = []
        for idx, community in communities.items():
            density = calculate_density(G, community)
            community_densities.append(community)
        
        # Sort communities by density and select the top ones
        densest_communities[method] = sorted(community_densities, key=lambda x: x[1], reverse=True)[:num_communities]
    return densest_communities
        
def find_largest(clusters, num_communities=10):
    largest_communities = {}
    smallest_communities = {}
    medium_communities = {}
    for method, cluster in clusters.items():
        communities = normalize_clusters(cluster)
        
        # Calculate size for each community
        community_sizes = []
        for idx, community in communities.items():
            community_sizes.append(community)
        
        # Sort communities by size and select the top ones
        largest_communities[method] = sorted(community_sizes, key=lambda x: len(x), reverse=True)[:num_communities]
        smallest_communities[method] = sorted(community_sizes, key=lambda x: len(x))[:num_communities]
        medium_communities[method] = sorted(community_sizes, key=lambda x: len(x))[num_communities//2:num_communities//2+num_communities]
    return largest_communities, smallest_communities, medium_communities

def find_random(clusters, num_communities=10):
    random_communities = {}
    for method, cluster in clusters.items():
        communities = normalize_clusters(cluster)
        community_list = list(communities.values())
        random_communities[method] = random.sample(community_list, min(num_communities, len(community_list)))
    return random_communities

def save_communities(communities, db_path, prefix):
    for method, community_list in communities.items():
        for i, community in enumerate(community_list):
            directory = f"../output/{method}/{prefix}"
            os.makedirs(directory, exist_ok=True)
            
            with open(f"../output/{method}/{prefix}/community_{i}.txt", "w") as f:
                f.write(f"Size: {len(community)}\n")
                f.write("\n")
                for product_id in community:
                    #print(f"{type(product_id)}")
                    product_metadata = database.get_metadata(product_id, db_path)
                    f.write(f"Product ID: {product_id}\n")
                    f.write(f"Title: {product_metadata.get('title', 'N/A')}\n")
                    f.write(f"Category: {product_metadata.get('main_category', 'N/A')}\n")
                    f.write(f"Average Rating: {product_metadata.get('average_rating', 'N/A')}\n")
                    f.write(f"Rating Number: {product_metadata.get('rating_number', 'N/A')}\n")
                    f.write("\n")
                else:
                    f.write(f"Product ID: {product_id}\n")
                    f.write("Metadata not found\n\n")
            print(f"Saved {prefix} {method} community {i} to {prefix}/{method}/community_{i}.txt")

def save_central_nodes(G, db_path, amount=10, output_dir="../output/centralities"):
    results = analyze_centrality(G, amount)
    os.makedirs(output_dir, exist_ok=True)
    for measure, nodes in results.items():
        filename = os.path.join(output_dir, f"{measure.replace(' ', '_').lower()}_centrality.txt")
        with open(filename, 'w') as f:
            for node, value in nodes:
                product_metadata = database.get_metadata(node, db_path)
                f.write(f"Centrality: {value}\n")
                f.write(f"Product ID: {node}\n")
                f.write(f"Title: {product_metadata.get('title', 'N/A')}\n")
                f.write(f"Category: {product_metadata.get('main_category', 'N/A')}\n")
                f.write(f"Average Rating: {product_metadata.get('average_rating', 'N/A')}\n")
                f.write(f"Rating Number: {product_metadata.get('rating_number', 'N/A')}\n")
                f.write("\n")
        print(f"{measure} saved to {filename}")
    compare_centralities(results)

def compare_centralities(results):
    
    metrics = list(results.keys())
    for i in range(len(metrics)):
        for j in range(i+1, len(metrics)):
            metric1 = metrics[i]
            metric2 = metrics[j]
            nodes1 = set([node for node, _ in results[metric1]])
            nodes2 = set([node for node, _ in results[metric2]])
            similarity = jaccard_similarity(nodes1, nodes2)
            print(f"Similarity between {metric1} and {metric2}: {similarity}")
        

def get_moderate_community(cluster):
    communities = normalize_clusters(cluster)
    sizes = [len(community) for community in communities.values()]
    mean_size = np.mean(sizes)
    std_dev = np.std(sizes)
    biggest_yet = 0
    index = 0
    for idx, community in communities.items():
        #print(idx)
        if len(community) > biggest_yet and len(community) < mean_size + std_dev:
            
            biggest_yet = len(community)
            index = idx
    #print(f"Biggest yet: {biggest_yet}")
    return communities.get(index, None)