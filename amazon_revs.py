import pandas as pd
import networkx as nx
from community import community_louvain
from cdlib import algorithms
import matplotlib.pyplot as plt
from textblob import TextBlob
from itertools import combinations
import numpy as np
import time
import random
import json
import sqlite3
import pickle

from review import Review

import networkx as nx

def save_graph(graph, filename):
    with open(filename, 'wb') as f:
        pickle.dump(graph, f)
    print(f"Graph saved to {filename}")

def load_graph(filename):
    with open(filename, 'rb') as f:
        graph = pickle.load(f)
    print(f"Graph loaded from {filename}")
    return graph

def process_reviews(input_path, error_log="error_lines.txt"):
    reviews = []
    rev = 0

    with open(input_path, 'r', encoding='utf-8') as infile, open(error_log, 'w', encoding='utf-8') as errorfile:
        for line_num, line in enumerate(infile, start=1):
            try:
                data = json.loads(line.strip())
                review = Review(
                    user_id=data["user_id"],
                    product_id=data["parent_asin"],
                    date=data["timestamp"]/1000,
                    score=data["rating"],
                    text=data["text"]
                )
                reviews.append(review)
                
                rev += 1
                if rev % 100000 == 0:
                    print(f"Processed {rev} reviews ({line_num} lines).")
            except Exception as e:
                errorfile.write(f"Exception in review {rev}, line {line_num}: {line}\n")
                errorfile.write(f"Error: {e}\n")
                continue

    print(f"Total reviews processed: {len(reviews)}")
    return reviews

def create_bipartite_graph(reviews):
    B = nx.Graph()
    i=0
    for review in reviews:

        #print(review)
        B.add_node(review.user_id, bipartite=0)
        B.add_node(review.product_id, bipartite=1)
        B.add_edge(review.user_id, review.product_id, review = review)
        i+=1
        # if i%100==0:
        #     print(f"{B[review.user_id][review.product_id]} edge added")
    return B


def analyze_communities_louvain(graph):
    start_time = time.time()
    partition = community_louvain.best_partition(graph)
    num_clusters = len(set(partition.values()))
    louvain_time = time.time() - start_time
    print(f"Louvain clusters: {num_clusters}, Time: {louvain_time:.4f}s")
    return partition, louvain_time


def analyze_communities_leiden(graph):
    start_time = time.time()
    result = algorithms.leiden(graph)
    partition = result.to_node_community_map() 
    num_clusters = len(result.communities)
    leiden_time = time.time() - start_time
    print(f"Leiden clusters: {num_clusters}, Time: {leiden_time:.4f}s")
    return partition, leiden_time

def apply_clustering_algorithms(G):
    clusters = {}
    print("Applying clustering algorithms...")
    clusters['louvain'] = community_louvain.best_partition(G)
    print("Louvain done")
    clusters['leiden'] = algorithms.leiden(G)
    print("Leiden done")
    clusters['label_propagation'] = algorithms.label_propagation(G)
    print("Label propagation done")
    # clusters['girvan_newman'] = list(nx.community.girvan_newman(G))
    # print("Girvan-Newman done")
    return clusters

def calculate_modularity(G, partition):
    return community_louvain.modularity(partition, G)

def calculate_density(G, community):
    subgraph = G.subgraph(community)
    return nx.density(subgraph)

def analyze_communities_label(graph):
    start_time = time.time()
    partition = list(nx.algorithms.community.label_propagation_communities(graph))
    num_clusters = len(partition)
    label_time = time.time() - start_time
    print(f"Label propagation clusters: {num_clusters}, Time: {label_time:.4f}s")
    return partition, label_time

def temporal_analysis(df):
    df['date'] = pd.to_datetime(df['date'], unit='s')
    df_grouped = df.groupby(pd.Grouper(key='date', freq='W'))['score'].mean()
    df_grouped.plot(title="Średnia ocena w czasie", ylabel="Średnia ocena", xlabel="Data")
    plt.show()

def generate_product_projection(bipartite_graph):
    product_graph = nx.Graph()
    #i=0
    for user in [n for n in bipartite_graph.nodes if n.startswith("A")]:
        #print(user)
        rated = list(bipartite_graph.neighbors(user))
        #print(rated)
        for p1, p2 in combinations(rated, 2):
            #i+=1
            if product_graph.has_edge(p1, p2):
                product_graph[p1][p2]['weight'] += 1
            else:
                product_graph.add_edge(p1, p2, weight=1)
                
    return product_graph


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


def visualize_statistics(analysis):
    methods = list(analysis.keys())
    means = [analysis[method]['mean'] for method in methods]
    variances = [analysis[method]['variance'] for method in methods]
    std_devs = [analysis[method]['std_dev'] for method in methods]

    plt.figure(figsize=(10, 6))
    plt.bar(methods, means, yerr=std_devs, capsize=5)
    plt.xlabel('Clustering Method')
    plt.ylabel('Cluster Size')
    plt.title('Cluster Size Statistics')
    plt.show()

def get_metadata(product_id, db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT data FROM metadata WHERE asin = ?', (product_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    else:
        return None

def save_communities(communities, db_path, prefix):
    for method, community_list in communities.items():
        for idx, (community, score, density, size) in enumerate(community_list):
            with open(f"{prefix}_{method}_community_{idx}.txt", "w") as f:
                f.write(f"Score: {score}\n")
                f.write(f"Density: {density}\n")
                f.write(f"Size: {size}\n")
                for product_id in community:
                    product_metadata = get_metadata(product_id, db_path)
                    if product_metadata:
                        f.write(f"Product ID: {product_id}\n")
                        f.write(f"Title: {product_metadata.get('title', 'N/A')}\n")
                        f.write(f"Category: {product_metadata.get('main_category', 'N/A')}\n")
                        f.write(f"Average Rating: {product_metadata.get('average_rating', 'N/A')}\n")
                        f.write(f"Rating Number: {product_metadata.get('rating_number', 'N/A')}\n")
                        f.write("\n")
                    else:
                        f.write(f"Product ID: {product_id}\n")
                        f.write("Metadata not found\n\n")
            print(f"Saved {prefix} {method} community {idx} with score {score} to {prefix}_{method}_community_{idx}.txt")

def save_largest_communities(clusters, db_path, num_communities=3):
    largest_communities = {}
    for method, cluster in clusters.items():
        if isinstance(cluster, dict):
            communities = {c: [k for k, v in cluster.items() if v == c] for c in set(cluster.values())}
        elif hasattr(cluster, 'communities'):
            communities = {i: c for i, c in enumerate(cluster.communities)}
        else:
            raise ValueError(f"Unexpected cluster type for method {method}: {type(cluster)}")
        
        # Sort communities by size and select the largest ones
        largest_communities[method] = sorted(communities.values(), key=len, reverse=True)[:num_communities]
    save_communities(largest_communities, db_path, "largest")

def save_random_communities(clusters, db_path, num_communities=3):
    random_communities = {}
    for method, cluster in clusters.items():
        if isinstance(cluster, dict):
            communities = {c: [k for k, v in cluster.items() if v == c] for c in set(cluster.values())}
        elif hasattr(cluster, 'communities'):
            communities = {i: c for i, c in enumerate(cluster.communities)}
        else:
            raise ValueError(f"Unexpected cluster type for method {method}: {type(cluster)}")
        
        # Select random communities
        random_communities[method] = random.sample(list(communities.values()), min(num_communities, len(communities)))
    save_communities(random_communities, db_path, "random")

def find_best_partitions(clusters, G, num_communities=3):
    best_partitions = {}
    for method, cluster in clusters.items():
        if isinstance(cluster, dict):
            communities = {c: [k for k, v in cluster.items() if v == c] for c in set(cluster.values())}
        elif hasattr(cluster, 'communities'):
            communities = {i: c for i, c in enumerate(cluster.communities)}
        else:
            raise ValueError(f"Unexpected cluster type for method {method}: {type(cluster)}")
        
        # Calculate modularity, density, and size for each community
        community_scores = []
        for idx, community in communities.items():
            partition = {node: idx for node in community}
            density = calculate_density(G, community)
            size = len(community)
            score = density + size  # Combine the criteria
            community_scores.append((community, score, density, size))
        
        # Sort communities by score and select the top ones
        best_communities = sorted(community_scores, key=lambda x: x[1], reverse=True)[:num_communities]
        best_partitions[method] = best_communities
    return best_partitions

if __name__ == "__main__":
    input_path = 'castrated.json'
    db_path = 'metadata.db'
    graph_filename = 'review_graph.pkl'
    try:
        review_graph = load_graph(graph_filename)
        print("Graph loaded from file")
    except FileNotFoundError:
        review_graph = nx.Graph()
        batch_reviews = process_reviews(input_path)
        B = create_bipartite_graph(batch_reviews)
        batch_graph = generate_product_projection(B)
        review_graph = nx.compose(review_graph, batch_graph)
        print(f"Processed batch of {len(batch_reviews)} reviews.")
        save_graph(review_graph, graph_filename)
        print("Graph generated and saved to file")
    
    print("Applying clustering algorithms...")
    clusters = apply_clustering_algorithms(review_graph)
    print("Clustering algorithms applied.")
    
    print("Analyzing clusters...")
    analysis = analyze_clusters(clusters)
    print("Cluster analysis done.")
    
    print("Visualizing statistics...")
    visualize_statistics(analysis)
    print("Statistics visualization done.")
    
    print("Finding best communities...")
    best_partitions = find_best_partitions(clusters, review_graph)
    print("Best communities found.")
    
    print("Saving best communities...")
    save_communities(best_partitions, db_path, "best")
    print("Best communities saved.")
    
    print("Saving largest communities...")
    save_largest_communities(clusters, db_path)
    print("Largest communities saved.")
    
    print("Saving random communities...")
    save_random_communities(clusters, db_path)
    print("Random communities saved.")