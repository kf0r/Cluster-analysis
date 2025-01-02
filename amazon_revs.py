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
import os
from review import Review
import database
from scipy import stats
import statistics
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

def save_communities(communities, db_path, prefix):
    for method, community_list in communities.items():
        for i, community in enumerate(community_list):
            directory = f"{method}/{prefix}"
            os.makedirs(directory, exist_ok=True)
            
            with open(f"{method}/{prefix}/community_{i}.txt", "w") as f:
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

def find_dense(G, clusters, num_communities=10):
    densest_communities = {}
    for method, cluster in clusters.items():
        communities = {c: [k for k, v in cluster.items() if v == c] for c in set(cluster.values())}
        
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
        communities = {c: [k for k, v in cluster.items() if v == c] for c in set(cluster.values())}
        
        # Calculate size for each community
        community_sizes = []
        for idx, community in communities.items():
            community_sizes.append(community)
        
        # Sort communities by size and select the top ones
        largest_communities[method] = sorted(community_sizes, key=lambda x: len(x), reverse=True)[:num_communities]
        smallest_communities[method] = sorted(community_sizes, key=lambda x: len(x))[:num_communities]
        medium_communities[method] = sorted(community_sizes, key=lambda x: len(x))[num_communities//2:num_communities//2+num_communities]
    return largest_communities, smallest_communities, medium_communities

def plot_community_sizes_distro(clusters):
    for method, cluster in clusters.items():
        communities = {c: [k for k, v in cluster.items() if v == c] for c in set(cluster.values())}
        sizes = [len(community) for community in communities.values()]
        print(f"Method: {method}")
        print(sizes)
        plt.hist(sizes, bins=len(set(sizes)))
        plt.yscale('log')
        plt.title(f"Rozkład wielkości klastrów metody ({method})")
        plt.xlabel("Wielkość klastra")
        plt.ylabel("Ilość klastrów")
        plt.grid(True)
        plot_filename = method + "/community_sizes_distro.png"
        plt.savefig(plot_filename)
        #plt.show()
        plt.close()

def plot_statistics_community_sizes(clusters, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    means = {}
    std_devs = {}
    variances = {}
    medians = {}
    modes = {}
    modularities = {}
    for method, cluster in clusters.items():
        communities = {c: [k for k, v in cluster.items() if v == c] for c in set(cluster.values())}
        sizes = [len(community) for community in communities.values()]
        means[method] = np.mean(sizes)
        std_devs[method] = np.std(sizes)
        variances[method] = np.var(sizes)
        medians[method] = np.median(sizes)
        modes[method] = statistics.mode(sizes)
        modularities[method] = calculate_modularity(review_graph, cluster)
    
    plot_from_data(modularities, "Modularność klastrów", "Metoda", "Modularność", output_dir)
    plot_from_data(means, "Średnia wielkość społeczności", "Metoda", "Średnia", output_dir)
    plot_from_data(std_devs, "Odchylenie standardowe wielkości społeczności", "Metoda", "Odchylenie standardowe", output_dir)
    plot_from_data(variances, "Wariancja wielkości społeczności", "Metoda", "Wariancja", output_dir)
    plot_from_data(medians, "Mediana wielkości społeczności", "Metoda", "Mediana", output_dir)
    plot_from_data(modes, "Dominanta wielkości społeczności", "Metoda", "Dominanta", output_dir)

def plot_from_data(data_dict, title, xlabel, ylabel, output_dir="plots"):
    plt.figure(figsize=(10, 6))
    methods = list(data_dict.keys())
    values = list(data_dict.values())
    plt.bar(methods, values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    filename = os.path.join(output_dir, title.replace(" ", "_").lower() + ".png")
    print(f"Saving plot to {filename}")
    plt.savefig(filename)
    plt.close()


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

    print("Finding dense communities...")
    dense = find_dense(review_graph, clusters)

    print("Finding communities based on size...")
    largest, smallest, medium = find_largest(clusters)

    print("Getting random communities...")
    #randos = get_random_communities(clusters)

    print("Saving communities...")
    save_communities(largest, db_path, "largest")
    save_communities(smallest, db_path, "smallest")
    save_communities(medium, db_path, "medium")
    save_communities(dense, db_path, "dense")
    #save_communities(randos, db_path, "random")

    print("Plotting community size distribution...")
    plot_community_sizes_distro(clusters)

    print("Mean community size graph...")
    print("Variance community size graph...")
    print("Standard deviation community size graph...")
    print("Calculating modularity...")
    plot_statistics_community_sizes(clusters)

    