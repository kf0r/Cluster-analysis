import matplotlib.pyplot as plt
import numpy as np
import os
import statistics
from clustering import calculate_modularity
import networkx as nx
from utility import get_moderate_community

# def visualize_statistics(analysis):
#     methods = list(analysis.keys())
#     means = [analysis[method]['mean'] for method in methods]
#     variances = [analysis[method]['variance'] for method in methods]
#     std_devs = [analysis[method]['std_dev'] for method in methods]

#     plt.figure(figsize=(10, 6))
#     plt.bar(methods, means, yerr=std_devs, capsize=5)
#     plt.xlabel('Clustering Method')
#     plt.ylabel('Cluster Size')
#     plt.title('Cluster Size Statistics')
#     plt.show()

def plot_community_sizes_distro(clusters):
    for method, cluster in clusters.items():
        communities = {c: [k for k, v in cluster.items() if v == c] for c in set(cluster.values())}
        sizes = [len(community) for community in communities.values()]
        #print(f"Method: {method}")
        #print(sizes)
        plt.hist(sizes, len(set(sizes)))
        plt.yscale('log')
        plt.title(f"Rozkład wielkości klastrów metody ({method})")
        plt.xlabel("Wielkość klastra")
        plt.ylabel("Ilość klastrów")
        plt.grid(True)
        plot_filename = method + "/community_sizes_distro.png"
        plt.savefig(plot_filename)
        #plt.show()
        plt.close()

def plot_statistics_community_sizes(review_graph, clusters, output_dir="plots"):
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

def plot_single_community(graph, clusters):
    for method, cluster in clusters.items():
        community = get_moderate_community(cluster)
        if community is None:
            print(f"No moderate community found for method {method}")
            continue
        subgraph = graph.subgraph(community)
        node_size = [3 * (1 + np.log(subgraph.degree[n])) for n in subgraph.nodes]
        pos = nx.kamada_kawai_layout(subgraph)
        print("Drawing subgraph")
        plt.figure(figsize=(15, 15))
        nx.draw(subgraph, pos, with_labels=False, node_size=node_size, width=0.3)
        
        plt.title(f"Przykładowa społeczność {method}")
        plt.savefig(f"{method}/single_community.png")
        plt.close()
        print("Subgraph drawn and saved")