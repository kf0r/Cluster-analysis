import matplotlib.pyplot as plt
import numpy as np
import random
import os
import statistics
from clustering import calculate_modularity
import networkx as nx
from utility import get_moderate_community, normalize_clusters
import database
from itertools import chain
from collections import Counter

def plot_community_sizes_distro(clusters, output_dir="../output"):
    '''
    Plot community sizes distribution for each clustering algorithm using matplotlib.
    Parameters:
        clusters (dict): dictionary where keys are method names, values are partition results.
    Returns:
        None
    '''
    for method, cluster in clusters.items():
        communities = {c: [k for k, v in cluster.items() if v == c] for c in set(cluster.values())}
        sizes = [len(community) for community in communities.values()]
        # print(f"Method: {method}")
        # print(sizes)
        # plt.hist(sizes, len(set(sizes)))

        size_counts = {size: sizes.count(size) for size in set(sizes)}

        sorted_sizes = sorted(size_counts.keys())
        sorted_counts = [size_counts[size] for size in sorted_sizes]

        print(f"Method: {method}")
        print(f"Sizes: {sorted_sizes}")
        print(f"Counts: {sorted_counts}")

        plt.plot(sorted_sizes, sorted_counts, marker='o', linestyle='-', label=method)

        plt.yscale('log')
        plt.xscale('log')
        plt.title(f"Rozkład wielkości klastrów metody ({method})")
        plt.xlabel("Wielkość klastra")
        plt.ylabel("Ilość klastrów")
        plt.grid(True)
        plot_filename = output_dir+ "/" + method + "/community_sizes_distro.png"
        plt.savefig(plot_filename)
        plt.close()
        #plt.show()
        

def plot_components_sizes_distro(review_graph, output_dir="../output/plots"):
    '''
    Plot distribution of components sizes in graph
    Parameters:
        review_graph (nx.Graph): given graph
        output_dir (str): path to diretory where plot will be saved
    Returns:
        None
    '''
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = output_dir + "/components_sizes_distro.png"
    component_sizes = [len(component) for component in nx.connected_components(review_graph)]

    plt.figure(figsize=(10, 6))
    plt.hist(component_sizes, bins=30, color='skyblue', edgecolor='black')
    plt.title("Rozkład rozmiarów składowych spójnych")
    plt.xlabel("Rozmiar składowej")
    plt.ylabel("Liczba składowych")
    plt.grid(True)
    plt.savefig(plot_filename)
    plt.close()

def plot_degree_distro(review_graph, output_dir="../output/plots"):
    '''
    Plot distribution of nodes degrees in graph
    Parameters:
        review_graph (nx.Graph): given graph
        output_dir (str): path to diretory where plot will be saved
    Returns:
        None
    '''
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = output_dir + "/degrees_distro.png"
    degrees = [degree for _, degree in review_graph.degree()]
    degree_counts = {degree: degrees.count(degree) for degree in set(degrees)}

    sorted_degrees = sorted(degree_counts.keys())
    sorted_counts = [degree_counts[degree] for degree in sorted_degrees]

    plt.figure(figsize=(10, 6))
    plt.plot(sorted_degrees, sorted_counts, marker='o', linestyle='-', label='Rozkład stopni wierzchołków')
    plt.xlabel("Stopień wierzchołka")
    plt.ylabel("Liczba wierzchołków")
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.savefig(plot_filename)
    plt.close()

    plot_filename = output_dir + "/degrees_distro_log.png"
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_degrees, sorted_counts, marker='o', linestyle='-', label='Rozkład stopni wierzchołków')
    plt.xlabel("Stopień wierzchołka (log)")
    plt.ylabel("Liczba wierzchołków (log)")
    plt.yscale('log')
    plt.xscale('log')
    #plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.savefig(plot_filename)
    plt.close()

def plot_statistics_community_sizes(review_graph, clusters, output_dir="../output/plots"):
    '''
    Plot statistics of community sizes for each clustering algorithm using matplotlib.
    Statistics are mean, standard deviation, variance, median and mode. Modularity is also plotted.
    Parameters:
        review_graph (nx.Graph): graph to analyze
        clusters (dict): dictionary where keys are method names, values are partition results.
        output_dir (str): directory to save plots
    Returns:
        None
    '''
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
        print(f"Method: {method}, mean: {means[method]}")
        std_devs[method] = np.std(sizes)
        print(f"Method: {method}, std_dev: {std_devs[method]}")
        variances[method] = np.var(sizes)
        print(f"Method: {method}, variance: {variances[method]}")
        medians[method] = np.median(sizes)
        print(f"Method: {method}, median: {medians[method]}")
        modes[method] = statistics.mode(sizes)
        print(f"Method: {method}, mode: {modes[method]}")
        modularities[method] = calculate_modularity(review_graph, cluster)
        print(f"Method: {method}, modularity: {modularities[method]}")
    
    plot_from_data(modularities, "Modularność klastrów", "Metoda", "Modularność", output_dir)
    plot_from_data(means, "Średnia wielkość społeczności", "Metoda", "Średnia", output_dir)
    plot_from_data(std_devs, "Odchylenie standardowe wielkości społeczności", "Metoda", "Odchylenie standardowe", output_dir)
    plot_from_data(variances, "Wariancja wielkości społeczności", "Metoda", "Wariancja", output_dir)
    plot_from_data(medians, "Mediana wielkości społeczności", "Metoda", "Mediana", output_dir)
    plot_from_data(modes, "Dominanta wielkości społeczności", "Metoda", "Dominanta", output_dir)

def plot_from_data(data_dict, title, xlabel, ylabel, output_dir="../output/plots"):
    '''
    Plot and save data from dictionary using matplotlib.
    Parameters: 
        data_dict (dict): dictionary where keys are x-axis labels, values are y-axis values
        title (str): title of the plot
        xlabel (str): label of x-axis
        ylabel (str): label of y-axis
        output_dir (str): directory to save plot
    Returns:
        None
    '''
    
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

def plot_single_community(graph, clusters, output_dir="../output"):
    '''
    Plot single community for each clustering algorithm using matplotlib.
    Community is chosen as the one within a standard deviation of the mean size.
    Parameters:
        graph (nx.Graph): graph to analyze
        clusters (dict): dictionary where keys are method names, values are partition results.
    Returns:
        None
    '''
    for method, cluster in clusters.items():
        community = get_moderate_community(cluster)
        if community is None:
            print(f"No moderate community found for method {method}")
            continue
        print(f"Moderate community of size {len(community)} found for method {method}")
        subgraph = graph.subgraph(community)
        node_size = [3 * (1 + np.log(subgraph.degree[n])) for n in subgraph.nodes]
        pos = nx.kamada_kawai_layout(subgraph)
        print("Drawing subgraph")
        plt.figure(figsize=(15, 15))
        nx.draw(subgraph, pos, with_labels=False, node_size=node_size, width=0.3)
        
        plt.title(f"Przykładowa społeczność {method}")
        plt.savefig(f"{output_dir}/{method}/single_community.png")
        plt.close()
        print("Subgraph drawn and saved")

def plot_clusters_categories(graph, clusters, db_path, output_dir ="../output/plots/"):
    '''
    Finds average cluster for each method and random set of nodes
    plots distribution of categories with use of plot_data_distro()
    Parameters:
        graph (nx.Graph): analysed graph
        clusters (dict): dictionary containing clusters as values and name of method as key
        output_dir (str): dictionary to store plots
    Returns:
        None
    '''
    size = 0
    iterations=0
    for method, cluster in clusters.items():
        iterations+=1
        communities = {c: [k for k, v in cluster.items() if v == c] for c in set(cluster.values())}
        sizes = [len(community) for community in communities.values()]
        mean = np.mean(sizes)
        std_dev = np.std(sizes)
        moderate = get_moderate_community(cluster, std_dev+mean, mean+5*std_dev)
        if not moderate: 
            print("no moderate :(")
            moderate = random.choice(communities)
        size += len(moderate)
        title = "Rozkład kategorii w społeczności " + method
        plot_data_distro(moderate, 'categories', 'Kategoria', 'Liczba produktów', title, db_path, output_dir)
    size //= iterations

    random_nodes = random.sample(list(graph.nodes), size)
    title = "Rozkład w losowej społeczności"
    plot_data_distro(random_nodes, 'categories', 'Kategoria', 'Liczba produktów', title, db_path, output_dir)

def plot_data_distro(community, data, xlab, ylab, title, db_path = "../data/metadata.db", output_dir = "../output/plots/"):
    '''
    Plots distribution of given metadata in given cluster
    Parameters:
        community (list): list of nodes IDs 
        data (str): data to fetch from metadata
        xlab (str): label for x axis
        ylab (str): label for y axis
        title (str): title for plot
        output_dir (str): directory to save plot
    Returns:
        None
    '''
    metadata = [
        database.get_metadata(node_id, db_path).get(data)
        for node_id in community
    ]
    flatten = list(chain.from_iterable(metadata))
    category_counts = Counter(flatten)
    category_counts.pop('Books', None)

    plt.bar(category_counts.keys(), category_counts.values(), color='skyblue')
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.tight_layout()
    plot_path = output_dir + title.replace(" ", "_")
    plt.gca().set_xticklabels([])
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot in {plot_path}")
    

