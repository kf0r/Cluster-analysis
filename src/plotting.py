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
        #print(f"Method: {method}")
        #print(sizes)
        plt.hist(sizes, len(set(sizes)))
        plt.yscale('log')
        plt.title(f"Rozkład wielkości klastrów metody ({method})")
        plt.xlabel("Wielkość klastra")
        plt.ylabel("Ilość klastrów")
        plt.grid(True)
        plot_filename = output_dir+ "/" + method + "/community_sizes_distro.png"
        plt.savefig(plot_filename)
        #plt.show()
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