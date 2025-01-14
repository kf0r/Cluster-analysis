import networkx as nx
from data_processing import load_graph, process_reviews, create_bipartite_graph, filter_bipart_graph, generate_product_projection, save_graph
from clustering import apply_clustering_algorithms
from plotting import plot_community_sizes_distro, plot_statistics_community_sizes, plot_single_community, plot_components_sizes_distro, plot_degree_distro, plot_clusters_categories
from utility import find_dense, find_largest, save_communities, find_random, save_central_nodes, save_basic_stats

if __name__ == "__main__":
    input_path = '../data/books.json'
    db_path = '../data/metadata.db'
    graph_filename = '../data/review_graph.pkl'
    try:
        review_graph = load_graph(graph_filename)
        print("Graph loaded from file")
    except FileNotFoundError:
        review_graph = nx.Graph()
        batch_reviews = process_reviews(input_path)
        B = create_bipartite_graph(batch_reviews)
        print(f"Bipart graph size before filtering: {len(B.edges)}")
        B = filter_bipart_graph(B)
        print(f"Bipart graph size after filtering: {len(B.edges)}")

        review_graph = generate_product_projection(B)
        
        save_graph(review_graph, graph_filename)
        print("Graph generated and saved to file")

    ##############################
    '''
    BASIC STATISTICS
    '''
    ##############################
    print(f"Graph size: {len(review_graph.nodes)} nodes, {len(review_graph.edges)} edges.")
    save_basic_stats(review_graph)
    plot_components_sizes_distro(review_graph)
    plot_degree_distro(review_graph)
    print("Basic statistics saved")
    
    ##############################
    '''
    CONNECTIVITY
    '''
    ##############################
    largest_component = max(nx.connected_components(review_graph), key=len)
    review_graph = review_graph.subgraph(largest_component)
    print("Graph is connected")

    ##############################
    '''
    CLUSTERING
    '''
    ##############################
    print("Applying clustering algorithms...")
    clusters = apply_clustering_algorithms(review_graph)
    print("Clustering algorithms applied.")

    print("Finding dense communities...")
    dense = find_dense(review_graph, clusters)

    print("Finding communities based on size...")
    largest, smallest, medium = find_largest(clusters)

    print("Getting random communities...")
    randos = find_random(clusters)

    ##############################
    '''
    SAVING COMMUNITIES
    '''
    ##############################
    print("Saving communities...")
    save_communities(largest, db_path, "largest")
    print("Largest communities saved.")
    save_communities(smallest, db_path, "smallest")
    print("Smallest communities saved")
    save_communities(medium, db_path, "medium")
    print("Medium communities saved")
    save_communities(dense, db_path, "dense")
    print("Dense communities saved")
    save_communities(randos, db_path, "random")
    print("Random communities saved")

    ##############################
    '''
    PLOTTING
    '''
    ##############################
    print("Plotting community size distribution...")
    plot_community_sizes_distro(clusters)

    print("Plotting categories distribution")
    plot_clusters_categories(review_graph, clusters, db_path)

    print("Calculating statistics of clusters...")
    plot_statistics_community_sizes(review_graph, clusters)

    print("Plotting single community...")
    plot_single_community(review_graph, clusters)

    ##############################
    '''
    CENTRAL NODES
    '''
    ##############################
    print("Looking for central nodes...")
    save_central_nodes(review_graph, db_path)
