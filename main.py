import networkx as nx
from data_processing import load_graph, process_reviews, create_bipartite_graph, filter_bipart_graph, generate_product_projection, save_graph
from clustering import apply_clustering_algorithms
from plotting import plot_community_sizes_distro, plot_statistics_community_sizes, plot_single_community
from utility import find_dense, find_largest, save_communities, find_random, save_central_nodes, compare_centralities

if __name__ == "__main__":
    input_path = 'books.json'
    db_path = 'metadata.db'
    graph_filename = 'review_graph.pkl'
    try:
        review_graph = load_graph(graph_filename)
        print("Graph loaded from file")
    except FileNotFoundError:
        review_graph = nx.Graph()
        batch_reviews = process_reviews(input_path)
        B = create_bipartite_graph(batch_reviews)
        B = filter_bipart_graph(B)

        review_graph = generate_product_projection(B)
        #review_graph = nx.compose(review_graph, batch_graph)
        #print(f"Processed batch of {len(batch_reviews)} reviews.")
        
        save_graph(review_graph, graph_filename)
        print("Graph generated and saved to file")

    print(f"Graph size: {len(review_graph.nodes)} nodes, {len(review_graph.edges)} edges.")
    print("Applying clustering algorithms...")
    clusters = apply_clustering_algorithms(review_graph)
    print("Clustering algorithms applied.")

    print("Finding dense communities...")
    dense = find_dense(review_graph, clusters)

    print("Finding communities based on size...")
    largest, smallest, medium = find_largest(clusters)

    print("Getting random communities...")
    randos = find_random(clusters)

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

    print("Plotting community size distribution...")
    plot_community_sizes_distro(clusters)

    print("Mean community size graph...")
    print("Variance community size graph...")
    print("Standard deviation community size graph...")
    print("Calculating modularity...")
    plot_statistics_community_sizes(review_graph, clusters)

    print("Plotting single community...")
    #plot_single_community(review_graph, clusters)

    print("Looking for central nodes...")
    save_central_nodes(review_graph, db_path)
