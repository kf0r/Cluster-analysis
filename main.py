import networkx as nx
from data_processing import load_graph, process_reviews, create_bipartite_graph, filter_bipart_graph, generate_product_projection, save_graph
from clustering import apply_clustering_algorithms
from plotting import plot_community_sizes_distro, plot_statistics_community_sizes
from utility import find_dense, find_largest, save_communities

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
        B = filter_bipart_graph(B)

        batch_graph = generate_product_projection(B)
        review_graph = nx.compose(review_graph, batch_graph)
        print(f"Processed batch of {len(batch_reviews)} reviews.")
        print(f"Graph size: {len(review_graph.nodes)} nodes, {len(review_graph.edges)} edges.")
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
    plot_statistics_community_sizes(review_graph, clusters)