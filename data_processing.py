import json
import pickle
import networkx as nx
from review import Review
from itertools import combinations

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

def filter_bipart_graph(graph, min_reviews=5):
    initial_product_count = len([node for node in graph.nodes if graph.nodes[node].get("bipartite") == 1])
    initial_user_count = len([node for node in graph.nodes if graph.nodes[node].get("bipartite") == 0])
    
    products_to_remove = [
        node for node in graph.nodes if graph.nodes[node].get("bipartite") == 1 and graph.degree(node) < min_reviews
    ]
    graph.remove_nodes_from(products_to_remove)

    users_to_remove = [
        node for node in graph.nodes if graph.nodes[node].get("bipartite") == 0 and graph.degree(node) < 2
    ]
    graph.remove_nodes_from(users_to_remove)

    final_product_count = len([node for node in graph.nodes if graph.nodes[node].get("bipartite") == 1])
    final_user_count = len([node for node in graph.nodes if graph.nodes[node].get("bipartite") == 0])

    print(f"Initial product count: {initial_product_count}, final product count: {final_product_count}")
    print(f"Initial user count: {initial_user_count}, final user count: {final_user_count}")
    print(f"Removed {len(products_to_remove)} products and {len(users_to_remove)} users.")
    return graph

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

def generate_product_projection(bipartite_graph):
    product_graph = nx.Graph()
    #i=0
    users = [n for n in bipartite_graph.nodes if n.startswith("A")]
    total = len(users)
    for i,user in enumerate(users):
        #print(user)
        rated = list(bipartite_graph.neighbors(user))
        #print(rated)
        for p1, p2 in combinations(rated, 2):
            #i+=1
            if product_graph.has_edge(p1, p2):
                product_graph[p1][p2]['weight'] += 1
            else:
                product_graph.add_edge(p1, p2, weight=1)
        if i%100==0:
            print(f"Projecting bipartiate {i/total*100:.2f}% done")
    return product_graph
