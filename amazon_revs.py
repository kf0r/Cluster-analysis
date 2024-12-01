import pandas as pd
import networkx as nx
from community import community_louvain
from cdlib import algorithms
import matplotlib.pyplot as plt
from textblob import TextBlob
from itertools import combinations
import time
from review import Review

import networkx as nx

def process_reviews(input_path, error_log="error_lines.txt"):
    reviews = []
    rev = 0

    with open(input_path, 'r', encoding='utf-8', errors='replace') as infile, open(error_log, 'w', encoding='utf-8') as errorfile:
        current_entry = {}
        line_num = 0
        for line in infile:
            exception = False
            line_num += 1
            line = line.strip()
            try:
                if line.startswith("review/time"):
                    current_entry["date"] = line.split(":", 1)[1].strip()
                elif line.startswith("review/userId"):
                    current_entry["user_id"] = line.split(":", 1)[1].strip()
                elif line.startswith("product/productId"):
                    current_entry["product_id"] = line.split(":", 1)[1].strip()
                elif line.startswith("review/score"):
                    current_entry["score"] = line.split(":", 1)[1].strip()
                elif line.startswith("review/text"):
                    current_entry["text"] = line.split(":", 1)[1].strip()
                elif not line: 
                    rev += 1
                    if not exception:
                        review = Review(
                            user_id=current_entry["user_id"],
                            product_id=current_entry["product_id"],
                            date=current_entry["date"],
                            score=current_entry["score"],
                            text=current_entry["text"]
                        )
                        reviews.append(review)
                       
                    exception = False
                    current_entry = {} 
                    if rev % 10000 == 0:
                        print(f"Processed {rev} reviews ({line_num} lines).")
            except Exception as e:
                errorfile.write(f"Exception in review {rev}, line {line_num}: {line}\n")
                exception = True
                continue

    return reviews

def generate_bipartiate(revs):
    B = nx.Graph()
    for rev in revs:
        B.add_edge(
            rev.user_id,
            rev.product_id,
            date=rev.date,
            score=rev.score,
            sentiment=rev.sentiment
        )
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
    for user in [n for n in bipartite_graph.nodes if n.startswith("A")]:
        rated_products = [neighbor for neighbor in bipartite_graph.neighbors(user) if neighbor.startswith("P")]
        for p1, p2 in combinations(rated_products, 2):
            if product_graph.has_edge(p1, p2):
                product_graph[p1][p2]['weight'] += 1
            else:
                product_graph.add_edge(p1, p2, weight=1)
    return product_graph


########################################################################################################################


file_path = "../data/faulty.txt"

reviews = process_reviews(file_path)

bipartite_graph = generate_bipartiate(reviews)

df = pd.DataFrame([rev.__dict__ for rev in reviews])

temporal_analysis(df)

print("projecting")
user_graph = nx.projected_graph(bipartite_graph, [n for n in bipartite_graph.nodes if n.startswith("A")])
product_graph = generate_product_projection(bipartite_graph)
print("projected, now analysing")



print("analized")
