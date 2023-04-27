import networkx as nx

# Create a new directed acyclic graph
graph = nx.DiGraph()
graph.add_nodes_from(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"])
graph.add_edges_from([("A", "B"), ("B", "C"), ("B", "D"), ("C", "E"), ("C", "F"), ("D", "G"), ("D", "H"), ("E", "I"), ("E", "J")])

# Define the reference clusters and their sizes
L = [("C", "F"), ("B", "E", "I"), ("D", "G", "H"), ("A", "J")]
sizes = [2, 3, 3, 2]

# Define the Topic Coherence fidelity function
def fcoherence(S):
    """Returns the Topic Coherence score for a set of topics S"""
    coherence = 0
    for cluster, size in zip(L, sizes):
        max_w = 0
        for s in S:
            w_p = (len(set(nx.descendants(graph, s)).intersection(cluster)) +1)/ (len(nx.descendants(graph, s))+1)
            w_r = len(set(nx.descendants(graph, s)).intersection(cluster)) / size
            w = 2 * w_p * w_r / (w_p + w_r) if w_p + w_r > 0 else 0
            if w > max_w:
                max_w = w
        coherence += max_w
    return coherence

# Define the Topic Specificity QC function
def fspecificity(s, alpha):
    """Returns 1 if the height of topic s is at least alpha, and 0 otherwise"""
    height = nx.shortest_path_length(graph, source="A", target=s)
    if height >= alpha:
        return 1
    else:
        return 0

# Define the combined function
def f(S, w1, w2, alpha):
    """Returns the combined score for a set of topics S"""
    coherence = fcoherence(S)
    specificity = sum([fspecificity(s, alpha) for s in S])
    return w1 * coherence + w2 * specificity

# Example usage
topics = ["C", "E", "G"]  # Set a list of topics
w1 = 0.8  # Set the weight of Topic Coherence
w2 = 0.2  # Set the weight of Topic Specificity
alpha = 3  # Set the value of alpha for Topic Specificity
score = f(topics, w1, w2, alpha)  # Calculate the combined score
print(score)