import networkx as nx
import pandas as pd
from itertools import combinations
import operator
from itertools import groupby


def collective_influence_centrality(Graph, weight=None):
    """
    Compute Collective Influence (CI) Centrality per each node (up to distance d=2).

    :param Graph: (Graph obj) Input Graph.
    :param weight : (string) None or string, optional (default=None)
      If None, all edge weights are considered equal.
      Otherwise holds the name of the edge attribute used as weight.
    :return: (dict) Dictionary of nodes with their respective CI Centrality values.
    """
    colinf = dict()
    for node in Graph:
        summatory = 0
        for iter_node in Graph.neighbors(node):
            if weight is None:
                summatory += Graph.degree(iter_node) - 1
            else:
                summatory += Graph.degree(iter_node, weight='weight') - 1
        if weight is None:
            colinf[node] = (Graph.degree(node) - 1) * summatory
        else:
            colinf[node] = (Graph.degree(node, weight='weight') - 1) * summatory
    return colinf

def lcc_size(Graph):
    """
    Compute Largest Connected Component (LCC) in a Graph.
    :param Graph: (Graph obj) Input Graph.
    :return: (int) Size of the LCC.
    """
    lcc_size = 0
    for c in nx.connected_components(Graph):
        if len(c) > lcc_size:
            lcc_size = len(c)
    return lcc_size

def max_centrality_nodes(Graph, centrality_function, tiebreaker_function=None, top_n=5, weight=None):
    top_nodes = []  # Initialize the list to store the top-n nodes

    # Calculate the primary centrality for each node in the graph
    if weight is None:
        primary_centrality = centrality_function(Graph)
        if tiebreaker_function is not None:
            secondary_centrality = tiebreaker_function(Graph)
    else:
        primary_centrality = centrality_function(Graph, weight='weight')
        if tiebreaker_function is not None:
            secondary_centrality = tiebreaker_function(Graph, weight='weight')

    # Sort the nodes based on their primary centrality values in descending order
    sorted_primary_centrality = sorted(primary_centrality.items(), key=operator.itemgetter(1), reverse=True)
    # print(sorted_primary_centrality)
    # print()
    final_sorted_nodes = []  # List to store the nodes in their final order

    if tiebreaker_function is not None:
        # Sort the nodes based on their secondary centrality values
        sorted_secondary_centrality = sorted(secondary_centrality.items(), key=operator.itemgetter(1), reverse=True)
        # print(sorted_secondary_centrality)
        secondary_centrality_dict = {key: value for key, value in sorted_secondary_centrality}

        # Group the nodes with identical primary centrality values
        groups = [dict(g) for k, g in groupby(sorted_primary_centrality, key=lambda x: x[1])]

        for group in groups:
            if len(group) == 1:  # If there's only one node in a group, add it to final_sorted_nodes
                final_sorted_nodes.extend(list(group.keys()))
            else:  # If there are multiple nodes, sort them based on their secondary centrality and add them to final_sorted_nodes
                group = [k for k in sorted(group, key=lambda k: list(secondary_centrality_dict.keys()).index(k))]
                final_sorted_nodes.extend(group)
    else:
        final_sorted_nodes = [t[0] for t in sorted_primary_centrality]

    # Add the first top_n nodes from the final_sorted_nodes list to top_nodes
    for i in range(0, top_n):
        top_nodes.append(final_sorted_nodes[i])

    return top_nodes  # Return the list containing the top-n nodes based on the primary centrality function (and tiebreaker_function, if provided)

# using APs 
def sorted_aps(G, tiebreaker_function=None):
    # Get a list of articulation points
    aps = list(nx.articulation_points(G))
    
    # Create a list to store the articulation points and their LCC sizes after removal
    aps_lcc = []

    # If a tie-breaker function is provided, calculate the tie-breaker metric for each node
    if tiebreaker_function is not None:
        tiebreaker_metric = tiebreaker_function(G)

    for ap in aps:
        # Create a copy of the graph and remove the current articulation point
        subgraph_nodes = [node for node in G.nodes if node != ap]
        G_subgraph = G.subgraph(subgraph_nodes)

        # Get the size of the largest connected component after the removal
        lcc_size_current = lcc_size(G_subgraph)

        # If a tie-breaker function is provided, store the articulation point, its LCC size and its tie-breaker metric
        # Otherwise, only store the articulation point and its LCC size
        if tiebreaker_function is not None:
            aps_lcc.append((ap, lcc_size_current, tiebreaker_metric[ap]))
        else:
            aps_lcc.append((ap, lcc_size_current))

    # If a tie-breaker function is provided, sort by LCC size in ascending order and then by the tie-breaker metric in descending order for ties
    # Otherwise, just sort by LCC size in ascending order
    if tiebreaker_function is not None:
        sorted_aps_lcc = sorted(aps_lcc, key=lambda x: (x[1], -x[2]))
    else:
        sorted_aps_lcc = sorted(aps_lcc, key=itemgetter(1))

    return sorted_aps_lcc

# when top_n is bigger than the number of aps, using tiebreadker_function to complete the top_n nodes
def top_aps(G, tiebreaker_function=None, top_n=1):
    # Get the ranked articulation points
    sorted_aps_lcc = sorted_aps(G, tiebreaker_function=tiebreaker_function)

    # Collect the top 'n' articulation points
    top_nodes = [ap for ap in (t[0] for t in sorted_aps_lcc)]

    # If top_n is bigger than the number of articulation points and no tiebreaker function is provided, raise an error
    if top_n > len(top_nodes) and tiebreaker_function is None:
        raise ValueError(f"The number of top nodes requested ({top_n}) is greater than the number of articulation points ({len(top_nodes)}), and no tiebreaker function was provided.")

    # If top_n is bigger than the number of articulation points, fill the top_nodes using tiebreaker_function
    if top_n > len(top_nodes):
        # Calculate the tiebreaker metric for all nodes in the graph
        all_nodes_metric = tiebreaker_function(G)

        # Remove the articulation points from the all_nodes_metric dict
        for ap in top_nodes:
            all_nodes_metric.pop(ap, None)

        # Sort the remaining nodes by the tiebreaker metric in descending order
        sorted_remaining_nodes = sorted(all_nodes_metric.items(), key=lambda x: x[1], reverse=True)

        # Append nodes to top_nodes until its length is top_n
        for node, _ in sorted_remaining_nodes:
            if len(top_nodes) >= top_n:
                break
            top_nodes.append(node)

    return top_nodes[:top_n]

# Using CoreHD
def core_degrees(G):
    # Find the 2-core of the network
    core = nx.k_core(G, k=2)

    # Get the degree of every node within this 2-core
    degrees = dict(core.degree())

    return degrees

def sorted_core_nodes(G, tiebreaker_function=None):
    # Get the degrees of the nodes in the 2-core
    core_degrees_dict = core_degrees(G)
    
    # If a tie-breaker function is provided, calculate the tie-breaker metric for each node
    if tiebreaker_function is not None:
        tiebreaker_metric = tiebreaker_function(G)
    
    # Create a list to store the nodes and their degrees
    core_nodes = []
    
    for node, degree in core_degrees_dict.items():
        # If a tie-breaker function is provided, store the node, its degree, and its tie-breaker metric
        # Otherwise, only store the node and its degree
        if tiebreaker_function is not None:
            core_nodes.append((node, degree, tiebreaker_metric[node]))
        else:
            core_nodes.append((node, degree))
    
    # If a tie-breaker function is provided, sort by degree in descending order and then by the tie-breaker metric in descending order for ties
    # Otherwise, just sort by degree in descending order
    if tiebreaker_function is not None:
        sorted_core_nodes = sorted(core_nodes, key=lambda x: (-x[1], -x[2]))
    else:
        sorted_core_nodes = sorted(core_nodes, key=lambda x: -x[1])

    return sorted_core_nodes

def top_core_nodes(G, tiebreaker_function=None, top_n=1):
    # Get the ranked core nodes
    sorted_nodes = sorted_core_nodes(G, tiebreaker_function=tiebreaker_function)

    # Collect the top 'n' core nodes
    top_nodes = [node for node in (t[0] for t in sorted_nodes)]

    # If top_n is bigger than the number of core nodes and no tiebreaker function is provided, raise an error
    if top_n > len(top_nodes) and tiebreaker_function is None:
        raise ValueError(f"The number of top nodes requested ({top_n}) is greater than the number of nodes in the 2-core ({len(top_nodes)}), and no tiebreaker function was provided.")

    # If top_n is bigger than the number of core nodes, fill the top_nodes using tiebreaker_function
    if top_n > len(top_nodes):
        # print("No Core existed, using DEG!")
        # Calculate the tiebreaker metric for all nodes in the graph
        all_nodes_metric = tiebreaker_function(G)

        # Remove the core nodes from the all_nodes_metric dict
        for node in top_nodes:
            all_nodes_metric.pop(node, None)

        # Sort the remaining nodes by the tiebreaker metric in descending order
        sorted_remaining_nodes = sorted(all_nodes_metric.items(), key=lambda x: x[1], reverse=True)

        # Append nodes to top_nodes until its length is top_n
        for node, _ in sorted_remaining_nodes:
            if len(top_nodes) >= top_n:
                break
            top_nodes.append(node)

    return top_nodes[:top_n]

def degree_centrality(Graph, weight=None):
    """Compute the degree centrality for nodes. From NetworkX, but adapted for weighted graphs.
    ----------
    Graph : graph
      A networkx graph
    weight : None or string, optional (default=None)
      If None, all edge weights are considered equal.
      Otherwise holds the name of the edge attribute used as weight.
    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with degree centrality as the value.
    """
    if Graph.number_of_edges() == 0:
            return {node: 1 for node in Graph}
        
    if weight is None:
        centrality = {node: d / (len(Graph) - 1.0) for node, d in Graph.degree()}
    else:
        degrees_dict = dict(nx.degree(Graph, weight='weight'))
        centrality = {node: d / max(degrees_dict.values()) for node, d in degrees_dict.items()}
    return centrality

# # GRD
def GRD(G, n=1):
    connected_components = sorted(nx.connected_components(G), key=len, reverse=True)
    if not connected_components:
        return None

    top_n_components_nodes = [node for component in connected_components[:n] for node in component]
    min_lcc_size = len(connected_components[0])
    nodes_to_remove = None

    for node_tuple in combinations(top_n_components_nodes, n):
        subgraph_nodes = [n for n in top_n_components_nodes if n not in node_tuple]
        subgraph = G.subgraph(subgraph_nodes)
        lcc_size_current = lcc_size(subgraph)
        if lcc_size_current <= min_lcc_size:
            min_lcc_size = lcc_size_current
            nodes_to_remove = node_tuple
                
    
    return list(nodes_to_remove)

# SF-GRD
def SF_GRD(G, n=1):
    connected_components = sorted(nx.connected_components(G), key=len, reverse=True)
    if not connected_components:
        return None

    # Create a subgraph with nodes from the top n components
    top_n_components_nodes = [node for component in connected_components[:n] for node in component]
    G_subgraph = G.subgraph(top_n_components_nodes)
    
    # Calculate properties for nodes in the subgraph
    betweenness_centrality = nx.betweenness_centrality(G_subgraph)
    degree_centrality = nx.degree_centrality(G_subgraph)
    articulation_points = list(nx.articulation_points(G_subgraph))

    # Select top 5 nodes by betweenness and degree centrality and all articulation points
    top_betweenness_nodes = sorted(betweenness_centrality, key=betweenness_centrality.get, reverse=True)[:5]
    top_degree_nodes = sorted(degree_centrality, key=degree_centrality.get, reverse=True)[:5]
    top_aps = sorted(articulation_points, key=lambda x: G_subgraph.degree(x), reverse=True)[:5]

    search_nodes = set(top_betweenness_nodes + top_degree_nodes + top_aps)

    min_lcc_size = lcc_size(G_subgraph)
    nodes_to_remove = None

    for node_tuple in combinations(search_nodes, n):
        subgraph_nodes = [node for node in top_n_components_nodes if node not in node_tuple]
        subgraph = G.subgraph(subgraph_nodes)
        lcc_size_current = lcc_size(subgraph)
        if lcc_size_current <= min_lcc_size:
            min_lcc_size = lcc_size_current
            nodes_to_remove = list(node_tuple)

    return list(nodes_to_remove)

def select_nodes_to_remove(graph, main_centr, second_centr=None, block_size=1, weight=None):
    if main_centr == 'APs':
        return top_aps(graph, tiebreaker_function=second_centr, top_n=block_size)
    elif main_centr == 'CoreHD':
        return top_core_nodes(graph, tiebreaker_function=second_centr, top_n=block_size)
    elif main_centr == 'GRD':
        return GRD(graph, n=block_size)
    elif main_centr == 'SF-GRD':
        return SF_GRD(graph, n=block_size)
    else:
        return max_centrality_nodes(graph, centrality_function=main_centr, tiebreaker_function=second_centr, top_n=block_size, weight=weight)

# including Direct Optimisation, AP, and Centrality measure
def disruption(Graph, main_centr, second_centr=None, block_size=1, within_LCC=False, weight=None, percentage=0.2):
    Graph = Graph.copy()
    N = Graph.number_of_nodes()  # Total number of nodes
    target_nodes_to_remove = int(N * percentage)
    
    graph_snapshots = {'Graphs': [], 'Nodes': []}
    lcc_sizes = dict()  
    kiter = 0
    lcc_sizes[kiter] = lcc_size(Graph)
    nodes_removed = 0
    while nodes_removed < target_nodes_to_remove and Graph.number_of_nodes() >= block_size:
        
        if within_LCC:
            largest_cc = max(nx.connected_components(Graph), key=len)
    
            if len(largest_cc) < block_size:
                subgraph = Graph
            else:    
                subgraph = Graph.subgraph(largest_cc)
        else:
            subgraph = Graph

        toremove = select_nodes_to_remove(subgraph, main_centr, second_centr, block_size, weight)
        
#         if not toremove:
#             break
        graph_snapshots['Graphs'].append(Graph.copy())
        graph_snapshots['Nodes'].append(toremove)
        
        Graph.remove_nodes_from(toremove)
        nodes_removed += len(toremove)
            
        kiter += block_size
        current_lcc_size = lcc_size(Graph)
        lcc_sizes[kiter] = current_lcc_size
    
    R = (sum(lcc_sizes.values()) - lcc_sizes[0]) / (N * (len(lcc_sizes) - 1))

    return R, lcc_sizes, graph_snapshots

# Store the properties of nodes found by direct optimisation method
def GRD_analysis(G, n=1):
    # 1. Calculate properties for all nodes
    betweenness_centrality = nx.betweenness_centrality(G)
    degree_centrality = nx.degree_centrality(G)
    articulation_points = list(nx.articulation_points(G))

    # Sort nodes by properties to get rankings
    sorted_by_betweenness = sorted(betweenness_centrality, key=betweenness_centrality.get, reverse=True)
    sorted_by_degree = sorted(degree_centrality, key=degree_centrality.get, reverse=True)

    connected_components = sorted(nx.connected_components(G), key=len, reverse=True)
    if not connected_components:
        return None, None

    top_n_components_nodes = [node for component in connected_components[:n] for node in component]
    min_lcc_size = len(connected_components[0])
    nodes_to_remove = None
    node_properties = None

    if n == 1:
        for node in top_n_components_nodes:
            subgraph_nodes = [n for n in top_n_components_nodes if n != node]
            subgraph = G.subgraph(subgraph_nodes)
            lcc_size_current = lcc_size(subgraph)
            if lcc_size_current < min_lcc_size:
                min_lcc_size = lcc_size_current
                nodes_to_remove = [node]
                if nodes_to_remove is not None:  # Add the condition here
                    node_properties = {
                        'betweenness_rank': sorted_by_betweenness.index(node) + 1,
                        'degree_rank': sorted_by_degree.index(node) + 1,
                        'is_ap': node in articulation_points,
                    }
    else:
        for node_tuple in combinations(top_n_components_nodes, n):
            subgraph_nodes = [n for n in top_n_components_nodes if n not in node_tuple]
            subgraph = G.subgraph(subgraph_nodes)
            lcc_size_current = lcc_size(subgraph)
            if lcc_size_current < min_lcc_size:
                min_lcc_size = lcc_size_current
                nodes_to_remove = list(node_tuple)
                if nodes_to_remove is not None:  # Add the condition here
                    node_properties = [{
                        'node': node,  
                        'betweenness_rank': sorted_by_betweenness.index(node) + 1,
                        'degree_rank': sorted_by_degree.index(node) + 1,
                        'is_ap': node in articulation_points,
                    } for node in node_tuple]
                
    return nodes_to_remove, node_properties

# add parameter percentage (remove a specific number of nodes according to the given percentage)
def disruption_GRD_analysis(Graph, main_centr, second_centr=None, block_size=1, within_LCC=False, weight=None, percentage=0.2):
    Graph = Graph.copy()
    N = Graph.number_of_nodes()  # Total number of nodes
    target_nodes_to_remove = int(N * percentage)

    graph_snapshots = {'Graphs': [], 'Nodes': []}
    directly_removed_nodes_properties = {}  # Separate dict to store properties of directly removed nodes
    lcc_sizes = dict()
    kiter = 0
    lcc_sizes[kiter] = lcc_size(Graph)

    nodes_removed = 0
    while nodes_removed < target_nodes_to_remove and Graph.number_of_nodes() >= block_size:
        if within_LCC:
            largest_cc = max(nx.connected_components(Graph), key=len)
            if len(largest_cc) < block_size:
                break
            subgraph = Graph.subgraph(largest_cc).copy()
        else:
            subgraph = Graph

        properties = None
        if main_centr == 'GRD-Analysis':
            toremove, properties = GRD_analysis(subgraph, n=block_size)
            if properties is not None:
                key = tuple(toremove) if isinstance(toremove, list) else toremove
                directly_removed_nodes_properties[key] = properties

        graph_snapshots['Graphs'].append(Graph.copy())
        graph_snapshots['Nodes'].append(toremove)

        if not toremove:
            break

        Graph.remove_nodes_from(toremove)
        nodes_removed += len(toremove)
        
        kiter += block_size
        current_lcc_size = lcc_size(Graph)
        lcc_sizes[kiter] = current_lcc_size

    R = (sum(lcc_sizes.values()) - lcc_sizes[0]) / (N * (len(lcc_sizes) - 1))

    return R, lcc_sizes, graph_snapshots, directly_removed_nodes_properties

def centrality_disruption_analysis(graph, centrality_measures, include_within_LCC=True, block_size=1, percentage=0.2):
    df_lcc_all = pd.DataFrame()
    graph_snapshots = {}
    R_values = {}  # Dictionary to store R values for each centrality measure

    for name, function in centrality_measures.items():
        R, lcc_sizes, dict_graphs_nodes = disruption(graph, main_centr=function[0], second_centr=function[1], block_size=block_size, within_LCC=False, weight=None, percentage=percentage)

        print(f"The value of R for {name} is {R}")
        R_values[name] = R  # Store R value in the dictionary

        # Store the dictionaries with a key corresponding to the centrality measure name
        graph_snapshots[name] = dict_graphs_nodes

        df_lcc_all[name] = pd.Series(lcc_sizes)
        df_lcc_all.index.name = 'Iteration (' + 'block size: ' + str(block_size) + ')'

        if include_within_LCC:
            R_within, lcc_sizes_within, dict_graphs_nodes_within = disruption(graph, main_centr=function[0], second_centr=function[1], block_size=block_size, within_LCC=True, weight=None, percentage=percentage)

            print(f"The value of R for {name} within LCC is {R_within}")
            R_values[name + " within LCC"] = R_within  # Store R value in the dictionary

            # Store the dictionaries with a key corresponding to the centrality measure name and a suffix
            graph_snapshots[name + " within LCC"] = dict_graphs_nodes_within

            df_lcc_all[name + " within LCC"] = pd.Series(lcc_sizes_within)

    return R_values, df_lcc_all, graph_snapshots

