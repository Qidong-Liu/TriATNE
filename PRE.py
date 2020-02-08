import numpy as np
import networkx as nx

# Perform train-test split
    # Takes in adjacency matrix in sparse format
    # Returns: adj_train, train_edges, val_edges, val_edges_false, 
        # test_edges, test_edges_false
def mask_test_edges(g, test_frac=.1, prevent_disconnect=True, verbose=False):
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    if verbose == True:
        print('preprocessing...')

    orig_num_cc = nx.number_connected_components(g.to_undirected())

    edges = g.edges()
    num_test = int(np.floor(len(edges) * test_frac))

    edge_tuples = [(edge[0], edge[1], g[edge[0]][edge[1]]['weight']) for edge in edges]
    all_edge_tuples = set(edge_tuples)

    train_edges = set(edge_tuples) # initialize train_edges to have all edges
    test_edges = set()

    if verbose == True:
        print('generating test/val sets...')

    # Iterate over shuffled edges, add to train/val sets
    np.random.shuffle(edge_tuples)
    for edge in edge_tuples:
        # print edge
        node1 = edge[0]
        node2 = edge[1]
        # If removing edge would disconnect a connected component, backtrack and move on
        ww = g[node1][node2]['weight']
        g.remove_edge(node1, node2)
        if prevent_disconnect == True:
            if nx.number_connected_components(g.to_undirected()) > orig_num_cc:
                g.add_edge(node1, node2, weight = ww)
                continue

        # Fill test_edges first
        if len(test_edges) < num_test:
            test_edges.add(edge)
            train_edges.remove(edge)

        # Both edge lists full --> break loop
        elif len(test_edges) == num_test:
            break

    if (len(test_edges) < num_test):
        print("WARNING: not enough removable edges to perform full train-test split!")
        print("Num. test edges requested: (", num_test, ")")
        print("Num. test edges returned: (", len(test_edges), ")")

    if prevent_disconnect == True:
        assert nx.number_connected_components(g.to_undirected()) == orig_num_cc

    if verbose == True:
        print('creating false test edges...')

    test_edges_false = set()
    while len(test_edges_false) < num_test:
        idx_i, idx_j = np.random.choice(g.nodes(), size=2, replace=False)
        false_edge = (idx_i, idx_j, 0)

        # Make sure false_edge not an actual edge, and not a repeat
        if false_edge in all_edge_tuples:
            continue
        if false_edge in test_edges_false:
            continue

        test_edges_false.add(false_edge)

    if verbose == True:
        print('creating false train edges...')

    train_edges_false = set()
    while len(train_edges_false) < len(train_edges):
        idx_i, idx_j = np.random.choice(g.nodes(), size=2, replace=False)
        false_edge = (idx_i, idx_j, 0)

        # Make sure false_edge in not an actual edge, not in test_edges_false, 
            # not in val_edges_false, not a repeat
        if false_edge in all_edge_tuples or \
            false_edge in test_edges_false or \
            false_edge in train_edges_false:
            continue

        train_edges_false.add(false_edge)

    if verbose == True:
        print('final checks for disjointness...')

    # assert: false_edges are actually false (not in all_edge_tuples)
    assert test_edges_false.isdisjoint(all_edge_tuples)
    assert train_edges_false.isdisjoint(all_edge_tuples)

    # assert: test, val, train false edges disjoint
    assert test_edges_false.isdisjoint(train_edges_false)

    # assert: test, val, train positive edges disjoint
    assert test_edges.isdisjoint(train_edges)

    if verbose == True:
        print('creating adj_train...')

    # Convert edge-lists to numpy arrays
    train_edges = np.array([list(edge_tuple) for edge_tuple in train_edges])
    train_edges_false = np.array([list(edge_tuple) for edge_tuple in train_edges_false])
    test_edges = np.array([list(edge_tuple) for edge_tuple in test_edges])
    test_edges_false = np.array([list(edge_tuple) for edge_tuple in test_edges_false])

    if verbose == True:
        print('Done with train-test split!')
        print('')

    return  train_edges, train_edges_false, test_edges, test_edges_false
