import networkx as nx
import numpy as np
from PRE import mask_test_edges

def load_edgelist(file, undirected=True):
    train = {}
    nodes = set()
    with open(file) as fin:
        for line in fin:
            cols = line.strip().split()
            node = cols[0]
            context = cols[1]
            if node in train:
                train[node].append(context)
            else:
                train[node] = [context]

            if undirected:
                if context in train:
                    train[context].append(node)
                else:
                    train[context] = [node]

            nodes.add(node)
            nodes.add(context)

    return train, nodes

def split_data(file_in, file_out1, file_out2, file_out3, file_out4, frac=0.1):
    G = nx.read_edgelist(file_in, nodetype=int, create_using=nx.DiGraph())
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1

    train_edges, train_edges_false, test_edges, test_edges_false = mask_test_edges(G, test_frac=frac, prevent_disconnect=True, verbose=True)
    np.savetxt(file_out1, train_edges, fmt='%d\t%d\t%d')
    np.savetxt(file_out2, train_edges_false, fmt='%d\t%d\t%d')
    np.savetxt(file_out3, test_edges, fmt='%d\t%d\t%d')
    np.savetxt(file_out4, test_edges_false, fmt='%d\t%d\t%d')

# Get batch data from training set
def get_batch(data, index, size):
    return data[index:index+size, 0], data[index:index+size, 1], data[index:index+size, 2]

def write_to_file(sess, model, file_out, FEATURE_SIZE, node_list, tag=1):
    print("write to file")
    with open(file_out, 'w') as file:
        if tag == 1:
            emb = sess.run(model.u_embedding, feed_dict={model.u: node_list})
        else:
            emb = sess.run(model.v_embedding, feed_dict={model.v: node_list})

        file.write('{} {}'.format(str(len(node_list)), str(FEATURE_SIZE)))
        for i, node in enumerate(node_list):
            line = ' '.join(map(str, emb[i]))
            line = '\n' + str(node) + ' ' + line
            file.writelines(line)
    file.close()


