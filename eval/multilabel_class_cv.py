
import networkx as nx
import settings as st

from eval import functions

def Cross_val(emb_file, cv=5, comments='#'):
    if st.format not in ('cora', 'blogcat'):
        exit('error: only label formats "cora" and "blogcat" are supported')

    print('reading {}...'.format(st.TRAIN_INPUT_FILENAME))
    orig_graph = nx.read_edgelist(st.TRAIN_INPUT_FILENAME, nodetype=str, data=(('weight', int),), create_using=nx.DiGraph(), comments=comments, delimiter='\t')

    print('reading {}...'.format(st.TRAIN_LABEL_FILENAME))
    if st.format == 'cora':
        labels, label_list = functions.read_cora_labels(st.TRAIN_LABEL_FILENAME)
    else:
        labels, label_list = functions.read_blogcat_labels(st.TRAIN_LABEL_FILENAME, delimiter='\t')
    assert len(labels) == orig_graph.number_of_nodes()

    print('reading {}...'.format(emb_file))
    emb = functions.read_w2v_emb(emb_file, False)

    f1_micro, f1_macro = functions.get_f1_cross_val(labels, label_list, cv, emb)
    return f1_micro, f1_macro
