# TriGAN
 Source code for KDD 2020 paper "TriGAN: Tripartite Adversarial Learning for Graph Embeddings"
## Requirements
The code has been tested under Python 3.7.5, with the following packages installed (along with their dependencies):
- tensorflow == 1.14.0
- gensim == 3.8.0
- networkx == 2.4
- numpy = 1.17.4
- scikit-learn == 0.21.3
- scipy == 1.3.1
- tqdm = 4.39.0 (for displaying the progress bar)
## Files in the folder
- **\data:** Store the example inputs.
- **\emb:** Store the learned embeddings.
- **\eval:** Store the evaluation scripts for different metric.
- **\model:** Store the trained models.
- **model.py:** Define each modules of the model.
- **PRE.py:** Train-test split for link prediction.
- **settings.py:** Defines different hyperparameters here.
- **TriGAN.py:** The main entrance of running.
- **utils.py:** Codes to load data, split data, prepare batches for training and write embeddings to files.
## Data

(1) TriGAN expect an edgelist for the input network, i.e.,
>node1   node2

>node1   node3

>...

Each row contains two IDs, which are separated by '\t'

(2) Two formats are supported for the input labels. By setting format = 'cora' , we expect a node with multi-labels:
>/label1/label2/label3/

>/label1/label5/

>...

The node ID defaults to line number and starts from 0 to N−1 (N is the number of nodes in the graph)

By setting format = 'blogcat', we expect a node with single label:
>node1   label1

>node2   label1

>...

>nodeN   label5

The first column is the node’s ID. The second column is the label of corresponding node. They are separated by'\t'.

(3) The output embeddings are as follows:
>N d

>node1 f1 f2 ... fd

>node2 f1 f2 ... fd

>...

>nodeN f1 f2 ... fd

where N denotes the number of nodes and d denotes the embeddings' dimension. Each subsequent row represents a node.

## Basic usage
### Classification
We take the Citeseer dataset as an exampleand show how to perform classification tasks for TriGAN:

(1) Modify the following parameters in settings.py:
>TRAIN_INPUT_FILENAME = '/data/Citeseer_input.txt'

>TRAIN_LABEL_FILENAME ='/data/Citeseer_label.txt'

>format ='blogcat'APP = 0

(2) Run the code by:

```python TriGAN.py```

(3) The learned embeddings will be stored in /emb/, and the evaluation results (micro F1 and macro F1) for embeddings will be printed on the screen for each epoch.

### Link Prediction
We take the wiki-vote dataset as an exampleand show how to perform link prediction tasks for TriGAN:

(1) If you have split the dataset, please move directly to step 2. Otherwise, modify the following parameters in settings.py:
>APP = 1

>split = True

>test_frac = 0.2
 
>FULL_FILENAME = '/data/wiki-vote.txt'

>TRAIN_POS_FILENAME = '/data/wiki_train_pos.txt'

>TRAIN_NEG_FILENAME = '/data/wiki_train_neg.txt'

>TEST_POS_FILENAME = '/data/wiki_test_pos.txt'

>TEST_NEG_FILENAME = '/data/wiki_test_neg.txt'

Note that splitting data can take a long time. Therefore, we recommend storing the divided data for later use.

(2) If you have finished parameter setting in step 1, skip to step 3. Otherwise, modify the following parameters in setting.py:
>APP = 1

>split = False

>TRAIN_POS_FILENAME = '/data/wiki_train_pos.txt'

>TEST_POS_FILENAME = '/data/wiki_test_pos.txt'

>TEST_NEG_FILENAME = '/data/wiki_test_neg.txt'

(3) Run the code by:

```python TriGAN.py```

(4) The learned embeddings will be stored in /emb/, and the evaluation results (AUC and NDCG@k) for embeddings will be printed on the screen for each epoch.
