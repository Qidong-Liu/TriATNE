import os

####parameters for learning
C_LEARNING_RATE = 0.0001
P_LEARNING_RATE = 0.0001
S_LEARNING_RATE = 0.00001
C_WEIGHT_DECAY = 0.000001
P_WEIGHT_DECAY = 0.000001
S_WEIGHT_DECAY = 0.000001
BATCH_SIZE = 1024


FEATURE_SIZE = 128              #dimensions

gama = 1.                     #Parameter for speed up traing of 'seller'. gama * N >= NNum
NNum = 5                       #Randomly choose NNum (K) nodes for training p_\phi
rate = 0.                       #dropout rate for customers
train_epochs = 50              #Total train epochs

###parameters for random walks
walk_length = 40
num_walks = 1
window_size = 10
negatives = 1

App = 1                         #0 denotes classification; 1 denotes link prediction
CP = False                      #Continue training from the checkpoint

workdir = os.path.abspath('.')  #workspace

######Classification Input
TRAIN_INPUT_FILENAME = workdir + "/data/Citeseer_input.txt"
TRAIN_LABEL_FILENAME = workdir + "/data/Citeseer_label.txt"
#TRAIN_INPUT_FILENAME = '/home/lqd/dataset/PubMed/cf/PubMed_input.txt'
#TRAIN_LABEL_FILENAME = '/home/lqd/dataset/PubMed/cf/PubMed_label.txt'
format = 'blogcat'

######Link prediction Input
split = False
test_frac = 0.5
#FULL_FILENAME = workdir + '/data/wiki-vote.txt'
FULL_FILENAME = '/home/lqd/dataset/Cora/lp/data_0.txt'

#TRAIN_POS_FILENAME = workdir + '/data/wiki_train_pos.txt'
TRAIN_POS_FILENAME = '/home/lqd/dataset/Cora/lp/train_edges_0.txt'
TRAIN_NEG_FILENAME = '/home/lqd/dataset/Cora/lp/train_edges_false_0.txt'
#TRAIN_NEG_FILENAME = workdir + '/data/wiki_train_neg.txt' # Optimal

#TEST_POS_FILENAME = workdir + "/data/wiki_test_pos.txt"
#TEST_NEG_FILENAME = workdir + "/data/wiki_test_neg.txt"
TEST_POS_FILENAME = '/home/lqd/dataset/Cora/lp/test_edges_0.txt'
TEST_NEG_FILENAME = '/home/lqd/dataset/Cora/lp/test_edges_false_0.txt'

#####Output
EMB_OUTPUT_FILENAME = workdir + "/emb/emb_lp.txt"
CONTEXT_OUTPUT_FILENAME = workdir + "/emb/context_lp.txt"
S_MODEL_BEST_FILE = workdir + '/model/s_best.model_lp'
P_MODEL_BEST_FILE = workdir + '/model/p_best.model_lp'
