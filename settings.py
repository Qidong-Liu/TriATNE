import os

####parameters for learning
D_LEARNING_RATE = 0.0001
G_LEARNING_RATE = 0.0001
D_WEIGHT_DECAY = 0.000001
G_WEIGHT_DECAY = 0.000001
BATCH_SIZE = 1024


FEATURE_SIZE = 128              #dimensions

gama = 0.01                     #Parameter for speed up traing of 'seller'. gama * N >= NNum
NNum = 10                       #Randomly choose NNum (K) nodes for training p_\phi
rate = 0.                       #dropout rate for customers
train_epochs = 20               #Total train epochs

###parameters for random walks
walk_length = 40
num_walks = 1
window_size = 10
negatives = 5

App = 0                         #0 denotes classification; 1 denotes link prediction
CP = False                      #Continue training from the checkpoint

workdir = os.path.abspath('.')  #workspace

######Classification Input
TRAIN_INPUT_FILENAME = workdir + "/data/Citeseer_input.txt"
TRAIN_LABEL_FILENAME = workdir + "/data/Citeseer_label.txt"
format = 'blogcat'

######Link prediction Input
split = False
test_frac = 0.2
FULL_FILENAME = workdir + '/data/wiki-vote.txt'

TRAIN_POS_FILENAME = workdir + '/data/wiki_train_pos.txt'
TRAIN_NEG_FILENAME = workdir + '/data/wiki_train_neg.txt' # Optimal

TEST_POS_FILENAME = workdir + "/data/wiki_test_pos.txt"
TEST_NEG_FILENAME = workdir + "/data/wiki_test_neg.txt"

#####Output
EMB_OUTPUT_FILENAME = workdir + "/emb/emb.txt"
CONTEXT_OUTPUT_FILENAME = workdir + "/emb/context.txt"
GEN_MODEL_BEST_FILE = workdir + '/model/gen_best.model'
DIS_MODEL_BEST_FILE = workdir + '/model/dis_best.model'
