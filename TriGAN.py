import _pickle as cPickle
import random
import tensorflow as tf
import numpy as np
import utils as ut
import datetime

from eval.auc import AUC
from eval.ndcg import ndcg_at_K
from model import MODEL

from tqdm import tqdm
import settings as st
from eval.multilabel_class_cv import Cross_val

class TriGAN(object):
    def __init__(self):
        if st.App == 0:
            self.nodes_contexts_train, self.nodes_set = ut.load_edgelist(st.TRAIN_INPUT_FILENAME)
        else:
            self.nodes_contexts_train, self.nodes_set = ut.load_edgelist(st.TRAIN_POS_FILENAME)
            self.test_edges = np.loadtxt(st.TEST_POS_FILENAME, dtype=int)
            self.test_edges_false = np.loadtxt(st.TEST_NEG_FILENAME, dtype=int)

        self.model = None
        self.build_model()

        self.CNum = int(st.gama * len(self.nodes_set))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess = tf.Session(config=config)
        self.sess.run(self.init_op)
        self.sess.graph.finalize()

    def build_model(self):
        if st.CP:               #### continue training from the checkpoint
            param_gen = cPickle.load(open(st.GEN_MODEL_BEST_FILE, 'rb'))
            param_dis = cPickle.load(open(st.DIS_MODEL_BEST_FILE, 'rb'))
            assert param_gen is not None
            assert param_dis is not None
        else:                    #### start a new train
            param_gen = None
            param_dis = None

        self.model = MODEL(len(self.nodes_set)+1, st.FEATURE_SIZE, st.G_WEIGHT_DECAY, st.D_WEIGHT_DECAY, st.G_LEARNING_RATE, st.D_LEARNING_RATE, G_param=param_gen, D_param=param_dis)

    def get_Pos(self, undirected=True):
        walks = []
        num = {}
        rand = random.Random()
        i = 0
        while i < st.num_walks:
            i += 1
            for node in tqdm(self.nodes_contexts_train):
                walk = [node]
                walk.append(rand.choice(self.nodes_contexts_train[node]))
                while len(walk) < st.walk_length:
                    cur = walk[-1]
                    if cur in self.nodes_contexts_train:
                        walk.append(rand.choice(self.nodes_contexts_train[cur]))
                    else:
                        break
                walks.append(walk)

        pos = []
        pos_contexts = []
        for walk in tqdm(walks):
            index = 0
            while index < len(walk):
                source = walk[index]

                left = max(0, index - st.window_size)
                right = min(len(walk), index + st.window_size + 1)
                if undirected:
                    targets = walk[left:index] + walk[index+1:right]
                else:
                    targets = walk[index + 1:right]

                if targets == []:
                    index += 1
                    continue

                pos_contexts.extend(targets)
                pos.extend([source]*len(targets))
                if source in num:
                    num[source].extend(targets)
                else:
                    num[source] = targets

                index += 1
        pos_labels = [1] * len(pos)
        return list(zip(pos, pos_contexts, pos_labels)), num

    def get_Neg(self, pos_Num):
        negs=[]
        nodes = []
        for node in tqdm(pos_Num):
            if node in self.nodes_contexts_train:
                all_contexts = list(self.nodes_set - set(pos_Num[node]) - set(self.nodes_contexts_train[node]))
            else:
                all_contexts = list(self.nodes_set)
            candidate_list = all_contexts

            prob = self.sess.run(self.model.prob, feed_dict={self.model.u: [node] * len(candidate_list), self.model.v: candidate_list})

            neg_list = np.random.choice(candidate_list, size= len(pos_Num[node]) * st.negatives, replace=True, p=prob)

            negs.extend(neg_list)
            nodes.extend([node]*len(neg_list))
        labels = [0]*len(nodes)
        return list(zip(nodes, negs, labels))

    def evaluation(self):
        if st.App == 0:
            f1_micro, f1_macro = Cross_val(st.EMB_OUTPUT_FILENAME)
            print('F1 (micro) = {}'.format(f1_micro))
            print('F1 (macro) = {}'.format(f1_macro))
        elif st.App == 1:
            self.nodes_contexts_test, temp = ut.load_edgelist(st.TEST_POS_FILENAME)
            d_score = AUC(self.sess, self.model, self.test_edges, self.test_edges_false)
            ndcg10 = ndcg_at_K(self.sess, self.model, self.nodes_contexts_test, self.nodes_contexts_train, self.nodes_set, k=10)
            ndcg20 = ndcg_at_K(self.sess, self.model, self.nodes_contexts_test, self.nodes_contexts_train, self.nodes_set, k=20)
            ndcg50 = ndcg_at_K(self.sess, self.model, self.nodes_contexts_test, self.nodes_contexts_train, self.nodes_set, k=50)
            print("D: AUC:", d_score)
            print("G: ndcg10:", ndcg10, "ndcg20:", ndcg20, "ndcg50:", ndcg50)
        else:
            print('please reset App as 0 or 1!')

    def train(self, verbose=True):
        print('start adversarial training')
        for epoch in range(st.train_epochs):
            print('epoch: {}'.format(epoch))
            [pos, num] = self.get_Pos()
            neg = self.get_Neg(num)
            train_data = pos + neg
            train_size = len(train_data)
            random.shuffle(train_data)
            train_data = np.asarray(train_data)

            for d_epoch in range(1):
                s_d_loss = 0
                index = 0
                print('Train D: {}'.format(d_epoch))
                while index < train_size:
                    if index + st.BATCH_SIZE <= train_size:
                        nodes, contexts, pred_data_label = ut.get_batch(train_data, index, st.BATCH_SIZE)
                    else:
                        nodes, contexts, pred_data_label = ut.get_batch(train_data, index, train_size - index)
                    index += st.BATCH_SIZE

                    d_loss, _ = self.sess.run(
                        [self.model.D_loss, self.model.D_updates],
                        feed_dict={self.model.u: nodes,
                                   self.model.v: contexts,
                                   self.model.pred_data_label: pred_data_label,
                                   self.model.rate: st.rate})

                    s_d_loss = s_d_loss + d_loss

                print('sum_loss: ', s_d_loss)

            # Train G
            starttime = datetime.datetime.now()
            for g_epoch in range(st.NNum):
                s_g_loss = 0
                s_p = 0
                s_r = 0
                all_list = list(self.nodes_set)
                random.shuffle(all_list)
                for node in all_list:
                    candidate_list = np.random.choice(list(self.nodes_set), size=self.CNum, replace=False, p=None)
                    prob = self.sess.run(self.model.prob, feed_dict={self.model.u: [node] * len(candidate_list), self.model.v: candidate_list})
                    choose_index = np.random.choice(range(len(candidate_list)), size=1, p=prob)
                    choose_index = np.asarray(choose_index)

                    p, r, g_loss, _, = self.sess.run(
                        [self.model.gan_prob, self.model.reward, self.model.g1_loss, self.model.G_updates],
                        feed_dict={self.model.u: [node]*len(candidate_list),
                                   self.model.v: candidate_list,
                                   self.model.sample_index: choose_index})

                    s_g_loss = s_g_loss + g_loss
                    s_p = s_p + np.sum(np.log(p))
                    s_r = s_r + np.sum(r)

                print('g_loss', s_g_loss, 's_p', s_p, 's_r', s_r)
            endtime = datetime.datetime.now()
            print('G-TIME:',(endtime-starttime).seconds)

            #########write embedding to file ###############
            if epoch % 1 == 0:
                ut.write_to_file(self.sess, self.model, st.EMB_OUTPUT_FILENAME, st.FEATURE_SIZE, list(self.nodes_set), tag=1)
                ut.write_to_file(self.sess, self.model, st.CONTEXT_OUTPUT_FILENAME, st.FEATURE_SIZE, list(self.nodes_set), tag=0)
                self.model.save_model(self.sess, st.GEN_MODEL_BEST_FILE, 0)
                self.model.save_model(self.sess, st.DIS_MODEL_BEST_FILE, 1)

            if verbose:
                self.evaluation()

        self.sess.close()

if __name__ == '__main__':
    if st.split:
        ut.split_data(st.FULL_FILENAME, st.TRAIN_POS_FILENAME, st.TRAIN_NEG_FILENAME, st.TEST_POS_FILENAME, st.TEST_NEG_FILENAME, st.test_frac)
    model = TriGAN()
    model.train()

