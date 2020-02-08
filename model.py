import tensorflow as tf
import _pickle as cPickle


class MODEL():
    def __init__(self, Num, feature_size, G_weight_decay, D_weight_decay, G_learning_rate, D_learning_rate, initdelta=0.05, G_param=None, D_param=None):
        self.d_params = []
        self.g_params = []

        self.u = tf.compat.v1.placeholder(tf.int32, shape=[None], name='u')
        self.v = tf.compat.v1.placeholder(tf.int32, shape=[None], name='v')
        self.pred_data_label = tf.compat.v1.placeholder(tf.float32, shape=[None], name="pred_data_label")
        self.sample_index = tf.compat.v1.placeholder(tf.int32, shape=[None], name='sample_index')
        self.rate = tf.compat.v1.placeholder(tf.float32)

        C_W_1 = tf.get_variable('C_weight_1', [feature_size, feature_size],
                                   initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))
        C_W_2 = tf.get_variable('C_weight_2', [feature_size, 1],
                                   initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        if D_param == None:
            self.node_embedding = tf.compat.v1.get_variable('N_embedd', [Num + 1, feature_size],
                                                            initializer=tf.random_uniform_initializer(minval=-initdelta, maxval=initdelta))
            self.context_embedding = tf.compat.v1.get_variable('parameters', [Num + 1, feature_size], initializer=tf.random_uniform_initializer(minval=-initdelta, maxval=initdelta))
        else:
            self.node_embedding = tf.Variable(D_param[0])
            self.context_embedding = tf.Variable(D_param[1])

        self.d_params.append(self.node_embedding)
        self.d_params.append(self.context_embedding)

        if G_param == None:
            self.W_1 = tf.get_variable('weight_1', [feature_size, feature_size],
                                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            self.W_2 = tf.get_variable('weight_2', [feature_size, 1],
                                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            self.b_1 = tf.get_variable('b_1', [feature_size], initializer=tf.constant_initializer(0.0))
            self.b_2 = tf.get_variable('b_2', [1], initializer=tf.constant_initializer(0.0))
        else:
            self.W_1 = tf.Variable(G_param[0])
            self.W_2 = tf.Variable(G_param[1])
            self.b_1 = tf.Variable(G_param[2])
            self.b_2 = tf.Variable(G_param[3])

        self.g_params.append(self.W_1)
        self.g_params.append(self.W_2)
        self.g_params.append(self.b_1)
        self.g_params.append(self.b_2)

        self.u_embedding = tf.nn.embedding_lookup(self.node_embedding, self.u)  # look up u's embedding
        self.v_embedding = tf.nn.embedding_lookup(self.context_embedding, self.v)  # look up v's embedding, mark

        self.u_sample_embedding = tf.gather(self.u_embedding, self.sample_index)
        self.v_sample_embedding = tf.gather(self.v_embedding, self.sample_index)

        #self.u_embedding_comp = tf.cast(self.u_embedding, dtype=tf.complex64)
        #self.v_embedding_comp = tf.cast(self.v_embedding, dtype=tf.complex64)
        #self.ccorr = tf.math.real(tf.signal.ifft(tf.math.conj(tf.fft(self.u_embedding_comp)) * tf.fft(self.v_embedding_comp)))
        self.ccorr = self.u_embedding * self.v_embedding

        #self.u_sample_embedding_comp = tf.cast(self.u_sample_embedding, dtype=tf.complex64)
        #self.v_sample_embedding_comp = tf.cast(self.v_sample_embedding, dtype=tf.complex64)
        #self.sample_ccorr = tf.math.real(tf.signal.ifft(tf.math.conj(tf.signal.fft(self.u_sample_embedding_comp)) * tf.signal.fft(self.v_sample_embedding_comp)))
        self.sample_ccorr = self.u_sample_embedding * self.v_sample_embedding

        C_layer1 = tf.nn.tanh(tf.matmul(self.ccorr, C_W_1))
        C_drop_out = tf.nn.dropout(C_layer1, rate=self.rate)

        R_layer1 = tf.nn.tanh(tf.matmul(self.sample_ccorr, C_W_1))

        self.D_pred_score = tf.reshape(tf.matmul(C_drop_out, C_W_2), [-1])
        self.G_pred_score = tf.reshape(tf.nn.sigmoid(tf.nn.xw_plus_b(tf.nn.tanh(tf.nn.xw_plus_b(self.ccorr, self.W_1, self.b_1)), self.W_2, self.b_2)), [-1])

        self.reward = tf.reshape(tf.log(tf.exp(tf.matmul(R_layer1, C_W_2)) + 1), [-1])
        #self.reward = (tf.sigmoid(tf.matmul(R_layer1, C_W_2)) - 0.5) * 2
        self.prob = tf.nn.softmax(self.G_pred_score)
        self.gan_prob = tf.gather(self.prob, self.sample_index)

        self.l1_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_pred_score, labels=self.pred_data_label))
        self.l2_loss = D_weight_decay * (tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.v_embedding))
        self.D_loss = self.l1_loss + self.l2_loss

        self.g1_loss = -tf.reduce_mean(tf.log(self.gan_prob) * self.reward)
        self.g2_loss = G_weight_decay * (tf.nn.l2_loss(self.W_1) + tf.nn.l2_loss(self.W_2)
                                         + tf.nn.l2_loss(self.b_1) + tf.nn.l2_loss(self.b_2))
        self.G_loss = self.g1_loss + self.g2_loss

        #self.G_optimizer = tf.train.AdamOptimizer(G_learning_rate)
        self.G_optimizer = tf.train.GradientDescentOptimizer(G_learning_rate)
        self.G_updates = self.G_optimizer.minimize(self.G_loss, var_list=self.g_params)

        self.D_optimizer = tf.train.AdamOptimizer(D_learning_rate)
        self.D_updates = self.D_optimizer.minimize(self.D_loss, var_list=self.d_params)

    def save_model(self, sess, filename, tag=1):
        param = None
        if tag == 1:
            param = sess.run(self.d_params)
        elif tag == 0:
            param = sess.run(self.g_params)
        else:
            print('tag should be 0 or 1')
        cPickle.dump(param, open(filename, 'wb'))
