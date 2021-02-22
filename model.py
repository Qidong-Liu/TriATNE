import tensorflow as tf
import _pickle as cPickle


class MODEL():
    def __init__(self, Num, feature_size, C_weight_decay, S_weight_decay, P_weight_decay, C_learning_rate, S_learning_rate, P_learning_rate, initdelta=0.05, S_param=None, P_param=None):
        self.p_params = []
        self.s_params = []
        self.c_params = []

        self.u = tf.compat.v1.placeholder(tf.int32, shape=[None], name='u')
        self.v = tf.compat.v1.placeholder(tf.int32, shape=[None], name='v')
        self.pred_data_label = tf.compat.v1.placeholder(tf.float32, shape=[None], name="pred_data_label")
        self.sample_index = tf.compat.v1.placeholder(tf.int32, shape=[None], name='sample_index')
        self.rate = tf.compat.v1.placeholder(tf.float32)

        with tf.variable_scope('customer'):
            C_W_1 = tf.get_variable('C_weight_1', [feature_size, 128],
                                   initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.4, seed=0))
            self.C_W_2 = tf.get_variable('C_weight_2', [128, 1],
                                         initializer=tf.random_uniform_initializer(minval=0.0, maxval=1.0, seed=0))

        self.c_params.append(self.C_W_2)

        self.soft_CW2 = tf.reshape(tf.nn.softmax(tf.reshape(self.C_W_2, [1, -1])), [-1, 1])

        with tf.variable_scope('producer'):
            if P_param == None:
                self.node_embedding = tf.compat.v1.get_variable('N_embedd', [Num + 1, feature_size],
                                                                initializer=tf.random_uniform_initializer(
                                                                    minval=-initdelta, maxval=initdelta))
                self.context_embedding = tf.compat.v1.get_variable('parameters', [Num + 1, feature_size],
                                                                   initializer=tf.random_uniform_initializer(
                                                                       minval=-initdelta, maxval=initdelta))
            else:
                self.node_embedding = tf.Variable(P_param[0])
                self.context_embedding = tf.Variable(P_param[1])

        self.p_params.append(self.node_embedding)
        self.p_params.append(self.context_embedding)

        with tf.variable_scope('seller'):
            if S_param == None:
                self.W_1 = tf.get_variable('weight_1', [feature_size, feature_size],
                                           initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.05))
                self.W_2 = tf.get_variable('weight_2', [feature_size, 1],
                                           initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.05))
                self.b_1 = tf.get_variable('b_1', [feature_size], initializer=tf.constant_initializer(0.0))
                self.b_2 = tf.get_variable('b_2', [1], initializer=tf.constant_initializer(0.0))
            else:
                self.W_1 = tf.Variable(S_param[0])
                self.W_2 = tf.Variable(S_param[1])
                self.b_1 = tf.Variable(S_param[2])
                self.b_2 = tf.Variable(S_param[3])


        self.s_params.append(self.W_1)
        self.s_params.append(self.W_2)
        self.s_params.append(self.b_1)
        self.s_params.append(self.b_2)

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

        C_layer1 = tf.matmul(self.ccorr, C_W_1)
        C_drop_out = tf.nn.dropout(C_layer1, rate=self.rate)

        R_layer1 = tf.matmul(self.sample_ccorr, C_W_1)

        self.P_logits = tf.reshape(tf.matmul(C_drop_out, self.soft_CW2), [-1])
        self.P_pred_score = tf.reshape(tf.reduce_sum(self.ccorr, 1), [-1])   #for test
        self.S_pred_score = tf.reshape(tf.nn.sigmoid(tf.nn.xw_plus_b(tf.nn.tanh(tf.nn.xw_plus_b(self.ccorr, self.W_1, self.b_1)), self.W_2, self.b_2)), [-1])

        self.reward = tf.reshape(tf.log(tf.exp(tf.matmul(R_layer1, self.soft_CW2)) + 1), [-1])
        #self.reward = (tf.sigmoid(tf.matmul(R_layer1, C_W_2)) - 0.5) * 2
        self.prob = tf.nn.softmax(self.S_pred_score)
        self.gan_prob = tf.gather(self.prob, self.sample_index)

        self.p1_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.P_logits, labels=self.pred_data_label))
        self.p2_loss = P_weight_decay * (tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.v_embedding))
        self.P_loss = self.p1_loss + self.p2_loss

        self.F_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.P_pred_score, labels=self.pred_data_label)) + self.p2_loss

        self.s1_loss = -tf.reduce_mean(tf.log(self.gan_prob) * self.reward)
        self.s2_loss = S_weight_decay * (tf.nn.l2_loss(self.W_1) + tf.nn.l2_loss(self.W_2)
                                         + tf.nn.l2_loss(self.b_1) + tf.nn.l2_loss(self.b_2))
        self.S_loss = self.s1_loss + self.s2_loss

        self.C_logits = tf.reshape(tf.matmul(C_drop_out, self.soft_CW2), [-1])
        self.C_loss = -tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.C_logits, labels=self.pred_data_label)) + C_weight_decay * tf.nn.l2_loss(self.C_W_2)

        C_optimizer = tf.train.GradientDescentOptimizer(C_learning_rate)
        self.C_updates = C_optimizer.minimize(self.C_loss, var_list=self.c_params)
        with tf.control_dependencies([self.C_updates]):
            self.clip_w  = tf.assign(self.C_W_2, tf.clip_by_value(self.C_W_2, 0.0, 1.0))

        #self.S_optimizer = tf.train.AdamOptimizer(S_learning_rate)
        self.S_optimizer = tf.train.GradientDescentOptimizer(S_learning_rate)
        self.S_updates = self.S_optimizer.minimize(self.S_loss, var_list=self.s_params)

        self.P_optimizer = tf.train.AdamOptimizer(P_learning_rate)
        self.P_updates = self.P_optimizer.minimize(self.P_loss, var_list=self.p_params)

        self.F_updates = self.P_optimizer.minimize(self.F_loss, var_list=self.p_params)

    def save_model(self, sess, filename, tag=1):
        param = None
        if tag == 1:
            param = sess.run(self.p_params)
        elif tag == 0:
            param = sess.run(self.s_params)
        else:
            print('tag should be 0 or 1')
        cPickle.dump(param, open(filename, 'wb'))
