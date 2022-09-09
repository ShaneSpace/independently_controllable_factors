import tensorflow as tf
import numpy as np
from gridworld import SimpleGridworld

EPS = 10**-6

class IndepFeatureLearner(object):

    def __init__(self, lmbda=1.0, learning_rate=0.0001, gpu_num=0):
        self.num_actions = 4
        self.num_factors = 4
        self.lbda = lmbda
        # for whatever reason we sample 1 state at a time.
        device_string = f'/gpu:{gpu_num}' if gpu_num >= 0 else f'/cpu:0'
        with tf.device(device_string):
            self.inp_s = tf.placeholder(tf.float32, shape=[None, 12, 12, 1], name='inp_s')
            #self.inp_a = tf.placeholder(tf.uint8, shape=[None, self.num_factors], name='inp_a')
            #a_onehot = []
            #for i in range(self.num_factors):
            #    a_onehot.append(tf.reshape(tf.one_hot(self.inp_a[:, i], self.num_actions), [-1, self.num_actions, 1]))
            #a_onehot = tf.concat(a_onehot, axis=-1) # [bs, num_actions, num_factors]
            self.inp_sp = tf.placeholder(tf.float32, shape=[None, 12, 12, 1, self.num_actions], name='inp_sp')

            # building autoencoder and decoder loss
            fc1, enc = self.build_encoder_gridworld(self.inp_s, 'encoder')
            recon = self.build_decoder_gridworld(enc, 'decoder')
            self.recon = recon
            decoder_loss = 0.5*tf.reduce_sum(tf.square(recon - self.inp_s), axis=[1,2,3])
            decoder_loss = tf.reduce_mean(decoder_loss, axis=0)
            self.decoder_loss = decoder_loss

            # build policy
            self.pi = pi = self.build_pi_network(fc1, 'pi') # [bs, num_actions, num_factors]

            # building selectivity terms and loss
            self.f = fs = enc
            fsp_list = []
            for i in range(self.num_actions):
                _, fsp = self.build_encoder_gridworld(self.inp_sp[:,:,:,:,i], 'encoder', reuse=True)
                fsp_list.append(tf.reshape(fsp, [-1, self.num_factors, 1]))
            fsp = tf.concat(fsp_list, axis=-1) # [bs, num_factors, num_actions]
            self.sel_encoder_loss = tf.reduce_mean(self.build_selectivity(fsp, fs, pi), axis=0)
            self.sel_pi_loss = tf.reduce_mean(self.build_selectivity_log(fsp, fs, pi), axis=0)

            encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
            decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
            policy_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pi')

            print('encoder', encoder_vars)
            print('decoder', decoder_vars)
            print('pi', policy_vars)

            self.train_recon = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(decoder_loss, var_list=encoder_vars + decoder_vars)
            #self.train_encoder_sel = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(-self.lbda*self.sel_encoder_loss, var_list=encoder_vars)
            self.train_pi_sel = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(-self.lbda*self.sel_pi_loss, var_list=policy_vars+encoder_vars)

            # TODO configure so we dont eat all the resources.
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())


    def train_step(self, s, sp):
        [_, recon_loss] = self.sess.run([self.train_recon, self.decoder_loss], feed_dict={self.inp_s: s})
        #[_, encoder_loss] = self.sess.run([self.train_encoder_sel, self.sel_encoder_loss], feed_dict={self.inp_s: s,
        #                                                                                              self.inp_sp: sp})
        [_, pi_loss] = self.sess.run([self.train_pi_sel, self.sel_pi_loss], feed_dict={self.inp_s: s, self.inp_sp: sp})
        #encoder_loss = 0
        #pi_loss = 0
        return recon_loss, pi_loss


    def get_f(self, s):
        return self.sess.run(self.f, {self.inp_s: s})

    def get_all_pi(self, s):
        return self.sess.run(self.pi, {self.inp_s: s})

    def get_actions(self, s):
        return np.argmax(self.sess.run(self.pi, {self.inp_s: s}), axis=1) # [bs, num_factors]

    def get_recon(self, s):
        return self.sess.run(self.recon, {self.inp_s: s})

    def build_encoder_gridworld(self, inp, name, reuse=None):
        # images are 12 x 12 x 1?
        inp_shape = [x.value for x in inp.get_shape()]
        assert inp_shape[1:] == [12, 12, 1]
        with tf.variable_scope(name, reuse=reuse):
            c1 = tf.layers.conv2d(inp, 16, 3, 2, padding='SAME', activation=tf.nn.relu, name='c1') # [bs, 6, 6, 16]
            c2 = tf.layers.conv2d(c1, 16, 3, 2, padding='SAME', activation=tf.nn.relu, name='c2') # [bs, 3, 3, 16]
            c2_flat = tf.reshape(c2, [-1, 3*3*16])
            fc1 = tf.layers.dense(c2_flat, 32, activation=tf.nn.relu, name='fc1')
            enc = tf.layers.dense(fc1, self.num_factors, activation=tf.nn.tanh, name='enc')
        return fc1, enc

    def build_decoder_gridworld(self, enc, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            fc1 = tf.layers.dense(enc, 32, activation=tf.nn.relu, name='fc1')
            c2_flat = tf.layers.dense(fc1, 3*3*16, activation=tf.nn.relu, name='c2_flat')
            c2 = tf.reshape(c2_flat, [-1, 3, 3, 16])
            c1 = tf.layers.conv2d_transpose(c2, 16, 3, 2, padding='SAME', activation=tf.nn.relu, name='c1') # [bs, 6, 6, 16]
            out = tf.layers.conv2d_transpose(c1, 1, 3, 2, padding='SAME', name='out') # [bs, 12, 12, 1]
        return out

    def build_pi_network(self, fc1, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            fc1 = tf.layers.dense(fc1, self.num_actions * self.num_factors, name='fc1')
            fc1 = tf.reshape(fc1, [-1, self.num_actions, self.num_factors])
            # apply softmax to each row
            policy_rows = []
            for i in range(self.num_actions):
                policy_rows.append(tf.reshape(tf.nn.softmax(fc1[:, :, i]), [-1, self.num_actions, 1]))
            fc1 = tf.concat(policy_rows, axis=2)
        return fc1

    def build_sel_term(self, fsp, fs, action_i, factor_k):
        numer = fsp[:, factor_k, action_i] - fs[:, factor_k]
        denom = tf.reduce_sum(tf.abs(fsp[:, :, action_i] - fs), axis=1) + EPS
        sel = numer / denom
        return sel

    def build_selectivity(self, fsp, fs, pi):
        # fs : [bs, num_factors]
        # fsp : [bs, num_factors (one for each factor), num_factors (one for each action from pi_k)]
        # returns the selectivity of the environment
        out = tf.zeros([tf.shape(fs)[0]])
        for k in range(self.num_factors):
            for i in range(self.num_actions):
                sel = self.build_sel_term(fsp, fs, i, k)
                out += pi[:, i, k] * sel
        return out




    def build_selectivity_log(self, fsp, fs, pi):
        # fs : [bs, num_factors]
        # fsp : [bs, num_factors, num_factors]
        # pi : [bs, num_actions, num_factors]
        # action : [bs, num_actions, num_factors]
        # returns the selectivity of the environment
        sel_log = tf.zeros([tf.shape(fs)[0]])
        for k in range(self.num_factors):
            for i in range(self.num_actions):
                sel = self.build_sel_term(fsp, fs, i, k)
                sel_log += pi[:, i, k] * sel
        return sel_log



