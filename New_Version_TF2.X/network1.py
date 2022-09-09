import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from gridworld import SimpleGridworld

EPS = 10**-6


class EncoderDecoder(tf.keras.Model):
    def __init__(self,lbda=1, num_factors=4,num_actions=4) -> None:
        super(EncoderDecoder,self).__init__()
        self.lbda = lbda
        self.num_factors = num_factors
        self.num_actions = num_actions
        self.encoder = self.build_encoder_gridworld()
        self.decoder = self.build_decoder_gridworld()
        self.pi_net = self.build_pi_network()

    def call(self, x, xp):
        fc1, enc = self.encoder(x)
        recon = self.decoder(enc)
        self.recon = recon

        decoder_loss = 0.5*tf.reduce_sum(tf.square(recon - x), axis=[1,2,3])
        decoder_loss = tf.reduce_mean(decoder_loss, axis=0)
        self.decoder_loss = decoder_loss

        self.pi = self.pi_net(fc1)
        self.f = fs = enc

        fsp_list = []
        for i in range(self.num_actions):
            _, fsp = self.encoder(xp[:,:,:,:,i]) # fsp_shape: (bs,num_factors)
            fsp_list.append(tf.reshape(fsp, [-1, self.num_factors, 1]))
        fsp = tf.concat(fsp_list, axis=-1) # [bs, num_factors, num_actions]
        self.sel_encoder_loss = -tf.reduce_mean(self.build_selectivity(fsp, fs, self.pi), axis=0)
        self.sel_pi_loss = -tf.reduce_mean(self.build_selectivity_log(fsp, fs, self.pi), axis=0)



        return self.decoder_loss, self.sel_encoder_loss, self.sel_pi_loss

    def get_parameters(self):
        self.encoder_para = self.encoder.trainable_variables
        self.decoder_para = self.decoder.trainable_variables
        self.pi_para      = self.pi_net.trainable_variables

        return self.encoder_para, self.decoder_para, self.pi_para

    def build_encoder_gridworld(self):
        '''
        modified it to TF.2X
        '''
        inp_s = layers.Input(shape = (12,12,1))
        c1 = layers.Conv2D(filters=16, kernel_size=(3,3), strides=(2,2), padding='same',activation='relu',name='conv1')(inp_s)
        c2 = layers.Conv2D(filters=16, kernel_size=(3,3), strides=(2,2), padding='same',activation='relu',name='conv2')(c1)
        c2_flat = layers.Flatten()(c2)
        fc1 = layers.Dense(units=32,name='fc1')(c2_flat)
        enc = layers.Dense(units=self.num_factors,activation='tanh',name='fc2')(fc1)

        encoder  = tf.keras.Model(inputs=[inp_s], outputs=[fc1, enc])

        return encoder

    def build_decoder_gridworld(self):
        '''
        modified it to TF2.X
        '''
        enc = layers.Input(shape=(self.num_factors))
        fc1 = layers.Dense(units=32, activation='relu')(enc)
        c2_flat = layers.Dense(units=3*3*16, activation='relu', name='c2_flat')(fc1)
        c2 = tf.reshape(c2_flat, [-1, 3, 3, 16])
        c1 = layers.Conv2DTranspose(filters=16, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu')(c2) # [bs, 6, 6, 16]
        out =layers.Conv2DTranspose(filters=1, kernel_size=(3,3), strides=(2,2), padding='same')(c1) # [bs, 12, 12, 1]
        decoder  = tf.keras.Model(inputs=[enc], outputs=[out])

        return decoder

    def build_pi_network(self):
        '''
        modified it to TF2.X
        '''
        # 注意此处是action数目乘以factor(应该等于policy)数目
        x = layers.Input(shape=(32))
        fc1 = layers.Dense(units = self.num_actions * self.num_factors, name='fc1')(x)
        # fc1 = tf.reshape(fc1, [-1, self.num_actions, self.num_factors])
        # out = tf.nn.softmax(fc1, axis=1) #存疑
        fc1 = tf.reshape(fc1, [-1, self.num_factors, self.num_actions])
        out = tf.nn.softmax(fc1, axis=2)

        pi_net = tf.keras.Model(inputs=[x], outputs=[out])

        return pi_net

    def build_sel_term(self, fsp, fs, action_i, factor_k):
        numer = fsp[:, factor_k, action_i] - fs[:, factor_k] #对应公式(1)的分子，此处有broadcast广播机制
        denom = tf.reduce_sum(tf.abs(fsp[:, :, action_i] - fs), axis=1) + EPS  #axis=1对应fsp的factor维度
        sel = numer / denom
        return sel #(batch,)

    def build_selectivity(self, fsp, fs, pi):
        # fs : [bs, num_factors]
        # fsp : [bs, num_factors (one for each factor), num_actions (one for each action from pi_k)]
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

# def grad(model, s, sp):
#     with tf.GradientTape() as tape:
#         loss_value = model(s, sp, training=True)
#     return loss_value, tape.gradient(loss_value, model.trainable_variables)

the_model = EncoderDecoder()
optimizer_recon = tf.keras.optimizers.Adam(learning_rate=0.0001)
optimizer_pi =  tf.keras.optimizers.Adam(learning_rate=0.0001)

def train_step(s,sp):

    with tf.GradientTape() as recon_tape, tf.GradientTape() as pi_tape:
        decoder_loss, sel_encoder_loss, sel_pi_loss = the_model(s,sp)
        encoder_para, decoder_para, pi_para =  the_model.get_parameters()


    gradients_of_recon = recon_tape.gradient(decoder_loss, encoder_para+decoder_para)
    gradients_of_pi = pi_tape.gradient(the_model.lbda*sel_pi_loss, pi_para)

    optimizer_recon.apply_gradients(zip(gradients_of_recon, encoder_para+decoder_para))
    optimizer_pi.apply_gradients(zip(gradients_of_pi, pi_para))

    return decoder_loss, sel_pi_loss




