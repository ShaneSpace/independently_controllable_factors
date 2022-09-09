from network import IndepFeatureLearner
from network1 import EncoderDecoder
from visualizations import visualize_correlation, visualize_policies, visualize_autoencoder
from gridworld import SimpleGridworld
import numpy as np
import itertools
import utils, replay_buffer
import os
from utils import build_directory_structure, LOG, add_implicit_name_arg
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser()
add_implicit_name_arg(parser)
parser.add_argument('--run-dir', type=str, default='runs')
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--lmbda', type=float, default=1)
parser.add_argument('--learning-rate', type=float, default=0.0001)
parser.add_argument('--gpu-num', type=int, default=0)
# args = parser.parse_args()

args = parser.parse_args(args=[])


build_directory_structure('.',
    {args.run_dir : {
        args.name : {
            'autoencoder': {},
            'correlation': {},
            'policies': {}
        }
    }
    })

# LOG.setup(os.path.join('.', args.run_dir, args.name))


env = SimpleGridworld()
dummy_env = SimpleGridworld()
# net = IndepFeatureLearner(lmbda=args.lmbda, learning_rate=args.learning_rate, gpu_num=args.gpu_num)

######
net = EncoderDecoder()
optimizer_recon = tf.keras.optimizers.Adam(learning_rate=0.0001)
optimizer_pi =  tf.keras.optimizers.Adam(learning_rate=0.0001)

def train_step(s,sp):

    with tf.GradientTape() as recon_tape, tf.GradientTape() as pi_tape:
        decoder_loss, sel_encoder_loss, sel_pi_loss = net(s,sp)
        encoder_para, decoder_para, pi_para =  net.get_parameters()


    gradients_of_recon = recon_tape.gradient(decoder_loss, encoder_para+decoder_para)
    gradients_of_pi = pi_tape.gradient(sel_pi_loss, pi_para)

    optimizer_recon.apply_gradients(zip(gradients_of_recon, encoder_para+decoder_para))
    optimizer_pi.apply_gradients(zip(gradients_of_pi, pi_para))

    return decoder_loss, sel_pi_loss
##############

buffer = replay_buffer.ReplayBuffer(10000)

visualization_freq = 10000
batch_size = args.batch_size

def run_training_step(buffer : replay_buffer.ReplayBuffer, net : IndepFeatureLearner):
    positions, _, _, _, _ = buffer.sample(batch_size)
    s_list = []
    sp_list = []
    action_list = []
    for pos in positions:
        s = dummy_env.get_observation(pos)
        #actions = net.([s]) # [1, num_factors]
        sp = []
        for i in range(net.num_actions):
            dummy_env.set_position(pos)
            sp.append(np.reshape(dummy_env.step(i), [12, 12, 1, 1]))
        sp = np.concatenate(sp, axis=-1)
        s_list.append(s)
        sp_list.append(sp)
    s_list  = tf.convert_to_tensor(s_list)
    sp_list = tf.convert_to_tensor(sp_list)
    return train_step(s_list, sp_list)


for i in itertools.count():
    s = env.get_observation()
    buffer.append(env.get_position(), None, None, None, None)
    a = np.random.randint(0, 4)
    sp = env.step(a)

    if buffer.length() < 1000:
        continue

    recon_loss, pi_loss = run_training_step(buffer, net)
    # LOG.add_line('recon_loss', recon_loss)
    # LOG.add_line('pi_loss', pi_loss)
    if i % 100 == 0:
        print(f'{i}: recon_loss: {recon_loss} pi_loss: {pi_loss}')
    # if i % visualization_freq == 0:
    #     visualize_policies(f'./vis/policies/{i}_', net, env)
    #     visualize_correlation(f'./vis/correlation/{i}_', net, env)
    #     visualize_autoencoder(f'./vis/autoencoder/', net, env)

