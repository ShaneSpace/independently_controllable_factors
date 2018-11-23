from network import IndepFeatureLearner
from visualizations import visualize_correlation, visualize_policies, visualize_autoencoder
from gridworld import SimpleGridworld
import numpy as np
import itertools
import utils, replay_buffer
import os

env = SimpleGridworld()
dummy_env = SimpleGridworld()
net = IndepFeatureLearner()
buffer = replay_buffer.ReplayBuffer(10000)

visualization_freq = 1000
batch_size = 64

def run_training_step(buffer : replay_buffer.ReplayBuffer, net : IndepFeatureLearner, env : SimpleGridworld):
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
    return net.train_step(s_list, sp_list)


for i in itertools.count():
    s = env.get_observation()
    buffer.append(env.get_position(), None, None, None, None)
    a = np.random.randint(0, 4)
    sp = env.step(a)

    if buffer.length() < 1000:
        continue

    recon_loss, pi_loss = run_training_step(buffer, net, env)
    print(f'{i}: recon_loss: {recon_loss} pi_loss: {pi_loss}')
    if i % visualization_freq == 0:
        visualize_policies(f'./vis/policies/{i}_', net, env)
        visualize_correlation(f'./vis/correlation/{i}_', net, env)
        visualize_autoencoder(f'./vis/autoencoder/', net, env)

