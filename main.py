from network import IndepFeatureLearner
from visualizations import visualize_correlation, visualize_policies
from gridworld import SimpleGridworld
import numpy as np
import itertools
import os

env = SimpleGridworld()
dummy_env = SimpleGridworld()
net = IndepFeatureLearner()
visualization_freq = 1000

def run_training_step(net: IndepFeatureLearner, env : SimpleGridworld):
    s = env.get_observation(env.position)
    actions = net.get_actions([s]) # [1, num_factors]
    sp = []
    for i in range(net.num_factors):
        dummy_env.set_position(env.get_position())
        sp.append(np.reshape(dummy_env.step(actions[0,i]), [12, 12, 1, 1]))
    sp = np.concatenate(sp, axis=-1)
    return net.train_step([s], actions, [sp])


for i in itertools.count():
    s = env.get_observation()
    a = np.random.randint(0, 4)
    sp = env.step(a)
    recon_loss, encoder_loss, pi_loss = run_training_step(net, env)
    print(f'{i}: recon_loss: {recon_loss} encoder_loss: {encoder_loss} pi_loss: {pi_loss}')
    if i % visualization_freq == 0:
        visualize_policies(f'./vis/policies/{i}_', net, env)
        visualize_correlation(f'./vis/correlation/{i}_', net, env)

