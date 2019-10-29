import gym
import tensorflow as tf
import numpy as np 

from PPO.PPO import *
from GAIL.GAIL import *

from models.discrete_policy import *
from models.discrete_value import *
from models.discriminator import *

def load_dataset(state_filename, action_filename):
	"""
	"""

	dataset = Dataset({'state': (4,), 'action': ()})

	states = np.loadtxt(state_filename)
	actions = np.loadtxt(action_filename)

	dataset.add({'state': states, 'action': actions})

	return dataset



expertDataset = load_dataset('states.csv', 'actions.csv')

## MAIN PART OF THE PROGRAM
sess = tf.compat.v1.InteractiveSession()

policyNetFactory = lambda name: PolicyNetwork((4,), 2, [20,20], name=name)
valueNetFactory = lambda name: ValueNetwork((4,), 2, [20,20], name=name)
discriminatorFactory = lambda name: DiscriminatorNetwork((4,), 2, [20,20,20], name=name)

environmentFactory = lambda: gym.make('CartPole-v0')

# Create the agent
ppoAgent = PPOAgent(environmentFactory, valueNetFactory, policyNetFactory)

gailAgent = GAILAgent(environmentFactory, ppoAgent, discriminatorFactory, expertDataset)

saver = tf.compat.v1.train.Saver()

sess.run(tf.global_variables_initializer())


for i in range(10000):
	gailAgent.train(i)

#	gailAgent.train_agent()

saver.save(sess, 'gail-model', global_step=10000)

# Evaluate the agent
env = environmentFactory()
env.reset()

done = False

while not done:
	state = np.array(env.state)
	action = int(ppoAgent.policyNet(state)[0])
	_,_,done,_ = env.step(action)
	env.render()