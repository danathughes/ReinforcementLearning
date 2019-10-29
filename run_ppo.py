import gym
import tensorflow as tf
import numpy as np 

from PPO.PPO import *

from models.discrete_policy import *
from models.discrete_value import *

## MAIN PART OF THE PROGRAM
sess = tf.compat.v1.InteractiveSession()

policyNetFactory = lambda name: PolicyNetwork((4,), 2, [20,20], name=name)
valueNetFactory = lambda name: ValueNetwork((4,), 2, [20,20], name=name)
environmentFactory = lambda: gym.make('CartPole-v0')

# Create the agent
ppoAgent = PPOAgent(environmentFactory, valueNetFactory, policyNetFactory)


# Create the session and initialize variables
sess.run(tf.compat.v1.global_variables_initializer())



# Train the agent
ppoAgent.train(10000)

# Evaluate the agent
env = environmentFactory()
env.reset()

done = False

while not done:
	state = np.array(env.state)
	action = int(ppoAgent.policyNet(state)[0])
	_,_,done,_ = env.step(action)
	env.render()


