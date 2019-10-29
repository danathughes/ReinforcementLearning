## discrete_policy.py
##
## 

import tensorflow as tf 
import numpy as np 
from factory import *

from dataset import *

class PolicyNetwork:
	"""
	A neural network to represent a policy
	"""

	def __init__(self, input_shape, action_size, hidden_layer_size, name=None, **kwargs):
		"""
		"""

		hidden_op = kwargs.get('hidden_op', tf.nn.relu)
	#	self.sess = sess

		# Construct the network, with the scope if provided
		if name is None:
			self.__construct(input_shape, action_size, hidden_layer_size, hidden_op)

			self.scope = None

		# Scope was provided
		else:
			with tf.compat.v1.variable_scope(name):
				self.__construct(input_shape, action_size, hidden_layer_size, hidden_op)

				self.scope = tf.compat.v1.get_variable_scope().name


	def __construct(self, input_shape, action_size, hidden_layer_size, hidden_op):
		"""
		"""

		self.input = tf.compat.v1.placeholder(tf.float32, shape=(None,) + input_shape)
		self.hidden_layers = []
		self.num_actions = action_size

		hidden_layer_size = [int(self.input.shape[-1])] + hidden_layer_size
		layer = self.input

		# Make the hidden layers
		for layer_num in range(1, len(hidden_layer_size)):
			print "Layer %d: " % layer_num,
			shape = hidden_layer_size[layer_num-1:layer_num+1]
			print "\tShape =", shape
			weight_name = "W%d" % layer_num
			bias_name = "b%d" % layer_num

			# Make the actual layer
			W = create_weight(shape, name=weight_name)
			b = create_bias((hidden_layer_size[layer_num],), name=bias_name)
			layer = hidden_op(tf.matmul(layer, W) + b)
			self.hidden_layers.append(layer)

		# Make the output layer
		shape = [hidden_layer_size[-1], action_size]
		W = create_weight(shape, name="W_out")
		b = create_bias(action_size, name="b_out")
		
		self.policy = tf.nn.softmax(tf.matmul(self.hidden_layers[-1], W) + b)		



	def getParams(self):
		"""

		"""

		return tf.compat.v1.trainable_variables(self.scope)



	def getActionDist(self, states):
		"""
		Return a distribution over actions for each of the provided state.
		"""

		if len(states.shape) == 1:
			states = states.reshape((1, -1))

		return tf.get_default_session().run(self.policy, {self.input: states})



	def getAction(self, states):
		"""
		"""

		actionDist = self.getActionDist(states)

		actions = np.zeros((actionDist.shape[0],))

		for i in range(actionDist.shape[0]):
			actions[i] = np.random.choice(range(self.num_actions), p=actionDist[i])

		return actions



	def __call__(self, states):
		"""
		"""

		return self.getAction(states)



