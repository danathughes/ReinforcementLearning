## discrete_value.py
##
## A class for networks that compute the state and state/action values for environments with 
## a discrete action space.  The network assumes the following input

import tensorflow as tf 
import numpy as np 
from factory import *

from dataset import *


class ValueNetwork:
	"""
	A neural network to calculate the value of state and state/action pairs
	NOTE:  This network calcualtes the 
	"""

	def __init__(self, input_shape, action_size, hidden_layer_size, name=None, **kwargs):
		"""
		"""

		hidden_op = kwargs.get('hidden_op', tf.nn.relu)

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

		self.state = tf.compat.v1.placeholder(tf.float32, shape=(None,) + input_shape)
		self.action = tf.compat.v1.placeholder(tf.int32, shape=(None,))
		self.hidden_layers = []
		self.num_actions = action_size

		hidden_layer_size = [int(self.state.shape[-1])] + hidden_layer_size
		layer = self.state

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
		
		self._Qs = tf.matmul(self.hidden_layers[-1], W) + b

		# V(s) = max_a Q(s,a)
		self._V = tf.reduce_max(self._Qs, axis=1)

		# Q(s,a) = Qs(s) @ a
		self._Q = tf.gather_nd(self._Qs, tf.reshape(self.action, (-1,1)), batch_dims=1)
		


	def getParams(self):
		"""
		"""

		return tf.compat.v1.trainable_variables(self.scope)


	def V(self, states):
		"""
		"""

		if len(states.shape) == 1:
			states = np.reshape(states,(1,-1))

		return tf.get_default_session().run(self._V, {self.state: states})



	def Q(self, states, actions=None):
		"""
		Calculate the Q value for the state / action pair.  If an action is not provided,
		return the Q value for all actions
		"""

		if len(states.shape) == 1:
			states = np.reshape(states,(1,-1))

		if actions is None:
			return tf.get_default_session().run(self._Qs, {self.state: states})
		else:
			if type(actions) is not np.ndarray:
				actions = np.array([actions])

			# TODO: Check the actions shape

			return tf.get_default_session().run(self._Q, {self.state:states, self.action:actions})



	def __call__(self, states, actions=None):
		"""
		"""

		if actions is None:
			return self.V(states)
		else:
			return self.Q(states, actions)