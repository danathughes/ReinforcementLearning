## discrete_policy.py
##
## 

import tensorflow as tf 
import numpy as np 

from factory import *

class DiscriminatorNetwork:
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

		self.state = tf.compat.v1.placeholder(tf.float32, shape=(None,) + input_shape)
		self.action = tf.compat.v1.placeholder(tf.int32, shape=(None,))
		self.action_one_hot = tf.one_hot(self.action, action_size)

		# Concatenate the state and action vectors
		self.state_action = tf.concat([self.state, self.action_one_hot], 1)

		self.hidden_layers = []
		self.num_actions = action_size

		hidden_layer_size = [int(self.state_action.shape[-1])] + hidden_layer_size
		layer = self.state_action

		# Make the hidden layers
		for layer_num in range(1, len(hidden_layer_size)):
			print "Layer %d: " % layer_num,
			shape = hidden_layer_size[layer_num-1:layer_num+1]
			print "\tShape =", shape
			weight_name = "W%d" % layer_num
			bias_name = "b%d" % layer_num

			# Make the actual layer
			W = tf.compat.v1.get_variable(weight_name, shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
			b = tf.compat.v1.get_variable(bias_name, shape=(hidden_layer_size[layer_num],), initializer=tf.zeros_initializer())

			layer = hidden_op(tf.matmul(layer, W) + b)
			self.hidden_layers.append(layer)

		# Make the output layer - a single sigmoid 
		shape = [hidden_layer_size[-1], 1]

		W = tf.compat.v1.get_variable('W_out', shape=shape, initializer=tf.truncated_normal_initializer(0.1))
		b = tf.compat.v1.get_variable('b_out', shape=(1,), initializer=tf.zeros_initializer())
		
		self.output = tf.sigmoid(tf.matmul(self.hidden_layers[-1], W) + b)		


	def getParams(self):
		"""
		"""

		return tf.compat.v1.trainable_variables(self.scope)


	def __call__(self, states, actions):
		"""
		"""

		sess = tf.get_default_session()
		out = sess.run(self.output, feed_dict={self.state: states, self.action: actions})

		return out



