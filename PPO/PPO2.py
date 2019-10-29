### TRPO.py
###
### Tensorflow implementation of TRPO

import gym
import tensorflow as tf
import numpy as np

from models.discrete_policy import *
from models.discrete_value import *


class CloneOperation(object):
  """
  Wrapper for Tensorflow operation to perform cloning operation between a source and target network
  """

  def __init__(self, source_network, target_network, scope='clone_operation'):
    """
    """

    self.source_network = source_network
    self.target_network = target_network
    self.scope = scope

    # Get the default session to perform operations in; raise an error if no session is available 
    self.sess = tf.compat.v1.get_default_session()
    assert self.sess != None, "No Tensorflow Session found, a Session is needed for the Clone operation"

    # Create an operator to update the target weights from the current DQN
    self.update_operations = []

    with tf.compat.v1.variable_scope(self.scope):
    	for source_param, target_param in zip(self.source_network.getParams(), self.target_network.getParams()):
    		self.update_operations.append(tf.compat.v1.assign(target_param, source_param))


  def __call__(self):
    """
    Perform the update operations
    """

    self.sess.run(self.update_operations)



class Batch(dict):
	"""
	A batch is simply a dictionary with a size attribute appended to it
	"""

	def __init__(self, *args):
		"""
		Create a new batch
		"""

		dict.__init__(self, args)
		self.size = 0


class Dataset:
	"""
	"""

	def __init__(self, schema):
		"""
		Create a new dataset with the provided schema.  The schema consists of 
		a dictionary mapping the name of each data element to the shape of a single
		entry of the element
		"""

		# Maintain the dictionary of element shapes, and create a dictionary of the
		# actual elements with no current entries
		self.schema = schema
		self.data = { name: np.zeros((0,) + shape) for name, shape in self.schema.items() }

		# Set the number of elements to 0, and create an empty index
		self.num_elements = 0
		self.indices = np.array([])
		self.batch_index = 0


	def add(self, data):
		"""
		Add the provided data to the current dataset.  Data needs to be formatted as
		a dictionary mapping element name to entries.
		"""

		# TODO: Check to make sure that the provided data 1) has all entries of the
		#       schema, 2) the shape of the data matches the schema, and 3) the number
		#       of datapoints in each of the elements are the same

		# Check the number of elements in the data


		# Append the provided data to the dataset
		for element in self.data.keys():
			self.data[element] = np.concatenate([self.data[element], data[element]])		

		# Recalculate the number of elements in the dataset
		self.num_elements = self.data.values()[0].shape[0]
		self.indices = np.array(range(self.num_elements))


	def shuffle(self):
		"""
		Randomize the indices
		"""

		# Reset the batch_index to the start of the indices
		self.batch_index = 0

		# Shuffle the indices
		self.indices = np.array(range(self.num_elements))
		np.random.shuffle(self.indices)


	def clear(self):
		"""
		Empty out the data
		"""

		self.data = { name: np.zeros((0,) + shape) for name, shape in self.schema.items() }

		self.num_elements = 0
		self.indices = np.array([])
		self.batch_index = 0


	def sample(self, batch_size):
		"""
		Get the next batch
		"""

		# Create a batch
		batch = Batch()

		# Get the indices into the dataset for this batch
		end_idx = min(self.batch_index + batch_size, self.num_elements)
		idx = self.indices[self.batch_index:end_idx]

		# Create the batch
		for name in self.data.keys():
			batch[name] = self.data[name][idx]

		batch.size = end_idx - self.batch_index

		# Increment the batch index
		self.batch_index = end_idx

#		batch = { element:self.data[element][idx] for element in self.data.keys() }

		return batch





class PPOAgent:
	"""
	"""

	def __init__(self, environmentFactory, valueNetFactory, policyNetFactory, **kwargs):
		"""
		"""

		# Create the environment
		self.environment = environmentFactory()

		# Create the network
		self.valueNet = valueNetFactory('valueNet')
		self.policyNet = policyNetFactory('policyNet')
		self.oldPolicyNet = policyNetFactory('oldPolicyNet')

		# Operation to copy parameters
		self.clonePolicy = CloneOperation(self.policyNet, self.oldPolicyNet)

		# Losses
		self.construct_value_loss()
		self.construct_policy_loss()

		# Optimization
		self.value_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=5e-5, epsilon=1e-5)
		self.train_value = self.value_optimizer.minimize(self.value_loss)

		self.policy_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=5e-5, epsilon=1e-5)
		self.train_policy = self.policy_optimizer.minimize(-self.policy_loss - 0.01*self.entropy_loss)

		# How long should trajectories be?
		self.trajectory_length = kwargs.get("trajectory_length", 200)

		# Discount factor
		self.discount_factor = kwargs.get("discount_factor", 0.9)

		# Create a dataset
		self.dataset = Dataset({'state': (4,),
			                    'action': (),
			                    'reward': (),
			                    'terminal': (),
			                    'advantage': (),
			                    'target_value': ()})

		# Create an operation to summarize
		with tf.compat.v1.variable_scope('summaries'):
			self.summaries = tf.compat.v1.summary.merge_all()

		self.epoch = 0

		sess = tf.compat.v1.get_default_session()
		self.summary_writer = tf.compat.v1.summary.FileWriter('./log', sess.graph)



	def construct_value_loss(self):
		"""
		"""

		with tf.compat.v1.variable_scope('value_loss'):

			# Create a placeholder for target value function
			self.target_value = tf.compat.v1.placeholder(tf.float32, (None,))
			self.value_loss = tf.reduce_mean(tf.math.squared_difference(self.target_value, self.valueNet._V))

		with tf.compat.v1.variable_scope('summaries'):
			tf.compat.v1.summary.scalar('value_loss', self.value_loss)


	def construct_policy_loss(self):
		"""
		"""

		with tf.compat.v1.variable_scope('policy_loss'):
			# Create a placeholder for actions
			self.actions = tf.compat.v1.placeholder(tf.float32, (None,))
			self.advantages = tf.compat.v1.placeholder(tf.float32, (None,))

			actions = tf.cast(self.actions, tf.int32)

			# Get the probabilities of the actions taken
			actions_one_hot = tf.one_hot(actions, 2)

			p_actions = tf.reduce_sum(self.policyNet.policy * actions_one_hot, axis=1)
			p_actions_old = tf.reduce_sum(self.oldPolicyNet.policy * actions_one_hot, axis=1)


			# Calculate the ratio of action probabilities taken by the policy and old policy network
			ratios = tf.exp(tf.math.log(tf.clip_by_value(p_actions, 1e-10, 1.0)) - 
							tf.math.log(tf.clip_by_value(p_actions_old, 1e-10, 1.0)))


			# Clip the ratios to constrain the size of updates
			clipped_ratios = tf.clip_by_value(ratios, clip_value_min=0.8, clip_value_max=1.2)

			clip_loss = tf.minimum(tf.multiply(self.advantages, ratios), 
				                   tf.multiply(self.advantages, clipped_ratios))

			# We want to maximize
			self.policy_loss = tf.reduce_mean(clip_loss)


		with tf.compat.v1.variable_scope('entropy_loss'):
			# Calculate a loss bonus based on the entropy of the policy
			entropy = -tf.reduce_sum(self.policyNet.policy * 
				                    tf.math.log(tf.clip_by_value(self.policyNet.policy, 1e-10, 1.0)), axis=1)
			self.entropy_loss = tf.reduce_mean(entropy, axis=0)


		with tf.compat.v1.variable_scope('summaries'):
			tf.compat.v1.summary.scalar('policy_loss', self.policy_loss)
			tf.compat.v1.summary.scalar('entropy_loss', self.entropy_loss)


	def train(self, num_iterations=1, num_epochs=4):
		"""
		"""

		for iteration in range(num_iterations):

			# Copy the new policy into the old policy
			self.clonePolicy()

			# Create a new dataset
			self.create_dataset()

			# Train on the dataset 
			# Keep training until at least 64 training points have been sampled
			# 4 times
			for epoch in range(num_epochs):

				total_training_points = 0

				self.dataset.shuffle()
				batch = self.dataset.sample(64)


				while total_training_points < 64:
					total_training_points += batch.size

					fd = {  self.valueNet.state: batch['state'],
							self.valueNet.action: batch['action'],
						    self.target_value: batch['target_value'],
						    self.policyNet.input: batch['state'],
					    	self.oldPolicyNet.input: batch['state'],
					    	self.actions: batch['action'],
					    	self.advantages: batch['advantage'] }

					sess = tf.get_default_session()
					sess.run([self.train_policy, self.train_value], feed_dict=fd)

					self.dataset.shuffle()
					batch = self.dataset.sample(64)


			# Write the summaries to the log
			self.summarize(iteration)


	def summarize(self, iteration_num):
		"""
		"""

		# Summarize over the whole dataset
		dataset = self.dataset.data
		episode_length = self.dataset.num_elements

		fd = {  self.valueNet.state: dataset['state'],
				self.valueNet.action: dataset['action'],
			    self.target_value: dataset['target_value'],
			    self.policyNet.input: dataset['state'],
		    	self.oldPolicyNet.input: dataset['state'],
		    	self.actions: dataset['action'],
		    	self.advantages: dataset['advantage'] }

		summary = tf.get_default_session().run(self.summaries, feed_dict=fd)

		self.summary_writer.add_summary(summary, iteration_num)
		self.summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_length', simple_value=episode_length)]), iteration_num)


	def create_dataset(self):
		"""
		Perform one iteration of training
		"""

		sess = tf.get_default_session()

		self.dataset.clear()


		# Generate a trajectory
		states, actions, rewards, next_states, terminals = self.generate_trajectory(True)

		# Calculate the advantage function
		advantages = self.calculate_advantages(states, actions, rewards, next_states, terminals)

		# Calculate the value of each state using the expected future value
		# If the state is terminal, the next state value is 0
		V_pred = rewards + self.discount_factor*self.valueNet(next_states)*(1.0 - terminals.astype(np.float32))

		# Add the relevent information to the dataset
		self.dataset.add({ 	'state': states,
							'action': actions,
							'reward': rewards,
							'terminal': terminals,
							'advantage': advantages,
							'target_value': V_pred
						 })


	def generate_trajectory(self, reset=False):
		"""
		"""

		states = []
		actions = []
		rewards = []
		next_states = []
		terminals = []

		done = False

		# Reset the environment, if desired
		if reset:
			self.environment.reset()

		# Continue collecting trajectories until 
		#   1) The max trajectory length is achieved, or
		#   2) The environment episode has ended
		while not done:
			state = np.array(self.environment.state)
			action = self.oldPolicyNet(state)[0]

			next_state, reward, terminal, _ = self.environment.step(int(action))

			states.append(state)
			actions.append(int(action))
			rewards.append(reward)
			next_states.append(next_state)
			terminals.append(terminal)

			# Finish if terminal or max trajectory length achieved
			if terminal:
				done = True
			if len(states) >= self.trajectory_length:
				done = True

		# Convert states, actions and reward to numpy arrays
		states = np.array(states)
		actions = np.array(actions)
		rewards = np.array(rewards)
		next_states = np.array(next_states)
		terminals = np.array(terminals)

		return states, actions, rewards, next_states, terminals


	def calculate_advantages(self, states, actions, rewards, next_states, terminals):
		"""
		"""

		# Calculate the value of the states and next states
		V_state = self.valueNet(states)
		V_next = self.valueNet(next_states)

		deltas = rewards + self.discount_factor*V_next - V_state

		# The Generalized Advatage Estimate can be calculated from deltas by:
		# A(t) = d(t) + (y)*d(t+1) + (y^2)*d(t+2) + ... + (y^T-t+1)*d(T-1)
		# This can be done recursively by calculating A(t) = d(t) + y*A(t+1)
		# NOTE: y is the discount factor (gamma)
		GAEs = np.zeros(deltas.shape)

		for t in reversed(range(len(deltas))):
			GAEs[:t+1] *= self.discount_factor
			GAEs[:t+1] += deltas[t]

		return GAEs





