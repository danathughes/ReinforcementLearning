### TRPO.py
###
### Tensorflow implementation of TRPO

import gym
import tensorflow as tf
import numpy as np

from models.discrete_policy import *
from models.discrete_value import *

from dataset import *
from operations import *
from trajectory import *



class PPOAgent:
	"""
	"""

	def __init__(self, environmentFactory, valueNetFactory, policyNetFactory, **kwargs):
		"""
		"""

		# Create the environment and 
		self.environment = environmentFactory()

		# Create the network
		self.valueNet = valueNetFactory('valueNet')
		self.policyNet = policyNetFactory('policyNet')
		self.oldPolicyNet = policyNetFactory('oldPolicyNet')

		# The trajectories will always be generated by the policy net and environment
		self.generateTrajectory = TrajectoryGenerator( {'state': (4,),
			                    						'action': (),
			                    						'reward': (),
			                    						'next_state': (4,),
			                    						'terminal': ()} )

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


	def train_step(self, dataset):
		"""
		"""

		fd = { self.valueNet.state: dataset['state'],
		       self.valueNet.action: dataset['action'],
		       self.target_value: dataset['target_value'],
		       self.policyNet.input: dataset['state'],
		       self.oldPolicyNet.input: dataset['state'],
		       self.actions: dataset['action'],
		       self.advantages: dataset['advantage'] }

		tf.get_default_session().run([self.train_policy, self.train_value], feed_dict=fd)


	def train(self, num_iterations=1, num_epochs=4):
		"""
		"""

		for iteration in range(num_iterations):
			# Copy the new policy into the old policy
			self.clonePolicy()

			# Create a new dataset
			dataset = self.create_dataset()

			# Train on the dataset 
			# Keep training until at least 64 training points have been sampled
			# 4 times
			for epoch in range(num_epochs):

				total_training_points = 0

				dataset.shuffle()
				batch = dataset.sample(64)


				while total_training_points < 64:
					total_training_points += batch.size

					self.train_step(batch)
					
					dataset.shuffle()
					batch = dataset.sample(64)


			# Write the summaries to the log
			self.summarize(iteration, dataset)


	def summarize(self, iteration, dataset):
		"""
		"""

		# Summarize over the whole dataset
		episode_length = dataset.size

		fd = {  self.valueNet.state: dataset['state'],
				self.valueNet.action: dataset['action'],
			    self.target_value: dataset['target_value'],
			    self.policyNet.input: dataset['state'],
		    	self.oldPolicyNet.input: dataset['state'],
		    	self.actions: dataset['action'],
		    	self.advantages: dataset['advantage'] }

		summary = tf.get_default_session().run(self.summaries, feed_dict=fd)

		self.summary_writer.add_summary(summary, iteration)
		self.summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_length', simple_value=episode_length)]), iteration)


	def create_dataset(self):
		"""
		Perform one iteration of training
		"""

		# Create the trajectories for the dataset
		dataset = self.generateTrajectory(self.environment, self.policyNet)

		# Calculate the advantage function
		advantages = self.calculate_advantages(dataset)

		# Calculate the value of each state using the expected future value
		# If the state is terminal, the next state value is 0
		V_pred = dataset['reward'] + self.discount_factor*self.valueNet(dataset['next_state'])*(1.0 - dataset['terminal'].astype(np.float32))

		# Add the relevent information to the dataset
		dataset.augment({'advantage': advantages, 'target_value': V_pred})

		return dataset


	def calculate_advantages(self, dataset):
		"""
		"""

		# Calculate the value of the states and next states
		V_state = self.valueNet(dataset['state'])
		V_next = self.valueNet(dataset['next_state'])

		deltas = dataset['reward'] + self.discount_factor*V_next - V_state

		# The Generalized Advatage Estimate can be calculated from deltas by:
		# A(t) = d(t) + (y)*d(t+1) + (y^2)*d(t+2) + ... + (y^T-t+1)*d(T-1)
		# This can be done recursively by calculating A(t) = d(t) + y*A(t+1)
		# NOTE: y is the discount factor (gamma)
		GAEs = np.zeros(deltas.shape)

		for t in reversed(range(len(deltas))):
			GAEs[:t+1] *= self.discount_factor
			GAEs[:t+1] += deltas[t]

		return GAEs