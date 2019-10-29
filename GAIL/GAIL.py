import gym
import tensorflow as tf 
import numpy as np 


from PPO.PPO import *
from dataset import *


class GAILAgent:
	"""
	"""

	def __init__(self, environmentFactory, policyAgent, discriminatorFactory, expertDataset, **kwargs):
		"""
		"""

		self.iter = 0

		# Store the agent that learns the policy, and create a discriminator network
		self.policyAgent = policyAgent
		self.environmentFactory = environmentFactory

		self.generateTrajectory = TrajectoryGenerator( {'state': (4,),
														'action': (),
														'reward': (),
														'next_state': (4,),
														'terminal': ()} )

		with tf.compat.v1.variable_scope('discriminator') as discriminator_scope:
			# Create two copies of the discriminator, sharing the variables between the two
			# NOTE: In this case, None *must* be passed to the factory to ensure that
			#       variables are indeed shared
			self.expertDiscriminator = discriminatorFactory(None)
			discriminator_scope.reuse_variables()
			self.agentDiscriminator = discriminatorFactory(None)

		self.expertDataset = expertDataset

		# Create the loss function and optimizer for the discriminator
		# NOTE: The discriminator optimizer wants to _maximize_ the discriminator loss,
		#       i.e., minimize the negative of the generated loss.
		self.construct_discriminator_loss()
		self.discriminator_optimizer = tf.compat.v1.train.AdamOptimizer()
		self.discriminator_trainer = self.discriminator_optimizer.minimize(self.discriminator_loss)

		# The agent reward is the log likelihood of the discriminator
		self.agent_reward = tf.math.log(tf.clip_by_value(self.agentDiscriminator.output, 1e-10, 1.0))


		# Create an operation to summarize
		with tf.compat.v1.variable_scope('summaries'):
			self.summaries = tf.compat.v1.summary.merge_all()

		self.summary_writer = policyAgent.summary_writer


	def construct_discriminator_loss(self):
		"""
		"""

		with tf.compat.v1.variable_scope('discriminator_loss'):

			# The discriminator assigns a 1 to expert state/action pairs, and a 0 to agent state/action pairs.
			# The loss function is basically the negative of the sum of the likelihoods---minimizing the
			# loss results in learning a way of discriminating between agent and expert behavior.
			expert_likelihood = tf.math.log(tf.clip_by_value(self.expertDiscriminator.output, 1e-10, 1.0))
			agent_likelihood = tf.math.log(tf.clip_by_value(1.0 - self.agentDiscriminator.output, 1e-10, 1.0))

			self.expert_likelihood = tf.reduce_mean(expert_likelihood)
			self.agent_likelihood = tf.reduce_mean(agent_likelihood)

			# The discriminator attempts to maximize each of the above cost functions, so the loss is 
			# simply the negative of the sum of the above
			self.discriminator_loss = -(self.expert_likelihood + self.agent_likelihood)

		with tf.compat.v1.variable_scope('summaries'):
			tf.compat.v1.summary.scalar('expert_likelihood', self.expert_likelihood)
			tf.compat.v1.summary.scalar('agent_likelihood', self.agent_likelihood)
			tf.compat.v1.summary.scalar('disciminator_loss', self.discriminator_loss)


	def train(self, step):
		"""
		"""

		# Sample the agent's policy
		dataset = self.policyAgent.create_dataset()

		# Update the discriminator
		for i in range(2):
			self.train_discriminator(dataset)

		# Modify the dataset to reflect the reward function of the agent, i.e.,
		# the agent's reward is the ability to fool the discriminator
		dataset = self.modify_dataset(dataset)

		# Train the PPO agent with this dataset
		self.policyAgent.clonePolicy()

		for i in range(6):
			dataset.shuffle()
			batch = dataset.sample(32)

			self.policyAgent.train_step(batch)

		self.policyAgent.summarize(step, dataset)


	def train_discriminator(self, dataset):
		"""
		"""

		agent_states = dataset['state']
		agent_actions = dataset['action']
		expert_states = self.expertDataset['state']
		expert_actions = self.expertDataset['action']

		sess = tf.get_default_session()
		fd = {self.agentDiscriminator.state: agent_states,
			  self.agentDiscriminator.action: agent_actions,
			  self.expertDiscriminator.state: expert_states,
			  self.expertDiscriminator.action: expert_actions}
		_, expert_ll, agent_ll, disc_loss = sess.run([self.discriminator_trainer, self.expert_likelihood, self.agent_likelihood, self.discriminator_loss], feed_dict=fd)


	def modify_dataset(self, dataset):
		"""
		Modify the agent's dataset by replacing the reward and advantage function 
		"""

		# Generate a trajectory
		states, actions, next_states, terminals = dataset['state'], dataset['action'], dataset['next_state'], dataset['terminal']

		# The reward is replaced by the prediction of the agent discriminator
		rewards = tf.get_default_session().run(self.agent_reward, feed_dict={self.agentDiscriminator.state: states,
														 self.agentDiscriminator.action: actions})

		rewards = np.reshape(rewards, [-1])
		dataset.replace({'reward': rewards})

		# Calculate the advantage function
		advantages = self.policyAgent.calculate_advantages(dataset)

		# Calculate the value of each state using the expected future value
		# If the state is terminal, the next state value is 0
		V_pred = rewards + self.policyAgent.discount_factor*self.policyAgent.valueNet(next_states)*(1.0 - terminals.astype(np.float32))

		# Set the advantage, target value and rewards of the dataset to the new values
		dataset.augment({'advantage': advantages, 'target_value': V_pred})

		return dataset


	def train_agent(self, num_iterations=1, num_epochs = 4):
		"""
		"""

		# Create a dataset
		dataset = policyAgent.create_dataset()

		# The dataset should already exist...
		dataset = self.modify_agent_dataset(dataset)

		# Train on the dataset 
		# Keep training until at least 64 training points have been sampled
		# 4 times
		for epoch in range(num_epochs):

			total_training_points = 0

			dataset.shuffle()
			batch = dataset.sample(64)


			while total_training_points < 64:
				total_training_points += batch.size

				fd = {  self.policyAgent.valueNet.state: batch['state'],
						self.policyAgent.valueNet.action: batch['action'],
					    self.policyAgent.target_value: batch['target_value'],
					    self.policyAgent.policyNet.input: batch['state'],
				    	self.policyAgent.oldPolicyNet.input: batch['state'],
				    	self.policyAgent.actions: batch['action'],
				    	self.policyAgent.advantages: batch['advantage'] }

				sess = tf.get_default_session()
				sess.run([self.policyAgent.train_policy, self.policyAgent.train_value], feed_dict=fd)

				dataset.shuffle()
				batch = self.policyAgent.dataset.sample(64)


		# Write the summaries to the log
		self.policyAgent.summarize(self.iter)
		self.iter += 1

