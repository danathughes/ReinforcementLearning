## trajectory.py
## 
## 

import numpy as np

from dataset import Dataset 

class TrajectoryGenerator:
	"""
	Trajectory Generator is used to produce a trajectory from an environment given a policy.
	"""

	def __init__(self, schema):
		"""
		"""

		# The schema is used to define the generate dataset.  The generator is responsible
		# for ensuring that this schema is followed when the object is called
		self.schema = schema


	def __call__(self, environment, policy, max_length=None, reset=True):
		"""

		Arguments:

		    environment   The environment the agent is interacting with.  The environment should 
		                  follow the OpenAI environment interface, i.e., implements the following
		                  methods:

		                  state          The current state variable of the environment
		                  reset()        Set the state of the environment to the start state
		                  step(action)   Perform an action on the environment and receive resulting
		                                 state evolution.


		    policy        The policy the agent is using to generate trajectories in the environment.
		                  The policy needs to implement a __call__ method, which takes a state as an
		                  argument, and generates an action.

		    max_length    The maximum length of the trajectory.  If None is provided, the trajectory
		                  will be generated until a terminal state is reached.

		    reset         If True, the call will reset the environment before generating trajectories,
		                  otherwise, trajectories will be generated from the current state of the 
		                  environment


		Return:

			Trajectory    A Dataset that follows the schema of the generator
		"""

		# Create lists for each component of the trajectory
		states, actions, rewards, next_states, terminals = [], [], [], [], []

		# Reset the environment, if desired
		if reset:
			environment.reset()

		# Perform actions on the environment until either the max_length is achieved, or a terminal
		# state is achieved
		done = False

		while not done:
			# Query the environment for the state, and use the policy to get an action
			state = np.array(environment.state)
			action = int(policy(state)[0])		# Single discrete action

			# Sample the next state
			next_state, reward, terminal, _ = environment.step(action)

			states.append(state)
			actions.append(action)
			next_states.append(next_state)
			rewards.append(reward)
			terminals.append(terminal)

			# Done sampling the trajectory?
			done = terminal
			if max_length is not None:
				done = done or len(states) > max_length

		# Convert the collected data to np arrays and create a dataset
		dataset = Dataset(self.schema.copy())
		dataset.add({'state': np.array(states),
					 'action': np.array(actions),
					 'next_state': np.array(next_states),
					 'reward': np.array(rewards),
					 'terminal': np.array(terminals)
					})

		return dataset




